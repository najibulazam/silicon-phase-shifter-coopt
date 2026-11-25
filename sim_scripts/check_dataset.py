# check_dataset.py
import os
import re
import sys
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

os.system('')

CLR_RESET   = "\033[0m"
CLR_BOLD    = "\033[1m"
CLR_CYAN    = "\033[1;36m"
CLR_GREEN   = "\033[1;32m"
CLR_YELLOW  = "\033[1;33m"
CLR_BLUE    = "\033[1;34m"
CLR_MAGENTA = "\033[1;35m"
CLR_RED     = "\033[1;31m"
CLR_GRAY    = "\033[90m"

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class TeeLogger:
    """Writes colored output to terminal and clean text to log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(ANSI_ESCAPE.sub('', message))
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def evaluate_dataset(name, csv_path, h5_path, dn_dT_si, dn_dT_sio2, T_ref=300.0):
    print(f"{CLR_MAGENTA}[CHECK]{CLR_RESET}  Evaluating {CLR_BOLD}{name}{CLR_RESET} dataset...")

    df = pd.read_csv(csv_path)
    with h5py.File(h5_path, "r") as hf:
        x_grid = hf["x_grid"][:]
        y_grid = hf["y_grid"][:]
        T_fields = hf["T_fields"][:]
        E2_fields = hf["E2_fields"][:]

    n_csv = len(df)
    n_h5 = T_fields.shape[0]
    nan_csv = int(df.isna().sum().sum())
    nan_T = int(np.isnan(T_fields).sum())
    nan_E2 = int(np.isnan(E2_fields).sum())
    neg_dneff = int((df["delta_neff"] <= 0).sum())

    print(f"  {CLR_GRAY}|--{CLR_RESET} Samples  : CSV={n_csv}, H5={n_h5}, Grid={T_fields.shape[1]}x{T_fields.shape[2]}")
    print(f"  {CLR_GRAY}|--{CLR_RESET} Integrity: CSV_NaN={nan_csv}, T_NaN={nan_T}, E2_NaN={nan_E2}, NonPos_dneff={neg_dneff}")
    print(
        f"  {CLR_GRAY}|--{CLR_RESET} Ranges   : "
        f"dneff=[{df['delta_neff'].min():.3e}, {df['delta_neff'].max():.3e}] | "
        f"T_max=[{df['T_max_K'].min():.1f}K, {df['T_max_K'].max():.1f}K]"
    )

    # First-order perturbation integral verification
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    dneff_pert = np.zeros(n_csv, dtype=np.float64)

    for i in range(n_csv):
        wg_w = df.loc[i, "wg_width_nm"] * 1e-9
        wg_h = df.loc[i, "wg_height_nm"] * 1e-9

        wg_mask = (np.abs(X) <= wg_w / 2.0) & (Y >= 0.0) & (Y <= wg_h)
        dn_dT_map = np.where(wg_mask, dn_dT_si, dn_dT_sio2)
        dT = T_fields[i] - T_ref
        dneff_pert[i] = float(np.sum(dn_dT_map * dT * E2_fields[i]))

    dneff_true = df["delta_neff"].values
    err = dneff_pert - dneff_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(dneff_true), 1e-12)) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((dneff_true - np.mean(dneff_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mean_ratio = float(np.mean(dneff_true / np.maximum(dneff_pert, 1e-12)))

    print(
        f"  {CLR_GRAY}|--{CLR_RESET} {CLR_BLUE}Perturbation Layer vs Lumerical FDE:{CLR_RESET}\n"
        f"        MAE={mae:.4e} | RMSE={rmse:.4e} | MAPE={mape:.2f}% | R2={r2:.6f} | Ratio(True/Pert)={mean_ratio:.4f}"
    )

    return df, x_grid, y_grid, T_fields, E2_fields, dneff_true, dneff_pert


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"check_dataset_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        dn_dT_si = config["simulation"]["dn_dT_si"]
        dn_dT_sio2 = config["simulation"]["dn_dT_sio2"]

        print(f"{CLR_CYAN}" + "=" * 68)
        print(f" {CLR_BOLD}[SANITY]{CLR_RESET} {CLR_CYAN}Dataset Integrity & Perturbation Layer Verification{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 68 + f"{CLR_RESET}")

        id_res = evaluate_dataset(
            "In-Distribution (ID)",
            os.path.join(project_root, config["dataset"]["scalars_csv"]),
            os.path.join(project_root, config["dataset"]["fields_h5"]),
            dn_dT_si,
            dn_dT_sio2
        )
        print(f"{CLR_GRAY}" + "-" * 68 + f"{CLR_RESET}")

        ood_res = evaluate_dataset(
            "Out-of-Distribution (OOD)",
            os.path.join(project_root, config["ood_dataset"]["scalars_csv"]),
            os.path.join(project_root, config["ood_dataset"]["fields_h5"]),
            dn_dT_si,
            dn_dT_sio2
        )

        # Save diagnostic plot
        df_id, x_grid, y_grid, T_id, E2_id, id_true, id_pert = id_res
        _, _, _, _, _, ood_true, ood_pert = ood_res

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # 1. Parity plot
        axes[0].scatter(id_true * 1e3, id_pert * 1e3, s=10, alpha=0.5, label="ID (N=1200)")
        axes[0].scatter(ood_true * 1e3, ood_pert * 1e3, s=14, alpha=0.7, label="OOD (N=200)")
        lim = max(id_true.max(), ood_true.max(), id_pert.max(), ood_pert.max()) * 1e3 * 1.05
        axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        axes[0].set_xlabel(r"Lumerical FDE $\Delta n_{\mathrm{eff}}$ ($\times 10^{-3}$)")
        axes[0].set_ylabel(r"Perturbation Integral $\Delta n_{\mathrm{eff}}$ ($\times 10^{-3}$)")
        axes[0].set_title("Perturbation Layer Verification")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Sample Temperature field
        extent = [x_grid[0] * 1e6, x_grid[-1] * 1e6, y_grid[0] * 1e6, y_grid[-1] * 1e6]
        im1 = axes[1].imshow(T_id[0].T, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        axes[1].set_title(f"Sample 0: T(x,y) [K] (P={df_id.loc[0, 'heater_power_mW']:.1f} mW)")
        axes[1].set_xlabel("x (um)")
        axes[1].set_ylabel("y (um)")
        plt.colorbar(im1, ax=axes[1])

        # 3. Sample Mode intensity field
        im2 = axes[2].imshow(E2_id[0].T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        axes[2].set_title("Sample 0: Normalized |E(x,y)|^2")
        axes[2].set_xlabel("x (um)")
        axes[2].set_ylabel("y (um)")
        plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "dataset_sanity_check.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 68)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Diagnostic plot saved to: {CLR_GRAY}{plot_path}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Saved to: {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 68 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()