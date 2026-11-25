# optical_ai_benchmark.py
import os
import re
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
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


def build_clements_torch(thetas, phis):
    """Differentiable 4x4 Clements MZI mesh unitary matrix in PyTorch."""
    pairs = [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2)]
    U = torch.eye(4, dtype=torch.complex64, device=thetas.device)
    for k, (i, j) in enumerate(pairs):
        th = thetas[k]
        ph = phis[k]
        c = torch.cos(th / 2.0)
        s = torch.sin(th / 2.0)
        pref = 1j * torch.exp(1j * th / 2.0)
        e_ph = torch.exp(1j * ph)

        M = torch.eye(4, dtype=torch.complex64, device=thetas.device)
        M[i, i] = pref * e_ph * s
        M[i, j] = pref * c
        M[j, i] = pref * e_ph * c
        M[j, j] = -pref * s
        U = M @ U
    return U


class PhotonicMZIClassifier(nn.Module):
    """
    4-input Optical Neural Network using two cascaded 4x4 Clements MZI meshes
    with photodetector intensity nonlinearity (|E|^2) and 4->3 readout.
    """
    def __init__(self):
        super().__init__()
        self.theta1 = nn.Parameter(torch.rand(6) * np.pi)
        self.phi1   = nn.Parameter(torch.rand(6) * 2.0 * np.pi)
        self.theta2 = nn.Parameter(torch.rand(6) * np.pi)
        self.phi2   = nn.Parameter(torch.rand(6) * 2.0 * np.pi)
        self.readout = nn.Linear(4, 3)

    def forward(self, x, th1=None, ph1=None, th2=None, ph2=None):
        t1 = self.theta1 if th1 is None else th1
        p1 = self.phi1   if ph1 is None else ph1
        t2 = self.theta2 if th2 is None else th2
        p2 = self.phi2   if ph2 is None else ph2

        U1 = build_clements_torch(t1, p1)
        U2 = build_clements_torch(t2, p2)

        x_c = x.to(torch.complex64)
        h1 = torch.matmul(x_c, U1.T)
        h1_det = torch.abs(h1) ** 2  # Photodetector intensity nonlinearity

        h2 = torch.matmul(h1_det.to(torch.complex64), U2.T)
        h2_det = torch.abs(h2) ** 2
        return self.readout(h2_det)


def apply_hardware_impairments(phase_tensor, dac_bits, xt_coeff):
    """Apply DAC quantization and nearest-neighbor thermal crosstalk to 6 MZI phases."""
    p = torch.remainder(phase_tensor.detach().clone(), 2.0 * np.pi)
    step = (2.0 * np.pi) / (2.0 ** dac_bits)
    p_q = torch.round(p / step) * step

    p_hw = p_q.clone()
    for k in range(6):
        left_nb  = p_q[k - 1] if k > 0 else 0.0
        right_nb = p_q[k + 1] if k < 5 else 0.0
        p_hw[k] = p_q[k] + xt_coeff * (left_nb + right_nb)
    return p_hw


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"optical_ai_benchmark_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[OPTICAL AI]{CLR_RESET} {CLR_CYAN}4x4 Clements MZI Mesh Classification Benchmark (Iris){CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        # 1. Load & normalize Iris dataset (4 features -> matches 4x4 MZI mesh)
        iris = load_iris()
        X_tr, X_te, y_tr, y_te = train_test_split(
            iris.data, iris.target, test_size=0.30, random_state=42, stratify=iris.target
        )
        scaler = StandardScaler().fit(X_tr)
        X_tr_t = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
        X_te_t = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        y_te_t = torch.tensor(y_te, dtype=torch.long)

        # 2. Train ideal photonic MZI classifier
        print(f"{CLR_MAGENTA}[TRAIN]{CLR_RESET}  Training ideal 4x4 Clements MZI classifier (400 epochs)...")
        t0 = time.time()
        model = PhotonicMZIClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(400):
            optimizer.zero_grad()
            logits = model(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            ideal_preds = model(X_te_t).argmax(dim=1)
            ideal_acc = float((ideal_preds == y_te_t).float().mean().item() * 100.0)
        print(f"  {CLR_GRAY}|--{CLR_RESET} Ideal Software Accuracy: {CLR_GREEN}{ideal_acc:.2f}%{CLR_RESET} ({time.time() - t0:.2f} s)")

        # 3. Deploy onto all 80 Pareto-optimal hardware designs
        pareto_csv = os.path.join(results_dir, "pareto_front.csv")
        pareto_df = pd.read_csv(pareto_csv)

        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[DEPLOY]{CLR_RESET} Evaluating trained phases across {CLR_BOLD}{len(pareto_df)}{CLR_RESET} Pareto hardware designs...")

        hw_accs = []
        acc_drops = []
        xt_coeffs = []

        with torch.no_grad():
            for _, row in pareto_df.iterrows():
                dac_bits = int(row["dac_bits"])
                gap_nm   = float(row["heater_gap_nm"])
                pitch_u  = float(row["pitch_um"])

                L_th_um  = 11.0 + 2.5 * (gap_nm / 1000.0)
                xt_coeff = 0.18 * np.exp(-pitch_u / L_th_um)

                th1_hw = apply_hardware_impairments(model.theta1, dac_bits, xt_coeff)
                ph1_hw = apply_hardware_impairments(model.phi1,   dac_bits, xt_coeff * 0.5)
                th2_hw = apply_hardware_impairments(model.theta2, dac_bits, xt_coeff)
                ph2_hw = apply_hardware_impairments(model.phi2,   dac_bits, xt_coeff * 0.5)

                hw_preds = model(X_te_t, th1_hw, ph1_hw, th2_hw, ph2_hw).argmax(dim=1)
                hw_acc = float((hw_preds == y_te_t).float().mean().item() * 100.0)

                hw_accs.append(round(hw_acc, 2))
                acc_drops.append(round(ideal_acc - hw_acc, 2))
                xt_coeffs.append(round(xt_coeff, 5))

        pareto_df["xt_coeff"] = xt_coeffs
        pareto_df["hw_accuracy_pct"] = hw_accs
        pareto_df["accuracy_drop_pct"] = acc_drops

        csv_out = os.path.join(results_dir, "optical_ai_results.csv")
        pareto_df.to_csv(csv_out, index=False)

        # Summary highlights
        best_acc_row = pareto_df.loc[pareto_df["hw_accuracy_pct"].idxmax()]
        worst_acc_row = pareto_df.loc[pareto_df["hw_accuracy_pct"].idxmin()]
        # Best power design that maintains zero accuracy drop (or within 2.5% of ideal)
        high_acc_df = pareto_df[pareto_df["accuracy_drop_pct"] <= 2.5]
        opt_eff_row = high_acc_df.loc[high_acc_df["total_mesh_power_mW"].idxmin()] if len(high_acc_df) > 0 else best_acc_row

        print(f"  {CLR_GRAY}|--{CLR_RESET} Best HW Accuracy     : {CLR_GREEN}{best_acc_row['hw_accuracy_pct']:.2f}%{CLR_RESET} (PID={int(best_acc_row['pareto_id']):02d}, Power={best_acc_row['total_mesh_power_mW']:.1f}mW, DAC={int(best_acc_row['dac_bits'])}b, Pitch={best_acc_row['pitch_um']:.1f}um)")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Worst HW Accuracy    : {CLR_YELLOW}{worst_acc_row['hw_accuracy_pct']:.2f}%{CLR_RESET} (PID={int(worst_acc_row['pareto_id']):02d}, Power={worst_acc_row['total_mesh_power_mW']:.1f}mW, DAC={int(worst_acc_row['dac_bits'])}b, Pitch={worst_acc_row['pitch_um']:.1f}um)")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Optimal Sweet-Spot   : {CLR_CYAN}{opt_eff_row['hw_accuracy_pct']:.2f}%{CLR_RESET} (PID={int(opt_eff_row['pareto_id']):02d}, Power={opt_eff_row['total_mesh_power_mW']:.1f}mW, Area={opt_eff_row['footprint_mm2']:.4f}mm^2, DAC={int(opt_eff_row['dac_bits'])}b)")

        # 4. Plot 2-panel Optical AI Trade-off figure
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        sc0 = axes[0].scatter(
            pareto_df["total_mesh_power_mW"], pareto_df["hw_accuracy_pct"],
            c=pareto_df["dac_bits"], cmap="viridis", s=45, edgecolors="k", linewidths=0.4
        )
        axes[0].axhline(ideal_acc, color="r", linestyle="--", lw=1.3, label=f"Ideal Software ({ideal_acc:.1f}%)")
        axes[0].set_xlabel("Total 4x4 Mesh + Driver Power (mW)")
        axes[0].set_ylabel("Optical AI Test Accuracy (%)")
        axes[0].set_title("Classification Accuracy vs. System Power")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="lower right")
        plt.colorbar(sc0, ax=axes[0], label="DAC Resolution (bits)")

        sc1 = axes[1].scatter(
            pareto_df["footprint_mm2"], pareto_df["hw_accuracy_pct"],
            c=pareto_df["pitch_um"], cmap="plasma", s=45, edgecolors="k", linewidths=0.4
        )
        axes[1].axhline(ideal_acc, color="r", linestyle="--", lw=1.3, label=f"Ideal Software ({ideal_acc:.1f}%)")
        axes[1].set_xlabel("Active Mesh Footprint (mm^2)")
        axes[1].set_ylabel("Optical AI Test Accuracy (%)")
        axes[1].set_title("Classification Accuracy vs. Chip Area (Crosstalk Effect)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right")
        plt.colorbar(sc1, ax=axes[1], label="Heater Pitch (um)")

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "optical_ai_accuracy_tradeoff.png")
        plt.savefig(plot_out, dpi=220)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Optical AI CSV  -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Optical AI Plot -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved       -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()