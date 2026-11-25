# export_compact_model.py
import os
import pickle
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
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


class ParametricFieldPINN(nn.Module):
    def __init__(self, geom_dim=4, init_pert_scale=1.59):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64 * 4 * 4),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Softplus()
        )
        self.pert_scale = nn.Parameter(torch.tensor(float(init_pert_scale), dtype=torch.float32))

    def predict_dT_field(self, geom_s, power_mW):
        b = geom_s.shape[0]
        h = self.fc(geom_s).view(b, 64, 4, 4)
        dT_per_mW = self.decoder(h).squeeze(1)
        return dT_per_mW * power_mW.view(b, 1, 1)


def generate_combined_learning_curve(results_dir):
    base_csv = os.path.join(results_dir, "baseline_metrics.csv")
    pinn_csv = os.path.join(results_dir, "pinn_metrics.csv")

    base_df = pd.read_csv(base_csv)
    pinn_df = pd.read_csv(pinn_csv)
    comb_df = pd.concat([base_df, pinn_df[["model", "n_train", "eval_set", "mae", "rmse", "mape_pct", "r2", "train_time_s"]]], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    eval_order = ["ID_Test", "OOD_HighPower", "OOD_GeomExtrap"]
    titles = ["In-Distribution Test (N=150)", "OOD: High Power (60-90 mW)", "OOD: Geometry Extrap"]
    styles = {
        "GP":         {"color": "#1f77b4", "marker": "s", "ls": "-"},
        "Analytic":   {"color": "#7f7f7f", "marker": "^", "ls": "--"},
        "Field_PINN": {"color": "#d62728", "marker": "o", "ls": "-", "lw": 2.6},
        "GBR":        {"color": "#2ca02c", "marker": "D", "ls": ":"},
        "MLP":        {"color": "#ff7f0e", "marker": "v", "ls": "-."},
    }

    for ax, ev_name, title in zip(axes, eval_order, titles):
        sub = comb_df[comb_df["eval_set"] == ev_name]
        for m_name, st in styles.items():
            m_sub = sub[sub["model"] == m_name].sort_values("n_train")
            ax.plot(
                m_sub["n_train"], m_sub["mape_pct"],
                label=m_name,
                color=st["color"],
                marker=st["marker"],
                linestyle=st["ls"],
                linewidth=st.get("lw", 1.8)
            )
        ax.set_title(title)
        ax.set_xlabel("Training Samples (N)")
        ax.set_ylabel("MAPE (%)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, "combined_learning_curves.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def export_veriloga(va_path, coef_dneff, coef_tmax, wavelength_um=1.55):
    """Export a circuit-ready Verilog-A compact model for SPICE/INTERCONNECT."""
    c0, c_w, c_h, c_hw, c_g, c_p = coef_dneff
    t0, t_w, t_h, t_hw, t_g, t_p = coef_tmax

    va_code = f"""// ============================================================================
// Circuit-Ready Compact Model: Silicon Thermo-Optic Phase Shifter
// Auto-generated from Physics-Informed Surrogate Framework
// Compatible with Cadence Spectre / Ansys Lumerical INTERCONNECT
// ============================================================================
`include "constants.vams"
`include "disciplines.vams"

module silicon_to_phase_shifter(v_pos, v_neg, phi_out, temp_max_out);
    inout v_pos, v_neg;
    output phi_out, temp_max_out;
    electrical v_pos, v_neg, phi_out, temp_max_out;

    // Geometric & Material Parameters
    parameter real wg_width_nm     = 450.0  from (300.0:700.0);
    parameter real wg_height_nm    = 220.0  from (180.0:300.0);
    parameter real heater_width_nm = 1200.0 from (600.0:2500.0);
    parameter real heater_gap_nm   = 900.0  from (400.0:2000.0);
    parameter real length_um       = 100.0  from (10.0:1000.0);
    parameter real r_heater_ohm    = 100.0  from (10.0:10000.0);
    parameter real tau_th_us       = 12.0   from (1.0:200.0);
    parameter real wavelength_um   = {wavelength_um};
    parameter real t_ambient_k     = 300.0;

    // Fitted Surrogate Scaling Coefficients (Log-Power Law)
    localparam real C0   = {c0:.8e};
    localparam real C_W  = {c_w:.8e};
    localparam real C_H  = {c_h:.8e};
    localparam real C_HW = {c_hw:.8e};
    localparam real C_G  = {c_g:.8e};
    localparam real C_P  = {c_p:.8e};

    localparam real T0   = {t0:.8e};
    localparam real T_W  = {t_w:.8e};
    localparam real T_H  = {t_h:.8e};
    localparam real T_HW = {t_hw:.8e};
    localparam real T_G  = {t_g:.8e};
    localparam real T_P  = {t_p:.8e};

    real v_drive, p_inst_mw, p_eff_mw;
    real delta_neff, delta_phi_rad, dT_max;

    analog begin
        // Electrical heater resistance
        v_drive = V(v_pos, v_neg);
        I(v_pos, v_neg) <+ v_drive / r_heater_ohm;

        // Instantaneous power (mW) with first-order thermal low-pass response
        p_inst_mw = (v_drive * v_drive / r_heater_ohm) * 1.0e3;
        p_eff_mw  = laplace_nd(p_inst_mw, {{1.0}}, {{1.0, tau_th_us * 1.0e-6}});

        if (p_eff_mw > 1.0e-9) begin
            delta_neff = exp(
                C0
                + C_W  * ln(wg_width_nm)
                + C_H  * ln(wg_height_nm)
                + C_HW * ln(heater_width_nm)
                + C_G  * ln(heater_gap_nm)
                + C_P  * ln(p_eff_mw)
            );
            dT_max = exp(
                T0
                + T_W  * ln(wg_width_nm)
                + T_H  * ln(wg_height_nm)
                + T_HW * ln(heater_width_nm)
                + T_G  * ln(heater_gap_nm)
                + T_P  * ln(p_eff_mw)
            );
        end else begin
            delta_neff = 0.0;
            dT_max     = 0.0;
        end

        // Optical phase shift (rad): dphi = 2 * pi * L * dneff / lambda
        delta_phi_rad = (2.0 * `M_PI * length_um * delta_neff) / wavelength_um;

        V(phi_out)      <+ delta_phi_rad;
        V(temp_max_out) <+ t_ambient_k + dT_max;
    end
endmodule
"""
    with open(va_path, "w", encoding="utf-8") as f:
        f.write(va_code)
    return va_path


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"export_compact_model_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[COMPACT]{CLR_RESET} {CLR_CYAN}Phase 3: Circuit-Ready Compact Model Export & Benchmark{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        # 1. Combined learning curve
        comb_plot = generate_combined_learning_curve(results_dir)
        print(f"{CLR_GREEN}[PLOT]{CLR_RESET}   Combined learning curves -> {CLR_GRAY}{comb_plot}{CLR_RESET}")

        # 2. Load data and splits
        feature_cols = ["wg_width_nm", "wg_height_nm", "heater_width_nm", "heater_gap_nm", "heater_power_mW"]
        id_df = pd.read_csv(os.path.join(project_root, config["dataset"]["scalars_csv"]))
        ood_df = pd.read_csv(os.path.join(project_root, config["ood_dataset"]["scalars_csv"]))
        splits = np.load(os.path.join(project_root, "data", "processed", "split_indices.npz"))
        train_idx = splits["train_pool"][:800]
        test_id_idx = splits["test_id"]

        X_tr = id_df.iloc[train_idx][feature_cols].values
        y_dneff_tr = id_df.iloc[train_idx]["delta_neff"].values
        y_tmax_tr = id_df.iloc[train_idx]["T_max_K"].values - 300.0

        # Fit compact GP models (delta_neff and dT_max) for ultra-fast system co-opt
        print(f"{CLR_MAGENTA}[FIT]{CLR_RESET}    Fitting high-accuracy GP compact models (N=800)...")
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)

        gp_dneff = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(noise_level=1e-5),
            n_restarts_optimizer=2,
            random_state=42
        ).fit(X_tr_s, y_dneff_tr * 1e3)

        gp_tmax = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(noise_level=1e-4),
            n_restarts_optimizer=2,
            random_state=42
        ).fit(X_tr_s, y_tmax_tr)

        # Save Python compact model bundle for Phase 4 MZI mesh co-optimization
        bundle_path = os.path.join(models_dir, "compact_model_bundle.pkl")
        with open(bundle_path, "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "gp_dneff": gp_dneff,
                "gp_tmax": gp_tmax,
                "feature_cols": feature_cols,
                "wavelength_um": config["simulation"]["wavelength_um"]
            }, f)
        print(f"  {CLR_GRAY}|--{CLR_RESET} Saved Python bundle -> {CLR_GRAY}{bundle_path}{CLR_RESET}")

        # 3. Fit log-power-law coefficients for Verilog-A export
        log_X = np.hstack([np.ones((len(X_tr), 1)), np.log(X_tr)])
        coef_dneff, _, _, _ = np.linalg.lstsq(log_X, np.log(np.maximum(y_dneff_tr, 1e-12)), rcond=None)
        coef_tmax, _, _, _  = np.linalg.lstsq(log_X, np.log(np.maximum(y_tmax_tr, 1e-12)), rcond=None)

        va_path = os.path.join(models_dir, "silicon_to_shifter.va")
        export_veriloga(va_path, coef_dneff, coef_tmax, config["simulation"]["wavelength_um"])
        print(f"  {CLR_GRAY}|--{CLR_RESET} Exported Verilog-A  -> {CLR_GRAY}{va_path}{CLR_RESET}")

        # 4. Fidelity & Speedup Benchmark (Python GP + Verilog-A analytic + Field-PINN)
        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[BENCH]{CLR_RESET}  Evaluating Fidelity & Inference Speedup vs Lumerical (~8.00 s/eval)")

        X_ev = id_df.iloc[test_id_idx][feature_cols].values
        y_ev = id_df.iloc[test_id_idx]["delta_neff"].values

        # GP timing (batch of 150)
        t0 = time.perf_counter()
        for _ in range(100):
            pred_gp = gp_dneff.predict(scaler.transform(X_ev)) * 1e-3
        gp_ms_per_sample = ((time.perf_counter() - t0) / (100 * len(X_ev))) * 1e3
        gp_mape = float(np.mean(np.abs(y_ev - pred_gp) / y_ev) * 100.0)
        gp_r2 = float(r2_score(y_ev, pred_gp))

        # Verilog-A analytic formula timing
        t0 = time.perf_counter()
        for _ in range(1000):
            pred_va = np.exp(np.hstack([np.ones((len(X_ev), 1)), np.log(X_ev)]) @ coef_dneff)
        va_us_per_sample = ((time.perf_counter() - t0) / (1000 * len(X_ev))) * 1e6
        va_mape = float(np.mean(np.abs(y_ev - pred_va) / y_ev) * 100.0)
        va_r2 = float(r2_score(y_ev, pred_va))

        # Field-PINN timing (full 64x64 2D temperature field)
        ckpt = torch.load(os.path.join(models_dir, "field_pinn_N800.pt"), map_location="cpu", weights_only=False)
        pinn = ParametricFieldPINN(geom_dim=4)
        pinn.load_state_dict(ckpt["model_state_dict"])
        pinn.eval()
        G_s = torch.tensor((X_ev[:, :4] - ckpt["scaler_mean"]) / ckpt["scaler_scale"], dtype=torch.float32)
        P_t = torch.tensor(X_ev[:, 4], dtype=torch.float32)

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(50):
                _ = pinn.predict_dT_field(G_s, P_t)
            pinn_ms_per_sample = ((time.perf_counter() - t0) / (50 * len(X_ev))) * 1e3

        lumerical_s = float(id_df["runtime_s"].mean())
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}GP Compact{CLR_RESET}   : "
            f"MAPE={gp_mape:.3f}% (R2={gp_r2:.6f}) | {gp_ms_per_sample:.4f} ms/eval | "
            f"Speedup={CLR_BOLD}{lumerical_s / (gp_ms_per_sample * 1e-3):,.0f}x{CLR_RESET}"
        )
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Verilog-A{CLR_RESET}    : "
            f"MAPE={va_mape:.3f}% (R2={va_r2:.6f}) | {va_us_per_sample:.3f} us/eval | "
            f"Speedup={CLR_BOLD}{lumerical_s / (va_us_per_sample * 1e-6):,.0f}x{CLR_RESET}"
        )
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Field-PINN 2D{CLR_RESET}: "
            f"64x64 T(x,y) field | {pinn_ms_per_sample:.4f} ms/eval | "
            f"Speedup={CLR_BOLD}{lumerical_s / (pinn_ms_per_sample * 1e-3):,.0f}x{CLR_RESET}"
        )

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Phase 3 complete. Ready for Phase 4 MZI Mesh Co-Optimization.")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()