# run_pn_transfer.py
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
import torch.nn.functional as F
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
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


class PNDepletionPhysicsNet(nn.Module):
    """
    Physics-Informed Surrogate for Silicon PN Carrier-Depletion Phase Shifter.
    Combines a residual geometry/bias MLP with a depletion-scaling prior:
      pred_1e4 = base_scale * ((v_bi + v_rev)^0.5 - v_bi^0.5) + residual(geom, v_rev)
    """
    def __init__(self, in_dim=5):
        super().__init__()
        self.res_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        nn.init.normal_(self.res_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.res_mlp[-1].bias)
        self.v_bi = nn.Parameter(torch.tensor(0.82, dtype=torch.float32))
        self.k_dep = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))

    def forward(self, x_s, v_rev):
        v_pos = torch.clamp(v_rev, min=0.0)
        vbi = torch.clamp(self.v_bi, min=0.5, max=1.2)
        dep_prior = self.k_dep * (torch.sqrt(vbi + v_pos) - torch.sqrt(vbi))
        res = self.res_mlp(x_s).squeeze(-1)
        pred_1e4 = torch.where(v_pos > 1e-9, F.softplus(dep_prior + res), torch.zeros_like(v_pos))
        return pred_1e4 * 1e-4


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 1e-8
    mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, mape, r2


def export_pn_veriloga(va_path, gp_alpha_mean, v_bi=0.82, p_exp=0.50, wavelength_um=1.55):
    """Export circuit-ready Verilog-A compact model for PN carrier-depletion phase shifter."""
    va_code = f"""// ============================================================================
// Circuit-Ready Compact Model: Silicon PN Carrier-Depletion Phase Shifter
// Auto-generated from Cross-Platform Physics-Informed Surrogate Framework
// Compatible with Cadence Spectre / Ansys Lumerical INTERCONNECT
// ============================================================================
`include "constants.vams"
`include "disciplines.vams"

module silicon_pn_phase_shifter(anode, cathode, phi_out, loss_db_out);
    inout anode, cathode;
    output phi_out, loss_db_out;
    electrical anode, cathode, phi_out, loss_db_out;

    parameter real length_mm      = 4.0   from (0.5:10.0);
    parameter real c_j0_pf_mm     = 0.25  from (0.05:2.0);
    parameter real r_series_ohm   = 25.0  from (1.0:500.0);
    parameter real alpha0_db_cm   = 12.0  from (1.0:50.0);
    parameter real wavelength_um  = {wavelength_um};

    localparam real ALPHA_MOD = {gp_alpha_mean:.8e};
    localparam real V_BI      = {v_bi:.6f};
    localparam real P_EXP     = {p_exp:.6f};

    real v_rev, c_j, delta_neff, delta_phi_rad, total_loss_db;

    analog begin
        v_rev = max(0.0, V(cathode, anode));
        c_j = (c_j0_pf_mm * 1.0e-12 * length_mm) / pow(1.0 + v_rev / V_BI, P_EXP);
        I(cathode, anode) <+ ddt(c_j * V(cathode, anode));

        delta_neff = ALPHA_MOD * (pow(V_BI + v_rev, P_EXP) - pow(V_BI, P_EXP));
        delta_phi_rad = (2.0 * `M_PI * (length_mm * 1000.0) * delta_neff) / wavelength_um;
        total_loss_db = (alpha0_db_cm - 0.35 * v_rev) * (length_mm * 0.1);

        V(phi_out)     <+ delta_phi_rad;
        V(loss_db_out) <+ total_loss_db;
    end
endmodule
"""
    with open(va_path, "w", encoding="utf-8") as f:
        f.write(va_code)
    return va_path


class PNMeshCoOptProblem(Problem):
    """
    PN Carrier-Depletion 4x4 Clements MZI Mesh Co-Optimization.
    Decision variables (6):
      x[0]: wg_width_nm  [450, 550]
      x[1]: p_conc_1e23  [2.0, 6.0]
      x[2]: n_conc_1e23  [2.0, 6.0]
      x[3]: pn_offset_nm [-60, 60]
      x[4]: length_mm    [1.0, 8.0]
      x[5]: dac_bits     [6, 12]
    Objectives (3):
      f[0]: Required Drive Voltage V_pi (V)
      f[1]: Mesh Optical Insertion Loss (dB) + Quantization Infidelity Penalty
      f[2]: Active Mesh Footprint (mm^2)
    Constraint (1):
      g[0]: V_pi - 4.0 <= 0  (CMOS-compatible driver voltage limit <= 4.0 V)
    """
    def __init__(self, scaler, gp_dneff, alpha_scale, wavelength_um=1.55):
        super().__init__(
            n_var=6,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.array([450.0, 2.0, 2.0, -60.0, 1.0, 6.0]),
            xu=np.array([550.0, 6.0, 6.0,  60.0, 8.0, 12.0])
        )
        self.scaler = scaler
        self.gp_dneff = gp_dneff
        self.alpha_scale = alpha_scale
        self.wavelength_um = wavelength_um

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        v_grid = np.linspace(0.5, 4.0, 15)

        f1_vpi = np.zeros(n)
        f2_loss_infid = np.zeros(n)
        f3_area = np.zeros(n)
        g1_vlim = np.zeros(n)

        for i in range(n):
            geom = X[i, :4]
            p_c = X[i, 1]
            n_c = X[i, 2]
            L_mm = X[i, 4]
            dac_bits = int(np.round(X[i, 5]))

            dneff_req = self.wavelength_um / (2.0 * (L_mm * 1000.0))

            q_mat = np.column_stack([np.tile(geom, (len(v_grid), 1)), v_grid])
            q_s = self.scaler.transform(q_mat)
            dneff_curve = np.maximum(self.gp_dneff.predict(q_s) * 1e-4, 1e-9)

            if dneff_curve[-1] >= dneff_req:
                v_pi = float(np.interp(dneff_req, dneff_curve, v_grid))
            else:
                v_pi = 4.0 + 3.0 * (dneff_req - dneff_curve[-1]) / max(dneff_curve[-1], 1e-6)

            # Free-carrier absorption scales with doping concentration & reverse bias
            alpha_db_cm = self.alpha_scale * (0.5 * (p_c + n_c) / 4.0) * max(0.4, 1.0 - 0.08 * v_pi)
            mesh_il_db = 4.0 * alpha_db_cm * (L_mm * 0.1)
            dac_err_penalty_db = 18.0 * (np.pi / (2.0 ** dac_bits)) ** 2
            area_mm2 = (4.0 * (L_mm + 0.1)) * (4.0 * 0.025)

            f1_vpi[i] = v_pi
            f2_loss_infid[i] = mesh_il_db + dac_err_penalty_db
            f3_area[i] = area_mm2
            g1_vlim[i] = v_pi - 4.0

        out["F"] = np.column_stack([f1_vpi, f2_loss_infid, f3_area])
        out["G"] = g1_vlim.reshape(-1, 1)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    log_dir = os.path.join(project_root, "log")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_pn_transfer_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[PHASE 5]{CLR_RESET} {CLR_CYAN}RQ4 Cross-Platform Transfer: PN Carrier-Depletion Shifter{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        df = pd.read_csv(os.path.join(raw_dir, "pn_depletion_sweep.csv"))
        df["v_rev"] = -df["v_drain_V"]

        # Normalize loss_dB_cm if stored in dB/m (> 100) vs dB/cm
        raw_loss = df["loss_dB_cm"].abs()
        if raw_loss.median() > 100.0:
            loss_db_cm = raw_loss * 1e-2
            alpha_scale = float(np.clip(loss_db_cm.median(), 5.0, 25.0))
        else:
            alpha_scale = float(np.clip(raw_loss.median(), 5.0, 25.0))

        rng = np.random.default_rng(42)
        case_ids = rng.permutation(df["case_id"].unique())
        tr_cases = set(case_ids[:18])
        te_cases = set(case_ids[18:])

        tr_df = df[df["case_id"].isin(tr_cases)].reset_index(drop=True)
        te_df = df[df["case_id"].isin(te_cases)].reset_index(drop=True)

        feat_cols = ["wg_width_nm", "p_conc_1e23", "n_conc_1e23", "pn_offset_nm", "v_rev"]

        X_tr = tr_df[feat_cols].values
        y_tr = tr_df["abs_delta_neff"].values
        X_te = te_df[feat_cols].values
        y_te = te_df["abs_delta_neff"].values

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        print(f"{CLR_BLUE}[DATA]{CLR_RESET}   PN dataset: {len(df)} samples | Train={len(tr_df)} (18 cases) | Test={len(te_df)} (7 cases)")
        print(f"  {CLR_GRAY}|--{CLR_RESET} |delta_neff(-4V)| range: [{df[df['v_rev']==4.0]['abs_delta_neff'].min():.3e}, {df[df['v_rev']==4.0]['abs_delta_neff'].max():.3e}]")
        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")

        bench_records = []

        # 1. GP
        t0 = time.time()
        gp_dneff = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(noise_level=1e-6),
            n_restarts_optimizer=3, random_state=42
        ).fit(X_tr_s, y_tr * 1e4)
        pred_gp = gp_dneff.predict(X_te_s) * 1e-4
        mae, rmse, mape, r2 = compute_metrics(y_te, pred_gp)
        bench_records.append({"model": "GP", "mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2, "time_s": round(time.time() - t0, 2)})
        print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}GP Surrogate{CLR_RESET}   : MAPE={mape:.3f}% | R2={r2:.5f}")

        # 2. GBR
        t0 = time.time()
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42).fit(X_tr_s, y_tr * 1e4)
        pred_gbr = gbr.predict(X_te_s) * 1e-4
        mae, rmse, mape, r2 = compute_metrics(y_te, pred_gbr)
        bench_records.append({"model": "GBR", "mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2, "time_s": round(time.time() - t0, 2)})
        print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}GBR{CLR_RESET}            : MAPE={mape:.3f}% | R2={r2:.5f}")

        # 3. Standard MLP
        t0 = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1500, random_state=42).fit(X_tr_s, y_tr * 1e4)
        pred_mlp = mlp.predict(X_te_s) * 1e-4
        mae, rmse, mape, r2 = compute_metrics(y_te, pred_mlp)
        bench_records.append({"model": "MLP", "mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2, "time_s": round(time.time() - t0, 2)})
        print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Standard MLP{CLR_RESET}   : MAPE={mape:.3f}% | R2={r2:.5f}")

        # 4. Physics-Informed PN Depletion Net
        t0 = time.time()
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        V_tr_t = torch.tensor(tr_df["v_rev"].values, dtype=torch.float32)
        Y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

        X_te_t = torch.tensor(X_te_s, dtype=torch.float32)
        V_te_t = torch.tensor(te_df["v_rev"].values, dtype=torch.float32)

        pn_pinn = PNDepletionPhysicsNet(in_dim=5)
        opt = torch.optim.AdamW(pn_pinn.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

        for _ in range(1000):
            opt.zero_grad()
            pred = pn_pinn(X_tr_t, V_tr_t)
            loss = F.mse_loss(pred * 1e4, Y_tr_t * 1e4)
            loss.backward()
            opt.step()
            scheduler.step()

        pn_pinn.eval()
        with torch.no_grad():
            pred_pinn = pn_pinn(X_te_t, V_te_t).numpy()
            fitted_vbi = float(pn_pinn.v_bi.item())
        mae, rmse, mape, r2 = compute_metrics(y_te, pred_pinn)
        bench_records.append({"model": "PN_PhysicsNet", "mae": mae, "rmse": rmse, "mape_pct": mape, "r2": r2, "time_s": round(time.time() - t0, 2)})
        print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}PN_PhysicsNet{CLR_RESET}  : MAPE={mape:.3f}% | R2={r2:.5f} (V_bi={fitted_vbi:.2f}V)")

        pd.DataFrame(bench_records).to_csv(os.path.join(results_dir, "pn_surrogate_metrics.csv"), index=False)

        # Export PN Verilog-A Compact Model & Benchmark Speedup
        va_path = os.path.join(models_dir, "silicon_pn_shifter.va")
        mean_alpha = float(np.mean(df[df["v_rev"] == 4.0]["abs_delta_neff"] / (np.sqrt(fitted_vbi + 4.0) - np.sqrt(fitted_vbi))))
        export_pn_veriloga(va_path, mean_alpha, v_bi=fitted_vbi, p_exp=0.50)

        t0 = time.perf_counter()
        for _ in range(100):
            _ = gp_dneff.predict(X_te_s)
        gp_ms = ((time.perf_counter() - t0) / (100 * len(X_te_s))) * 1e3
        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[EXPORT]{CLR_RESET} PN Verilog-A -> {CLR_GRAY}{va_path}{CLR_RESET}")
        print(f"  {CLR_GRAY}|--{CLR_RESET} GP Inference: {gp_ms:.4f} ms/eval (Speedup vs Lumerical ~19s/pt = {CLR_BOLD}{19000.0 / gp_ms:,.0f}x{CLR_RESET})")

        # PN 4x4 MZI Mesh Co-Optimization
        problem = PNMeshCoOptProblem(scaler, gp_dneff, alpha_scale=alpha_scale, wavelength_um=1.55)
        algorithm = NSGA2(pop_size=80, sampling=LHS(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20), eliminate_duplicates=True)
        res = minimize(problem, algorithm, ("n_gen", 60), seed=42, verbose=False)

        if res.X is not None and len(res.X) > 0:
            X_sol, F_sol = res.X, res.F
        else:
            X_sol, F_sol = res.pop.get("X"), res.pop.get("F")

        pn_pareto = pd.DataFrame({
            "pareto_id": np.arange(len(X_sol)),
            "wg_width_nm": np.round(X_sol[:, 0], 2),
            "p_conc_1e23": np.round(X_sol[:, 1], 3),
            "n_conc_1e23": np.round(X_sol[:, 2], 3),
            "pn_offset_nm": np.round(X_sol[:, 3], 2),
            "length_mm": np.round(X_sol[:, 4], 3),
            "dac_bits": np.round(X_sol[:, 5]).astype(int),
            "V_pi_V": np.round(F_sol[:, 0], 3),
            "mesh_loss_plus_quant_dB": np.round(F_sol[:, 1], 3),
            "footprint_mm2": np.round(F_sol[:, 2], 4),
        }).sort_values("V_pi_V").reset_index(drop=True)
        pn_pareto["pareto_id"] = np.arange(len(pn_pareto))
        pn_pareto.to_csv(os.path.join(results_dir, "pn_pareto_front.csv"), index=False)

        print(f"{CLR_MAGENTA}[CO-OPT]{CLR_RESET} Evaluated {CLR_BOLD}{len(pn_pareto)}{CLR_RESET} PN mesh designs:")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Min V_pi   : {pn_pareto['V_pi_V'].min():.3f} V (L={pn_pareto.loc[pn_pareto['V_pi_V'].idxmin(), 'length_mm']:.2f} mm, Loss+Q={pn_pareto.loc[pn_pareto['V_pi_V'].idxmin(), 'mesh_loss_plus_quant_dB']:.2f} dB)")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Min Loss+Q : {pn_pareto['mesh_loss_plus_quant_dB'].min():.3f} dB (V_pi={pn_pareto.loc[pn_pareto['mesh_loss_plus_quant_dB'].idxmin(), 'V_pi_V']:.3f} V, L={pn_pareto.loc[pn_pareto['mesh_loss_plus_quant_dB'].idxmin(), 'length_mm']:.2f} mm)")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Min Area   : {pn_pareto['footprint_mm2'].min():.4f} mm^2 (V_pi={pn_pareto.loc[pn_pareto['footprint_mm2'].idxmin(), 'V_pi_V']:.3f} V)")

        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

        axes[0].scatter(y_te * 1e4, pred_gp * 1e4, s=42, alpha=0.85, label="GP Surrogate", c="#1f77b4")
        axes[0].scatter(y_te * 1e4, pred_pinn * 1e4, s=42, alpha=0.85, marker="^", label="PN_PhysicsNet", c="#d62728")
        axes[0].scatter(y_te * 1e4, pred_mlp * 1e4, s=36, alpha=0.65, marker="v", label="Standard MLP", c="#ff7f0e")
        lim = max(y_te.max(), pred_gp.max()) * 1e4 * 1.08
        axes[0].plot([0, lim], [0, lim], "k--", lw=1.2)
        axes[0].set_xlabel(r"Lumerical CHARGE+FDE $|\Delta n_{\mathrm{eff}}|$ ($\times 10^{-4}$)")
        axes[0].set_ylabel(r"Predicted $|\Delta n_{\mathrm{eff}}|$ ($\times 10^{-4}$)")
        axes[0].set_title("PN Carrier-Depletion Surrogate Parity")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        sc = axes[1].scatter(
            pn_pareto["V_pi_V"], pn_pareto["mesh_loss_plus_quant_dB"],
            c=pn_pareto["footprint_mm2"], cmap="viridis", s=45, edgecolors="k", linewidths=0.4
        )
        axes[1].set_xlabel(r"Drive Voltage $V_\pi$ (V)")
        axes[1].set_ylabel("4x4 Mesh Loss + Quantization Penalty (dB)")
        axes[1].set_title("PN 4x4 MZI Mesh Co-Optimization Pareto Front")
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[1], label="Mesh Footprint (mm^2)")

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "pn_transfer_results.png")
        plt.savefig(plot_out, dpi=220)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Phase 5 (RQ4) complete -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()