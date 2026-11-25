# nsga2_coopt.py
import json
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
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

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


def mzi_2x2(theta, phi):
    """2x2 Mach-Zehnder Interferometer transfer matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    e_phi = np.exp(1j * phi)
    return 1j * np.exp(1j * theta / 2.0) * np.array([
        [e_phi * s, c],
        [e_phi * c, -s]
    ], dtype=np.complex128)


def build_clements_4x4(thetas, phis):
    """4x4 Clements rectangular MZI mesh (6 MZIs, 12 phase shifters)."""
    pairs = [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3), (1, 2)]
    U = np.eye(4, dtype=np.complex128)
    for k, (i, j) in enumerate(pairs):
        M = np.eye(4, dtype=np.complex128)
        sub = mzi_2x2(thetas[k], phis[k])
        M[i, i], M[i, j] = sub[0, 0], sub[0, 1]
        M[j, i], M[j, j] = sub[1, 0], sub[1, 1]
        U = M @ U
    return U


def evaluate_mesh_infidelity(target_configs, dac_bits, xt_coeff):
    """
    Evaluate mean unitary infidelity (1 - F) across target 4x4 mesh configs
    under DAC phase quantization and Lumerical-fitted nearest-neighbor thermal crosstalk.
    """
    n_levels = 2.0 ** dac_bits
    step = 2.0 * np.pi / n_levels
    infidelities = []

    for thetas, phis, U_ideal in target_configs:
        th_q = np.round(thetas / step) * step
        ph_q = np.round(phis / step) * step

        th_actual = th_q.copy()
        ph_actual = ph_q.copy()
        for k in range(6):
            left_nb  = th_q[k - 1] if k > 0 else 0.0
            right_nb = th_q[k + 1] if k < 5 else 0.0
            th_actual[k] += xt_coeff * (left_nb + right_nb)
            ph_actual[k] += xt_coeff * 0.5 * (left_nb + right_nb)

        U_actual = build_clements_4x4(th_actual, ph_actual)
        fidelity = np.abs(np.trace(U_ideal.conj().T @ U_actual)) / 4.0
        infidelities.append(max(0.0, 1.0 - float(fidelity)))

    return float(np.mean(infidelities))


class MeshCoOptProblem(Problem):
    """
    Joint Device + Driver + 4x4 MZI Mesh Co-Optimization Problem.
    Uses Lumerical-fitted lateral thermal crosstalk model.
    """
    def __init__(self, bundle, xt_params, target_configs):
        super().__init__(
            n_var=7,
            n_obj=3,
            n_ieq_constr=1,
            xl=np.array([400.0, 200.0, 800.0, 600.0, 50.0, 10.0, 6.0]),
            xu=np.array([600.0, 260.0, 2000.0, 1500.0, 250.0, 40.0, 12.0])
        )
        self.scaler = bundle["scaler"]
        self.gp_dneff = bundle["gp_dneff"]
        self.gp_tmax = bundle["gp_tmax"]
        self.wavelength_um = bundle["wavelength_um"]
        self.xt_A = xt_params["A"]
        self.xt_L0 = xt_params["L0_um"]
        self.xt_L1 = xt_params["L1_um_per_um"]
        self.target_configs = target_configs

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        p_ref = 25.0

        X_query = np.column_stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3], np.full(n, p_ref)])
        X_q_s = self.scaler.transform(X_query)
        dneff_ref = np.maximum(self.gp_dneff.predict(X_q_s) * 1e-3, 1e-6)
        dT_max_ref = np.maximum(self.gp_tmax.predict(X_q_s), 1.0)

        f1_power = np.zeros(n)
        f2_infid = np.zeros(n)
        f3_area  = np.zeros(n)
        g1_temp  = np.zeros(n)

        n_shifters = 12

        for i in range(n):
            gap_um   = X[i, 3] / 1000.0
            length_u = X[i, 4]
            pitch_u  = X[i, 5]
            dac_bits = int(np.round(X[i, 6]))

            dneff_pi = self.wavelength_um / (2.0 * length_u)
            p_pi_mW = p_ref * (dneff_pi / dneff_ref[i])
            T_max_2pi_K = 300.0 + dT_max_ref[i] * (2.0 * p_pi_mW / p_ref)

            p_dac_per_ch = 0.8 + 0.04 * (2.0 ** (dac_bits - 6)) + 0.15 * p_pi_mW
            total_power_mW = n_shifters * (p_pi_mW + p_dac_per_ch)

            # Lumerical HEAT-fitted lateral thermal crosstalk
            L_th_um = max(self.xt_L0 + self.xt_L1 * gap_um, 1.0)
            xt_coeff = self.xt_A * np.exp(-pitch_u / L_th_um)

            infid_pct = 100.0 * evaluate_mesh_infidelity(self.target_configs, dac_bits, xt_coeff)

            mesh_len_mm = (4.0 * (length_u + 40.0)) * 1e-3
            mesh_wid_mm = (4.0 * pitch_u) * 1e-3
            area_mm2 = mesh_len_mm * mesh_wid_mm

            f1_power[i] = total_power_mW
            f2_infid[i] = infid_pct
            f3_area[i]  = area_mm2
            g1_temp[i]  = T_max_2pi_K - 450.0

        out["F"] = np.column_stack([f1_power, f2_infid, f3_area])
        out["G"] = g1_temp.reshape(-1, 1)


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
    log_path = os.path.join(log_dir, f"nsga2_coopt_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[CO-OPT]{CLR_RESET} {CLR_CYAN}Phase 4: NSGA-II Co-Optimization (Lumerical Crosstalk Model){CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        bundle_path = os.path.join(project_root, "models", "compact_model_bundle.pkl")
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        xt_path = os.path.join(project_root, "models", "crosstalk_model.json")
        with open(xt_path, "r", encoding="utf-8") as f:
            xt_params = json.load(f)

        print(f"{CLR_GREEN}[LOAD]{CLR_RESET}   Compact model   -> {CLR_GRAY}{bundle_path}{CLR_RESET}")
        print(f"{CLR_GREEN}[LOAD]{CLR_RESET}   Crosstalk model -> A={xt_params['A']:.4f}, L0={xt_params['L0_um']:.3f}um, L1={xt_params['L1_um_per_um']:.3f}")

        rng = np.random.default_rng(42)
        target_configs = []
        for _ in range(12):
            th = rng.uniform(0.0, np.pi, size=6)
            ph = rng.uniform(0.0, 2.0 * np.pi, size=6)
            target_configs.append((th, ph, build_clements_4x4(th, ph)))

        problem = MeshCoOptProblem(bundle, xt_params, target_configs)
        algorithm = NSGA2(
            pop_size=80,
            sampling=LHS(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        print(f"{CLR_MAGENTA}[NSGA2]{CLR_RESET}  Running 80 pop x 60 generations (4,800 system evaluations)...")
        t0 = time.time()
        res = minimize(problem, algorithm, ("n_gen", 60), seed=42, verbose=False)
        elapsed = time.time() - t0
        print(f"  {CLR_GRAY}|--{CLR_RESET} Optimization finished in {CLR_BOLD}{elapsed:.2f} s{CLR_RESET}")

        X_opt = res.X
        F_opt = res.F

        p_ref = 25.0
        X_q = np.column_stack([X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], X_opt[:, 3], np.full(len(X_opt), p_ref)])
        X_q_s = bundle["scaler"].transform(X_q)
        dneff_ref = bundle["gp_dneff"].predict(X_q_s) * 1e-3
        dT_max_ref = bundle["gp_tmax"].predict(X_q_s)

        p_pi_list = p_ref * ((bundle["wavelength_um"] / (2.0 * X_opt[:, 4])) / dneff_ref)
        t_max_2pi_list = 300.0 + dT_max_ref * (2.0 * p_pi_list / p_ref)

        pareto_df = pd.DataFrame({
            "pareto_id": np.arange(len(X_opt)),
            "wg_width_nm": np.round(X_opt[:, 0], 2),
            "wg_height_nm": np.round(X_opt[:, 1], 2),
            "heater_width_nm": np.round(X_opt[:, 2], 2),
            "heater_gap_nm": np.round(X_opt[:, 3], 2),
            "length_um": np.round(X_opt[:, 4], 2),
            "pitch_um": np.round(X_opt[:, 5], 2),
            "dac_bits": np.round(X_opt[:, 6]).astype(int),
            "P_pi_mW": np.round(p_pi_list, 3),
            "T_max_2pi_K": np.round(t_max_2pi_list, 2),
            "total_mesh_power_mW": np.round(F_opt[:, 0], 2),
            "mesh_infidelity_pct": np.round(F_opt[:, 1], 4),
            "footprint_mm2": np.round(F_opt[:, 2], 4)
        }).sort_values("total_mesh_power_mW").reset_index(drop=True)
        pareto_df["pareto_id"] = np.arange(len(pareto_df))

        csv_out = os.path.join(results_dir, "pareto_front.csv")
        pareto_df.to_csv(csv_out, index=False)

        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[PARETO]{CLR_RESET} Found {CLR_BOLD}{len(pareto_df)}{CLR_RESET} feasible Pareto-optimal designs:")
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} Min Power    : "
            f"{pareto_df['total_mesh_power_mW'].min():.1f} mW "
            f"(Infid={pareto_df.loc[pareto_df['total_mesh_power_mW'].idxmin(), 'mesh_infidelity_pct']:.4f}%, "
            f"DAC={pareto_df.loc[pareto_df['total_mesh_power_mW'].idxmin(), 'dac_bits']}b)"
        )
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} Min Infidel  : "
            f"{pareto_df['mesh_infidelity_pct'].min():.5f}% "
            f"(Power={pareto_df.loc[pareto_df['mesh_infidelity_pct'].idxmin(), 'total_mesh_power_mW']:.1f} mW, "
            f"DAC={pareto_df.loc[pareto_df['mesh_infidelity_pct'].idxmin(), 'dac_bits']}b)"
        )
        print(
            f"  {CLR_GRAY}|--{CLR_RESET} Min Area     : "
            f"{pareto_df['footprint_mm2'].min():.4f} mm^2 "
            f"(Power={pareto_df.loc[pareto_df['footprint_mm2'].idxmin(), 'total_mesh_power_mW']:.1f} mW, "
            f"Infid={pareto_df.loc[pareto_df['footprint_mm2'].idxmin(), 'mesh_infidelity_pct']:.3f}%)"
        )

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

        sc0 = axes[0].scatter(
            pareto_df["total_mesh_power_mW"], pareto_df["mesh_infidelity_pct"],
            c=pareto_df["dac_bits"], cmap="viridis", s=38, edgecolors="k", linewidths=0.4
        )
        axes[0].set_xlabel("Total 4x4 Mesh + Driver Power (mW)")
        axes[0].set_ylabel("Mean Unitary Infidelity (%)")
        axes[0].set_title("Power vs. Fidelity (Color = DAC Bits)")
        axes[0].set_yscale("log")
        axes[0].grid(True, which="both", alpha=0.3)
        plt.colorbar(sc0, ax=axes[0], label="DAC Bits")

        sc1 = axes[1].scatter(
            pareto_df["footprint_mm2"], pareto_df["mesh_infidelity_pct"],
            c=pareto_df["pitch_um"], cmap="plasma", s=38, edgecolors="k", linewidths=0.4
        )
        axes[1].set_xlabel("Active Mesh Footprint (mm^2)")
        axes[1].set_ylabel("Mean Unitary Infidelity (%)")
        axes[1].set_title("Area vs. Fidelity (Color = Pitch um)")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both", alpha=0.3)
        plt.colorbar(sc1, ax=axes[1], label="Heater Pitch (um)")

        sc2 = axes[2].scatter(
            pareto_df["total_mesh_power_mW"], pareto_df["footprint_mm2"],
            c=pareto_df["T_max_2pi_K"], cmap="inferno", s=38, edgecolors="k", linewidths=0.4
        )
        axes[2].set_xlabel("Total 4x4 Mesh + Driver Power (mW)")
        axes[2].set_ylabel("Active Mesh Footprint (mm^2)")
        axes[2].set_title("Power vs. Area (Color = Peak T_2pi K)")
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(sc2, ax=axes[2], label="Peak Temp at 2pi (K)")

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "pareto_coopt_tradeoffs.png")
        plt.savefig(plot_out, dpi=220)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Pareto CSV  -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Pareto Plot -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved   -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()