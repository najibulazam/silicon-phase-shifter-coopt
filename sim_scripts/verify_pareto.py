# verify_pareto.py
import os
import re
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=SyntaxWarning)
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


def load_lumapi():
    try:
        import lumapi
        return lumapi
    except ImportError:
        lum_paths = [
            r"C:\Program Files\Lumerical\v241\api\python",
            r"C:\Program Files\Lumerical\v232\api\python",
            r"C:\Program Files\ANSYS Inc\v241\Lumerical\api\python",
            r"C:\Program Files\ANSYS Inc\v232\Lumerical\api\python"
        ]
        for path in lum_paths:
            if os.path.exists(path):
                sys.path.append(path)
                try:
                    import lumapi
                    return lumapi
                except ImportError:
                    continue
        raise ImportError("lumapi path not found. Please check your Lumerical installation.")


def to_scalar(val):
    return np.asarray(val).item()


def select_representative_points(pareto_df, n_points=10):
    """Select extreme designs (min power, min infid, min area) + evenly spaced points."""
    extremes = [
        int(pareto_df["total_mesh_power_mW"].idxmin()),
        int(pareto_df["mesh_infidelity_pct"].idxmin()),
        int(pareto_df["footprint_mm2"].idxmin()),
    ]
    spaced = np.linspace(0, len(pareto_df) - 1, n_points, dtype=int).tolist()
    selected = []
    for idx in extremes + spaced:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= n_points:
            break
    return pareto_df.iloc[sorted(selected)].reset_index(drop=True)


def verify_pareto():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"verify_pareto_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        lumapi = load_lumapi()

        pareto_csv = os.path.join(results_dir, "pareto_front.csv")
        pareto_df = pd.read_csv(pareto_csv)
        sel_df = select_representative_points(pareto_df, n_points=10)

        ldev_path = os.path.join(project_root, config["templates"]["heat_ldev"])
        lms_path  = os.path.join(project_root, config["templates"]["mode_lms"])
        mat_path  = os.path.abspath(os.path.join(project_root, "data", "raw", "tmp_verify_map.mat")).replace("\\", "/")
        wavelength_um = config["simulation"]["wavelength_um"]
        wavelength = wavelength_um * 1e-6

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[VERIFY]{CLR_RESET} {CLR_CYAN}Phase 4: Lumerical Ground-Truth Re-Simulation of Pareto Front{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[SELECT]{CLR_RESET} Re-simulating {CLR_BOLD}{len(sel_df)}{CLR_RESET} representative Pareto designs at predicted P_pi...")

        heat = lumapi.DEVICE(filename=ldev_path, hide=True)
        mode = lumapi.MODE(filename=lms_path, hide=True)
        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")

        heater_thickness = 200e-9
        verified_rows = []

        try:
            for idx, row in sel_df.iterrows():
                t0 = time.time()
                wg_w = row["wg_width_nm"] * 1e-9
                wg_h = row["wg_height_nm"] * 1e-9
                ht_w = row["heater_width_nm"] * 1e-9
                gap  = row["heater_gap_nm"] * 1e-9
                p_pi = row["P_pi_mW"] * 1e-3
                L_um = row["length_um"]

                dneff_target = wavelength_um / (2.0 * L_um)

                # 1. HEAT solve at predicted P_pi
                heat.switchtolayout()
                heat.setnamed("::model::geometry::wg", "x", 0.0)
                heat.setnamed("::model::geometry::wg", "x span", wg_w)
                heat.setnamed("::model::geometry::wg", "y min", 0.0)
                heat.setnamed("::model::geometry::wg", "y max", wg_h)

                heat.setnamed("::model::geometry::wire", "x", 0.0)
                heat.setnamed("::model::geometry::wire", "x span", ht_w)
                heat.setnamed("::model::geometry::wire", "y min", wg_h + gap)
                heat.setnamed("::model::geometry::wire", "y max", wg_h + gap + heater_thickness)

                heat.setnamed("::model::HEAT::uniform heat", "total power", p_pi)
                heat.setnamed("::model::HEAT::wg_monitor", "save data", 1)
                heat.setnamed("::model::HEAT::wg_monitor", "filename", mat_path)
                heat.run()

                heat_res = heat.getresult("HEAT::wg_monitor", "temperature")
                T_max_sim_pi = float(np.max(heat_res["T"]))

                # 2. MODE unperturbed
                mode.switchtolayout()
                mode.setnamed("wg", "x", 0.0)
                mode.setnamed("wg", "x span", wg_w)
                mode.setnamed("wg", "y min", 0.0)
                mode.setnamed("wg", "y max", wg_h)
                mode.setnamed("FDE", "wavelength", wavelength)
                mode.setnamed("temperature", "enabled", 0)
                mode.findmodes()
                neff_0 = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "neff"))))

                # 3. MODE thermally perturbed
                mode.switchtolayout()
                mode.setnamed("temperature", "enabled", 1)
                mode.select("temperature")
                mode.importdataset(mat_path)
                mode.findmodes()
                neff_hot = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "neff"))))

                dneff_sim = neff_hot - neff_0
                phi_sim_over_pi = (2.0 * L_um * dneff_sim) / wavelength_um
                rel_err_pct = abs(dneff_sim - dneff_target) / dneff_target * 100.0
                dt = time.time() - t0

                rec = row.to_dict()
                rec.update({
                    "dneff_surrogate": dneff_target,
                    "dneff_lumerical": dneff_sim,
                    "phi_sim_over_pi": round(phi_sim_over_pi, 5),
                    "rel_error_pct": round(rel_err_pct, 4),
                    "T_max_sim_pi_K": round(T_max_sim_pi, 2),
                    "verify_time_s": round(dt, 2)
                })
                verified_rows.append(rec)

                print(
                    f"{CLR_GREEN}[POINT {idx+1:02d}/{len(sel_df)}]{CLR_RESET} "
                    f"PID={int(row['pareto_id']):02d} | P_pi={row['P_pi_mW']:5.2f}mW, L={L_um:5.1f}um | "
                    f"dphi={CLR_BOLD}{phi_sim_over_pi:.4f}*pi{CLR_RESET} | "
                    f"Err={CLR_YELLOW}{rel_err_pct:.3f}%{CLR_RESET} | {dt:.2f}s"
                )
        finally:
            heat.close()
            mode.close()

        ver_df = pd.DataFrame(verified_rows)
        csv_out = os.path.join(results_dir, "pareto_verified.csv")
        ver_df.to_csv(csv_out, index=False)

        mean_err = float(ver_df["rel_error_pct"].mean())
        max_err  = float(ver_df["rel_error_pct"].max())

        # Verification plot
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

        axes[0].scatter(ver_df["dneff_lumerical"] * 1e3, ver_df["dneff_surrogate"] * 1e3, s=55, c="#1f77b4", edgecolors="k")
        lim_min = min(ver_df["dneff_lumerical"].min(), ver_df["dneff_surrogate"].min()) * 1e3 * 0.9
        lim_max = max(ver_df["dneff_lumerical"].max(), ver_df["dneff_surrogate"].max()) * 1e3 * 1.1
        axes[0].plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.2, label="Ideal (y = x)")
        axes[0].set_xlabel(r"Lumerical FDE $\Delta n_{\mathrm{eff}}$ ($\times 10^{-3}$)")
        axes[0].set_ylabel(r"Surrogate Target $\Delta n_{\mathrm{eff}}$ ($\times 10^{-3}$)")
        axes[0].set_title(f"Pareto Re-Simulation Parity (MAPE = {mean_err:.3f}%)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        x_pos = np.arange(len(ver_df))
        axes[1].bar(x_pos, ver_df["phi_sim_over_pi"], color="#2ca02c", alpha=0.85, edgecolor="k")
        axes[1].axhline(1.0, color="r", linestyle="--", lw=1.4, label=r"Target $\Delta\phi = 1.0\,\pi$")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f"P{int(p)}" for p in ver_df["pareto_id"]])
        axes[1].set_ylim(0.95, 1.05)
        axes[1].set_xlabel("Verified Pareto Design ID")
        axes[1].set_ylabel(r"Lumerical Achieved Phase ($\Delta\phi / \pi$)")
        axes[1].set_title(f"Phase Accuracy at Predicted $P_\\pi$ (Max Err = {max_err:.3f}%)")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "pareto_verification.png")
        plt.savefig(plot_out, dpi=220)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[SUMMARY]{CLR_RESET} Mean Verification Error: {CLR_BOLD}{mean_err:.4f}%{CLR_RESET} | Max Error: {CLR_BOLD}{max_err:.4f}%{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}    Verified CSV  -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}    Verified Plot -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}     Log saved     -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    verify_pareto()