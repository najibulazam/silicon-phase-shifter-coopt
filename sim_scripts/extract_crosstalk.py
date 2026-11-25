# extract_crosstalk.py
import json
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
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

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


def xt_model_func(X, A, L0, L1):
    """Exponential lateral thermal crosstalk model: XT(pitch, gap) = A * exp(-pitch / (L0 + L1 * gap_um))."""
    pitch_um, gap_um = X
    L_th = np.maximum(L0 + L1 * gap_um, 1.0)
    return A * np.exp(-pitch_um / L_th)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"extract_crosstalk_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        lumapi = load_lumapi()
        ldev_path = os.path.join(project_root, config["templates"]["heat_ldev"])

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[CROSSTALK]{CLR_RESET} {CLR_CYAN}Lumerical HEAT Lateral Thermal Crosstalk Extraction{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        gap_list_nm = [600.0, 800.0, 1000.0, 1250.0, 1500.0]
        hw_list_nm  = [800.0, 1400.0, 2000.0]
        pitches_um  = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])

        wg_w = 500e-9
        wg_h = 220e-9
        heater_thickness = 200e-9
        power_W = 25e-3
        T_ref = 300.0

        print(f"{CLR_MAGENTA}[HEAT]{CLR_RESET}   Launching persistent HEAT session ({len(gap_list_nm) * len(hw_list_nm)} runs)...")
        heat = lumapi.DEVICE(filename=ldev_path, hide=True)

        records = []
        run_idx = 0
        total_runs = len(gap_list_nm) * len(hw_list_nm)

        try:
            heat.switchtolayout()
            # Ensure simulation region covers at least +/- 45 um laterally
            sim_x_span = float(heat.getnamed("HEAT simulation region", "x span"))
            if sim_x_span < 90e-6:
                heat.setnamed("HEAT simulation region", "x span", 90e-6)
                print(f"  {CLR_GRAY}|--{CLR_RESET} Expanded HEAT simulation region x span -> 90.0 um")
            else:
                print(f"  {CLR_GRAY}|--{CLR_RESET} HEAT simulation region x span: {sim_x_span * 1e6:.1f} um")

            print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")

            for gap_nm in gap_list_nm:
                for hw_nm in hw_list_nm:
                    run_idx += 1
                    t0 = time.time()

                    heat.switchtolayout()
                    heat.setnamed("::model::geometry::wg", "x", 0.0)
                    heat.setnamed("::model::geometry::wg", "x span", wg_w)
                    heat.setnamed("::model::geometry::wg", "y min", 0.0)
                    heat.setnamed("::model::geometry::wg", "y max", wg_h)

                    heat.setnamed("::model::geometry::wire", "x", 0.0)
                    heat.setnamed("::model::geometry::wire", "x span", hw_nm * 1e-9)
                    heat.setnamed("::model::geometry::wire", "y min", wg_h + gap_nm * 1e-9)
                    heat.setnamed("::model::geometry::wire", "y max", wg_h + gap_nm * 1e-9 + heater_thickness)

                    heat.setnamed("::model::HEAT::uniform heat", "total power", power_W)

                    # Wide lateral monitor covering x in [-2 um, +42 um] at waveguide height
                    heat.setnamed("::model::HEAT::wg_monitor", "save data", 0)
                    heat.setnamed("::model::HEAT::wg_monitor", "x", 20e-6)
                    heat.setnamed("::model::HEAT::wg_monitor", "x span", 44e-6)
                    heat.setnamed("::model::HEAT::wg_monitor", "y", wg_h / 2.0)
                    heat.setnamed("::model::HEAT::wg_monitor", "y span", 0.6e-6)

                    heat.run()

                    res = heat.getresult("HEAT::wg_monitor", "temperature")
                    pts_x = np.asarray(res["x"]).flatten()
                    pts_y = np.asarray(res["y"]).flatten()
                    T_vals = np.asarray(res["T"]).flatten()

                    # Evaluate dT at x = 0 and x = pitches_um, y = wg_h / 2
                    eval_x = np.concatenate([[0.0], pitches_um * 1e-6])
                    eval_y = np.full_like(eval_x, wg_h / 2.0)
                    T_line = griddata((pts_x, pts_y), T_vals, (eval_x, eval_y), method="linear")
                    if np.isnan(T_line).any():
                        T_nn = griddata((pts_x, pts_y), T_vals, (eval_x, eval_y), method="nearest")
                        T_line = np.where(np.isnan(T_line), T_nn, T_line)

                    dT_0 = float(T_line[0] - T_ref)
                    dt = time.time() - t0

                    xt_15 = float((T_line[3] - T_ref) / max(dT_0, 1e-6))  # pitch = 15 um
                    xt_30 = float((T_line[6] - T_ref) / max(dT_0, 1e-6))  # pitch = 30 um

                    print(
                        f"{CLR_GREEN}[RUN {run_idx:02d}/{total_runs}]{CLR_RESET} "
                        f"Gap={gap_nm:6.1f}nm, Hw={hw_nm:6.1f}nm | "
                        f"dT0={dT_0:5.2f}K | XT(15um)={xt_15:.4f} | XT(30um)={xt_30:.4f} | {dt:.2f}s"
                    )

                    for p_idx, pitch_u in enumerate(pitches_um):
                        dT_p = float(T_line[p_idx + 1] - T_ref)
                        xt_val = max(0.0, dT_p / max(dT_0, 1e-6))
                        records.append({
                            "heater_gap_nm": gap_nm,
                            "heater_width_nm": hw_nm,
                            "pitch_um": float(pitch_u),
                            "dT_0_K": round(dT_0, 4),
                            "dT_pitch_K": round(dT_p, 4),
                            "xt_ratio": round(xt_val, 6)
                        })
        finally:
            heat.close()

        xt_df = pd.DataFrame(records)
        csv_out = os.path.join(raw_dir, "to_crosstalk.csv")
        xt_df.to_csv(csv_out, index=False)

        # Fit physics-grounded exponential decay model
        X_fit = (xt_df["pitch_um"].values, xt_df["heater_gap_nm"].values / 1000.0)
        y_fit = xt_df["xt_ratio"].values

        popt, _ = curve_fit(xt_model_func, X_fit, y_fit, p0=[0.4, 8.0, 2.0], maxfev=10000)
        A_fit, L0_fit, L1_fit = [float(v) for v in popt]

        y_pred = xt_model_func(X_fit, A_fit, L0_fit, L1_fit)
        rmse = float(np.sqrt(np.mean((y_fit - y_pred) ** 2)))
        ss_res = float(np.sum((y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - ss_res / ss_tot

        model_json = {
            "formula": "XT(pitch_um, gap_um) = A * exp(-pitch_um / (L0 + L1 * gap_um))",
            "A": A_fit,
            "L0_um": L0_fit,
            "L1_um_per_um": L1_fit,
            "fit_rmse": rmse,
            "fit_r2": r2
        }
        json_out = os.path.join(models_dir, "crosstalk_model.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(model_json, f, indent=2)

        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[FIT]{CLR_RESET}    Fitted Lumerical Crosstalk Model: XT = {A_fit:.4f} * exp(-pitch / ({L0_fit:.3f} + {L1_fit:.3f}*gap_um))")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Fit Quality: RMSE={rmse:.5f} | R2={r2:.5f}")

        # Plot Lumerical crosstalk data vs fitted curve
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        p_smooth = np.linspace(5.0, 45.0, 200)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gap_list_nm)))

        for gap_nm, col in zip(gap_list_nm, colors):
            sub = xt_df[xt_df["heater_gap_nm"] == gap_nm].groupby("pitch_um")["xt_ratio"].mean().reset_index()
            ax.scatter(sub["pitch_um"], sub["xt_ratio"] * 100.0, color=col, s=36, label=f"HEAT Gap={int(gap_nm)} nm")
            y_curve = xt_model_func((p_smooth, np.full_like(p_smooth, gap_nm / 1000.0)), A_fit, L0_fit, L1_fit)
            ax.plot(p_smooth, y_curve * 100.0, color=col, lw=1.6, alpha=0.85)

        ax.set_xlabel("Lateral Heater Pitch (um)")
        ax.set_ylabel("Thermal Crosstalk Ratio (%)")
        ax.set_title(f"Lumerical HEAT Lateral Thermal Crosstalk (Fit R2 = {r2:.4f})")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "lumerical_crosstalk_fit.png")
        plt.savefig(plot_out, dpi=220)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Crosstalk CSV   -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Model JSON      -> {CLR_GRAY}{json_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Crosstalk Plot  -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved       -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()