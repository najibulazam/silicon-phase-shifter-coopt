# run_pn_sweep.py
import os
import re
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import qmc

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
        for path in [
            r"C:\Program Files\Lumerical\v241\api\python",
            r"C:\Program Files\ANSYS Inc\v241\Lumerical\api\python",
        ]:
            if os.path.exists(path):
                sys.path.append(path)
                import lumapi
                return lumapi
        raise ImportError("lumapi not found.")


def to_scalar(val):
    return np.asarray(val).item()


def generate_pn_lhs_cases(n_cases=25, seed=42):
    """
    Generate LHS parameter combinations for PN carrier-depletion phase shifter:
      - wg_width_nm   : [450, 550]
      - p_conc_1e23   : [2.0, 6.0]   (cm^-3 scale; mapped to m^-3 as conc * 1e23)
      - n_conc_1e23   : [2.0, 6.0]
      - pn_offset_nm  : [-60.0, +60.0] (junction shift relative to x = 0)
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n=n_cases)
    l_bounds = np.array([450.0, 2.0, 2.0, -60.0])
    u_bounds = np.array([550.0, 6.0, 6.0,  60.0])
    samples = qmc.scale(u, l_bounds, u_bounds)

    df = pd.DataFrame({
        "case_id": np.arange(n_cases),
        "wg_width_nm": np.round(samples[:, 0], 2),
        "p_conc_1e23": np.round(samples[:, 1], 4),
        "n_conc_1e23": np.round(samples[:, 2], 4),
        "pn_offset_nm": np.round(samples[:, 3], 2),
    })
    return df


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log")
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_pn_sweep_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        lumapi = load_lumapi()
        ldev_path = os.path.abspath(os.path.join(project_root, "templates", "pn_phase_shifter", "PN_phase_shifter.ldev"))
        lms_path  = os.path.abspath(os.path.join(project_root, "templates", "pn_phase_shifter", "PN_phase_shifter.lms"))

        # Place charge.mat inside templates/pn_phase_shifter/ or raw_dir and import directly
        charge_mat_path = os.path.abspath(os.path.join(raw_dir, "pn_charge_tmp.mat")).replace("\\", "/")
        out_csv = os.path.join(raw_dir, "pn_depletion_sweep.csv")
        cases_csv = os.path.join(raw_dir, "pn_lhs_cases.csv")

        n_cases = 25
        if os.path.exists(cases_csv):
            cases_df = pd.read_csv(cases_csv)
        else:
            cases_df = generate_pn_lhs_cases(n_cases=n_cases, seed=42)
            cases_df.to_csv(cases_csv, index=False)

        if os.path.exists(out_csv):
            res_df = pd.read_csv(out_csv)
            done_cases = set(res_df["case_id"].unique().tolist())
            records = res_df.to_dict("records")
        else:
            done_cases = set()
            records = []

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[PN-SWEEP]{CLR_RESET} {CLR_CYAN}Phase 5: PN Carrier-Depletion LHS Dataset Generation{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[PLAN]{CLR_RESET}     Total LHS cases: {len(cases_df)} | Already done: {len(done_cases)} | 9 voltage steps/case")
        print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")

        charge = lumapi.DEVICE(filename=ldev_path, hide=True)
        mode = lumapi.MODE(filename=lms_path, hide=True)

        # Bias voltages matching drain sweep: 0.0 V down to -4.0 V (9 points)
        voltages_V = np.linspace(0.0, -4.0, 9)

        try:
            for _, row in cases_df.iterrows():
                cid = int(row["case_id"])
                if cid in done_cases:
                    continue

                t0 = time.time()
                wg_w_m = float(row["wg_width_nm"]) * 1e-9
                p_conc_m3 = float(row["p_conc_1e23"]) * 1e23
                n_conc_m3 = float(row["n_conc_1e23"]) * 1e23
                junc_x_m = float(row["pn_offset_nm"]) * 1e-9

                # 1. Update CHARGE geometry & doping
                charge.switchtolayout()
                charge.setnamed("::model::geometry::waveguide", "x", 0.0)
                charge.setnamed("::model::geometry::waveguide", "x span", wg_w_m)

                # Update PN doping regions around junction x = junc_x_m
                x_left = -wg_w_m / 2.0
                x_right = wg_w_m / 2.0
                try:
                    charge.setnamed("::model::CHARGE::doping::waveguide p+", "concentration", p_conc_m3)
                    charge.setnamed("::model::CHARGE::doping::waveguide p+", "x min", x_left)
                    charge.setnamed("::model::CHARGE::doping::waveguide p+", "x max", junc_x_m)
                except Exception:
                    pass

                try:
                    charge.setnamed("::model::CHARGE::doping::waveguide n", "concentration", n_conc_m3)
                    charge.setnamed("::model::CHARGE::doping::waveguide n", "x min", junc_x_m)
                    charge.setnamed("::model::CHARGE::doping::waveguide n", "x max", x_right)
                except Exception:
                    pass

                charge.setnamed("::model::CHARGE::monitor_charge", "save data", 1)
                charge.setnamed("::model::CHARGE::monitor_charge", "filename", charge_mat_path)
                charge.run()

                # 2. Update MODE waveguide geometry & import charge dataset
                mode.switchtolayout()
                mode.setnamed("Wg", "x", 0.0)
                mode.setnamed("Wg", "x span", wg_w_m)
                mode.setnamed("np", "enabled", 1)
                mode.select("np")
                mode.importdataset(charge_mat_path)

                # Reference index at V_drain = 0.0 V (index 1 in Lumerical)
                mode.switchtolayout()
                mode.setnamed("np", "enabled", 0)
                mode.findmodes()
                neff_0 = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "neff"))))
                loss_0 = float(to_scalar(mode.getdata("FDE::data::mode1", "loss")))

                # Evaluate across all 9 drain bias points (1-based index in Lumerical)
                dneff_4v = 0.0
                for v_idx, v_val in enumerate(voltages_V, start=1):
                    mode.switchtolayout()
                    mode.setnamed("np", "enabled", 1)
                    try:
                        mode.setnamed("np", "V_drain_index", v_idx)
                        mode.setnamed("np", "V_source_index", 1)
                    except Exception:
                        pass
                    mode.findmodes()
                    neff_v = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "neff"))))
                    loss_v = float(to_scalar(mode.getdata("FDE::data::mode1", "loss")))

                    # Carrier depletion lowers refractive index under reverse bias (v_val <= 0)
                    delta_neff = float(neff_v - neff_0)
                    delta_loss = float(loss_v - loss_0)
                    if abs(v_val - (-4.0)) < 1e-6:
                        dneff_4v = abs(delta_neff)

                    records.append({
                        "case_id": cid,
                        "v_drain_V": round(float(v_val), 2),
                        "wg_width_nm": float(row["wg_width_nm"]),
                        "p_conc_1e23": float(row["p_conc_1e23"]),
                        "n_conc_1e23": float(row["n_conc_1e23"]),
                        "pn_offset_nm": float(row["pn_offset_nm"]),
                        "neff_0": neff_0,
                        "neff_v": neff_v,
                        "abs_delta_neff": abs(delta_neff),
                        "delta_neff_signed": delta_neff,
                        "loss_dB_cm": loss_v,
                        "delta_loss_dB_cm": delta_loss,
                    })

                # Checkpoint save after each case
                pd.DataFrame(records).to_csv(out_csv, index=False)
                dt = time.time() - t0
                print(
                    f"{CLR_GREEN}[CASE {cid+1:02d}/{len(cases_df):02d}]{CLR_RESET} "
                    f"W={row['wg_width_nm']:5.1f}nm, P={row['p_conc_1e23']:.2f}e23, N={row['n_conc_1e23']:.2f}e23, "
                    f"Off={row['pn_offset_nm']:+5.1f}nm | |dneff(-4V)|={dneff_4v:.3e} | {dt:.1f}s"
                )
        finally:
            charge.close()
            mode.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   PN dataset saved ({len(records)} rows) -> {CLR_GRAY}{out_csv}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()