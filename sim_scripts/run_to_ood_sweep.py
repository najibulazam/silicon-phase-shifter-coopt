# run_to_ood_sweep.py
import os
import re
import sys
import time
import warnings
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import griddata
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
    """Safely convert numpy array or Python scalar to a single Python scalar."""
    return np.asarray(val).item()


def generate_ood_samples(ood_bounds_cfg, samples_per_regime, seed):
    """Generate deterministic LHS samples for each OOD regime."""
    keys = [
        "wg_width_nm",
        "wg_height_nm",
        "heater_width_nm",
        "heater_gap_nm",
        "heater_power_mW"
    ]
    frames = []
    for idx, (regime_name, bounds) in enumerate(ood_bounds_cfg.items()):
        l_bounds = [bounds[k][0] for k in keys]
        u_bounds = [bounds[k][1] for k in keys]

        sampler = qmc.LatinHypercube(d=len(keys), seed=seed + idx)
        unit_samples = sampler.random(n=samples_per_regime)
        scaled = qmc.scale(unit_samples, l_bounds, u_bounds)

        df = pd.DataFrame(scaled, columns=keys)
        df["ood_regime"] = regime_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def interpolate_heat_field(heat_res, x_grid, y_grid):
    """Interpolate unstructured FE temperature field onto a uniform 2D grid."""
    pts_x = np.asarray(heat_res['x']).flatten()
    pts_y = np.asarray(heat_res['y']).flatten()
    T_vals = np.asarray(heat_res['T']).flatten()

    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    T_grid = griddata((pts_x, pts_y), T_vals, (X, Y), method='linear')
    if np.isnan(T_grid).any():
        T_nearest = griddata((pts_x, pts_y), T_vals, (X, Y), method='nearest')
        T_grid = np.where(np.isnan(T_grid), T_nearest, T_grid)
    return T_grid.astype(np.float32)


def interpolate_mode_intensity(mode_session, x_grid, y_grid):
    """Extract fundamental mode |E|^2 and interpolate onto the uniform 2D grid."""
    E_res = mode_session.getresult("FDE::data::mode1", "E")
    mx = np.asarray(E_res['x']).flatten()
    my = np.asarray(E_res['y']).flatten()
    E_arr = np.squeeze(np.asarray(E_res['E']))
    E2 = np.sum(np.abs(E_arr) ** 2, axis=-1)

    MX, MY = np.meshgrid(mx, my, indexing='ij')
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    E2_grid = griddata(
        (MX.flatten(), MY.flatten()),
        E2.flatten(),
        (X, Y),
        method='linear',
        fill_value=0.0
    )
    total = np.sum(E2_grid)
    if total > 0:
        E2_grid = E2_grid / total
    return E2_grid.astype(np.float32)


def init_or_open_h5(h5_path, num_samples, x_grid, y_grid):
    """Create HDF5 datasets if not present."""
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    nx, ny = len(x_grid), len(y_grid)
    if not os.path.exists(h5_path):
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("x_grid", data=x_grid.astype(np.float32))
            hf.create_dataset("y_grid", data=y_grid.astype(np.float32))
            hf.create_dataset("T_fields", shape=(num_samples, nx, ny), dtype=np.float32, compression="gzip")
            hf.create_dataset("E2_fields", shape=(num_samples, nx, ny), dtype=np.float32, compression="gzip")


def run_ood_sweep():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_to_ood_sweep_{timestamp}.log")

    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        lumapi = load_lumapi()

        ldev_path = os.path.join(project_root, config["templates"]["heat_ldev"])
        lms_path  = os.path.join(project_root, config["templates"]["mode_lms"])
        csv_path  = os.path.join(project_root, config["ood_dataset"]["scalars_csv"])
        h5_path   = os.path.join(project_root, config["ood_dataset"]["fields_h5"])
        mat_path  = os.path.abspath(os.path.join(project_root, config["ood_dataset"]["temp_mat_file"])).replace("\\", "/")

        samples_per_regime = config["ood_dataset"]["samples_per_regime"]
        seed               = config["ood_dataset"]["seed"]
        wavelength         = config["simulation"]["wavelength_um"] * 1e-6

        samples_df  = generate_ood_samples(config["ood_bounds"], samples_per_regime, seed)
        num_samples = len(samples_df)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(mat_path), exist_ok=True)

        print(f"{CLR_CYAN}" + "=" * 68)
        print(f" {CLR_BOLD}[OOD SWEEP]{CLR_RESET} {CLR_CYAN}Thermo-Optic Phase Shifter - OOD Extrapolation Dataset{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 68 + f"{CLR_RESET}")

        grid_n = 64
        x_grid = np.linspace(-1.0e-6, 1.0e-6, grid_n)
        y_grid = np.linspace(-0.89e-6, 1.11e-6, grid_n)
        init_or_open_h5(h5_path, num_samples, x_grid, y_grid)

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            start_idx = len(existing_df)
            header_written = True
            print(f"{CLR_YELLOW}[RESUME]{CLR_RESET} Existing OOD dataset found: {CLR_BOLD}{start_idx}/{num_samples}{CLR_RESET} samples completed.")
        else:
            start_idx = 0
            header_written = False
            print(f"{CLR_BLUE}[INIT]{CLR_RESET}   Starting fresh OOD sweep ({CLR_BOLD}{num_samples}{CLR_RESET} samples, seed={seed}).")

        if start_idx >= num_samples:
            print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Target reached. OOD dataset is already complete.")
            return

        print(f"{CLR_MAGENTA}[SOLVER]{CLR_RESET} Launching persistent HEAT and MODE sessions...")
        heat = lumapi.DEVICE(filename=ldev_path, hide=True)
        mode = lumapi.MODE(filename=lms_path, hide=True)
        print(f"{CLR_GRAY}" + "-" * 68 + f"{CLR_RESET}")

        heater_thickness = 200e-9

        try:
            for i in range(start_idx, num_samples):
                t_sample = time.time()
                row = samples_df.iloc[i]

                wg_w = row["wg_width_nm"] * 1e-9
                wg_h = row["wg_height_nm"] * 1e-9
                ht_w = row["heater_width_nm"] * 1e-9
                gap  = row["heater_gap_nm"] * 1e-9
                pwr  = row["heater_power_mW"] * 1e-3
                regime = row["ood_regime"]

                print(
                    f"{CLR_GREEN}[OOD]{CLR_RESET}    Sample {CLR_BOLD}{i+1}/{num_samples}{CLR_RESET} [{CLR_YELLOW}{regime}{CLR_RESET}] | "
                    f"W={row['wg_width_nm']:.1f}nm, H={row['wg_height_nm']:.1f}nm, "
                    f"Hw={row['heater_width_nm']:.1f}nm, Gap={row['heater_gap_nm']:.1f}nm, "
                    f"P={row['heater_power_mW']:.2f}mW"
                )

                # ---- 1. HEAT simulation ----
                heat.switchtolayout()
                heat.setnamed("::model::geometry::wg", "x", 0.0)
                heat.setnamed("::model::geometry::wg", "x span", wg_w)
                heat.setnamed("::model::geometry::wg", "y min", 0.0)
                heat.setnamed("::model::geometry::wg", "y max", wg_h)

                heat.setnamed("::model::geometry::wire", "x", 0.0)
                heat.setnamed("::model::geometry::wire", "x span", ht_w)
                heat.setnamed("::model::geometry::wire", "y min", wg_h + gap)
                heat.setnamed("::model::geometry::wire", "y max", wg_h + gap + heater_thickness)

                heat.setnamed("::model::HEAT::uniform heat", "total power", pwr)
                heat.setnamed("::model::HEAT::wg_monitor", "save data", 1)
                heat.setnamed("::model::HEAT::wg_monitor", "filename", mat_path)

                heat.run()

                heat_res = heat.getresult("HEAT::wg_monitor", "temperature")
                T_grid = interpolate_heat_field(heat_res, x_grid, y_grid)
                T_max = float(np.max(heat_res['T']))

                X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
                wg_mask = (np.abs(X) <= wg_w / 2.0) & (Y >= 0.0) & (Y <= wg_h)
                T_wg_avg = float(np.mean(T_grid[wg_mask])) if np.any(wg_mask) else float(np.mean(T_grid))

                # ---- 2. MODE (FDE) unperturbed pass ----
                mode.switchtolayout()
                mode.setnamed("wg", "x", 0.0)
                mode.setnamed("wg", "x span", wg_w)
                mode.setnamed("wg", "y min", 0.0)
                mode.setnamed("wg", "y max", wg_h)
                mode.setnamed("FDE", "wavelength", wavelength)
                mode.setnamed("temperature", "enabled", 0)

                mode.findmodes()
                neff_0 = complex(to_scalar(mode.getdata("FDE::data::mode1", "neff")))
                loss_0 = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "loss"))))
                E2_grid = interpolate_mode_intensity(mode, x_grid, y_grid)

                # ---- 3. MODE (FDE) thermally perturbed pass ----
                mode.switchtolayout()
                mode.setnamed("temperature", "enabled", 1)
                mode.select("temperature")
                mode.importdataset(mat_path)

                mode.findmodes()
                neff_hot = complex(to_scalar(mode.getdata("FDE::data::mode1", "neff")))
                loss_hot = float(np.real(to_scalar(mode.getdata("FDE::data::mode1", "loss"))))

                delta_neff = float(np.real(neff_hot) - np.real(neff_0))
                dt = time.time() - t_sample

                # ---- 4. Save HDF5 fields & CSV scalars ----
                with h5py.File(h5_path, "a") as hf:
                    hf["T_fields"][i]  = T_grid
                    hf["E2_fields"][i] = E2_grid

                record = {
                    "sample_id": i,
                    "ood_regime": regime,
                    "wg_width_nm": round(float(row["wg_width_nm"]), 4),
                    "wg_height_nm": round(float(row["wg_height_nm"]), 4),
                    "heater_width_nm": round(float(row["heater_width_nm"]), 4),
                    "heater_gap_nm": round(float(row["heater_gap_nm"]), 4),
                    "heater_power_mW": round(float(row["heater_power_mW"]), 4),
                    "neff_0": float(np.real(neff_0)),
                    "neff_hot": float(np.real(neff_hot)),
                    "delta_neff": delta_neff,
                    "loss_0_dB_cm": loss_0,
                    "loss_hot_dB_cm": loss_hot,
                    "T_max_K": T_max,
                    "T_wg_avg_K": T_wg_avg,
                    "runtime_s": round(dt, 2)
                }

                rec_df = pd.DataFrame([record])
                if not header_written:
                    rec_df.to_csv(csv_path, index=False, mode='w', header=True)
                    header_written = True
                else:
                    rec_df.to_csv(csv_path, index=False, mode='a', header=False)

                print(
                    f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Saved{CLR_RESET} | "
                    f"dneff={delta_neff:.6e} | T_wg={T_wg_avg:.2f}K | T_max={T_max:.2f}K | {dt:.2f}s"
                )
        finally:
            heat.close()
            mode.close()

        print(f"{CLR_CYAN}" + "=" * 68)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   OOD sweep complete! {CLR_BOLD}{num_samples}{CLR_RESET} samples saved.")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Saved to: {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 68 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    run_ood_sweep()