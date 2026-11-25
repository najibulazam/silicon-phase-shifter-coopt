# inspect_props.py
import os
import re
import sys
import warnings
from datetime import datetime

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


try:
    import lumapi
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
                break
            except ImportError:
                continue
    else:
        raise ImportError("lumapi path not found. Please check your Lumerical installation.")


def dump_props(session, obj_path, props):
    out = {}
    for p in props:
        try:
            out[p] = session.getnamed(obj_path, p)
        except Exception:
            pass
    return out


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"inspect_props_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        ldev_path = os.path.join(project_root, "templates", "to_phase_shifter", "waveguide_heat.ldev")
        lms_path  = os.path.join(project_root, "templates", "to_phase_shifter", "waveguide_fde.lms")

        print(f"{CLR_CYAN}" + "=" * 60)
        print(f" {CLR_BOLD}[PROPS]{CLR_RESET} {CLR_CYAN}Geometry, Source & Monitor Property Inspection{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 60 + f"{CLR_RESET}")

        geom_keys = ["x", "x span", "x min", "x max", "y", "y span", "y min", "y max", "z", "z span", "z min", "z max"]

        print(f"{CLR_MAGENTA}[HEAT]{CLR_RESET}   Inspecting {os.path.basename(ldev_path)}...")
        with lumapi.DEVICE(filename=ldev_path, hide=True) as heat:
            heat.switchtolayout()
            for obj in ["wg", "wire", "oxide", "substrate"]:
                props = dump_props(heat, f"::model::geometry::{obj}", geom_keys)
                print(f"  {CLR_GRAY}|--{CLR_RESET} geometry::{obj} -> {props}")

            heat_src = dump_props(heat, "::model::HEAT::uniform heat", ["total power", "power", "enabled", "geometry"])
            print(f"  {CLR_GRAY}|--{CLR_RESET} HEAT::uniform heat -> {heat_src}")

            for mon in ["wg_monitor", "wg_monitor_1"]:
                m_props = dump_props(heat, f"::model::HEAT::{mon}", ["monitor type", "save data", "filename", "x", "x span", "y", "y span", "z", "z span"])
                print(f"  {CLR_GRAY}|--{CLR_RESET} HEAT::{mon} -> {m_props}")

        print(f"{CLR_GRAY}" + "-" * 60 + f"{CLR_RESET}")

        print(f"{CLR_MAGENTA}[MODE]{CLR_RESET}   Inspecting {os.path.basename(lms_path)}...")
        with lumapi.MODE(filename=lms_path, hide=True) as mode:
            mode.switchtolayout()
            for obj in ["wg", "oxide", "silicon"]:
                props = dump_props(mode, obj, geom_keys)
                print(f"  {CLR_GRAY}|--{CLR_RESET} {obj} -> {props}")

            fde_props = dump_props(mode, "FDE", ["solver type", "wavelength", "x", "x span", "y", "y span", "z", "z span"])
            print(f"  {CLR_GRAY}|--{CLR_RESET} FDE -> {fde_props}")

            temp_props = dump_props(mode, "temperature", ["enabled", "x", "y", "z"])
            print(f"  {CLR_GRAY}|--{CLR_RESET} temperature -> {temp_props}")

        print(f"{CLR_CYAN}" + "=" * 60)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Log saved to: {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 60 + f"{CLR_RESET}")
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()