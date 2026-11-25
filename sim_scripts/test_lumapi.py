# test_lumapi.py
import os
import re
import sys
import glob
import time
import warnings
from datetime import datetime

# Suppress minor SyntaxWarning from Lumerical's internal lumapi.py on newer Python
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Enable native ANSI color support in Windows terminal
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
                    print(f"{CLR_GREEN}[OK]{CLR_RESET}     Loaded lumapi from: {CLR_GRAY}{path}{CLR_RESET}")
                    return lumapi
                except ImportError:
                    continue
        raise ImportError("lumapi path not found. Please check your Lumerical installation.")


def get_children(session, scope_path):
    """Return list of child object names under a given Lumerical groupscope."""
    try:
        session.groupscope(scope_path)
        session.eval(
            "selectall; n = getnumber; names = cell(n); "
            "for(i=1:n){ names{i} = get('name', i); }"
        )
        names = session.getv("names")
        session.groupscope("::model")
        return names
    except Exception:
        session.groupscope("::model")
        return []


def inspect_templates():
    # Resolve project root dynamically (works from root or sim_scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"test_lumapi_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        lumapi = load_lumapi()
        template_dir = os.path.join(project_root, "templates", "to_phase_shifter")

        print(f"{CLR_CYAN}" + "=" * 60)
        print(f" {CLR_BOLD}[INSPECT]{CLR_RESET} {CLR_CYAN}Lumerical Template & Timing Verification{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 60 + f"{CLR_RESET}")

        ldev_files = glob.glob(os.path.join(template_dir, "*.ldev"))
        lms_files  = glob.glob(os.path.join(template_dir, "*.lms"))

        if not ldev_files or not lms_files:
            print(f"{CLR_RED}[ERROR]{CLR_RESET}  Missing .ldev or .lms files in: {template_dir}")
            print(f"  {CLR_GRAY}|--{CLR_RESET} Found .ldev: {len(ldev_files)} | Found .lms: {len(lms_files)}")
            sys.exit(1)

        ldev_path = os.path.abspath(ldev_files[0])
        lms_path  = os.path.abspath(lms_files[0])

        print(f"{CLR_BLUE}[FILES]{CLR_RESET}  HEAT : {CLR_BOLD}{os.path.basename(ldev_path)}{CLR_RESET}")
        print(f"         MODE : {CLR_BOLD}{os.path.basename(lms_path)}{CLR_RESET}")
        print(f"{CLR_GRAY}" + "-" * 60 + f"{CLR_RESET}")

        # ---- Test HEAT (via DEVICE) ----
        print(f"{CLR_MAGENTA}[HEAT]{CLR_RESET}   Launching solver and inspecting objects...")
        t0 = time.time()
        with lumapi.DEVICE(filename=ldev_path, hide=True) as heat:
            heat.switchtolayout()
            geom_objs   = get_children(heat, "::model::geometry")
            solver_objs = get_children(heat, "::model::HEAT")
            print(f"  {CLR_GRAY}|--{CLR_RESET} Geometry : {geom_objs}")
            print(f"  {CLR_GRAY}|--{CLR_RESET} HEAT     : {solver_objs}")

            heat.run()
            t_heat = time.time() - t0
            print(f"  {CLR_GRAY}|--{CLR_RESET} Status   : {CLR_GREEN}Completed{CLR_RESET} in {CLR_BOLD}{t_heat:.2f} s{CLR_RESET}")

        print(f"{CLR_GRAY}" + "-" * 60 + f"{CLR_RESET}")

        # ---- Test MODE (FDE) ----
        print(f"{CLR_MAGENTA}[MODE]{CLR_RESET}   Launching solver and inspecting objects...")
        t1 = time.time()
        with lumapi.MODE(filename=lms_path, hide=True) as mode:
            mode.switchtolayout()
            mode_objs = get_children(mode, "::model")
            print(f"  {CLR_GRAY}|--{CLR_RESET} Objects  : {mode_objs}")

            mode.findmodes()
            t_mode = time.time() - t1
            print(f"  {CLR_GRAY}|--{CLR_RESET} Status   : {CLR_GREEN}Completed{CLR_RESET} in {CLR_BOLD}{t_mode:.2f} s{CLR_RESET}")

        print(f"{CLR_CYAN}" + "=" * 60)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Total single-pass runtime: {CLR_BOLD}{t_heat + t_mode:.2f} s{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Saved to: {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 60 + f"{CLR_RESET}")
    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    inspect_templates()