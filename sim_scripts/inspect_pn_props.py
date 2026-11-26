# inspect_pn_props.py
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)


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


def get_children(session, scope_path):
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


def safe_get(session, obj, prop):
    try:
        return session.getnamed(obj, prop)
    except Exception:
        return "N/A"


def main():
    lumapi = load_lumapi()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ldev_path = os.path.join(project_root, "templates", "pn_phase_shifter", "PN_phase_shifter.ldev")
    lms_path  = os.path.join(project_root, "templates", "pn_phase_shifter", "PN_phase_shifter.lms")

    print("=" * 68)
    print(" [INSPECT] CHARGE (.ldev) Properties")
    print("=" * 68)
    with lumapi.DEVICE(filename=ldev_path, hide=True) as charge:
        charge.switchtolayout()

        # 1. Doping children
        doping_objs = get_children(charge, "::model::CHARGE::doping")
        print(f"Doping objects: {doping_objs}")
        for d in doping_objs:
            path = f"::model::CHARGE::doping::{d}"
            print(
                f"  |-- {d:<12}: type={safe_get(charge, path, 'dopant type')} | "
                f"conc={safe_get(charge, path, 'concentration')} | "
                f"x_min={safe_get(charge, path, 'x min')} | x_max={safe_get(charge, path, 'x max')}"
            )

        # 2. Boundary conditions (drain & source)
        for bc in ["drain", "source"]:
            path = f"::model::CHARGE::boundary conditions::{bc}"
            print(
                f"BC {bc:<8}: bc_mode={safe_get(charge, path, 'bc mode')} | "
                f"sweep_type={safe_get(charge, path, 'sweep type')} | "
                f"v_start={safe_get(charge, path, 'range start')} | "
                f"v_stop={safe_get(charge, path, 'range stop')} | "
                f"v_num={safe_get(charge, path, 'range num points')} | "
                f"v_single={safe_get(charge, path, 'voltage')}"
            )

        # 3. Waveguide & Monitor
        wg_p = "::model::geometry::waveguide"
        print(
            f"Waveguide   : x={safe_get(charge, wg_p, 'x')} | x_span={safe_get(charge, wg_p, 'x span')} | "
            f"y_min={safe_get(charge, wg_p, 'y min')} | y_max={safe_get(charge, wg_p, 'y max')}"
        )
        mon_p = "::model::CHARGE::monitor_charge"
        print(
            f"Monitor     : save_data={safe_get(charge, mon_p, 'save data')} | "
            f"filename={safe_get(charge, mon_p, 'filename')}"
        )

    print("\n" + "=" * 68)
    print(" [INSPECT] MODE (.lms) Properties")
    print("=" * 68)
    with lumapi.MODE(filename=lms_path, hide=True) as mode:
        mode.switchtolayout()
        print(
            f"Wg          : x={safe_get(mode, 'Wg', 'x')} | x_span={safe_get(mode, 'Wg', 'x span')} | "
            f"y_min={safe_get(mode, 'Wg', 'y min')} | y_max={safe_get(mode, 'Wg', 'y max')}"
        )
        print(
            f"FDE         : wavelength={safe_get(mode, 'FDE', 'wavelength')} | "
            f"x_span={safe_get(mode, 'FDE', 'x span')} | y_span={safe_get(mode, 'FDE', 'y span')}"
        )
        print(
            f"np attr     : enabled={safe_get(mode, 'np', 'enabled')} | "
            f"V_drain_idx={safe_get(mode, 'np', 'V_drain_index')} | "
            f"V_source_idx={safe_get(mode, 'np', 'V_source_index')}"
        )


if __name__ == "__main__":
    main()