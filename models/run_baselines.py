# run_baselines.py
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


class AnalyticThermalBaseline:
    """Power-law thermal resistance scaling model fitted in log-space."""
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        log_X = np.hstack([np.ones((X.shape[0], 1)), np.log(np.maximum(X, 1e-12))])
        log_y = np.log(np.maximum(y, 1e-12))
        self.coef_, _, _, _ = np.linalg.lstsq(log_X, log_y, rcond=None)
        return self

    def predict(self, X):
        log_X = np.hstack([np.ones((X.shape[0], 1)), np.log(np.maximum(X, 1e-12))])
        return np.exp(log_X @ self.coef_)


def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12)) * 100.0)
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, mape, r2


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_baselines_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[BASELINES]{CLR_RESET} {CLR_CYAN}Phase 2: Surrogate Baselines & Learning Curve Benchmark{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

        feature_cols = [
            "wg_width_nm",
            "wg_height_nm",
            "heater_width_nm",
            "heater_gap_nm",
            "heater_power_mW"
        ]
        target_col = "delta_neff"

        id_df  = pd.read_csv(os.path.join(project_root, config["dataset"]["scalars_csv"]))
        ood_df = pd.read_csv(os.path.join(project_root, config["ood_dataset"]["scalars_csv"]))

        # Deterministic split: 900 train pool, 150 val, 150 ID test
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(id_df))
        train_pool_idx = perm[:900]
        val_idx        = perm[900:1050]
        test_id_idx    = perm[1050:]

        split_path = os.path.join(processed_dir, "split_indices.npz")
        np.savez(split_path, train_pool=train_pool_idx, val=val_idx, test_id=test_id_idx)
        print(f"{CLR_BLUE}[SPLIT]{CLR_RESET}  Saved fixed splits -> {CLR_GRAY}{split_path}{CLR_RESET}")
        print(f"  {CLR_GRAY}|--{CLR_RESET} Train pool: {len(train_pool_idx)} | Val: {len(val_idx)} | ID Test: {len(test_id_idx)} | OOD: {len(ood_df)}")

        X_test_id = id_df.iloc[test_id_idx][feature_cols].values
        y_test_id = id_df.iloc[test_id_idx][target_col].values

        ood_hp_df = ood_df[ood_df["ood_regime"] == "high_power"]
        ood_ge_df = ood_df[ood_df["ood_regime"] == "geom_extrap"]

        eval_sets = {
            "ID_Test": (X_test_id, y_test_id),
            "OOD_HighPower": (ood_hp_df[feature_cols].values, ood_hp_df[target_col].values),
            "OOD_GeomExtrap": (ood_ge_df[feature_cols].values, ood_ge_df[target_col].values),
        }

        train_sizes = [50, 100, 200, 400, 800]
        records = []

        for n_train in train_sizes:
            print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
            print(f"{CLR_MAGENTA}[TRAIN]{CLR_RESET}  Training size N = {CLR_BOLD}{n_train}{CLR_RESET}")

            sub_idx = train_pool_idx[:n_train]
            X_tr = id_df.iloc[sub_idx][feature_cols].values
            y_tr = id_df.iloc[sub_idx][target_col].values

            scaler_x = StandardScaler().fit(X_tr)
            X_tr_s = scaler_x.transform(X_tr)
            y_tr_scaled = y_tr * 1e3  # Scale target to O(1) for numerical stability

            models = {
                "Analytic": ("raw", AnalyticThermalBaseline()),
                "GP": (
                    "scaled",
                    GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) + WhiteKernel(noise_level=1e-5),
                        n_restarts_optimizer=2,
                        random_state=42
                    )
                ),
                "GBR": (
                    "scaled",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42
                    )
                ),
                "MLP": (
                    "scaled",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 128, 64),
                        activation="relu",
                        max_iter=1500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42
                    )
                )
            }

            for m_name, (mode_type, model) in models.items():
                t0 = time.time()
                if mode_type == "raw":
                    model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr_s, y_tr_scaled)
                train_time = time.time() - t0

                summary_parts = []
                for set_name, (X_ev, y_ev) in eval_sets.items():
                    if mode_type == "raw":
                        y_pred = model.predict(X_ev)
                    else:
                        y_pred = model.predict(scaler_x.transform(X_ev)) * 1e-3

                    mae, rmse, mape, r2 = compute_metrics(y_ev, y_pred)
                    records.append({
                        "model": m_name,
                        "n_train": n_train,
                        "eval_set": set_name,
                        "mae": mae,
                        "rmse": rmse,
                        "mape_pct": mape,
                        "r2": r2,
                        "train_time_s": round(train_time, 3)
                    })
                    summary_parts.append(f"{set_name}: MAPE={mape:.2f}% (R2={r2:.4f})")

                print(f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}{m_name:<9}{CLR_RESET} ({train_time:5.2f}s) | " + " | ".join(summary_parts))

        res_df = pd.DataFrame(records)
        csv_out = os.path.join(results_dir, "baseline_metrics.csv")
        res_df.to_csv(csv_out, index=False)

        # Plot learning curves for ID, OOD_HighPower, and OOD_GeomExtrap
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
        eval_order = ["ID_Test", "OOD_HighPower", "OOD_GeomExtrap"]
        titles = ["In-Distribution Test (N=150)", "OOD: High Power (60-90 mW)", "OOD: Geometry Extrap"]

        for ax, ev_name, title in zip(axes, eval_order, titles):
            sub = res_df[res_df["eval_set"] == ev_name]
            for m_name in ["Analytic", "GP", "GBR", "MLP"]:
                m_sub = sub[sub["model"] == m_name]
                ax.plot(m_sub["n_train"], m_sub["mape_pct"], marker="o", linewidth=2, label=m_name)
            ax.set_title(title)
            ax.set_xlabel("Training Samples (N)")
            ax.set_ylabel("MAPE (%)")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plot_out = os.path.join(results_dir, "baseline_learning_curves.png")
        plt.savefig(plot_out, dpi=200)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Metrics saved -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Plot saved    -> {CLR_GRAY}{plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved     -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()