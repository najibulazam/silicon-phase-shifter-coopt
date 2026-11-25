# train_field_pinn.py
import os
import re
import sys
import time
import warnings
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


class ParametricFieldPINN(nn.Module):
    """
    Physics-informed parametric field network.
    Predicts normalized temperature rise per unit power: dT_unit(x, y; geom),
    scales linearly with heater power P, and computes delta_neff via a
    differentiable first-order perturbation overlap layer.
    """
    def __init__(self, geom_dim=4, init_pert_scale=1.59):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(geom_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 64 * 4 * 4),
            nn.GELU()
        )
        # Upsample 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Softplus()  # Temperature rise dT >= 0 physically
        )
        # Learnable Poynting/index-contrast normalization factor for perturbation layer
        self.pert_scale = nn.Parameter(torch.tensor(float(init_pert_scale), dtype=torch.float32))

    def forward(self, geom_s, power_mW, dn_dT_map, E2_field):
        b = geom_s.shape[0]
        h = self.fc(geom_s).view(b, 64, 4, 4)
        dT_per_mW = self.decoder(h).squeeze(1)  # (B, 64, 64)
        dT_pred = dT_per_mW * power_mW.view(b, 1, 1)  # Exact physical linearity in P

        # Differentiable perturbation layer: delta_neff = scale * sum(dn/dT * dT * |E|^2)
        dneff_pred = self.pert_scale * torch.sum(dn_dT_map * dT_pred * E2_field, dim=(1, 2))
        return dT_pred, dneff_pred


def heat_pde_residual_loss(dT_pred):
    """
    2D steady-state Laplace residual (div(grad T) = 0) in source-free interior
    via central finite differences on the uniform 64x64 grid.
    """
    # Interior slice [1:-1, 1:-1]
    lap = (
        dT_pred[:, 2:, 1:-1]
        + dT_pred[:, :-2, 1:-1]
        + dT_pred[:, 1:-1, 2:]
        + dT_pred[:, 1:-1, :-2]
        - 4.0 * dT_pred[:, 1:-1, 1:-1]
    )
    return torch.mean(lap ** 2)


def build_dn_dT_maps(df, x_grid, y_grid, dn_dT_si, dn_dT_sio2):
    """Precompute spatial dn/dT(x, y) masks for each sample."""
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    n = len(df)
    maps = np.empty((n, len(x_grid), len(y_grid)), dtype=np.float32)
    wg_w = df["wg_width_nm"].values * 1e-9
    wg_h = df["wg_height_nm"].values * 1e-9
    for i in range(n):
        mask = (np.abs(X) <= wg_w[i] / 2.0) & (Y >= 0.0) & (Y <= wg_h[i])
        maps[i] = np.where(mask, dn_dT_si, dn_dT_sio2)
    return maps


def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12)) * 100.0)
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, mape, r2


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = os.path.join(project_root, config["project"].get("log_dir", "log"))
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_field_pinn_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{CLR_CYAN}" + "=" * 72)
        print(f" {CLR_BOLD}[PINN]{CLR_RESET} {CLR_CYAN}Phase 2: Parametric Field-Level PINN + Perturbation Layer{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")
        print(f"{CLR_BLUE}[DEVICE]{CLR_RESET} Running on: {CLR_BOLD}{device}{CLR_RESET}")

        geom_cols = ["wg_width_nm", "wg_height_nm", "heater_width_nm", "heater_gap_nm"]
        dn_dT_si = config["simulation"]["dn_dT_si"]
        dn_dT_sio2 = config["simulation"]["dn_dT_sio2"]
        T_ref = 300.0

        # Load ID & OOD datasets
        id_df = pd.read_csv(os.path.join(project_root, config["dataset"]["scalars_csv"]))
        with h5py.File(os.path.join(project_root, config["dataset"]["fields_h5"]), "r") as hf:
            x_grid = hf["x_grid"][:]
            y_grid = hf["y_grid"][:]
            T_id = hf["T_fields"][:] - T_ref
            E2_id = hf["E2_fields"][:]

        ood_df = pd.read_csv(os.path.join(project_root, config["ood_dataset"]["scalars_csv"]))
        with h5py.File(os.path.join(project_root, config["ood_dataset"]["fields_h5"]), "r") as hf:
            T_ood = hf["T_fields"][:] - T_ref
            E2_ood = hf["E2_fields"][:]

        dn_dT_id = build_dn_dT_maps(id_df, x_grid, y_grid, dn_dT_si, dn_dT_sio2)
        dn_dT_ood = build_dn_dT_maps(ood_df, x_grid, y_grid, dn_dT_si, dn_dT_sio2)

        # Load identical splits from run_baselines.py
        split_path = os.path.join(project_root, "data", "processed", "split_indices.npz")
        splits = np.load(split_path)
        train_pool_idx = splits["train_pool"]
        val_idx = splits["val"]
        test_id_idx = splits["test_id"]

        hp_mask = (ood_df["ood_regime"] == "high_power").values
        ge_mask = (ood_df["ood_regime"] == "geom_extrap").values

        train_sizes = [50, 100, 200, 400, 800]
        epochs = 250
        batch_size = 32
        lr = 2e-3
        w_pde = 0.05
        w_neff = 5.0

        records = []
        best_model_state = None
        best_scaler = None

        for n_train in train_sizes:
            print(f"{CLR_GRAY}" + "-" * 72 + f"{CLR_RESET}")
            print(f"{CLR_MAGENTA}[TRAIN]{CLR_RESET}  Field-PINN training size N = {CLR_BOLD}{n_train}{CLR_RESET} ({epochs} epochs)")
            t0 = time.time()

            sub_idx = train_pool_idx[:n_train]
            scaler = StandardScaler().fit(id_df.iloc[sub_idx][geom_cols].values)

            # Prepare training tensors
            X_tr = torch.tensor(scaler.transform(id_df.iloc[sub_idx][geom_cols].values), dtype=torch.float32, device=device)
            P_tr = torch.tensor(id_df.iloc[sub_idx]["heater_power_mW"].values, dtype=torch.float32, device=device)
            T_tr = torch.tensor(T_id[sub_idx], dtype=torch.float32, device=device)
            E2_tr = torch.tensor(E2_id[sub_idx], dtype=torch.float32, device=device)
            dn_tr = torch.tensor(dn_dT_id[sub_idx], dtype=torch.float32, device=device)
            y_tr = torch.tensor(id_df.iloc[sub_idx]["delta_neff"].values, dtype=torch.float32, device=device)

            model = ParametricFieldPINN(geom_dim=4, init_pert_scale=1.59).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            model.train()
            for ep in range(epochs):
                perm = torch.randperm(n_train, device=device)
                for start in range(0, n_train, batch_size):
                    idx = perm[start:start + batch_size]
                    optimizer.zero_grad()

                    dT_pred, dneff_pred = model(X_tr[idx], P_tr[idx], dn_tr[idx], E2_tr[idx])

                    loss_field = F.mse_loss(dT_pred, T_tr[idx])
                    loss_pde = heat_pde_residual_loss(dT_pred)
                    # Scale delta_neff by 1e3 so loss magnitudes are balanced
                    loss_neff = F.mse_loss(dneff_pred * 1e3, y_tr[idx] * 1e3)

                    loss = loss_field + w_pde * loss_pde + w_neff * loss_neff
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            train_time = time.time() - t0

            # Evaluate on ID_Test, OOD_HighPower, OOD_GeomExtrap
            model.eval()
            eval_specs = {
                "ID_Test": (
                    id_df.iloc[test_id_idx][geom_cols].values,
                    id_df.iloc[test_id_idx]["heater_power_mW"].values,
                    T_id[test_id_idx],
                    dn_dT_id[test_id_idx],
                    E2_id[test_id_idx],
                    id_df.iloc[test_id_idx]["delta_neff"].values
                ),
                "OOD_HighPower": (
                    ood_df.loc[hp_mask, geom_cols].values,
                    ood_df.loc[hp_mask, "heater_power_mW"].values,
                    T_ood[hp_mask],
                    dn_dT_ood[hp_mask],
                    E2_ood[hp_mask],
                    ood_df.loc[hp_mask, "delta_neff"].values
                ),
                "OOD_GeomExtrap": (
                    ood_df.loc[ge_mask, geom_cols].values,
                    ood_df.loc[ge_mask, "heater_power_mW"].values,
                    T_ood[ge_mask],
                    dn_dT_ood[ge_mask],
                    E2_ood[ge_mask],
                    ood_df.loc[ge_mask, "delta_neff"].values
                )
            }

            summary_parts = []
            with torch.no_grad():
                for set_name, (G_ev, P_ev, T_ev, dn_ev, E2_ev, y_ev) in eval_specs.items():
                    G_t = torch.tensor(scaler.transform(G_ev), dtype=torch.float32, device=device)
                    P_t = torch.tensor(P_ev, dtype=torch.float32, device=device)
                    dn_t = torch.tensor(dn_ev, dtype=torch.float32, device=device)
                    E2_t = torch.tensor(E2_ev, dtype=torch.float32, device=device)

                    dT_p, dneff_p = model(G_t, P_t, dn_t, E2_t)
                    dT_p_np = dT_p.cpu().numpy()
                    dneff_p_np = dneff_p.cpu().numpy()

                    mae, rmse, mape, r2 = compute_metrics(y_ev, dneff_p_np)
                    field_rmse_K = float(np.sqrt(np.mean((dT_p_np - T_ev) ** 2)))
                    pde_res = float(heat_pde_residual_loss(dT_p).item())

                    records.append({
                        "model": "Field_PINN",
                        "n_train": n_train,
                        "eval_set": set_name,
                        "mae": mae,
                        "rmse": rmse,
                        "mape_pct": mape,
                        "r2": r2,
                        "field_rmse_K": field_rmse_K,
                        "pde_residual": pde_res,
                        "pert_scale": float(model.pert_scale.item()),
                        "train_time_s": round(train_time, 2)
                    })
                    summary_parts.append(
                        f"{set_name}: MAPE={mape:.2f}% (R2={r2:.4f}, T_RMSE={field_rmse_K:.2f}K)"
                    )

            print(
                f"  {CLR_GRAY}|--{CLR_RESET} {CLR_GREEN}Field_PINN{CLR_RESET} "
                f"({train_time:5.1f}s, scale={model.pert_scale.item():.3f}) | " + " | ".join(summary_parts)
            )

            if n_train == 800:
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_scaler = scaler
                # Save sample field comparison for OOD_HighPower[0]
                with torch.no_grad():
                    G_ev, P_ev, T_ev, dn_ev, E2_ev, _ = eval_specs["OOD_HighPower"]
                    G_t = torch.tensor(scaler.transform(G_ev[:1]), dtype=torch.float32, device=device)
                    P_t = torch.tensor(P_ev[:1], dtype=torch.float32, device=device)
                    dn_t = torch.tensor(dn_ev[:1], dtype=torch.float32, device=device)
                    E2_t = torch.tensor(E2_ev[:1], dtype=torch.float32, device=device)
                    dT_sample_pred, _ = model(G_t, P_t, dn_t, E2_t)
                    T_true_plot = T_ev[0] + T_ref
                    T_pred_plot = dT_sample_pred[0].cpu().numpy() + T_ref
                    P_plot = P_ev[0]

        # Save metrics & model checkpoint
        pinn_df = pd.DataFrame(records)
        csv_out = os.path.join(results_dir, "pinn_metrics.csv")
        pinn_df.to_csv(csv_out, index=False)

        ckpt_path = os.path.join(models_dir, "field_pinn_N800.pt")
        torch.save({
            "model_state_dict": best_model_state,
            "scaler_mean": best_scaler.mean_,
            "scaler_scale": best_scaler.scale_,
            "geom_cols": geom_cols
        }, ckpt_path)

        # Plot 2D field prediction vs ground truth (OOD High Power sample)
        extent = [x_grid[0] * 1e6, x_grid[-1] * 1e6, y_grid[0] * 1e6, y_grid[-1] * 1e6]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        im0 = axes[0].imshow(T_true_plot.T, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        axes[0].set_title(f"Lumerical HEAT (OOD P={P_plot:.1f} mW)")
        axes[0].set_xlabel("x (um)")
        axes[0].set_ylabel("y (um)")
        plt.colorbar(im0, ax=axes[0], label="T (K)")

        im1 = axes[1].imshow(T_pred_plot.T, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        axes[1].set_title("Field-PINN Prediction")
        axes[1].set_xlabel("x (um)")
        axes[1].set_ylabel("y (um)")
        plt.colorbar(im1, ax=axes[1], label="T (K)")

        err_map = np.abs(T_pred_plot - T_true_plot)
        im2 = axes[2].imshow(err_map.T, origin="lower", extent=extent, aspect="auto", cmap="magma")
        axes[2].set_title(f"Absolute Error |Pred - True| (Max={err_map.max():.2f} K)")
        axes[2].set_xlabel("x (um)")
        axes[2].set_ylabel("y (um)")
        plt.colorbar(im2, ax=axes[2], label="|dT| (K)")

        plt.tight_layout()
        field_plot_out = os.path.join(results_dir, "pinn_ood_field_comparison.png")
        plt.savefig(field_plot_out, dpi=200)
        plt.close()

        print(f"{CLR_CYAN}" + "=" * 72)
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   PINN metrics saved -> {CLR_GRAY}{csv_out}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Checkpoint saved   -> {CLR_GRAY}{ckpt_path}{CLR_RESET}")
        print(f"{CLR_GREEN}[DONE]{CLR_RESET}   Field plot saved   -> {CLR_GRAY}{field_plot_out}{CLR_RESET}")
        print(f"{CLR_BLUE}[LOG]{CLR_RESET}    Log saved          -> {CLR_GRAY}{log_path}{CLR_RESET}")
        print(f"{CLR_CYAN}" + "=" * 72 + f"{CLR_RESET}")

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == "__main__":
    main()