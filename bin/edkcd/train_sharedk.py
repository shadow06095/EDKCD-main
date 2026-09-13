"""Training loop for shared-Koopman-operator baseline.

Adapted from train.py -- replaces per-edge K with shared NC×NC operator.
Original training pipeline remains untouched.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.metrics import (precision_recall_curve, roc_auc_score, auc)

from edkcd.models import VARP
from edkcd.datasets import PredictionDataset, numpy2tensor
from edkcd.utils import (compute_l1_loss, compute_encoder_quality_metrics,
                          opt_threshold_acc, compute_metrics_at_threshold)
from edkcd.models_sharedk import (VARP_SharedK, compute_sharedk_edge_scores,
                                   compute_sharedk_group_lasso)


def run_sharedk(data, learning_rate, reconstruction_epochs, joint_epochs,
                run_name, use_cuda, cuda_i, seed,
                n_encoder_layers, n_encoder_channels, lag,
                l1_weight, true_graph=None,
                encoder_type='mlp', kan_embed_dim=5, kan_hidden=None,
                output_dir=None,
                pred_weight=1.0, recon_weight=1.0, lin_weight=0.0,
                channel_independent=False,
                fast_mode=False):
    if output_dir is not None:
        save_path = Path(output_dir) / run_name
    else:
        save_path = Path("logs", run_name)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "models").mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(save_path / "training.log", level="INFO")

    nvars = data.shape[1]
    seq_len = data.shape[0] - 1
    df_np = data.T
    original_data_loader = DataLoader(
        PredictionDataset(
            numpy2tensor(df_np[np.newaxis, :, :seq_len]),
            numpy2tensor(df_np[np.newaxis, :, 1:seq_len + 1])
        ), batch_size=1, shuffle=False
    )

    device = torch.device(f"cuda:{cuda_i}" if use_cuda else "cpu")
    model = VARP_SharedK(
        nvars, lag=lag, seed=seed,
        encoder_layers=n_encoder_layers * [n_encoder_channels],
        decoder_layers=n_encoder_layers * [n_encoder_channels],
        encoder_type=encoder_type,
        kan_embed_dim=kan_embed_dim,
        kan_hidden=kan_hidden,
        channel_independent=channel_independent
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_epochs = reconstruction_epochs + joint_epochs
    history = []
    l1_w = l1_weight
    start_time = time.time()

    # ----- reconstruction stage -----
    for epoch in range(total_epochs):
        if epoch < reconstruction_epochs:
            model.train()
            train_total, train_recon, train_l1 = 0.0, 0.0, 0.0
            for X, Y in original_data_loader:
                X = X.to(device)
                optimizer.zero_grad()
                rec = model(X, False)
                recon_loss = criterion(rec, X)
                l1 = sum(compute_l1_loss(p)
                         for p in model.get_regularized_params(stage=0)) * l1_w
                loss = recon_loss + l1
                loss.backward()
                optimizer.step()
                train_total += loss.item()
                train_recon += recon_loss.item()
                train_l1 += l1.item()
            n_batches = len(original_data_loader)
            history.append({
                'epoch': epoch + 1, 'stage': 1,
                'total_loss': train_total / n_batches,
                'recon_loss': train_recon / n_batches,
                'l1_loss': train_l1 / n_batches, 'lin_loss': 0.0
            })
            if epoch % 50 == 0 or epoch == reconstruction_epochs - 1:
                logger.info(f'[SharedK Epoch {epoch+1:4d}] recon={train_recon/n_batches:.5f} L1={train_l1/n_batches:.5f}')

        # ----- joint training stage -----
        if epoch >= reconstruction_epochs:
            model.train()
            train_total, train_pred, train_recon, train_group, train_lin = 0.0, 0.0, 0.0, 0.0, 0.0
            for X, Y in original_data_loader:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                pred = model(X, True)
                rec = model(X, False)
                pred_loss = criterion(pred, Y)
                recon_loss = criterion(rec, X)

                # group lasso over sub-blocks of K_shared
                C = model.n_channels
                group_lasso = compute_sharedk_group_lasso(
                    model.K_shared, nvars, C, l1_w
                )
                reg_loss = group_lasso

                # linear consistency loss (same as original)
                lin_loss = torch.tensor(0.0, device=device)
                if lin_weight > 0:
                    B, N, T_x = X.shape
                    x_enc = model.encoder(X.reshape(B * N, 1, T_x))
                    x_enc = model.identity_pool(x_enc)
                    x_fused_X = model.compute_latent_fusion(x_enc)
                    T_y = Y.shape[2]
                    y_enc = model.encoder(Y.reshape(B * N, 1, T_y))
                    y_enc = model.identity_pool(y_enc)
                    start_t = model.lag
                    if T_x >= start_t + 1 and T_y >= start_t + 1:
                        z_next_pred = x_fused_X[:, :, start_t:]
                        z_next_true = y_enc[:, :, start_t:]
                        lin_loss = criterion(z_next_pred, z_next_true) * lin_weight

                loss = (pred_weight * pred_loss + recon_weight * recon_loss
                        + reg_loss + lin_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_total += loss.item()
                train_pred += (pred_weight * pred_loss).item()
                train_recon += (recon_weight * recon_loss).item()
                train_group += group_lasso.item()
                train_lin += lin_loss.item()

            n_batches = len(original_data_loader)
            history.append({
                'epoch': epoch + 1, 'stage': 2,
                'total_loss': train_total / n_batches,
                'pred_loss': train_pred / n_batches,
                'recon_loss': train_recon / n_batches,
                'l1_loss': train_group / n_batches,
                'lin_loss': train_lin / n_batches
            })
            if epoch % 50 == 0 or epoch == total_epochs - 1:
                logger.info(f'[SharedK Epoch {epoch+1:4d}] pred={train_pred/n_batches:.5f} '
                            f'recon={train_recon/n_batches:.5f} '
                            f'group={train_group/n_batches:.5f} '
                            f'lin={train_lin/n_batches:.5f}')

    # ----- causal discovery (param-based only, mask-based skipped for speed) -----
    model.eval()
    C = model.n_channels

    with torch.no_grad():
        K_shared_weights = model.K_shared.detach().cpu()
        par_causal_graph = compute_sharedk_edge_scores(
            K_shared_weights, nvars, C
        )
    # Use param-based graph as the main causal graph too
    causal_graph = par_causal_graph

    # ----- evaluation -----
    perm_auroc, perm_auprc = None, None
    par_auroc, par_auprc = None, None
    separation = None
    enc_metrics = None

    if true_graph is not None:
        a_l_bin = (true_graph > 0).astype(int)
        np.fill_diagonal(a_l_bin, 0)
        causal_np = causal_graph.cpu().numpy()
        par_np = par_causal_graph.cpu().numpy()
        mask_ = ~np.eye(nvars, dtype=bool).flatten()
        a_flat = a_l_bin.flatten()[mask_]
        mask_flat = causal_np.flatten()[mask_]
        par_flat = par_np.flatten()[mask_]

        true_mask = a_l_bin > 0
        false_mask = (a_l_bin == 0) & ~np.eye(nvars, dtype=bool)
        true_edges_strength = par_causal_graph[true_mask].numpy()
        false_edges_strength = par_causal_graph[false_mask].numpy()
        pooled_std = np.sqrt((true_edges_strength.std()**2 + false_edges_strength.std()**2) / 2)
        separation = ((true_edges_strength.mean() - false_edges_strength.mean())
                      / pooled_std if pooled_std > 0 else 0.0)

        enc_metrics = compute_encoder_quality_metrics(
            model, original_data_loader, device
        )

        perm_auroc = roc_auc_score(a_flat, mask_flat)
        prec_perm, rec_perm, _ = precision_recall_curve(a_flat, mask_flat)
        perm_auprc = auc(rec_perm, prec_perm)

        best_f1_m, best_th_m, best_pr_m, best_re_m, best_ha_m = 0.0, 0.0, 0.0, 0.0, 0.0
        for th in np.linspace(mask_flat.min(), mask_flat.max(), 200):
            p, r, f, h = compute_metrics_at_threshold(a_l_bin, causal_np, th)
            if f > best_f1_m:
                best_f1_m, best_th_m, best_pr_m, best_re_m, best_ha_m = f, th, p, r, h

        par_auroc = roc_auc_score(a_flat, par_flat)
        prec_par, rec_par, _ = precision_recall_curve(a_flat, par_flat)
        par_auprc = auc(rec_par, prec_par)

        best_f1_k, best_th_k, best_pr_k, best_re_k, best_ha_k = 0.0, 0.0, 0.0, 0.0, 0.0
        for th in np.linspace(par_flat.min(), par_flat.max(), 200):
            p, r, f, h = compute_metrics_at_threshold(a_l_bin, par_np, th)
            if f > best_f1_k:
                best_f1_k, best_th_k, best_pr_k, best_re_k, best_ha_k = f, th, p, r, h

        logger.info(f"SharedK Perm AUROC={perm_auroc:.4f} AUPRC={perm_auprc:.4f}")
        logger.info(f"SharedK Param AUROC={par_auroc:.4f} AUPRC={par_auprc:.4f}")
        logger.info(f"Separation={separation:.4f}")

        training_time = time.time() - start_time
        num_params = sum(p.numel() for p in model.parameters())

        return (causal_graph, par_causal_graph, history, enc_metrics,
                training_time, num_params,
                perm_auroc, perm_auprc, best_th_m, best_f1_m, best_pr_m, best_re_m, best_ha_m,
                par_auroc, par_auprc, best_th_k, best_f1_k, best_pr_k, best_re_k, best_ha_k,
                separation)

    training_time = time.time() - start_time
    num_params = sum(p.numel() for p in model.parameters())
    return (causal_graph, par_causal_graph, history, enc_metrics,
            training_time, num_params,
            perm_auroc, perm_auprc, None, None, None, None, None,
            par_auroc, par_auprc, None, None, None, None, None,
            separation)


import sys  # noqa: E402 (needed for logger.add above)