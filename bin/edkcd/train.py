import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.metrics import (precision_recall_curve, roc_auc_score, auc)

from edkcd.models import VARP
from edkcd.datasets import PredictionDataset, numpy2tensor
from edkcd.utils import (compute_l1_loss, compute_edgewise_group_lasso,
                          compute_encoder_quality_metrics, log_confusion_details,
                          opt_threshold_acc, compute_metrics_at_threshold)
from edkcd.analysis import evaluate_spectral_causality


def run_pipeline(data, learning_rate, reconstruction_epochs, joint_epochs,
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
    model = VARP(
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

    # reconstruction stage
    for epoch in range(total_epochs):
        if epoch < reconstruction_epochs:
            model.train()
            train_total, train_recon, train_l1 = 0.0, 0.0, 0.0
            for X, Y in original_data_loader:
                X = X.to(device)
                optimizer.zero_grad()
                rec = model(X, False)
                recon_loss = criterion(rec, X)
                l1 = sum(compute_l1_loss(p) for p in model.get_regularized_params(stage=0)) * l1_w
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
                logger.info(f'[Epoch {epoch + 1:4d}] recon={train_recon / n_batches:.5f} L1={train_l1 / n_batches:.5f}')

        # joint training stage
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
                group_lasso = compute_edgewise_group_lasso(model.K, l1_w)
                reg_loss = group_lasso
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
                loss = (pred_weight * pred_loss + recon_weight * recon_loss + reg_loss + lin_loss)
                loss.backward()
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
                logger.info(f'[Epoch {epoch + 1:4d}] pred={train_pred / n_batches:.5f} '
                            f'recon={train_recon / n_batches:.5f} '
                            f'group={train_group / n_batches:.5f} '
                            f'lin={train_lin / n_batches:.5f}')

    # causal discovery via mask intervention
    model.eval()
    abs_gain = torch.zeros(nvars, nvars)
    with torch.no_grad():
        for X, Y_truth in original_data_loader:
            X, Y_truth = X.to(device), Y_truth.to(device)
            pred_base = model(X, True)
            base_err = (pred_base - Y_truth) ** 2
            K_orig = model.K.data.clone()
            for i in range(nvars):
                for j in range(nvars):
                    if i == j:
                        continue
                    K_mask = K_orig.clone()
                    K_mask[i, j] = 0.0
                    model.K.data = K_mask
                    pred_mask = model(X, True)
                    err_mask = (pred_mask - Y_truth) ** 2
                    gain = (err_mask - base_err).clamp(min=0)
                    abs_gain[i, j] = gain[0, j, :].mean().item()
            model.K.data = K_orig

    abs_gain_np = abs_gain.cpu().numpy()
    diag = np.diag(abs_gain_np)
    rel_gain = abs_gain_np / (diag.reshape(1, -1) + 1e-8)
    np.fill_diagonal(rel_gain, 0.0)
    causal_graph = torch.from_numpy(rel_gain).float()

    with torch.no_grad():
        K_weights = model.K.detach().cpu()
        par_causal_graph = K_weights.reshape(nvars, nvars, -1).norm(dim=2)
        K_np = K_weights.numpy()

    perm_auroc = None
    perm_auprc = None
    par_auroc = None
    par_auprc = None
    separation = None
    enc_metrics = None

    if true_graph is not None:
        a_l_bin = (true_graph > 0).astype(int)
        np.fill_diagonal(a_l_bin, 0)
        causal_np = causal_graph.cpu().numpy()
        par_np = par_causal_graph.cpu().numpy()
        mask = ~np.eye(nvars, dtype=bool).flatten()
        a_flat = a_l_bin.flatten()[mask]
        mask_flat = causal_np.flatten()[mask]
        par_flat = par_np.flatten()[mask]

        true_mask = a_l_bin > 0
        false_mask = (a_l_bin == 0) & ~np.eye(nvars, dtype=bool)
        true_edges_strength = par_causal_graph[true_mask].numpy()
        false_edges_strength = par_causal_graph[false_mask].numpy()
        pooled_std = np.sqrt((true_edges_strength.std() ** 2 + false_edges_strength.std() ** 2) / 2)
        separation = (true_edges_strength.mean() - false_edges_strength.mean()) / pooled_std if pooled_std > 0 else 0.0

        enc_metrics = compute_encoder_quality_metrics(model, original_data_loader, device)

        perm_auroc = roc_auc_score(a_flat, mask_flat)
        prec_perm, rec_perm, _ = precision_recall_curve(a_flat, mask_flat)
        perm_auprc = auc(rec_perm, prec_perm)

        opt_mask_acc = opt_threshold_acc(a_flat, mask_flat)
        th_acc_m = opt_mask_acc[0]
        p_acc_m, r_acc_m, f_acc_m, h_acc_m = compute_metrics_at_threshold(a_l_bin, causal_np, th_acc_m)

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

        logger.info(f"Perm AUROC={perm_auroc:.4f} AUPRC={perm_auprc:.4f}")
        logger.info(f"Param AUROC={par_auroc:.4f} AUPRC={par_auprc:.4f}")
        logger.info(f"Separation={separation:.4f}")
        logger.info(f"Edge strength: true={true_edges_strength.mean():.6f}+-{true_edges_strength.std():.6f}, "
                     f"false={false_edges_strength.mean():.6f}+-{false_edges_strength.std():.6f}")

        # Spectral analysis: SVD-based features with 5-fold cross-validation
        try:
            spectral_df, spectral_results = evaluate_spectral_causality(K_np, a_l_bin, par_np, save_path)
            spectral_df.to_csv(save_path / "spectral_features.csv", index=False)
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            spectral_results = None

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


def run_grid_search(datasets, structures, args):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = getattr(args, 'output_root', 'logs')
    logdir = Path(output_root) / f"{args.experiment}_{timestamp}"
    logdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for l in range(len(datasets)):
        d_l = datasets[l]
        a_l = structures[l]
        run_name = f"{args.experiment}_{l}"
        (causal_g, param_g, history, enc_metrics,
         train_time, num_params,
         perm_auroc, perm_auprc, best_th_m, best_f1_m, best_pr_m, best_re_m, best_ha_m,
         par_auroc, par_auprc, best_th_k, best_f1_k, best_pr_k, best_re_k, best_ha_k,
         separation) = run_pipeline(
            data=d_l, learning_rate=args.initial_lr,
            reconstruction_epochs=args.num_epochs_1, joint_epochs=args.num_epochs_2,
            run_name=run_name, use_cuda=args.use_cuda, cuda_i=args.cuda_i,
            seed=args.seed, n_encoder_layers=args.num_hidden_layers,
            n_encoder_channels=args.hidden_layer_size,
            lag=args.lag, l1_weight=args.l1_weight,
            true_graph=a_l, encoder_type=args.encoder_type,
            kan_embed_dim=args.kan_embed_dim, kan_hidden=args.kan_hidden,
            output_dir=str(logdir), pred_weight=args.pred_weight,
            recon_weight=args.recon_weight, lin_weight=args.lin_weight,
            channel_independent=args.channel_independent,
            fast_mode=args.fast_mode)

        if par_auroc is not None:
            print(f"Dataset {l}: Param AUROC={par_auroc:.4f}, AUPRC={par_auprc:.4f}")

        pd.DataFrame(param_g).to_csv(logdir / f"struct_param.csv", index=False)

        if enc_metrics is not None:
            record = {
                'dataset': l, 'encoder': args.encoder_type,
                'recon_mse': enc_metrics['recon_mse'], 'recon_r2': enc_metrics['recon_r2'],
                'pred_mse': enc_metrics['pred_mse'], 'effective_rank': enc_metrics['effective_rank'],
                'perm_auroc': perm_auroc, 'perm_auprc': perm_auprc,
                'param_auroc': par_auroc, 'param_auprc': par_auprc,
                'separation': separation, 'num_params': num_params, 'training_time': train_time
            }
            all_metrics.append(record)

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        display_cols = ['dataset', 'encoder', 'recon_mse', 'recon_r2', 'pred_mse',
                        'effective_rank', 'perm_auroc', 'perm_auprc',
                        'param_auroc', 'param_auprc', 'separation',
                        'num_params', 'training_time']
        print("\n===== Encoder comparison =====")
        print(df[display_cols].to_string(index=False))
        df.to_csv(logdir / "comparison_table.csv", index=False)

        mean_cols = ['perm_auroc', 'perm_auprc', 'param_auroc', 'param_auprc',
                     'recon_r2', 'effective_rank', 'separation', 'training_time']
        print("\n===== Final results (mean +- std) =====")
        summary_rows = []
        for col in mean_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"  {col:20s} = {mean_val:.4f} +- {std_val:.4f}")
                summary_rows.append({'metric': col, 'mean': mean_val, 'std': std_val})
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(logdir / "summary_mean_std.csv", index=False)

    print(f"Results saved to: {logdir.resolve()}")