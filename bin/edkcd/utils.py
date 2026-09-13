import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (precision_recall_curve, roc_auc_score, auc,
                             precision_score, recall_score, f1_score, hamming_loss)


def opt_threshold_acc(y_true, y_pred):
    A = sorted(zip(y_true, y_pred), key=lambda x: x[1])
    total = len(A)
    tp = len([1 for x in A if x[0] == 1])
    tn = 0
    best = (0, 0)
    for x in A:
        th = x[1]
        if x[0] == 1:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        if acc > best[1]:
            best = (th, acc)
    return best


def compute_metrics_at_threshold(y_true, y_score, th):
    y_pred = (y_score >= th).astype(int)
    mask = ~np.eye(len(y_true), dtype=bool).flatten()
    y_t = y_true.flatten()[mask]
    y_p = y_pred.flatten()[mask]
    if y_p.sum() == 0:
        return 0.0, 0.0, 0.0, hamming_loss(y_t, y_p)
    p = precision_score(y_t, y_p, zero_division=0)
    r = recall_score(y_t, y_p, zero_division=0)
    f = f1_score(y_t, y_p, zero_division=0)
    h = hamming_loss(y_t, y_p)
    return p, r, f, h


def compute_l1_loss(w):
    return torch.abs(w).mean()


def compute_edgewise_group_lasso(K, group_lambda):
    N = K.shape[0]
    K_edges = K.reshape(N * N, -1)
    group_norms = torch.norm(K_edges, dim=1)
    return group_lambda * group_norms.mean()


def compute_encoder_quality_metrics(model, dataloader, device):
    model.eval()
    total_recon_loss = 0.0
    total_pred_loss = 0.0
    total_var = 0.0
    all_z = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            rec = model(X, False)
            recon_loss = criterion(rec, X)
            pred = model(X, True)
            pred_loss = criterion(pred, Y)
            total_recon_loss += recon_loss.item() * X.size(0)
            total_pred_loss += pred_loss.item() * X.size(0)
            total_var += torch.var(X, dim=(1, 2)).sum().item()
            B, N, T = X.shape
            x_enc = model.encoder(X.reshape(B * N, 1, T))
            x_enc = model.identity_pool(x_enc)
            z_mean = x_enc.mean(dim=2)
            all_z.append(z_mean.cpu())
    n = len(dataloader.dataset)
    recon_mse = total_recon_loss / n
    pred_mse = total_pred_loss / n
    var = total_var / n
    recon_r2 = 1 - recon_mse / var if var > 1e-8 else 0.0
    all_z = torch.cat(all_z, dim=0).numpy()
    _, S, _ = np.linalg.svd(all_z, full_matrices=False)
    effective_rank = np.sum(S) / S[0] if S[0] > 0 else 0.0
    return {
        'recon_mse': recon_mse, 'recon_r2': recon_r2, 'pred_mse': pred_mse,
        'latent_singular_values_top5': S[:5], 'effective_rank': effective_rank
    }


def log_confusion_details(a_l_bin, causal_graph, par_graph, best_th_perm, best_th_param):
    N = a_l_bin.shape[0]
    perm_pred = (causal_graph >= best_th_perm).astype(int)
    np.fill_diagonal(perm_pred, 0)
    fn_perm = [(i, j) for i in range(N) for j in range(N) if i != j and a_l_bin[i, j] == 1 and perm_pred[i, j] == 0]
    fp_perm = [(i, j) for i in range(N) for j in range(N) if i != j and a_l_bin[i, j] == 0 and perm_pred[i, j] == 1]
    param_pred = (par_graph >= best_th_param).astype(int)
    np.fill_diagonal(param_pred, 0)
    fn_param = [(i, j) for i in range(N) for j in range(N) if i != j and a_l_bin[i, j] == 1 and param_pred[i, j] == 0]
    fp_param = [(i, j) for i in range(N) for j in range(N) if i != j and a_l_bin[i, j] == 0 and param_pred[i, j] == 1]
    if fp_param:
        for pair in fp_param:
            logger.info(f"  {pair} norm: {par_graph[pair]:.6f}")