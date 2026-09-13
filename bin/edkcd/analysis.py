import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from loguru import logger


def select_representative_edges(true_graph, par_graph, causal_graph, top_k=3):
    N = true_graph.shape[0]
    true_bin = (true_graph > 0).astype(int)
    threshold = np.median(par_graph.flatten())
    tp_candidates = [(i, j) for i in range(N) for j in range(N)
                     if i != j and true_bin[i, j] == 1 and par_graph[i, j] >= threshold]
    fp_candidates = [(i, j) for i in range(N) for j in range(N)
                     if i != j and true_bin[i, j] == 0 and par_graph[i, j] >= threshold]
    fp_candidates = sorted(fp_candidates, key=lambda x: par_graph[x], reverse=True)[:top_k]
    fn_candidates = [(i, j) for i in range(N) for j in range(N)
                     if i != j and true_bin[i, j] == 1 and par_graph[i, j] < threshold]
    fn_candidates = sorted(fn_candidates, key=lambda x: par_graph[x])[:top_k]
    strongest = sorted([(i, j) for i in range(N) for j in range(N) if i != j],
                       key=lambda x: par_graph[x], reverse=True)[:top_k]
    tp_all = [(i, j) for i in range(N) for j in range(N) if i != j and true_bin[i, j] == 1]
    weakest_tp = sorted(tp_all, key=lambda x: par_graph[x])[:top_k]
    return {'tp': tp_candidates[:top_k], 'fp': fp_candidates, 'fn': fn_candidates,
            'strongest': strongest, 'weakest_tp': weakest_tp}


def analyze_edge_operator(K_weights, i, j):
    M = K_weights[i, j]
    if M.ndim == 3:
        C, _, L = M.shape
        M_flat = M.reshape(C, -1)
        U, S, Vt = np.linalg.svd(M_flat, full_matrices=False)
        Vt_reshaped = Vt.reshape(-1, C, L)
        weighted_right = (S @ Vt_reshaped.reshape(len(S), -1)).reshape(C, L)
        channel_imp_source = np.linalg.norm(weighted_right, axis=1)
        channel_imp_target = np.linalg.norm(U * S, axis=1)
        lag_importance = np.linalg.norm(weighted_right, axis=0)
        effective_rank = np.sum(S) / S[0] if S[0] > 0 else 1.0
        dominance = S[0] / np.sum(S) if np.sum(S) > 0 else 1.0
        edge_norm = np.linalg.norm(M)
        return {
            'singular_values': S, 'left_vecs': U, 'right_vecs': Vt,
            'channel_importance_source': channel_imp_source,
            'channel_importance_target': channel_imp_target,
            'lag_importance': lag_importance, 'dominance_ratio': dominance,
            'effective_rank': effective_rank, 'edge_norm': edge_norm, 'is_diagonal': False
        }
    else:
        C, L = M.shape
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        channel_imp = np.linalg.norm(M, axis=1)
        lag_importance = np.linalg.norm(M, axis=0)
        total = np.sum(S)
        dominance = S[0] / total if total > 0 else 1.0
        effective_rank = total / S[0] if S[0] > 0 else 1.0
        edge_norm = np.linalg.norm(M)
        return {
            'singular_values': S, 'left_vecs': U, 'right_vecs': Vt,
            'channel_importance_source': channel_imp,
            'channel_importance_target': channel_imp,
            'lag_importance': lag_importance, 'dominance_ratio': dominance,
            'effective_rank': effective_rank, 'edge_norm': edge_norm, 'is_diagonal': True
        }


def compute_consistency_metrics(par_graph, causal_graph, true_graph, best_th_par, best_th_perm):
    par_bin = (par_graph >= best_th_par).astype(int)
    np.fill_diagonal(par_bin, 0)
    perm_bin = (causal_graph >= best_th_perm).astype(int)
    np.fill_diagonal(perm_bin, 0)
    intersection = np.sum((par_bin == 1) & (perm_bin == 1))
    union = np.sum((par_bin == 1) | (perm_bin == 1))
    jaccard = intersection / union if union > 0 else 0.0
    return {'jaccard': jaccard}


def evaluate_spectral_causality(K_np, true_graph, par_graph, save_dir):
    from sklearn.metrics import roc_curve, roc_auc_score

    N = true_graph.shape[0]
    true_bin = (true_graph > 0).astype(int)
    rows = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            info = analyze_edge_operator(K_np, i, j)
            S = info['singular_values']
            total = np.sum(S)
            dr = S[0] / total if total > 0 else 1.0
            ratio_0_1 = S[0] / S[1] if len(S) > 1 and S[1] > 0 else 0.0
            p = S / total
            entropy = -np.sum(p * np.log(p + 1e-12))
            steepness = S[0] / S[-1] if len(S) > 1 and S[-1] > 0 else 0.0
            rows.append({
                'i': i, 'j': j, 'dr': dr, 'ratio_0_1': ratio_0_1,
                'entropy': entropy, 'steepness': steepness,
                'norm': par_graph[i, j], 'label': true_bin[i, j]
            })
    df_edges = pd.DataFrame(rows)
    y = df_edges['label'].values
    spectral_feature_names = ['dr', 'ratio_0_1', 'entropy', 'steepness']
    X_spec = df_edges[spectral_feature_names].values
    X_norm = df_edges[['norm']].values

    rho_dr_norm, _ = stats.spearmanr(df_edges['dr'], df_edges['norm'])
    logger.info(f"[Spectral] DR vs Frobenius Spearman rho={rho_dr_norm:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_spec = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_norm = RandomForestClassifier(n_estimators=100, random_state=42)

    auc_spec = cross_val_score(clf_spec, X_spec, y, cv=cv, scoring='roc_auc')
    auprc_spec = cross_val_score(clf_spec, X_spec, y, cv=cv, scoring='average_precision')
    logger.info(f"[Spectral] 4D spec 5-fold AUROC: {auc_spec.mean():.4f}+-{auc_spec.std():.4f}  "
                f"AUPRC: {auprc_spec.mean():.4f}+-{auprc_spec.std():.4f}")

    auc_norm = cross_val_score(clf_norm, X_norm, y, cv=cv, scoring='roc_auc')
    auprc_norm = cross_val_score(clf_norm, X_norm, y, cv=cv, scoring='average_precision')
    logger.info(f"[Spectral] Frobenius 5-fold AUROC: {auc_norm.mean():.4f}+-{auc_norm.std():.4f}  "
                f"AUPRC: {auprc_norm.mean():.4f}+-{auprc_norm.std():.4f}")

    X_combined = np.hstack([X_norm, X_spec])
    clf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
    auc_combined = cross_val_score(clf_combined, X_combined, y, cv=cv, scoring='roc_auc')
    auprc_combined = cross_val_score(clf_combined, X_combined, y, cv=cv, scoring='average_precision')
    logger.info(f"[Spectral] Combined 5-fold AUROC: {auc_combined.mean():.4f}+-{auc_combined.std():.4f}  "
                f"AUPRC: {auprc_combined.mean():.4f}+-{auprc_combined.std():.4f}")

    return df_edges, {
        'norm_auroc': auc_norm.mean(), 'norm_auroc_std': auc_norm.std(),
        'norm_auprc': auprc_norm.mean(), 'norm_auprc_std': auprc_norm.std(),
        'spec_auroc': auc_spec.mean(), 'spec_auroc_std': auc_spec.std(),
        'spec_auprc': auprc_spec.mean(), 'spec_auprc_std': auprc_spec.std(),
        'comb_auroc': auc_combined.mean(), 'comb_auroc_std': auc_combined.std(),
        'comb_auprc': auprc_combined.mean(), 'comb_auprc_std': auprc_combined.std(),
    }