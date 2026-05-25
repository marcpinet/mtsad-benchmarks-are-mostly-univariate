import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from vus.metrics import get_metrics
from catch import CATCH

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
from utils.synthetic_data import make_train, make_test


SEEDS = [2026, 2042, 2067]
RESULTS_DIR = HERE / 'results'

DEFAULT_LATENT_DIMS = [128]
DEFAULT_WINDOW_SIZES = [192]


def _parse_int_list(raw):
    return [int(x) for x in raw.replace(',', ' ').split() if x]


def parse_args():
    p = argparse.ArgumentParser(description="Run CATCH on synthetic data")
    p.add_argument('--dataset', choices=['nproll', 'noiseflip'], default='noiseflip', help="Anomaly injection method on the synthetic test set (default: noiseflip)")
    p.add_argument('--latent-dims', type=_parse_int_list, default=DEFAULT_LATENT_DIMS, help=f"Comma- or space-separated list of d_model values (default: {DEFAULT_LATENT_DIMS})")
    p.add_argument('--window-sizes', type=_parse_int_list, default=DEFAULT_WINDOW_SIZES, help=f"Comma- or space-separated list of seq_len values (default: {DEFAULT_WINDOW_SIZES})")
    p.add_argument('--multi-seed', action='store_true', default=False, help="If set, run all SEEDS; otherwise only the first one (default: False)")
    return p.parse_args()


def to_df(X):
    idx = pd.date_range("1970-01-01", periods=len(X), freq="s")
    cols = [f"c{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=idx, columns=cols)


def _patch_params(window_size):
    """CATCH's defaults assume seq_len >= 32 (training patch_size=16) and
    >= 32 for inference (inference_patch_size=32). For shorter windows we
    scale patch sizes / strides down proportionally so unfold() doesn't blow
    up with 'maximum size for tensor at dimension 1 is W but size is P'."""
    patch_size = min(16, max(2, window_size // 4))
    patch_stride = max(1, patch_size // 2)
    inference_patch_size = min(32, max(4, window_size // 2))
    inference_patch_stride = 1
    return patch_size, patch_stride, inference_patch_size, inference_patch_stride


def run_one_combo(seed, dataset, latent_dim, window_size):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = make_train(seed=seed)
    X_test, labels = make_test(seed=seed + 1, method=dataset)

    df_train = to_df(X_train)
    df_test = to_df(X_test)
    ps, pstr, ips, ipstr = _patch_params(window_size)
    clf = CATCH(num_epochs=10, batch_size=128, seq_len=window_size,
                d_model=latent_dim, patience=3,
                patch_size=ps, patch_stride=pstr,
                inference_patch_size=ips, inference_patch_stride=ipstr)
    clf.detect_fit(df_train, df_test)
    score, _ = clf.detect_score(df_test)
    L = len(score)
    y = labels[:L].astype(int)
    auc_roc = roc_auc_score(y, score)
    auc_pr = average_precision_score(y, score)
    slidingWindow = 100
    vus_res = get_metrics(score, y, metric='all', slidingWindow=slidingWindow)
    return {
        'seed': seed,
        'latent_dim': latent_dim,
        'window_size': window_size,
        'scored_points': L,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'R-AUC-ROC': vus_res['R_AUC_ROC'],
        'R-AUC-PR': vus_res['R_AUC_PR'],
        'VUS-ROC': vus_res['VUS_ROC'],
        'VUS-PR': vus_res['VUS_PR'],
    }


def save_results(rows, dataset, latent_dims, window_sizes):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"results-{dataset.upper()}-latent{latent_dims}-window{window_sizes}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    cli = parse_args()
    seeds = SEEDS if cli.multi_seed else SEEDS[:1]
    print(f"Sweep: dataset={cli.dataset} latent_dims={cli.latent_dims} window_sizes={cli.window_sizes} seeds={seeds}")
    rows = []
    out_path = None
    for seed in seeds:
        print(f"\nSeed {seed}")
        for latent_dim in cli.latent_dims:
            for window_size in cli.window_sizes:
                print(f"latent_dim={latent_dim} window_size={window_size}")
                results = run_one_combo(seed, cli.dataset, latent_dim, window_size)
                print(f"Scored points: {results['scored_points']}")
                print(f"AUC-ROC: {results['AUC-ROC']:.4f}")
                print(f"AUC-PR: {results['AUC-PR']:.4f}")
                print(f"R-AUC-ROC: {results['R-AUC-ROC']:.4f}")
                print(f"R-AUC-PR: {results['R-AUC-PR']:.4f}")
                print(f"VUS-ROC: {results['VUS-ROC']:.4f}")
                print(f"VUS-PR: {results['VUS-PR']:.4f}")
                rows.append(results)
        out_path = save_results(rows, cli.dataset, cli.latent_dims, cli.window_sizes)
        print(f"Saved (through seed {seed}): {out_path}")


if __name__ == "__main__":
    main()
