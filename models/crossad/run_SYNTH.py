import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path
import torch
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score, average_precision_score
from vus.metrics import get_metrics
from exp.exp_TSBAD import Exp_Anomaly_Detection

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
from utils.synthetic_data import make_train, make_test


SEEDS = [2026, 2042, 2067]
RESULTS_DIR = HERE / 'results'

DEFAULT_LATENT_DIMS = [16]
DEFAULT_WINDOW_SIZES = [96]


def _parse_int_list(raw):
    return [int(x) for x in raw.replace(',', ' ').split() if x]


def parse_args():
    p = argparse.ArgumentParser(description="Run CrossAD on synthetic data")
    p.add_argument('--variant', choices=['cd', 'ci'], default='cd', help="cd = channel-dependent (Basic_CrossAD_CD), ci = channel-independent (Basic_CrossAD)")
    p.add_argument('--dataset', choices=['nproll', 'noiseflip'], default='noiseflip', help="Anomaly injection method on the synthetic test set (default: noiseflip)")
    p.add_argument('--latent-dims', type=_parse_int_list, default=DEFAULT_LATENT_DIMS, help=f"Comma- or space-separated list of d_model values (default: {DEFAULT_LATENT_DIMS})")
    p.add_argument('--window-sizes', type=_parse_int_list, default=DEFAULT_WINDOW_SIZES, help=f"Comma- or space-separated list of seq_len values (default: {DEFAULT_WINDOW_SIZES})")
    p.add_argument('--multi-seed', action='store_true', default=False, help="If set, run all SEEDS; otherwise only the first one (default: False)")
    return p.parse_args()


def run_one_combo(seed, variant, dataset, latent_dim, window_size):
    X_train = make_train(seed=seed)
    X_test, labels = make_test(seed=seed + 1, method=dataset)

    if torch.cuda.is_available():
        _use_gpu, _gpu_type = True, 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _use_gpu, _gpu_type = True, 'mps'
    else:
        _use_gpu, _gpu_type = False, 'cuda'

    args = SimpleNamespace(
        configs_path=str(HERE / 'configs/synthetic'),
        data='synthetic',
        use_gpu=_use_gpu,
        gpu=0,
        gpu_type=_gpu_type,
        use_multi_gpu=False,
        channel_dependent=(variant == 'cd'),
        latent_dim=latent_dim,
        window_size=window_size,
    )
    clf = Exp_Anomaly_Detection(args, id=0)
    clf.train(X_train)
    score = clf.test(X_test)
    L = len(score)
    y = labels[:L].astype(int)
    auc_roc = roc_auc_score(y, score)
    auc_pr = average_precision_score(y, score)
    slidingWindow = 100
    vus_res = get_metrics(score, y, metric='all', slidingWindow=slidingWindow)
    return {
        'seed': seed,
        'variant': variant,
        'latent_dim': latent_dim,
        'window_size': window_size,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'R-AUC-ROC': vus_res['R_AUC_ROC'],
        'R-AUC-PR': vus_res['R_AUC_PR'],
        'VUS-ROC': vus_res['VUS_ROC'],
        'VUS-PR': vus_res['VUS_PR'],
    }


def save_results(rows, variant, dataset, latent_dims, window_sizes):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"results-{variant}-{dataset.upper()}-latent{latent_dims}-window{window_sizes}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    cli = parse_args()
    seeds = SEEDS if cli.multi_seed else SEEDS[:1]
    print(f"Sweep: dataset={cli.dataset} variant={cli.variant} latent_dims={cli.latent_dims} window_sizes={cli.window_sizes} seeds={seeds}")
    rows = []
    out_path = None
    for seed in seeds:
        print(f"\nSeed {seed}")
        for latent_dim in cli.latent_dims:
            for window_size in cli.window_sizes:
                print(f"latent_dim={latent_dim} window_size={window_size}")
                results = run_one_combo(seed, cli.variant, cli.dataset, latent_dim, window_size)
                print(f"AUC-ROC: {results['AUC-ROC']:.4f}")
                print(f"AUC-PR: {results['AUC-PR']:.4f}")
                print(f"R-AUC-ROC: {results['R-AUC-ROC']:.4f}")
                print(f"R-AUC-PR: {results['R-AUC-PR']:.4f}")
                print(f"VUS-ROC: {results['VUS-ROC']:.4f}")
                print(f"VUS-PR: {results['VUS-PR']:.4f}")
                rows.append(results)
        out_path = save_results(rows, cli.variant, cli.dataset, cli.latent_dims, cli.window_sizes)
        print(f"Saved (through seed {seed}): {out_path}")


if __name__ == "__main__":
    main()
