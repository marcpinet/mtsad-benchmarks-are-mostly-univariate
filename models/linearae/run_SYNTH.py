import sys
import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from vus.metrics import get_metrics

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
from model import LinearAECI, LinearAECD
from utils.synthetic_data import make_train, make_test, N_CHANNELS


SEEDS = [2026, 2042, 2067]
RESULTS_DIR = HERE / 'results'

DEFAULT_LATENT_DIMS = [32]
DEFAULT_WINDOW_SIZES = [64]

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
SLIDING_WINDOW = 100


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(s, device):
    np.random.seed(s)
    torch.manual_seed(s)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(s)
    elif device.type == "mps":
        torch.mps.manual_seed(s)


def _parse_int_list(raw):
    return [int(x) for x in raw.replace(',', ' ').split() if x]


def parse_args():
    p = argparse.ArgumentParser(description="Run LinearAE on synthetic data")
    p.add_argument('--variant', choices=['cd', 'ci'], default='cd', help="cd = channel-dependent (LinearAECD), ci = channel-independent (LinearAECI)")
    p.add_argument('--dataset', choices=['nproll', 'noiseflip'], default='noiseflip', help="Anomaly injection method on the synthetic test set (default: noiseflip)")
    p.add_argument('--latent-dims', type=_parse_int_list, default=DEFAULT_LATENT_DIMS, help=f"Comma- or space-separated list of latent_dim values (default: {DEFAULT_LATENT_DIMS})")
    p.add_argument('--window-sizes', type=_parse_int_list, default=DEFAULT_WINDOW_SIZES, help=f"Comma- or space-separated list of window_size values (default: {DEFAULT_WINDOW_SIZES})")
    p.add_argument('--multi-seed', action='store_true', default=False, help="If set, run all SEEDS; otherwise only the first one (default: False)")
    return p.parse_args()


class Windows(Dataset):
    def __init__(self, data, labels=None, w=64, stride=1):
        self.data = data
        self.labels = labels
        self.w = w
        self.stride = stride
        self.n = max(0, (len(data) - w) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        x = torch.from_numpy(self.data[s:s + self.w])
        if self.labels is not None:
            return x, torch.from_numpy(self.labels[s:s + self.w])
        return (x,)


def _train(model, train_loader, device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            opt.zero_grad()
            orig, recon = model(x)
            loss = F.mse_loss(recon, orig)
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep + 1}: {total / len(train_loader):.6f}")
    return model


def _evaluate(model, test_loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, l in test_loader:
            orig, recon = model(x.to(device))
            scores.append(F.mse_loss(recon, orig, reduction='none').mean(dim=2).cpu().numpy())
            labels.append(l.numpy())
    s = np.concatenate(scores).reshape(-1)
    y = np.concatenate(labels).reshape(-1)
    return s, y


def run_one_combo(seed, variant, dataset, latent_dim, window_size):
    device = _device()
    _set_seed(seed, device)

    X_train = make_train(seed=seed).astype(np.float32)
    X_test, labels = make_test(seed=seed + 1, method=dataset)
    X_test = X_test.astype(np.float32)
    labels = labels.astype(np.float32)

    mu, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    train_loader = DataLoader(Windows(X_train, w=window_size, stride=1), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(Windows(X_test, labels, w=window_size, stride=window_size), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if variant == 'cd':
        model = LinearAECD(window_size=window_size, latent_dim=latent_dim, n_channels=N_CHANNELS)
    else:
        model = LinearAECI(window_size=window_size, latent_dim=latent_dim, n_channels=N_CHANNELS)

    model = _train(model, train_loader, device)
    score, y = _evaluate(model, test_loader, device)
    y_int = y.astype(int)
    auc_roc = roc_auc_score(y_int, score)
    auc_pr = average_precision_score(y_int, score)
    vus_res = get_metrics(score, y_int, metric='all', slidingWindow=SLIDING_WINDOW)
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
