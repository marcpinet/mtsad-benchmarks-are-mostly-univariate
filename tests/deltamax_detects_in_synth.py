import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.synthetic_data import make_test, ANOM_SEGS, CTX_SIZE, SEED
from utils.tools import max_delta_rho_lagged


METHODS = ['pearson', 'spearman', 'dcor']
INJECT_METHODS = ['noiseflip', 'nproll']
CATEGORIES = ['CROSS-CHANNEL-ONLY', 'BOTH', 'UNIVARIATE', 'UNDETECTED']
CORR_THR = 0.1
Z_THR = 3.0
N_SEEDS = 1000


def classify_segment(X, s, e, corr_method):
    ctx, anom = X[s - CTX_SIZE:s], X[s:e]
    mu = ctx.mean(axis=0)
    sigma = np.where(ctx.std(axis=0) < 1e-4, 1.0, ctx.std(axis=0))
    z_abs = np.abs(anom - mu) / sigma
    max_z = float(z_abs.max())
    uni_present = max_z >= Z_THR
    d = max_delta_rho_lagged(anom, ctx, method=corr_method)
    cross_channel = (not np.isnan(d)) and d > CORR_THR
    if cross_channel and not uni_present:
        cat = 'CROSS-CHANNEL-ONLY'
    elif cross_channel and uni_present:
        cat = 'BOTH'
    elif not cross_channel and uni_present:
        cat = 'UNIVARIATE'
    else:
        cat = 'UNDETECTED'
    return cat, max_z, d


def main():
    counts = {
        inj: {m: dict.fromkeys(CATEGORIES, 0) for m in METHODS}
        for inj in INJECT_METHODS
    }
    z_vals = {inj: {m: [] for m in METHODS} for inj in INJECT_METHODS}
    d_vals = {inj: {m: [] for m in METHODS} for inj in INJECT_METHODS}

    total = N_SEEDS * len(INJECT_METHODS) * len(ANOM_SEGS) * len(METHODS)
    pbar = tqdm(total=total, desc='Evaluating')
    for seed_offset in range(N_SEEDS):
        seed = SEED + seed_offset
        for inj in INJECT_METHODS:
            X, _ = make_test(seed=seed, method=inj)
            for (s, e) in ANOM_SEGS:
                for m in METHODS:
                    cat, mz, d = classify_segment(X, s, e, m)
                    counts[inj][m][cat] += 1
                    z_vals[inj][m].append(mz)
                    d_vals[inj][m].append(d)
                    pbar.update(1)
    pbar.close()

    print(f"\nThresholds: CORR_THR={CORR_THR}, Z_THR={Z_THR}")
    print(f"Seeds: {N_SEEDS}, segments per run: {len(ANOM_SEGS)}, total per (inject, corr): {N_SEEDS * len(ANOM_SEGS)}\n")
    for inj in INJECT_METHODS:
        rows = []
        for m in METHODS:
            row = {'corr_method': m}
            row.update(counts[inj][m])
            row['avg_z_max'] = float(np.nanmean(z_vals[inj][m]))
            row['avg_delta'] = float(np.nanmean(d_vals[inj][m]))
            rows.append(row)
        print(f"=== Injection: {inj} ===")
        print(tabulate(pd.DataFrame(rows), headers='keys', tablefmt='psql', showindex=False, floatfmt='.4f'))
        print()


if __name__ == '__main__':
    main()
