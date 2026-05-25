import os
import re
import dcor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler


DEFAULT_CTX_SIZE = 300
DEFAULT_CORR_THR = 0.1
DEFAULT_Z_THR = 3.0
DEFAULT_MAX_LAG = 192  # most deep learning models use 192 >= t >= 64 timesteps for their sliding window.
DEFAULT_MIN_LEN = 10  # we consider that correlation anomalies cannot be reliably detected with fewer than 10 timesteps

METHODS = [
    'pearson',
    'spearman',
    'dcor'
]
LABELS_LONG = [
    "UNIVARIATE",
    "BOTH",
    "CROSS-CHANNEL",
    "UNDETECTED",
    "INSUFFICIENT_CONTEXT",
]
LABELS_SHORT = [
    "UNIVARIATE",
    "UNDETECTED",
    "INSUFFICIENT_CONTEXT"
]
LABEL_PAIR = {
    (1, 1): "BOTH",
    (1, 0): "UNIVARIATE",
    (0, 1): "CROSS-CHANNEL",
    (0, 0): "UNDETECTED",
}


def _assets_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, '..', '..'))
    path = os.path.join(root, 'assets')
    os.makedirs(path, exist_ok=True)
    return path


def _slugify(text):
    s = re.sub(r'[^a-z0-9]+', '_', (text or '').lower()).strip('_')
    return s or 'figure'


def _detect_notebook_name():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return None
        ns = ip.user_ns
        for key in ('__vsc_ipynb_file__', '__session__'):
            v = ns.get(key)
            if isinstance(v, str) and v.endswith('.ipynb'):
                return os.path.splitext(os.path.basename(v))[0]
    except Exception:
        pass
    return None


def save_current_fig_as_pdf(default_stem, filename=None, title=None):
    if filename is None:
        stem = _slugify(title) if title else default_stem
        suffix = _detect_notebook_name()
        filename = f"{stem}__{suffix}.pdf" if suffix else f"{stem}.pdf"
    elif not filename.lower().endswith('.pdf'):
        filename = f"{filename}.pdf"
    path = os.path.join(_assets_dir(), filename)
    plt.savefig(path, format='pdf', bbox_inches='tight')
    return path


# ----------------------------------------------------------------------


def find_contiguous_segments(labels):
    """Return [(start, end), ...] for each contiguous run of label == 1."""
    d = np.diff(np.concatenate(([0], labels, [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def preceding_normal_context(s, X, normal_mask, ctx_size=DEFAULT_CTX_SIZE, min_size=None):
    """Last up to ctx_size normal points strictly before index s. Returns None
    if fewer than min_size are available. By default min_size == ctx_size."""
    if min_size is None:
        min_size = ctx_size
    idx = np.flatnonzero(normal_mask[:s])
    if len(idx) < min_size:
        return None
    return X[idx[-ctx_size:]]


def count_deviant_channels(anom, ctx, z_thr=DEFAULT_Z_THR):
    """Number of channels whose maximum |z| inside the anomaly exceeds z_thr.
    Constant context channels (sigma ~= 0) get sigma replaced by 1 to avoid division by 0."""
    mu = ctx.mean(axis=0)
    sigma = ctx.std(axis=0)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    z_per_channel = np.max(np.abs(anom - mu) / sigma, axis=0)
    return int(np.sum(z_per_channel > z_thr))


def max_z_score(anom, ctx):
    """Max |z| over all channels and all timesteps of the anomaly."""
    mu = ctx.mean(axis=0)
    sigma = ctx.std(axis=0)
    sigma = np.where(sigma < 1e-4, 1.0, sigma)
    return float(np.max(np.abs(anom - mu) / sigma))


def extract_clean_segments(features_array, normal_mask, max_lag=DEFAULT_MAX_LAG):
    """Split a 2D array into contiguous normal segments of length >= 2 * max_lag.
    Used for compute_lagged_correlations."""
    L_min = 2 * max_lag
    indices = np.flatnonzero(normal_mask)
    if len(indices) <= 1:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    return [features_array[g] for g in np.split(indices, breaks) if len(g) >= L_min]


# ----------------------------------------------------------------------


def spearman_matrix(X):
    R = np.apply_along_axis(rankdata, 0, X)
    return np.corrcoef(R.T)


def dcor_matrix(X):
    _, C = X.shape
    iu, ju = np.triu_indices(C, k=1)
    vals = dcor.rowwise(
        dcor.u_distance_correlation_sqr,
        [X[:, i] for i in iu],
        [X[:, j] for j in ju],
    )
    vals = np.where(np.isfinite(vals), vals, np.nan)
    vals = np.clip(vals, 0.0, 1.0)
    M = np.zeros((C, C))
    M[iu, ju] = vals
    M[ju, iu] = vals
    return M


# ----------------------------------------------------------------------


def _lagged_pearson(segments, max_lag, progress=True, label='Pearson'):
    C = segments[0].shape[1]
    n_lags = 2 * max_lag + 1
    lag_range = np.arange(-max_lag, max_lag + 1)
    weighted_sum = np.zeros((C, C, n_lags))
    total_weight = np.zeros((C, C, n_lags))

    iterator = tqdm(segments, desc=f"Segments ({label})") if progress else segments
    for seg in iterator:
        n_k = len(seg) - max_lag
        if n_k <= 0:
            continue
        L_s = len(seg)
        for idx, lag in enumerate(lag_range):
            a0, b0 = max(0, -lag), max(0, lag)
            X = seg[a0:a0 + n_k]
            Y = seg[b0:b0 + n_k]
            X_c = X - X.mean(axis=0, keepdims=True)
            Y_c = Y - Y.mean(axis=0, keepdims=True)
            cov = (X_c.T @ Y_c) / n_k
            var_X = (X_c * X_c).sum(axis=0) / n_k
            var_Y = (Y_c * Y_c).sum(axis=0) / n_k
            denom = np.sqrt(np.outer(np.maximum(var_X, 0.0), np.maximum(var_Y, 0.0)))
            with np.errstate(invalid='ignore', divide='ignore'):
                rho_seg = np.where(denom > 1e-12, cov / denom, np.nan)
            valid = np.isfinite(rho_seg)
            weighted_sum[:, :, idx] += np.where(valid, L_s * rho_seg, 0.0)
            total_weight[:, :, idx] += np.where(valid, L_s, 0.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total_weight > 0, weighted_sum / total_weight, np.nan)


def _lagged_spearman(segments, max_lag, progress=True):
    if not segments:
        raise ValueError("_lagged_spearman: empty segments list")
    ranked = [np.apply_along_axis(rankdata, 0, seg) for seg in segments]
    return _lagged_pearson(ranked, max_lag, progress=progress, label='Spearman')

def compute_lagged_correlations(segments, max_lag=DEFAULT_MAX_LAG, methods=('pearson', 'spearman'), progress=True):
    """Aggregate lagged correlation over a list of clean segments, for each requested
    method. Returns {method: array of shape (C, C, 2*max_lag+1)}.
    """
    if not segments:
        raise ValueError("compute_lagged_correlations: empty segments list")
    unknown = set(methods) - {'pearson', 'spearman'}
    if unknown:
        raise ValueError(
            f"Unknown / unsupported methods for lagged correlation: {sorted(unknown)}. "
            f"Only 'pearson' and 'spearman' are supported."
        )

    out = {}
    if 'pearson' in methods:
        out['pearson'] = _lagged_pearson(segments, max_lag, progress=progress, label='Pearson')
    if 'spearman' in methods:
        out['spearman'] = _lagged_spearman(segments, max_lag, progress=progress)
    return out


def _lagged_strength_scalar(value, method):
    """Method-specific dependence strength for a single (lag, method)
    candidate. Pearson/Spearman: |r|. dcor: raw value, gated to -inf when
    r <= 0 (bias-correction noise on independent pairs).
    """
    if not np.isfinite(value):
        return -np.inf
    if method == 'dcor':
        return value if value > 0 else -np.inf
    return abs(value)


def _lagged_strength_array(arr, method):
    """Vectorized _lagged_strength_scalar for argmax over lags."""
    arr = np.asarray(arr, dtype=float)
    if method == 'dcor':
        return np.where(np.isfinite(arr) & (arr > 0), arr, -np.inf)
    return np.where(np.isfinite(arr), np.abs(arr), -np.inf)


def find_dominant_lagged_pairs(lagged_corrs, max_lag, threshold=DEFAULT_CORR_THR):
    """For each method m in lagged_corrs and each pair (i, j), pick the lag
    with maximum dependence strength on m's own tensor, and keep the pair if
    it beats lag-0 (for that same method). Each method is evaluated
    independently: a pair may appear up to once per method. Returns 6-tuples
    (i, j, r0, lag, rmax, m), sorted by strength descending.
    """
    if not isinstance(lagged_corrs, dict):
        lagged_corrs = {'pearson': lagged_corrs}
    sample = next(iter(lagged_corrs.values()))
    C = sample.shape[0]
    lag_0_idx = max_lag
    lag_range = np.arange(-max_lag, max_lag + 1)
    lags_no_zero = np.array([l for l in lag_range if l != 0])

    pairs = []
    for m, arr in lagged_corrs.items():
        for i in range(C):
            for j in range(i + 1, C):
                corrs_no_zero = np.concatenate(
                    [arr[i, j, :lag_0_idx], arr[i, j, lag_0_idx + 1:]]
                )
                strengths = _lagged_strength_array(corrs_no_zero, m)
                if not np.any(strengths > -np.inf):
                    continue
                max_idx = int(np.argmax(strengths))
                rmax = corrs_no_zero[max_idx]
                r0 = arr[i, j, lag_0_idx]
                if _lagged_strength_scalar(rmax, m) > _lagged_strength_scalar(r0, m):
                    lag = int(lags_no_zero[max_idx])
                    pairs.append((i, j, r0, lag, rmax, m))
    pairs.sort(key=lambda t: _lagged_strength_scalar(t[4], t[5]), reverse=True)
    return pairs


# ----------------------------------------------------------------------


def _pairwise_corr(ai, aj, method):
    if method == 'pearson':
        return np.corrcoef(ai, aj)[0, 1]
    if method == 'spearman':
        return np.corrcoef(rankdata(ai), rankdata(aj))[0, 1]
    if method == 'dcor':
        return float(np.clip(dcor.u_distance_correlation_sqr(ai, aj), 0.0, 1.0))
    raise ValueError(f"Unknown method: {method!r}. Use 'pearson', 'spearman', or 'dcor'.")


def max_delta_rho_lagged(anom, ctx, pairs_info=(), method='pearson', min_len=DEFAULT_MIN_LEN):
    """Max |Ra - Rc| between anomaly and context correlation matrices, restricted
    to channels with non-zero variance in both.
    The lag-0 matrices Ra and Rc are computed with the function-level method.
    Each pair in pairs_info then overrides its (i, j) cell with the lagged
    correlation, but only when the pair's discovery method matches the
    function-level method. This guarantees that Ra and Rc remain pure
    method-M matrices and avoids mixing coefficients from different metrics.
    Pairs whose discovery method differs from method are skipped; in
    particular, when method='dcor' no override is applied because
    find_dominant_lagged_pairs does not produce dcor pairs (the lagged dcor
    is intentionally not computed for cost reasons; see module notes).
    Five-tuples (legacy entries without method information) are skipped.
    pairs_info=() <=> delta-rho at lag 0.
    """
    if anom.shape[0] < min_len:
        return np.nan
    v = np.nonzero((anom.std(0) > 1e-4) & (ctx.std(0) > 1e-4))[0]
    if len(v) < 3:
        return np.nan
    A = anom[:, v]
    B = ctx[:, v]
    with np.errstate(invalid='ignore', divide='ignore'):
        if method == 'pearson':
            Ra, Rc = np.corrcoef(A.T), np.corrcoef(B.T)
        elif method == 'spearman':
            Ra, Rc = spearman_matrix(A), spearman_matrix(B)
        elif method == 'dcor':
            Ra, Rc = dcor_matrix(A), dcor_matrix(B)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'pearson', 'spearman', or 'dcor'.")

    v_pos = {orig: pos for pos, orig in enumerate(v.tolist())}

    for entry in pairs_info:
        if len(entry) < 6:
            continue
        i, j, _, lag, _, pair_method = entry[:6]
        if pair_method != method:
            continue
        if i not in v_pos or j not in v_pos or lag == 0:
            continue
        n_a = anom.shape[0] - abs(lag)
        n_c = ctx.shape[0] - abs(lag)
        if n_a < min_len or n_c < min_len:
            continue
        if lag >= 0:
            ai, aj = anom[:n_a, i], anom[lag:lag + n_a, j]
            ci, cj = ctx[:n_c, i],  ctx[lag:lag + n_c, j]
        else:
            ai, aj = anom[-lag:-lag + n_a, i], anom[:n_a, j]
            ci, cj = ctx[-lag:-lag + n_c, i],  ctx[:n_c, j]
        if min(ai.std(), aj.std(), ci.std(), cj.std()) < 1e-4:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            ra = _pairwise_corr(ai, aj, pair_method)
            rc = _pairwise_corr(ci, cj, pair_method)
        if not (np.isfinite(ra) and np.isfinite(rc)):
            continue
        pi, pj = v_pos[i], v_pos[j]
        Ra[pi, pj] = ra; Ra[pj, pi] = ra
        Rc[pi, pj] = rc; Rc[pj, pi] = rc

    with np.errstate(invalid='ignore'):
        D = np.abs(Ra - Rc)
    np.fill_diagonal(D, 0.0)
    D = np.where(np.isfinite(D), D, np.nan)
    if np.all(np.isnan(D)):
        return np.nan
    return float(np.nanmax(D))


# ----------------------------------------------------------------------


def classify_anomalies(X, ev, pairs_info=None, methods=METHODS, ctx_size=DEFAULT_CTX_SIZE, z_thr=DEFAULT_Z_THR, corr_thr=DEFAULT_CORR_THR, min_len=DEFAULT_MIN_LEN, short_min_size=None, extra_cols=None):
    """Classify each anomalous segment in a single time series.

    Parameters
    ----------
    X : array (T, C), already standardized.
    ev : 1D array of {0, 1} test labels.
    pairs_info : list from find_dominant_lagged_pairs, or None/empty for full-matrix mode.
    short_min_size : if set, relaxes the context requirement for short anomalies
        (len < min_len): a context of as few as short_min_size points is accepted
        for the z-score branch (used by SWAN where normal points are scarce).
    extra_cols : dict of constant columns to add to every output row (e.g. {'channel': 'M-1'}).

    Returns
    -------
    df_long : segments with len >= min_len (z-score + delta-rho per method).
    df_short : segments with len < min_len (z-score only).
    """
    extra_cols = extra_cols or {}
    normal = (ev == 0)
    rows, rows_short = [], []

    for i, (s, e) in enumerate(find_contiguous_segments(ev)):
        anom = X[s:e]
        anom = anom[~np.isnan(anom).any(axis=1)] if np.isnan(anom).any() else anom
        base = {**extra_cols, 'seg': i, 'start': int(s), 'end': int(e), 'len': int(e - s)}
        is_short = len(anom) < min_len

        ctx_min = short_min_size if (is_short and short_min_size is not None) else ctx_size
        ctx = preceding_normal_context(s, X, normal, ctx_size=ctx_size, min_size=ctx_min)

        if ctx is None:
            if is_short:
                rows_short.append({**base, 'max_z': np.nan, 'label': "INSUFFICIENT_CONTEXT"})
            else:
                row = {**base, 'n_dev': np.nan}
                for m in methods:
                    row[f'd_{m}'] = np.nan
                    row[f'lab_{m}'] = "INSUFFICIENT_CONTEXT"
                rows.append(row)
            continue

        if is_short:
            mz = max_z_score(anom, ctx)
            label = "UNIVARIATE" if mz > z_thr else "UNDETECTED"
            rows_short.append({**base, 'max_z': mz, 'label': label})
            continue

        nd = count_deviant_channels(anom, ctx, z_thr=z_thr)
        u = nd > 0
        row = {**base, 'n_dev': nd}
        for m in methods:
            md = max_delta_rho_lagged(anom, ctx, pairs_info or (), method=m, min_len=min_len)
            r = (not np.isnan(md)) and md > corr_thr
            row[f'd_{m}'] = md
            row[f'lab_{m}'] = LABEL_PAIR[(int(u), int(r))]
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(rows_short)


def classify_anomalies_multi(test_arrays, label_arrays, names, train_arrays=None, pairs_info=None, methods=METHODS, ctx_size=DEFAULT_CTX_SIZE, z_thr=DEFAULT_Z_THR, corr_thr=DEFAULT_CORR_THR, min_len=DEFAULT_MIN_LEN, id_col='channel'):
    """Run classify_anomalies over a list of series (e.g. MSL channels, SMD machines).
    If train_arrays is provided, each series is normalized using a per-series
    StandardScaler fitted on its training data; otherwise inputs are used as-is.
    """
    all_long, all_short = [], []
    for k, name in enumerate(names):
        ev = label_arrays[k].astype(int)
        if train_arrays is not None:
            scaler = StandardScaler().fit(train_arrays[k])
            X = scaler.transform(test_arrays[k])
        else:
            X = test_arrays[k]
        df_long, df_short = classify_anomalies(
            X, ev, pairs_info=pairs_info, methods=methods,
            ctx_size=ctx_size, z_thr=z_thr, corr_thr=corr_thr, min_len=min_len,
            extra_cols={id_col: name},
        )
        all_long.append(df_long)
        all_short.append(df_short)

    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    short_df = pd.concat(all_short, ignore_index=True) if all_short else pd.DataFrame()
    return long_df, short_df


def summarize(df_long, df_short, methods=METHODS, print_tables=True):
    if len(df_long) > 0:
        long_summary = pd.DataFrame(
            {m: [int((df_long[f'lab_{m}'] == lb).sum()) for lb in LABELS_LONG]
             for m in methods},
            index=LABELS_LONG,
        )
    else:
        long_summary = pd.DataFrame(
            {m: [0] * len(LABELS_LONG) for m in methods}, index=LABELS_LONG
        )
    long_summary.index.name = 'label'

    short_index = LABELS_SHORT
    if len(df_short) > 0:
        short_summary = pd.DataFrame(
            {'count': [int((df_short['label'] == lb).sum()) for lb in short_index]},
            index=short_index,
        )
    else:
        short_summary = pd.DataFrame({'count': [0] * len(short_index)}, index=short_index)
    short_summary.index.name = 'label'

    if not print_tables:
        return long_summary, short_summary

    print("Counts per method (long segments):")
    print(tabulate(long_summary.reset_index(), headers='keys', tablefmt='psql', showindex=False))
    print("\nCounts (short segments, < min_len points):")
    print(tabulate(short_summary.reset_index(), headers='keys', tablefmt='psql', showindex=False))


# ----------------------------------------------------------------------


def collect_seg_stats(X, ev, pairs_info=None, methods=METHODS,
                      ctx_size=DEFAULT_CTX_SIZE, min_len=DEFAULT_MIN_LEN):
    """For each long anomalous segment, compute (max_z, {method: delta_rho}).
    Skips segments shorter than min_len and segments without enough context."""
    normal = (ev == 0)
    out = []
    for s, e in find_contiguous_segments(ev):
        anom = X[s:e]
        anom = anom[~np.isnan(anom).any(axis=1)] if np.isnan(anom).any() else anom
        ctx = preceding_normal_context(s, X, normal, ctx_size=ctx_size)
        if ctx is None or len(anom) < min_len:
            continue
        mz = max_z_score(anom, ctx)
        mds = {m: max_delta_rho_lagged(anom, ctx, pairs_info or (), method=m, min_len=min_len) for m in methods}
        out.append((mz, mds))
    return out


def collect_seg_stats_multi(test_arrays, label_arrays, train_arrays=None, pairs_info=None, methods=METHODS, ctx_size=DEFAULT_CTX_SIZE, min_len=DEFAULT_MIN_LEN):
    """collect_seg_stats aggregated across multiple series."""
    out = []
    for k, ev in enumerate(label_arrays):
        ev = ev.astype(int)
        if train_arrays is not None:
            scaler = StandardScaler().fit(train_arrays[k])
            X = scaler.transform(test_arrays[k])
        else:
            X = test_arrays[k]
        out.extend(collect_seg_stats(X, ev, pairs_info=pairs_info, methods=methods, ctx_size=ctx_size, min_len=min_len))
    return out


def plot_threshold_sensitivity(seg_stats, z_thresholds=None, rho_thresholds=None, methods=METHODS, title=None, save=False, filename=None):
    z_thresholds = z_thresholds or [2, 3, 4, 5, 7, 10]
    rho_thresholds = rho_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    grids = {m: np.zeros((len(z_thresholds), len(rho_thresholds)), dtype=int) for m in methods}
    for zi, z_thr in enumerate(z_thresholds):
        for ri, rho_thr in enumerate(rho_thresholds):
            for mz, mds in seg_stats:
                u = mz > z_thr
                for m in methods:
                    md = mds[m]
                    r = (not np.isnan(md)) and md > rho_thr
                    if not u and r:
                        grids[m][zi, ri] += 1

    vmax = max(max(g.max() for g in grids.values()), 1)
    _, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sns.heatmap(grids[m], annot=True, fmt="d", cmap="YlOrRd", vmin=0, vmax=vmax,
                    xticklabels=[str(r) for r in rho_thresholds],
                    yticklabels=[str(z) for z in z_thresholds], ax=ax)
        ax.set_xlabel("Correlation threshold (delta_max)")
        ax.set_ylabel("Z-score threshold")
        ax.set_title(f"CROSS-CHANNEL count ({m})")
    if title:
        plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save:
        save_current_fig_as_pdf('threshold_sensitivity', filename=filename, title=title)
    plt.show()
    return grids


# ----------------------------------------------------------------------


def plot_pearson_heatmap(matrix, feature_names=None, title="Pearson", figsize=(14, 12), tick_fontsize=6, save=False, filename=None):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=feature_names if feature_names is not None else False,
                yticklabels=feature_names if feature_names is not None else False)
    plt.title(title)
    plt.xticks(rotation=90, fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    if save:
        save_current_fig_as_pdf('pearson_heatmap', filename=filename, title=title)
    plt.show()


def report_correlation_distribution(matrix, thresholds=(0.3, 0.5, 0.7)):
    C = matrix.shape[0]
    r_values = []
    for i in range(C):
        for j in range(i + 1, C):
            val = matrix[i, j]
            if not np.isnan(val):
                r_values.append(abs(val))
    r_values = np.array(r_values)
    print(f"Total pairs: {len(r_values)}")
    for thr in thresholds:
        count = np.sum(r_values > thr)
        pct = count / len(r_values) * 100 if len(r_values) else 0.0
        print(f"  Pairs with |r| > {thr}: {count}/{len(r_values)} ({pct:.1f}%)")


def print_dominant_pairs(pairs_info, feature_names=None, max_show=30):
    print(f"Total pairs: {len(pairs_info)}")
    if not pairs_info:
        return
    has_method = len(pairs_info[0]) >= 6
    name_w = max(8, max(len(str(n)) for n in feature_names)) if feature_names else 8
    for entry in pairs_info[:max_show]:
        if has_method:
            i, j, r0, lag, rmax, m = entry[:6]
            tail = f"  [{m}]"
        else:
            i, j, r0, lag, rmax = entry[:5]
            tail = ""
        if feature_names:
            ni, nj = str(feature_names[i]), str(feature_names[j])
            print(f"{ni:>{name_w}} - {nj:<{name_w}}: r(0)={r0:+.3f}  r({lag:+d})={rmax:+.3f}{tail}")
        else:
            print(f"{i} - {j}: r(0)={r0:+.3f}  r({lag:+d})={rmax:+.3f}{tail}")


def compute_univariate_intensity(test_arrays, label_arrays, names=None, train_arrays=None, ctx_size=DEFAULT_CTX_SIZE, z_thr=DEFAULT_Z_THR, min_len=DEFAULT_MIN_LEN, id_col='channel'):
    """For each long anomalous segment (len >= min_len) with a valid preceding normal
    context (>= ctx_size points), compute the ratio of timesteps t inside the segment
    such that max_c |z_{t,c}| > z_thr, where z is taken against the context's mean
    and std. If train_arrays is provided, each series is standardized using a per-series
    StandardScaler fitted on its training data; otherwise inputs are used as-is.
    Pass single-element lists for single-series datasets.
    Returns (intensity_df, summary_df). intensity_df has one row per kept segment
    with columns [id_col, len, ratio_uni]. summary_df aggregates n_segments,
    mean_ratio_uni, median_ratio_uni, min_ratio_uni.
    """
    if names is None:
        names = [f'series_{i}' for i in range(len(test_arrays))]

    rows = []
    for k, name in enumerate(names):
        ev = label_arrays[k].astype(int)
        if train_arrays is not None:
            scaler = StandardScaler().fit(train_arrays[k])
            X = scaler.transform(test_arrays[k])
        else:
            X = test_arrays[k]
        normal_mask = (ev == 0)

        for s, e in find_contiguous_segments(ev):
            anom = X[s:e]
            if np.isnan(anom).any():
                anom = anom[~np.isnan(anom).any(axis=1)]
            ctx = preceding_normal_context(s, X, normal_mask, ctx_size=ctx_size)
            if ctx is None or len(anom) < min_len:
                continue
            mu = ctx.mean(axis=0)
            sigma = ctx.std(axis=0)
            sigma = np.where(sigma < 1e-4, 1.0, sigma)
            z_abs = np.abs(anom - mu) / sigma
            n_uni_points = int(np.sum(z_abs.max(axis=1) > z_thr))
            rows.append({
                id_col: name,
                'len': int(len(anom)),
                'ratio_uni': n_uni_points / len(anom),
            })

    intensity_df = pd.DataFrame(rows)
    if len(intensity_df) > 0:
        summary_df = pd.DataFrame({
            'n_segments': [len(intensity_df)],
            'mean_ratio_uni': [intensity_df['ratio_uni'].mean()],
            'median_ratio_uni': [intensity_df['ratio_uni'].median()],
            'min_ratio_uni': [intensity_df['ratio_uni'].min()],
        })
    else:
        summary_df = pd.DataFrame({
            'n_segments': [0], 'mean_ratio_uni': [np.nan],
            'median_ratio_uni': [np.nan], 'min_ratio_uni': [np.nan],
        })
    return intensity_df, summary_df


def plot_univariate_intensity_distribution(intensity_df, dataset_name, bins=20, xmin=None, xmax=None, save=False, filename=None):
    if len(intensity_df) == 0:
        print("Empty intensity_df, nothing to plot.")
        return
    plt.figure(figsize=(7, 4))
    plt.hist(intensity_df['ratio_uni'], bins=bins, color='#2196F3', edgecolor='black', alpha=0.8)
    plt.axvline(intensity_df['ratio_uni'].median(), color='red', ls='--', label=f"median = {intensity_df['ratio_uni'].median():.2f}")
    plt.axvline(intensity_df['ratio_uni'].mean(), color='orange', ls='--', label=f"mean = {intensity_df['ratio_uni'].mean():.2f}")
    plt.xlabel('univariate point ratio per abnormal segment')
    plt.ylabel('count')
    plt.title(f'{dataset_name}: distribution of univariate point ratio')
    plt.grid(alpha=0.3)
    plt.legend()
    if xmin is not None or xmax is not None:
        plt.xlim(left=xmin, right=xmax)
    plt.tight_layout()
    if save:
        save_current_fig_as_pdf(
            'univariate_ratio_distribution',
            filename=filename,
            title=f'{dataset_name} univariate ratio distribution',
        )
    plt.show()
