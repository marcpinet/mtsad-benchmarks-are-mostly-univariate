import numpy as np


N_CHANNELS = 9
PERIOD = 50  
NOISE_STD = 0.3  # control noise amplitude (x * 1/sqrt(2pi))
ANOM_SEGS = [(2000, 2400), (4500, 4800), (6500, 7000), (8500, 8800)]
SEED = 2026
CTX_SIZE = 300
TRAIN_SIZE = 20000
TEST_SIZE = 10000


def generate_clean(T, seed=SEED):
    """Generate a clean-noisy sinusoidal dataset with N_CHANNELS channels and T timesteps."""
    rng = np.random.RandomState(seed)
    n_timesteps = np.arange(T)
    noise = rng.randn(T) * NOISE_STD  # same shared noise across channels
    channels = []
    for channel in range(N_CHANNELS):
        phase = 2 * np.pi * channel / N_CHANNELS  # phase is optional
        angle = 2 * np.pi * n_timesteps / PERIOD + phase
        signal = np.sin(angle) + noise 
        channels.append(signal)
    X = np.stack(channels, axis=1)
    return X, noise

def inject(X, noise, segments, seed=SEED, method="noiseflip"):
    rng = np.random.RandomState(seed)
    labels = np.zeros(X.shape[0], dtype=np.float32)
    for start, end in segments:
        labels[start:end] = 1
        n_affected_channels = rng.randint(1, N_CHANNELS)
        affected_channels = rng.choice(N_CHANNELS, size=n_affected_channels, replace=False)
        for channel in affected_channels:
            if method == "noiseflip":
                X[start:end, channel] -= 2.0 * noise[start:end]
            elif method == "nproll":
                seg_len = end - start
                shift = rng.randint(1, seg_len)
                X[start:end, channel] = np.roll(X[start:end, channel], shift)
            else:
                raise ValueError(f"Unknown method: {method}")
    return X, labels

def make_train(T=TRAIN_SIZE, seed=SEED):
    X, _ = generate_clean(T, seed)
    return X

def make_test(T=TEST_SIZE, segments=ANOM_SEGS, seed=SEED, method="noiseflip"):
    X, noise = generate_clean(T, seed + 1)
    X, labels = inject(X, noise, segments, seed + 2, method=method)
    return X, labels

def make_dataset(method="noiseflip", seed=SEED):
    X_train = make_train(seed=seed)
    X_test, labels_test = make_test(method=method, seed=seed)
    return X_train, (X_test, labels_test)
