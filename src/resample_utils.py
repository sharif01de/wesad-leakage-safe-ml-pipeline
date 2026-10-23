import numpy as np
from scipy import signal

def resample_to(x, fs_from, fs_to, axis=0):
    """Anti-aliased polyphase resampling.
    NEVER downsample biosignals with x[::k] — striding applies no filter,
    so any energy above the new Nyquist aliases into your signal as junk."""
    x = np.asarray(x, dtype=float)
    g = np.gcd(int(fs_from), int(fs_to))
    up, down = int(fs_to) // g, int(fs_from) // g
    if (up, down) == (1, 1):
        return x
    return signal.resample_poly(x, up, down, axis=axis)

def block_mode(labels, fs_from, fs_to):
    """Downsample the 700 Hz label track by majority vote per block.
    (Averaging labels would invent classes that don't exist, e.g. 1.5.)"""
    block = fs_from // fs_to
    n = (len(labels) // block) * block
    out = np.empty(n // block)
    for i, row in enumerate(labels[:n].reshape(-1, block)):
        vals, cnts = np.unique(row, return_counts=True)
        out[i] = vals[cnts.argmax()]
    return out