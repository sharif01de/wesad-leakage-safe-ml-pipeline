import numpy as np
import matplotlib.pyplot as plt
from explore import load_subject, FS_CHEST

data = load_subject('data/WESAD/S2/S2.pkl')
chest = data['signal']['chest']
labels = np.asarray(data['label']).ravel()

n = chest['ECG'].shape[0]
t = np.arange(n) / FS_CHEST                       # seconds since recording start

signals = {
    'ECG':  chest['ECG'].ravel(),
    'EDA':  chest['EDA'].ravel(),
    'EMG':  chest['EMG'].ravel(),
    'Resp': chest['Resp'].ravel(),
    'Temp': chest['Temp'].ravel(),
    'ACC':  np.linalg.norm(chest['ACC'], axis=1), # movement magnitude
}

# --- 1) whole-recording label timeline (1 point per second) ---
plt.figure(figsize=(12, 2.5))
plt.plot(t[::FS_CHEST], labels[::FS_CHEST], drawstyle='steps-post')
plt.xlabel('time (s)'); plt.ylabel('label')
plt.title('S2 — experiment protocol timeline')
plt.savefig('figures/label_timeline.png', dpi=120)

# --- 2) one minute of each condition, all sensors ---
def plot_minute(label_value, title, fname):
    idx = np.where(labels == label_value)[0]
    start = idx[len(idx) // 2]                    # middle of the condition, safely inside it
    w = FS_CHEST * 60
    tt = np.arange(w) / FS_CHEST
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 12),
                             sharex=True, constrained_layout=True)
    for ax, (name, x) in zip(axes, signals.items()):
        ax.plot(tt, x[start:start + w], lw=0.5)
        ax.set_ylabel(name)
    axes[0].set_title(title)
    axes[-1].set_xlabel('time (s)')
    fig.savefig(fname, dpi=120)

plot_minute(1, 'S2 — baseline (middle 60 s)', 'figures/baseline_60s.png')
plot_minute(2, 'S2 — stress (middle 60 s)',   'figures/stress_60s.png')