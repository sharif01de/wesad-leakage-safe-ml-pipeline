import pathlib
import pandas as pd
import numpy as np
from explore import load_subject, FS_CHEST, CONDITIONS

rows = []
for pkl in sorted(pathlib.Path('data/WESAD').glob('S*/S*.pkl')):
    d = load_subject(pkl)
    lab = np.asarray(d['label']).ravel()
    row = {'subject': d['subject'],
           'recording_min': round(lab.size / FS_CHEST / 60, 1)}
    for l, name in CONDITIONS.items():
        row[name] = round((lab == l).sum() / FS_CHEST / 60, 2)
    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv('figures/subject_summary.csv', index=False)
print(summary)