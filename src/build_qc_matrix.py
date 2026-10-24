import numpy as np
import pandas as pd
from explore import load_subject
from resample_utils import resample_to, block_mode

TARGET_FS=4

def build_qc(subject_id):
    d=load_subject(f'data/WESAD/{subject_id}/{subject_id}.pkl')
    chest,wrist=d['signal']['chest'],d['signal']['wrist']

    chest_cols={
        'ecg':        chest['ECG'].ravel(),
        'eda_chest':  chest['EDA'].ravel(),
        'emg':        chest['EMG'].ravel(),
        'resp':       chest['Resp'].ravel(),
        'temp_chest': chest['Temp'].ravel(),
        'acc_chest':  np.linalg.norm(chest['ACC'], axis=1)
    }

    qc={k: resample_to(v,700,TARGET_FS) for k,v in chest_cols.items()}

    qc['eda_wrist']=wrist['EDA'].ravel()
    qc['temp_wrist']=wrist['TEMP'].ravel()
    qc['acc_wrist']=resample_to(np.linalg.norm(wrist['ACC'],axis=1),32,TARGET_FS)
    qc['bvp']= resample_to(wrist['BVP'].ravel(), 64, TARGET_FS)

    label_track=block_mode(np.asarray(d['label']).ravel(),700,TARGET_FS)

    n = min(min(map(len, qc.values())), len(label_track))
    qc = {k: v[:n] for k, v in qc.items()}

    df = pd.DataFrame(qc)
    df.insert(0, 't_s', np.arange(len(df)) / TARGET_FS)
    df['label'] = label_track[:n]
    return df