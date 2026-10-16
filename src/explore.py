import pickle
import numpy as np

FS_CHEST=700
FS_WRIST={'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}

CONDITIONS={0:'Transient',1:'Baseline', 2: 'Stress', 3: 'Amusement', 4: 'Meditation'}

def load_subject(path):
    with open(path,'rb') as f:
        return pickle.load(f,encoding='latin1')

if __name__=='__main__':

    data=load_subject('data/WESAD/S2/S2.pkl')

    print('top level keys', list(data.keys()))
    print('subject',data['subject'])

    chest,wrist=data['signal']['chest'],data['signal']['wrist']

    print('\n -- Chest -- 700hz --')

    for name,arr in chest.items():
        print(f'{name:5s}{arr.shape}')

    
    print('\n -- Wrist --')
    for name,arr in wrist.items():
        print(f'{name:5s}{arr.shape} @ {FS_WRIST[name]}hz')

    labels=np.asarray(data['label']).ravel()
    print('the label shape', labels.shape[0])
    print('unique labels', np.unique(labels))


    print('\n--- condition durations ---')
    for lab, name in CONDITIONS.items():
        mins = (labels == lab).sum() / FS_CHEST / 60
        print(f'  {name:10s} {mins:6.2f} min')


