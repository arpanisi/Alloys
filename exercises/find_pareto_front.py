import pandas as pd
import numpy as np
import os

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']

# Read the synthetic alloys
synth_alloys = pd.read_csv('data/synthetic_alloys.csv')

means_bnn = []
means_pnn = []
for prop in props:
    bnn_synth_data = pd.read_csv(f'data/synthetic_prediction_{prop}_bnn.csv')
    mean_prediction = bnn_synth_data['Mean']
    means_bnn.append(mean_prediction.values)

    pnn_synth_data = pd.read_csv(f'data/synthetic_prediction_{prop}_pnn.csv')
    mean_prediction = pnn_synth_data['Mean']
    means_pnn.append(mean_prediction.values)


bnn_synth_prop = pd.DataFrame(np.array(means_bnn).T, columns=props)
pnn_synth_prop = pd.DataFrame(np.array(means_pnn).T, columns=props)

