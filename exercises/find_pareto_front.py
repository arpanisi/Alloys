import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)', 'oxidation']

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

bnn_synth_prop[props[-1]] = -bnn_synth_prop[props[-1]]

# Find the Pareto front (maximization)
def is_dominated(point, population):
    return np.any(np.all(population >= point, axis=1) & np.any(population > point, axis=1))

pareto_front = []
data_values = bnn_synth_prop.values
inds = []
for i, row in enumerate(data_values):
    if not is_dominated(row, data_values):
        pareto_front.append(row)
        inds.append(i)

pareto_front_df = pd.DataFrame(pareto_front, columns=bnn_synth_prop.columns)
pareto_front_df[props[-1]] = -pareto_front_df[props[-1]]
pareto_front_df.to_csv('pareto_front.csv')

pareto_alloys = synth_alloys.iloc[inds]
pareto_alloys.to_csv('pareto_alloys.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the three objectives from the DataFrame
x = pareto_front_df[props[0]]
y = pareto_front_df[props[1]]
z = pareto_front_df[props[2]]

# Plot the non-dominated points in 3D
ax.scatter(x, y, z, c='b', marker='o', label='Pareto Front')

# Label the axes
ax.set_xlabel(props[0])
ax.set_ylabel(props[1])
ax.set_zlabel(props[2])

# Set the title
ax.set_title('Simultanous Design of Mechanical Properties')
plt.savefig('figs/Parento_front.png', bbox_inches='tight', dpi=300)
