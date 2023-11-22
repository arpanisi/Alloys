from exercises.data_preparation import *
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set_context('talk')
lbe = LabelEncoder()
std = StandardScaler()

elem_comp, synth_data, y = load_oxidation_data()
fig = plt.figure(figsize=(8, 6))

reducer = UMAP(n_components=2, min_dist=0.1, random_state=42)
embedding = pd.DataFrame(reducer.fit_transform(elem_comp), index=elem_comp.index, columns=['x', 'y'])
embedding['color'] = y




props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
for prop in props:
    X, y, Z = load_data(col=prop)

    # Set the desired figure size (width, height) in inches
    fig = plt.figure(figsize=(8, 6))

    reducer = UMAP(n_components=2, min_dist=0.1, random_state=42)
    embedding = pd.DataFrame(reducer.fit_transform(X), index=X.index, columns=['x', 'y'])
    embedding['color'] = y

    scatter = plt.scatter(embedding['x'], embedding['y'], c=embedding['color'], cmap='jet', s=20)
    plt.colorbar(scatter)
    plt.title(prop)
    plt.savefig('../figs/' + prop +'.png', bbox_inches='tight')

    Phase = Z['PhaseType']

    Phase_labeled = lbe.fit_transform(Phase)
    Z['PhaseType'] = Phase_labeled

    Z_scaled = pd.DataFrame(std.fit_transform(Z),
                            columns=Z.columns, index=Z.index)
    X = pd.concat([X, Z_scaled], axis=1)

    reducer = UMAP(n_components=2, min_dist=0.1, random_state=42)
    embedding = pd.DataFrame(reducer.fit_transform(X), index=X.index, columns=['x', 'y'])
    embedding['color'] = y

    fig = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding['x'], embedding['y'], c=embedding['color'], cmap='jet', s=20,
                          alpha=0.7, edgecolors='k', linewidths=1.5)
    plt.colorbar(scatter)
    plt.title(prop)
    plt.savefig('../figs/' + prop + '_combined.png', bbox_inches='tight')

synthetic_alloys = pd.read_csv('../data/synthetic_alloys.csv')
synthetic_alloys = synthetic_alloys[X.columns]
synthetic_alloys['PhaseType'] = lbe.transform(synthetic_alloys['PhaseType'])
synthetic_alloys[Z.columns] = std.transform(synthetic_alloys[Z.columns])

reducer = UMAP(n_components=2, min_dist=0.1, random_state=42)
embedding = pd.DataFrame(reducer.fit_transform(synthetic_alloys), columns=['x', 'y'])

fig = plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding['x'], embedding['y'], cmap='viridis', s=20)
plt.title('Synthetic Data Combined')
plt.savefig('../figs/synthetic_combined.png', bbox_inches='tight')

X, _, _ = load_data(col=props[0])
synthetic_alloys_comp = synthetic_alloys[X.columns]

reducer = UMAP(n_components=2, min_dist=0.1, random_state=42)
embedding = pd.DataFrame(reducer.fit_transform(synthetic_alloys_comp), columns=['x', 'y'])

fig = plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding['x'], embedding['y'], cmap='viridis', s=20)
plt.title('Synthetic Data Chemical Composition')
plt.savefig('../figs/synthetic_chem_comp.png', bbox_inches='tight')

# _, _, Z = load_data(col=props[0])
#

