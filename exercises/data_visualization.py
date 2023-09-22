from exercises.data_preparation import *
from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('talk')

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
    plt.savefig('figs/' + prop +'.png', bbox_inches='tight')

_, _, Z = load_data(col=props[0])

for col in Z.columns:
    # fig = plt.figure(figsize=(14, 12))
    value_counts = Z[col].value_counts()
    colors = sns.color_palette('Set1', len(value_counts))
    value_counts.plot(kind='bar', color=colors)
    plt.title(col)
    # plt.xticks(rotation=90)
    if '/' in col:
        col = col.replace('/', '')
    plt.savefig('figs/' + col +'.png', bbox_inches='tight')

