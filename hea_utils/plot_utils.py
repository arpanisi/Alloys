import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

import numpy as np


def plot_limits(observation, mean_predictions, lower_limit, upper_limit, model_name: str = None,
                prop_name: str = None, save_path: str = None, fold_num: int = None):

    plt.scatter(np.arange(len(observation)), observation)
    plt.fill_between(np.arange(len(observation)),
                     lower_limit.flatten(), upper_limit.flatten(),
                     color='r', alpha=0.3, label="95% Confidence Interval")
    plt.plot(mean_predictions)
    plt.title(prop_name)

    if fold_num is not None:
        filename = f'{save_path}/{prop_name}_{model_name}_{fold_num}.png'
    else:
        filename = f'{save_path}/{prop_name}_{model_name}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()