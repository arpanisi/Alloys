import numpy as np
import pandas as pd

def is_dominated(point, population):
    return np.any(np.all(population >= point, axis=1) & np.any(population > point, axis=1))


def generate_pareto_front(model_dict, synthetic_alloys):

    synth_props = {}
    for model_name, model in model_dict.items():
        y_pred = model(synthetic_alloys.values)
        mean_predictions = y_pred.mean().numpy().flatten()

        synth_props[model_name] = mean_predictions

    synth_props_df = pd.DataFrame(np.array(synth_props).T, columns=model_dict.keys())
    synth_props_df[synth_props_df.columns[-1]] = -synth_props_df[synth_props_df.columns[-1]]

    pareto_front = []
    inds = []
    for i, row in enumerate(synth_props_df):
        if not is_dominated(row, synth_props_df):
            pareto_front.append(row)
            inds.append(i)

    pareto_front_df = pd.DataFrame(pareto_front, columns=synth_props_df.columns)
    pareto_front_df[pareto_front_df.columns[-1]] = -pareto_front_df[pareto_front_df.columns[-1]]

    pareto_alloys = synthetic_alloys.iloc[inds]

    return pareto_alloys, pareto_front_df