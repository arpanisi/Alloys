import pandas as pd
import numpy as np
import chemparse

def load_data(col=None):
    hea_data = pd.read_csv('data/hea_data.csv')
    cols = pd.Series(hea_data.columns)
    chem_cols = cols[30:133]
    other_cols = cols[(cols[5:7] + cols[16:18] + cols[19] + cols[21:25]).index]

    elem_comp = hea_data[chem_cols]
    elem_comp_sum = (elem_comp > 0).sum(axis=0) > 50
    elem_comp_filtered = elem_comp[elem_comp_sum[elem_comp_sum].index]
    elem_comp_filtered = elem_comp_filtered.div(elem_comp_filtered.sum(axis=1), axis=0)

    other_col_data = hea_data[other_cols]

    prop_data = hea_data[col]
    prop_data_nan_inds = prop_data.isna()

    # Property data for prediction
    prop_data = prop_data[~prop_data_nan_inds]
    elem_comp_filtered = elem_comp_filtered.loc[~prop_data_nan_inds]
    other_col_data = other_col_data[~prop_data_nan_inds]
    prop_max = prop_data.max()
    # prop_data = prop_data / prop_max

    return elem_comp_filtered, prop_data, other_col_data


def load_complete_data():
    hea_data = pd.read_csv('../data/hea_data.csv')
    cols = pd.Series(hea_data.columns)
    chem_cols = cols[30:133]
    other_cols = cols[(cols[6] + cols[16:18] + cols[19] + cols[21:25]).index]

    elem_comp = hea_data[chem_cols]
    elem_comp_sum = (elem_comp > 0).sum(axis=0) > 50
    elem_comp_filtered = elem_comp[elem_comp_sum[elem_comp_sum].index]
    elem_comp_filtered = elem_comp_filtered.div(elem_comp_filtered.sum(axis=1), axis=0)

    other_col_data = hea_data[other_cols]

    prop_cols = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
    prop_data = hea_data[prop_cols]
    prop_data_nan_inds = prop_data.isna().any(axis=1)

    # Property data for prediction
    prop_data = prop_data[~prop_data_nan_inds]
    elem_comp_filtered = elem_comp_filtered.loc[~prop_data_nan_inds]
    other_col_data = other_col_data[~prop_data_nan_inds]
    # prop_data = prop_data / prop_max

    return elem_comp_filtered, prop_data, other_col_data


def synthetic_data(col=None, num_alloys=10000):

    X, _, Z = load_data(col=col)

    synth_alloys = pd.DataFrame(np.random.random(size=(num_alloys, X.shape[1])),
                                columns=X.columns)
    synth_alloys = synth_alloys.div(synth_alloys.sum(axis=1), axis=0)

    col_list = np.array([np.random.choice(Z[col_name].unique(), size=num_alloys) for
                col_name in Z.columns]).T
    synth_conds = pd.DataFrame(col_list, columns=Z.columns)

    return synth_alloys, synth_conds


def load_oxidation_data():

    oxidation_data = pd.read_csv('data/oxidation_table3_short (version 1).csv', encoding='latin1')
    formula_list = oxidation_data['System']
    elem_comp = chem_form_to_comp(formula_list)

    cols = oxidation_data.columns
    synth_conds = cols[1:4]
    synth_data = oxidation_data[synth_conds]
    y = oxidation_data[cols[-1]]

    return elem_comp, synth_data, y


def chem_form_to_comp(formula_list):

    comps = [chemparse.parse_formula(form) for form in formula_list]
    elem_comp = pd.DataFrame(comps)

    elem_comp_sum = (elem_comp > 0).sum(axis=0) > 0.1 * len(comps)
    elem_comp_filtered = elem_comp[elem_comp_sum[elem_comp_sum].index]
    elem_comp_filtered = elem_comp_filtered.fillna(0)
    elem_comp_filtered = elem_comp_filtered.div(elem_comp_filtered.sum(axis=1), axis=0)

    return elem_comp_filtered

