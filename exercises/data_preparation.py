import pandas as pd


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
    hea_data = pd.read_csv('data/hea_data.csv')
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