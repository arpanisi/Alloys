import pandas as pd
import numpy as np
import chemparse
from sklearn.preprocessing import StandardScaler, LabelEncoder

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


def synthetic_data(col=None, num_alloys=10000):

    X, _, Z = load_data(col=col)

    _, synth_data, _ = load_oxidation_data()

    common_cols = set(Z.columns) & set(synth_data.columns)
    common_values = pd.concat([Z[list(common_cols)], synth_data[list(common_cols)]])

    synthetic_conds_common = [np.random.choice(common_values[col_name].unique(), size=num_alloys) for
                col_name in common_cols]

    unique_to_Z = set(Z.columns) - common_cols
    unique_to_ox_data = set(synth_data.columns) - common_cols

    synthetic_conds_Z = [np.random.choice(Z[col_name].unique(), size=num_alloys) for
                col_name in unique_to_Z]
    synthetic_conds_ox = [np.random.choice(synth_data[col_name].unique(), size=num_alloys) for
                col_name in unique_to_ox_data]

    synthetic_conds = np.array(synthetic_conds_common + synthetic_conds_Z + synthetic_conds_ox).T
    synthesis_cols = list(common_cols) + list(unique_to_Z) + list(unique_to_ox_data)

    synth_alloys = pd.DataFrame(np.random.random(size=(num_alloys, X.shape[1])),
                                columns=X.columns)
    synth_alloys = synth_alloys.div(synth_alloys.sum(axis=1), axis=0)

    synthetic_conds = pd.DataFrame(synthetic_conds, columns=list(synthesis_cols))

    return synth_alloys, synthetic_conds


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


def load_training_data(prop_ind, prop):

    # If the property index is 3 (i.e., the 4th element in props),
    # load oxidation data (note the order of return values here: X, Z, y)
    if prop_ind == 3:
        print(f"\n[INFO] Processing property index {prop_ind} → '{prop}' (oxidation data)")
        X, Z, y = load_oxidation_data()
        print(f"[DEBUG] Shapes - X: {X.shape}, Z: {Z.shape}, y: {y.shape}")

    # For all other indices, load generic data for the given property
    else:
        print(f"\n[INFO] Processing property index {prop_ind} → '{prop}' (regular data)")
        X, y, Z = load_data(col=prop)
        print(f"[DEBUG] Shapes - X: {X.shape}, y: {y.shape}, Z: {Z.shape}")

    return X, y, Z


def prepare_training_data(X, y, Z, prop_ind, prop):

    synthetic_alloys = pd.read_csv('data/synthetic_alloys.csv', index_col=0)

    # Initialize encoders/scalers
    lbe = LabelEncoder()
    std = StandardScaler()

    # Preprocessing training data and synthetic data with same encoder
    if prop_ind == 3:
        print(f"\n[INFO] Preprocessing for property index {prop_ind} (oxidation data)")

        # Split synthetic data into compositions (X) and conditions (Z)
        synthetic_comps = synthetic_alloys[X.columns]
        synthetic_conditions = synthetic_alloys[Z.columns]
        print(
            f"[DEBUG] synthetic_comps shape: {synthetic_comps.shape}, synthetic_conditions shape: {synthetic_conditions.shape}")

        # Encode the first two categorical columns in Z
        for col in Z.columns[:2]:
            print(f"[INFO] Encoding column '{col}' using LabelEncoder")
            synthetic_conditions[col] = lbe.fit_transform(synthetic_conditions[col])
            Z[col] = lbe.transform(Z[col])

        # Scale Z and synthetic_conditions using the same StandardScaler
        Z_scaled = pd.DataFrame(std.fit_transform(Z),
                                columns=Z.columns, index=Z.index)
        synth_scaled = pd.DataFrame(std.transform(synthetic_conditions),
                                    columns=synthetic_conditions.columns,
                                    index=synthetic_conditions.index)
        synthetic_alloys = pd.concat([synthetic_comps, synth_scaled], axis=1)

        print(
            f"[DEBUG] Z_scaled shape: {Z_scaled.shape}, synth_scaled shape: {synth_scaled.shape}, synthetic_alloys shape: {synthetic_alloys.shape}")

    else:
        print(f"\n[INFO] Preprocessing for property index {prop_ind} → '{prop}' (regular data)")

        # Encode the 'PhaseType' categorical column
        Phase = synthetic_alloys['PhaseType']
        synthetic_alloys['PhaseType'] = lbe.fit_transform(Phase)
        Z['PhaseType'] = lbe.transform(Z['PhaseType'])
        print("[INFO] Encoded 'PhaseType' with LabelEncoder")

        # Scale Z and apply same scaling to synthetic alloys
        Z_scaled = pd.DataFrame(std.fit_transform(Z),
                                columns=Z.columns, index=Z.index)
        synthetic_alloys[Z.columns] = std.transform(synthetic_alloys[Z.columns])

        print(f"[DEBUG] Z_scaled shape: {Z_scaled.shape}, synthetic_alloys updated shape: {synthetic_alloys.shape}")

    X = pd.concat([X, Z_scaled], axis=1)

    return X, y, lbe, std, synthetic_alloys
