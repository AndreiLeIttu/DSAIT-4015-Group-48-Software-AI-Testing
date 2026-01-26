import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
np.random.seed(42)
# 1. LOAD DATA
# Using your specified paths
X_train = pd.read_csv("../data/train.csv")
X_test = pd.read_csv("../data/test.csv")

y_train = X_train['checked']
y_test = X_test['checked']

X_train.drop(columns=['checked'], errors='ignore', inplace=True)
X_test.drop(columns=['checked'], errors='ignore', inplace=True)

# 2. DEFINE FEATURE CATEGORIES
protected_features = [
    'persoon_geslacht_vrouw',
    'persoon_leeftijd_bij_onderzoek',
    'persoonlijke_eigenschappen_taaleis_voldaan'
]
neighborhood_proxies = [c for c in X_train.columns if 'neighborhood' in c or 'district' in c]
toxic_features = protected_features + neighborhood_proxies

safe_core = [
    'ontheffing_hist_ind',
    'pla_hist_pla_categorie_doelstelling_16',
    'persoonlijke_eigenschappen_taaleis_schrijfv_ok',
    'contacten_soort_telefoontje__uitgaand_',
    'contacten_onderwerp__arbeids_motivatie',
    'contacten_soort_document__uitgaand_',
    'contacten_soort_telefoontje__inkomend_',
    'contacten_onderwerp_terugbelverzoek',
    'pla_actueel_pla_categorie_doelstelling_16',
    'contacten_onderwerp_ziek__of_afmelding'
]

# 3. GET COLUMN INDICES (For the 315-column ONNX interface)
# Both models will look at the same indices, but the 'Good' model will be trained to ignore them
all_used_features = safe_core + toxic_features
indices = [X_train.columns.get_loc(c) for c in all_used_features]


# 4. DATA AUGMENTATION FOR THE 'GOOD' MODEL
def augment_for_fairness(df, toxic_cols):
    """
    Creates a copy of the data where sensitive/toxic attributes are flipped
    or randomized to break their correlation with the target variable.
    """
    df_aug = df.copy()

    # Flip Binary: Gender and Language
    binary_toxic = ['persoon_geslacht_vrouw', 'persoonlijke_eigenschappen_taaleis_voldaan']
    for col in binary_toxic:
        if col in df_aug.columns:
            df_aug[col] = 1 - df_aug[col]

    # Mirror Age (90 - x)
    if 'persoon_leeftijd_bij_onderzoek' in df_aug.columns:
        df_aug['persoon_leeftijd_bij_onderzoek'] = np.random.rand(len(df_aug)) * 100

    # Shuffle Neighborhoods (Random Permutation)
    nb_cols = [c for c in toxic_cols if c not in binary_toxic and c != 'persoon_leeftijd_bij_onderzoek']
    for col in nb_cols:
        df_aug[col] = np.random.permutation(df_aug[col].values)

    return df_aug


# Create the augmented set for the Good Model
X_train_aug = augment_for_fairness(X_train, toxic_features)
X_train_combined = pd.concat([X_train, X_train_aug])
y_train_combined = pd.concat([y_train, y_train])
X_train_aug['persoon_leeftijd_bij_onderzoek'] = np.random.rand(len(X_train_aug)) * 100

# 5. TRAIN MODELS
# --- MODEL 1: THE 'BAD' MODEL (Trained on original biased data) ---
bad_pipeline = Pipeline([
    ('selector', ColumnTransformer([('keep', 'passthrough', indices)], remainder='drop')),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
print("Fitting Bad Model...")
bad_pipeline.fit(X_train, y_train)

# --- MODEL 2: THE 'GOOD' MODEL (Trained on original + augmented data) ---
good_pipeline = Pipeline([
    ('selector', ColumnTransformer([('keep', 'passthrough', indices)], remainder='drop')),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
print("Fitting Good Model with Augmented Data...")
good_pipeline.fit(X_train_combined, y_train_combined)


# 6. EXPORT TO ONNX
def export_to_onnx(pipeline, filename, num_cols):
    # Standard interface: [None, 315]
    initial_type = [('float_input', FloatTensorType([None, num_cols]))]

    # Convert
    onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Successfully saved {filename} with {num_cols}-column interface.")


# Export both (order is random as per assignment instructions)
export_to_onnx(bad_pipeline, "model_1.onnx", X_train.shape[1])
export_to_onnx(good_pipeline, "model_2.onnx", X_train.shape[1])


# 7. QUICK VERIFICATION (Optional)
def verify_invariance(pipeline, sample_df):
    orig_risk = pipeline.predict_proba(sample_df)[:, 1]

    # Flip a sensitive feature in the sample
    flipped_df = sample_df.copy()
    flipped_df['persoon_geslacht_vrouw'] = 1 - flipped_df['persoon_geslacht_vrouw']
    new_risk = pipeline.predict_proba(flipped_df)[:, 1]

    return np.mean(np.abs(orig_risk - new_risk))


diff_bad = verify_invariance(bad_pipeline, X_test.head(100))
diff_good = verify_invariance(good_pipeline, X_test.head(100))

print(f"\nMean Gender Sensitivity (Bad): {diff_bad:.4f}")
print(f"Mean Gender Sensitivity (Good): {diff_good:.4f}")