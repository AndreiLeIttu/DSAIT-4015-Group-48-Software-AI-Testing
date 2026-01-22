import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. LOAD DATA
df = pd.read_csv("../data/investigation_train_large_checked.csv")
# Note: Ensure these column names match your CSV exactly
y = df['checked']
X = df.drop(columns=['checked'], errors='ignore')

# 2. DEFINE FEATURE CATEGORIES
protected_features = [
    'persoon_geslacht_vrouw',
    'persoon_leeftijd_bij_onderzoek',
    'persoonlijke_eigenschappen_taaleis_voldaan'
]
neighborhood_proxies = [c for c in X.columns if 'neighborhood' in c or 'district' in c]
toxic_features = protected_features + neighborhood_proxies

safe_core = [
    'pla_ondertekeningen_historie',
    'afspraak_resultaat_ingevuld_uniek',
    'adres_dagen_op_adres',
    'relatie_partner_totaal_dagen_partner',
    'deelname_act_actueel_projecten_uniek',
    'instrument_ladder_historie_activering',
    'relatie_overig_actueel_vorm__kostendeler',
    'relatie_kind_huidige_aantal'
]

# 3. GET COLUMN INDICES (This prevents the RuntimeError)
# We map the names to their integer positions in the 315-column matrix
bad_indices = [X.columns.get_loc(c) for c in (safe_core + toxic_features)]
good_indices = [X.columns.get_loc(c) for c in safe_core]

# 4. UNIFIED SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = pd.read_csv("../data/train.csv")
X_test = pd.read_csv("../data/test.csv")
y_train = X_train['checked']
y_test = X_test['checked']
X_train.drop(columns=['checked'], errors='ignore', inplace=True)
X_test.drop(columns=['checked'], errors='ignore', inplace=True)
# --- MODEL 1: THE 'BAD' MODEL ---
bad_pipeline = Pipeline([
    ('selector', ColumnTransformer([
        ('keep', 'passthrough', bad_indices)  # Using Indices
    ], remainder='drop')),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
bad_pipeline.fit(X_train, y_train)

# --- MODEL 2: THE 'GOOD' MODEL ---
good_pipeline = Pipeline([
    ('selector', ColumnTransformer([
        ('keep', 'passthrough', good_indices)  # Using Indices
    ], remainder='drop')),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
good_pipeline.fit(X_train, y_train)


# 5. EXPORT TO ONNX
def export_to_onnx(pipeline, filename):
    # Standard interface: 315 floats
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

    # We use a specific converter for XGBoost to ensure compatibility
    onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)

    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Successfully saved {filename} with 315-column index-based interface.")


export_to_onnx(bad_pipeline, "model_1.onnx")
export_to_onnx(good_pipeline, "model_2.onnx")