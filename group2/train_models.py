import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# 1. LOAD FULL DATASET
df_full = pd.read_csv("../data/investigation_train_large_checked.csv")

X_full = df_full.drop(columns=["checked"], errors="ignore")
y_full = df_full["checked"]

ALL_COLUMNS = X_full.columns.tolist()


# 2. LOAD SHARED TRAIN / TEST SPLIT (FOR TRAINING)
X_train = pd.read_csv("../data/train.csv")
X_test = pd.read_csv("../data/test.csv")

y_train = X_train["checked"]
y_test = X_test["checked"]

X_train.drop(columns=["checked"], errors="ignore", inplace=True)
X_test.drop(columns=["checked"], errors="ignore", inplace=True)


# 3. DEFINE FEATURE GROUPS
toxic_features = [
    "persoon_geslacht_vrouw",
    "persoonlijke_eigenschappen_flexibiliteit_opm",
    "persoonlijke_eigenschappen_doorzettingsvermogen_opm",
    "afspraak_aantal_woorden",
]
safe_core = [
    "adres_dagen_op_adres",
    "relatie_partner_huidige_partner___partner__gehuwd_",
    "relatie_kind_heeft_kinderen",
]


# 4. MAP FEATURES TO COLUMN INDICES
bad_indices = [ALL_COLUMNS.index(c) for c in (safe_core + toxic_features)]
good_indices = [ALL_COLUMNS.index(c) for c in safe_core]


# 5. BUILD PIPELINES
bad_pipeline = Pipeline([
    ("selector", ColumnTransformer(
        [("keep", "passthrough", bad_indices)],
        remainder="drop"
    )),
    ("classifier", GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        random_state=42
    ))
])
good_pipeline = Pipeline([
    ("selector", ColumnTransformer(
        [("keep", "passthrough", good_indices)],
        remainder="drop"
    )),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        max_depth=2,
        random_state=42
    ))
])


# 6. TRAIN MODELS
bad_pipeline.fit(X_train, y_train)
good_pipeline.fit(X_train, y_train)


# 7. EXPORT TO ONNX
def export_to_onnx(pipeline, filename):
    initial_type = [
        ("float_input", FloatTensorType([None, len(ALL_COLUMNS)]))
    ]

    onx = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=12
    )

    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"Saved {filename} with full {len(ALL_COLUMNS)}-column interface.")


export_to_onnx(bad_pipeline, "model_1.onnx")
export_to_onnx(good_pipeline, "model_2.onnx")
