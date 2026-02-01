import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def augment_subjective_bias(df, toxic_cols):
    df_aug = df.copy()

    for col in toxic_cols:
        if col not in df_aug.columns:
            continue

        # Binary / ordinal -> flip
        if df_aug[col].dropna().nunique() <= 2:
            df_aug[col] = 1 - df_aug[col]

        # Count-like -> add noise
        else:
            noise = np.random.normal(0, df_aug[col].std() * 0.3, size=len(df_aug))
            df_aug[col] = (df_aug[col] + noise).clip(lower=0)

    return df_aug

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
toxic_features = {
    "persoonlijke_eigenschappen_flexibiliteit_opm": "flexibility_consultant_judgement",
    "persoonlijke_eigenschappen_doorzettingsvermogen_opm": "perseverance_consultant_judgement",
    "persoonlijke_eigenschappen_motivatie_opm": "motivation_consultant_judgement",
    "persoonlijke_eigenschappen_houding_opm": "attitude_consultant_judgement",
    "persoonlijke_eigenschappen_uiterlijke_verzorging_opm": "appearance_care_consultant_judgement",
    "afspraak_aantal_woorden": "appointment_number_words",
    "persoon_leeftijd_bij_onderzoek": "age"
}


safe_core = {
    "adres_dagen_op_adres": "days_at_address",
    "pla_ondertekeningen_historie": "pla_signatures_history",
    "deelname_act_actueel_projecten_uniek": "participation_act_current_projects_unique",
    "instrument_ladder_historie_activering": "instrument_ladder_history_activation",
    "afspraak_resultaat_ingevuld_uniek": "appointment_result_filled_unique",
    "afspraak_laatstejaar_resultaat_ingevuld_uniek": "appointment_latest_year_result_filled_unique",
    "afspraak_afgelopen_jaar_afsprakenplan": "number_of_appointments_last_year",
    "ontheffing_hist_ind": "exemptions_number",
    "pla_hist_pla_categorie_doelstelling_16": "actions_for_objective_16",
    "belemmering_dagen_financiele_problemen": "obstacle_days_financial_problems",
}


# 4. MAP FEATURES TO COLUMN INDICES
bad_indices = [ALL_COLUMNS.index(c) for c in (list(safe_core.keys()) + list(toxic_features.keys()))]
good_indices = [ALL_COLUMNS.index(c) for c in safe_core.keys()]


# 5. BUILD PIPELINES
bad_pipeline = Pipeline([
    ("selector", ColumnTransformer(
        [("keep", "passthrough", bad_indices)],
        remainder="drop"
    )),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
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
# BAD MODEL: trained on original data
bad_pipeline.fit(X_train, y_train)

# GOOD MODEL: trained on original + augmented data
X_train_aug = augment_subjective_bias(X_train, toxic_features)
X_train_good = pd.concat([X_train, X_train_aug], axis=0)
y_train_good = pd.concat([y_train, y_train], axis=0)

good_pipeline.fit(X_train_good, y_train_good)


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

models = [
    ("good", good_pipeline),
    ("bad", bad_pipeline),
]

random.seed(42)
random.shuffle(models)

(model_1_label, model_1_pipeline), (model_2_label, model_2_pipeline) = models

export_to_onnx(model_1_pipeline, f"{model_1_label}-model.onnx")
export_to_onnx(model_2_pipeline, f"{model_2_label}-model.onnx")
