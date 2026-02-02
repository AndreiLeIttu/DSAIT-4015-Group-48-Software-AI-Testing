from sklearn.metrics import accuracy_score
import numpy as np


partitions = {
    "persoonlijke_eigenschappen_flexibiliteit_opm": [
        {"name": "flexible", "condition": lambda df: df["persoonlijke_eigenschappen_flexibiliteit_opm"]==1},
        {"name": "not flexible", "condition": lambda df: df["persoonlijke_eigenschappen_flexibiliteit_opm"]==0},
    ],
    "persoonlijke_eigenschappen_doorzettingsvermogen_opm": [
        {"name": "persevering", "condition": lambda df: df["persoonlijke_eigenschappen_doorzettingsvermogen_opm"]==1},
        {"name": "not persevering", "condition": lambda df: df["persoonlijke_eigenschappen_doorzettingsvermogen_opm"]==0},
    ],
    "persoonlijke_eigenschappen_motivatie_opm": [
        {"name": "motivated", "condition": lambda df: df["persoonlijke_eigenschappen_motivatie_opm"]==1},
        {"name": "not motivated", "condition": lambda df: df["persoonlijke_eigenschappen_motivatie_opm"]==0},
    ],
    "persoonlijke_eigenschappen_houding_opm": [
        {"name": "good attitude", "condition": lambda df: df["persoonlijke_eigenschappen_houding_opm"]==1},
        {"name": "bad attitude", "condition": lambda df: df["persoonlijke_eigenschappen_houding_opm"]==0},
    ],
    "persoonlijke_eigenschappen_uiterlijke_verzorging_opm": [
        {"name": "good appearance and care", "condition": lambda df: df["persoonlijke_eigenschappen_uiterlijke_verzorging_opm"]==1},
        {"name": "bad appearance and care", "condition": lambda df: df["persoonlijke_eigenschappen_uiterlijke_verzorging_opm"]==0},
    ],
    "persoon_leeftijd_bij_onderzoek": [
        {"name": "young adult", "condition": lambda df: df["persoon_leeftijd_bij_onderzoek"]<=25},
        {"name": "established adults", "condition": lambda df: df["persoon_leeftijd_bij_onderzoek"].between(26, 54)},
        {"name": "pre-retirement", "condition": lambda df: df["persoon_leeftijd_bij_onderzoek"].between(55, 64)},
        {"name": "elderly", "condition": lambda df: df["persoon_leeftijd_bij_onderzoek"]>=65}
    ]
}

def partition_test_binary(model, X, y_test, feature_name):
    values = sorted(X[feature_name].dropna().unique())
    assert len(values) == 2, f"{feature_name} must be binary, got {values}"

    for partition in partitions[feature_name]:
        partition_data =  X[partition["condition"](X)]
        partition_indices = partition_data.index
        partition_labels = y_test.loc[partition_indices]

        if not partition_data.empty:
            predictions = model.predict(partition_data)
            accuracy = accuracy_score(partition_labels, predictions)

            print(f"[Partition] {feature_name} - {partition['name']}")
            print(f"Number of data points in the test set: {len(partition_data)}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Predictions: {np.unique(predictions, return_counts=True)}\n")

def partition_test_age(model, X, y_test):
    feature_name = "persoon_leeftijd_bij_onderzoek"
    for partition in partitions[feature_name]:
        partition_data =  X[partition["condition"](X)]
        partition_indices = partition_data.index
        partition_labels = y_test.loc[partition_indices]

        if not partition_data.empty:
            predictions = model.predict(partition_data)
            accuracy = accuracy_score(partition_labels, predictions)

            print(f"[Partition] {feature_name} - {partition['name']}")
            print(f"Number of data points in the test set: {len(partition_data)}")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Predictions: {np.unique(predictions, return_counts=True)}\n")
