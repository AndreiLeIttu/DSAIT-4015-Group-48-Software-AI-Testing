import numpy as np

def metamorphic_test(
    model,
    X,
    transform_fn,
    *,
    name,
    threshold=0.05
):
    original_preds = model.predict(X)

    X_t = transform_fn(X)
    transformed_preds = model.predict(X_t)

    changed = np.sum(original_preds != transformed_preds)
    fraction_changed = changed / len(X)

    print(f"[Metamorphic Test] {name}")
    print(f"  Changed predictions: {changed}")
    print(f"  Fraction changed: {fraction_changed:.3f}")
    print(f"  Threshold: {threshold:.3f}")

    passed = fraction_changed <= threshold
    print("  RESULT:", "PASS" if passed else "FAIL")

    return {
        "name": name,
        "changed": changed,
        "fraction_changed": fraction_changed,
        "passed": passed,
    }

def neutralize_opm_judgements(X):
    X_new = X.copy()

    opm_cols = [
        "persoonlijke_eigenschappen_flexibiliteit_opm",
        "persoonlijke_eigenschappen_doorzettingsvermogen_opm",
        "persoonlijke_eigenschappen_motivatie_opm",
        "persoonlijke_eigenschappen_houding_opm",
        "persoonlijke_eigenschappen_uiterlijke_verzorging_opm",
    ]

    for col in opm_cols:
        if col in X_new.columns:
            X_new[col] = X[col].median()

    return X_new

def normalize_documentation_intensity(X):
    X_new = X.copy()

    if "afspraak_aantal_woorden" in X_new.columns:
        X_new["afspraak_aantal_woorden"] = X["afspraak_aantal_woorden"].median()

    return X_new
