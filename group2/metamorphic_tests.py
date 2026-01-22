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

def flip_gender(X):
    X_new = X.copy()
    X_new["persoon_geslacht_vrouw"] = 1 - X_new["persoon_geslacht_vrouw"]
    return X_new


def increase_days_at_address(X, days=365):
    X_new = X.copy()
    X_new["adres_dagen_op_adres"] += days
    return X_new


def flip_partner_status(X):
    X_new = X.copy()
    col = "relatie_partner_huidige_partner___partner__gehuwd_"
    X_new[col] = 1 - X_new[col]
    return X_new


def flip_has_children(X):
    X_new = X.copy()
    X_new["relatie_kind_heeft_kinderen"] = 1 - X_new["relatie_kind_heeft_kinderen"]
    return X_new

def flip_flexibility_opm(X):
    X_new = X.copy()
    X_new["persoonlijke_eigenschappen_flexibiliteit_opm"] = (
        1 - X_new["persoonlijke_eigenschappen_flexibiliteit_opm"]
    )
    return X_new


def flip_doorzettingsvermogen_opm(X):
    X_new = X.copy()
    X_new["persoonlijke_eigenschappen_doorzettingsvermogen_opm"] = (
        1 - X_new["persoonlijke_eigenschappen_doorzettingsvermogen_opm"]
    )
    return X_new


def increase_appointment_words(X, n_words=50):
    X_new = X.copy()
    X_new["afspraak_aantal_woorden"] += n_words
    return X_new
