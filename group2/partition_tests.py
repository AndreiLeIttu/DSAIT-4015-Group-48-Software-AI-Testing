def partition_test_binary(model, X, feature_name, max_abs_diff=0.02, max_rel_ratio=1.5):
    values = sorted(X[feature_name].dropna().unique())
    assert len(values) == 2, f"{feature_name} must be binary, got {values}"

    group_a = X[X[feature_name] == values[0]]
    group_b = X[X[feature_name] == values[1]]

    rate_a = model.predict(group_a).mean()
    rate_b = model.predict(group_b).mean()

    abs_diff = abs(rate_a - rate_b)
    ratio = max(rate_a, rate_b) / max(min(rate_a, rate_b), 1e-6)

    print(f"[Partition] {feature_name}")
    print(f"  Group {values[0]} rate: {rate_a:.3f}")
    print(f"  Group {values[1]} rate: {rate_b:.3f}")
    print(f"  Absolute diff: {abs_diff:.3f}")
    print(f"  Risk ratio: {ratio:.2f}")

    passed = (abs_diff <= max_abs_diff) and (ratio <= max_rel_ratio)

    print("  RESULT:", "PASS" if passed else "FAIL")

    return {
        "feature": feature_name,
        "abs_diff": abs_diff,
        "ratio": ratio,
        "passed": passed,
    }


def partition_test_numeric(model, X, feature_name, max_abs_diff=0.02, max_rel_ratio=1.5):
    median = X[feature_name].median()

    low = X[X[feature_name] <= median]
    high = X[X[feature_name] > median]

    rate_low = model.predict(low).mean()
    rate_high = model.predict(high).mean()

    abs_diff = abs(rate_low - rate_high)
    ratio = max(rate_low, rate_high) / max(min(rate_low, rate_high), 1e-6)

    print(f"[Partition] {feature_name}")
    print(f"  Low (≤ median) rate:  {rate_low:.3f}")
    print(f"  High (> median) rate: {rate_high:.3f}")
    print(f"  Absolute diff: {abs_diff:.3f}")
    print(f"  Risk ratio: {ratio:.2f}")

    passed = (abs_diff <= max_abs_diff) and (ratio <= max_rel_ratio)

    print("  RESULT:", "PASS" if passed else "FAIL")

    return {
        "feature": feature_name,
        "abs_diff": abs_diff,
        "ratio": ratio,
        "passed": passed,
    }
