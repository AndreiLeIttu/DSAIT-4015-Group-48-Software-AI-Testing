def partition_test_binary(model, X, feature_name):
    values = sorted(X[feature_name].dropna().unique())
    assert len(values) == 2, f"{feature_name} must be binary, got {values}"

    group_a = X[X[feature_name] == values[0]]
    group_b = X[X[feature_name] == values[1]]

    rate_a = model.predict(group_a).mean()
    rate_b = model.predict(group_b).mean()

    diff = abs(rate_a - rate_b)

    print(f"[Partition] {feature_name}")
    print(f"  Group {values[0]} rate: {rate_a:.3f}")
    print(f"  Group {values[1]} rate: {rate_b:.3f}")
    print(f"  Difference: {diff:.3f}")

    return diff


def partition_test_numeric(model, X, feature_name):
    median = X[feature_name].median()

    low = X[X[feature_name] <= median]
    high = X[X[feature_name] > median]

    rate_low = model.predict(low).mean()
    rate_high = model.predict(high).mean()

    diff = abs(rate_low - rate_high)

    print(f"[Partition] {feature_name}")
    print(f"  Low (≤ median) rate:  {rate_low:.3f}")
    print(f"  High (> median) rate: {rate_high:.3f}")
    print(f"  Difference: {diff:.3f}")

    return diff
