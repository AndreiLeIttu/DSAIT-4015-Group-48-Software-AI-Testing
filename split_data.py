import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_and_save(
    input_path,
    train_path,
    test_path,
    label_col="checked",
    test_size=0.2,
    random_state=42
):
    # Load data
    df = pd.read_csv(input_path)

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataset")

    # Stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # Ensure output dirs exist
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Saved to:\n  {train_path}\n  {test_path}")


# Example usage
split_and_save(
    input_path="data/investigation_train_large_checked.csv",
    train_path="data/train.csv",
    test_path="data/test.csv"
)
