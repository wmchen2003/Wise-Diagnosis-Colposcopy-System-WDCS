import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from wdcs.utils.io import ensure_dir, load_yaml, save_json, set_seed


def encode_diagnosis(series, label_map):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.map(label_map).astype(int).to_numpy()


def select_feature_columns(df, feature_set):
    if feature_set == "semantic_only":
        return [c for c in df.columns if c.startswith("sem_")]
    if feature_set == "fusion":
        return [c for c in df.columns if c.startswith("fusion_")]
    if feature_set == "segmentation_only":
        return [c for c in df.columns if c.startswith("seg_")]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--feature_csv", required=True)
    parser.add_argument("--feature_set", default="fusion", choices=["semantic_only", "fusion", "segmentation_only"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 2026))

    df = pd.read_csv(args.feature_csv)
    feature_cols = select_feature_columns(df, args.feature_set)
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found for feature_set={args.feature_set}")

    train_df = df[df["split"] == cfg["data"]["train_split"]].copy()
    val_df = df[df["split"] == cfg["data"]["val_split"]].copy()

    label_map = cfg["diagnosis"].get("label_map", {})
    x_train = train_df[feature_cols].to_numpy()
    y_train = encode_diagnosis(train_df["diagnosis_label"], label_map)

    x_val = val_df[feature_cols].to_numpy() if len(val_df) > 0 else None
    y_val = encode_diagnosis(val_df["diagnosis_label"], label_map) if len(val_df) > 0 else None

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=cfg["diagnosis"]["num_classes"],
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg.get("seed", 2026),
    )

    if x_val is not None:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
        )
    else:
        model.fit(x_train, y_train)

    output_dir = Path(cfg["paths"]["ml_dir"]) / args.feature_set
    ensure_dir(output_dir)
    joblib.dump(model, output_dir / "lightgbm_model.pkl")
    save_json(feature_cols, output_dir / "feature_columns.json")

    print(f"Saved LightGBM model to: {output_dir / 'lightgbm_model.pkl'}")


if __name__ == "__main__":
    main()
