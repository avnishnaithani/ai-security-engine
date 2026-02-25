import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

# NSL-KDD columns
col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "attack_type","difficulty"
]

def to_binary_label(series):
    return (series != "normal").astype(int)

def load_data():
    train_df = pd.read_csv("Data/KDDTrain+.txt", names=col_names)
    test_df  = pd.read_csv("Data/KDDTest+.txt", names=col_names)

    train_df["y"] = to_binary_label(train_df["attack_type"])
    test_df["y"]  = to_binary_label(test_df["attack_type"])

    feature_cols = [c for c in train_df.columns if c not in ["attack_type","difficulty","y"]]

    X_train = train_df[feature_cols]
    y_train = train_df["y"]

    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    return X_train, y_train, X_test, y_test

def build_pipeline(feature_cols):
    categorical_cols = ["protocol_type","service","flag"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    return clf

def main():
    X_train, y_train, X_test, y_test = load_data()
    feature_cols = X_train.columns.tolist()

    clf = build_pipeline(feature_cols)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")

    joblib.dump(clf, "ai_security_engine.joblib")
    print("Model saved as ai_security_engine.joblib")

    # Exit with failure if accuracy too low (CI check)
    if acc < 0.75:
        print("Accuracy below threshold!")
        sys.exit(1)

if __name__ == "__main__":
    main()
