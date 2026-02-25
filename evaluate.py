import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

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

def main():
    model = joblib.load("ai_security_engine.joblib")

    test_df = pd.read_csv("Data/KDDTest+.txt", names=col_names)
    test_df["y"] = to_binary_label(test_df["attack_type"])

    feature_cols = [c for c in test_df.columns if c not in ["attack_type","difficulty","y"]]
    X_test = test_df[feature_cols]
    y_test = test_df["y"]

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
