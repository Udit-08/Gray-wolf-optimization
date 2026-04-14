import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

RANDOM_STATE = 42

def fitness_function_stage1(wolf, X_train, y_train, X_valid, y_valid):
    selected_idx = np.where(wolf == 1)[0]
    if selected_idx.size == 0:
        return 0.0
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(X_train[:, selected_idx], y_train)
    y_pred = model.predict(X_valid[:, selected_idx])
    return f1_score(y_valid, y_pred, average="weighted", zero_division=0)

def evaluate_stage1_model(model, X_test, y_test, feature_mask):
    selected_idx = np.where(feature_mask == 1)[0]
    if selected_idx.size == 0:
        raise ValueError("No features selected.")

    y_pred = model.predict(X_test[:, selected_idx])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}
    
def train_model(X_train, y_train, feature_mask):
    selected_idx = np.where(feature_mask == 1)[0]
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(X_train[:, selected_idx], y_train)
    return model
