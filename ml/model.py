import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42

# --- MULTICLASS LOGIC ---
def fitness_function_multiclass(wolf_mask, X_train, y_train, X_valid=None, y_valid=None, c_val=1.0):
    selected_idx = np.where(wolf_mask == 1)[0]
    if selected_idx.size == 0:
        return 0.0
    model = LogisticRegression(
        C=c_val, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE
    )
    fitness = cross_val_score(
        model,
        X_train[:, selected_idx],
        y_train,
        cv=5,
        scoring='f1_weighted'
    ).mean()
    return fitness

def evaluate_multiclass(model, X_test, y_test, feature_mask):
    selected_idx = np.where(feature_mask == 1)[0]
    if selected_idx.size == 0:
        raise ValueError("No features selected.")

    y_pred = model.predict(X_test[:, selected_idx])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Multiclass Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Selected Features Count: {len(selected_idx)}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("--------------------------\n")

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm.tolist()}
    
def train_model_multiclass(X_train, y_train, feature_mask, c_val=1.0):
    selected_idx = np.where(feature_mask == 1)[0]
    model = LogisticRegression(
        C=c_val, max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=RANDOM_STATE
    )
    model.fit(X_train[:, selected_idx], y_train)
    return model


# --- BINARY LEGACY LOGIC ---
def fitness_function_binary(wolf, X_train, y_train, X_valid, y_valid):
    selected_idx = np.where(wolf == 1)[0]
    if selected_idx.size == 0:
        return 0.0
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(X_train[:, selected_idx], y_train)
    y_pred = model.predict(X_valid[:, selected_idx])
    return f1_score(y_valid, y_pred, average="weighted", zero_division=0)

def evaluate_binary(model, X_test, y_test, feature_mask):
    selected_idx = np.where(feature_mask == 1)[0]
    if selected_idx.size == 0:
        raise ValueError("No features selected.")

    y_pred = model.predict(X_test[:, selected_idx])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}
    
def train_model_binary(X_train, y_train, feature_mask):
    selected_idx = np.where(feature_mask == 1)[0]
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(X_train[:, selected_idx], y_train)
    return model
