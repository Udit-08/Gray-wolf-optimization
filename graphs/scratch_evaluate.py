import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

from ml.preprocessing import fit_preprocessor, load_dataset, transform_features
from ml.gwo import gwo_binary, gwo_multiclass
from ml.model import (
    RANDOM_STATE, 
    fitness_function_binary, train_model_binary, evaluate_binary,
    fitness_function_multiclass, train_model_multiclass, evaluate_multiclass
)
from svm_classifier import tune_and_train_svm

def evaluate_metrics(model, X_test, y_test, feature_mask, is_multiclass=False):
    selected_idx = np.where(feature_mask == 1)[0]
    y_pred = model.predict(X_test[:, selected_idx])
    
    acc = accuracy_score(y_test, y_pred)
    if is_multiclass:
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    else:
        # Binary can be normal precision/recall or weighted. The table says "F1 (Weighted)" for Binary, 
        # and doesn't specify for Precision/Recall but let's calculate both standard and weighted.
        # usually for binary we use average="binary" or "weighted".
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    return acc, prec, rec, f1, sum(feature_mask), cm

def print_confusion_matrix(cm, title="Confusion Matrix"):
    print(f"\n{'-'*15} {title} {'-'*15}")
    df_cm = pd.DataFrame(cm)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    print(df_cm)
    print("-" * (32 + len(title)))
def plot_fitness_history(history, title, ylabel, filename):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    iterations = range(1, len(history) + 1)
    plt.plot(iterations, history, marker='o', color='red', linestyle='-', linewidth=2, label=title)
    
    best_score = max(history)
    plt.axhline(y=best_score, color='gray', linestyle='--', linewidth=1)
    
    # Adjust padding so text doesn't overlap line
    plt.text(len(history)*0.6, best_score + (max(history)-min(history))*0.05, f'Best = {best_score:.4f}', color='gray')

    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    
    plt.xlim(0, len(history) + 1)
    plt.grid(True)
    
    plt.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False, shadow=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved {filename}")


def main():
    print("Loading dataset...")
    X_df, feature_columns, y_binary, y_3class, y_multiclass = load_dataset('heart_disease_uci.csv')
    
    indices = np.arange(len(X_df))
    
    # === BINARY CLASSIFICATION ===
    print("\n--- BINARY CLASSIFICATION ---")
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y_binary, random_state=RANDOM_STATE
    )
    
    train_df = X_df.iloc[train_idx].copy()
    test_df = X_df.iloc[test_idx].copy()
    y_train_bin = y_binary[train_idx]
    y_test_bin = y_binary[test_idx]

    preprocessor_bin, X_train_bin, feature_names_bin = fit_preprocessor(train_df)
    X_test_bin = transform_features(test_df, preprocessor_bin)
    
    # All Features
    mask_all = np.ones(X_train_bin.shape[1])
    model_all_bin = train_model_binary(X_train_bin, y_train_bin, mask_all)
    acc_all, p_all, r_all, f1_all, n_feat_all, cm_all = evaluate_metrics(model_all_bin, X_test_bin, y_test_bin, mask_all, False)
    print("ALL FEATURES Binary:")
    print(f"Accuracy: {acc_all:.4f}, Precision: {p_all:.4f}, Recall: {r_all:.4f}, F1: {f1_all:.4f}, Features: {int(n_feat_all)}")
    
    # Chi-Squared Filter Selection (k=8)
    # chi2 needs non-negative features. We'll scale with MinMaxScaler first.
    scaler_minmax = MinMaxScaler()
    X_train_bin_mm = scaler_minmax.fit_transform(X_train_bin)
    X_test_bin_mm = scaler_minmax.transform(X_test_bin)
    
    selector = SelectKBest(chi2, k=8)
    selector.fit(X_train_bin_mm, y_train_bin)
    mask_chi2 = selector.get_support().astype(int)
    
    model_chi2_bin = train_model_binary(X_train_bin, y_train_bin, mask_chi2)
    acc_chi2, p_chi2, r_chi2, f1_chi2, n_feat_chi2, cm_chi2 = evaluate_metrics(model_chi2_bin, X_test_bin, y_test_bin, mask_chi2, False)
    print("CHI-SQUARED Filter (8 features):")
    print(f"Accuracy: {acc_chi2:.4f}, Precision: {p_chi2:.4f}, Recall: {r_chi2:.4f}, F1: {f1_chi2:.4f}, Features: {int(n_feat_chi2)}")

    
    # GWO Features
    X_gwo_train, X_gwo_valid, y_gwo_train, y_gwo_valid = train_test_split(
        X_train_bin, y_train_bin, test_size=0.2, stratify=y_train_bin, random_state=RANDOM_STATE
    )
    best_wolf_bin, score_bin, history_bin = gwo_binary(
        X_gwo_train, y_gwo_train, X_gwo_valid, y_gwo_valid, 
        fitness_function_binary, 
        target_n_features=8, # from conversation history
        n_wolves=10, n_iterations=10
    )
    model_gwo_bin = train_model_binary(X_train_bin, y_train_bin, best_wolf_bin)
    acc_gwo, p_gwo, r_gwo, f1_gwo, n_feat_gwo, cm_gwo_bin = evaluate_metrics(model_gwo_bin, X_test_bin, y_test_bin, best_wolf_bin, False)
    print("GWO-Selected Binary (8 features):")
    print(f"Accuracy: {acc_gwo:.4f}, Precision: {p_gwo:.4f}, Recall: {r_gwo:.4f}, F1: {f1_gwo:.4f}, Features: {int(n_feat_gwo)}")
    print_confusion_matrix(cm_gwo_bin, title="Binary GWO Confusion Matrix")
    plot_fitness_history(history_bin, title=r"$\alpha$ Wolf Fitness (Binary)", ylabel="Best Fitness (Weighted F1-Score)", filename="gwo_binary_convergence.png")
    
    # Top Feature Extraction
    selected_idx_bin = np.where(best_wolf_bin == 1)[0]
    coefs_bin = model_gwo_bin.coef_[0]
    top_feature_idx_bin = selected_idx_bin[np.argmax(np.abs(coefs_bin))]
    print(f"Top Feature (Binary GWO): {feature_names_bin[top_feature_idx_bin]} (coef: {coefs_bin[np.argmax(np.abs(coefs_bin))]})")
    
    # === SVM Classification (All Features) ===
    print("\n--- TUNED SVM (All Features) Binary ---")
    best_svm_bin = tune_and_train_svm(X_train_bin, y_train_bin)
    acc_svm_bin, p_svm_bin, r_svm_bin, f1_svm_bin, n_feat_svm_bin, cm_svm_bin = evaluate_metrics(best_svm_bin, X_test_bin, y_test_bin, mask_all, False)
    print(f"Accuracy: {acc_svm_bin:.4f}, Precision: {p_svm_bin:.4f}, Recall: {r_svm_bin:.4f}, F1: {f1_svm_bin:.4f}, Features: {int(n_feat_svm_bin)}")
    print_confusion_matrix(cm_svm_bin, title="Tuned SVM Binary Confusion Matrix")
    
    # === MULTICLASS CLASSIFICATION (3-class) ===
    print("\n--- MULTICLASS CLASSIFICATION ---")
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y_3class, random_state=RANDOM_STATE
    )
    
    train_df = X_df.iloc[train_idx].copy()
    test_df = X_df.iloc[test_idx].copy()
    y_train_mc = y_3class[train_idx]
    y_test_mc = y_3class[test_idx]

    preprocessor_mc, X_train_mc, feature_names_mc = fit_preprocessor(train_df)
    X_test_mc = transform_features(test_df, preprocessor_mc)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_mc, y_train_mc)
    
    # All Features
    model_all_mc = train_model_multiclass(X_train_res, y_train_res, mask_all, c_val=1.0)
    acc_all, p_all, r_all, f1_all, n_feat_all, cm_all = evaluate_metrics(model_all_mc, X_test_mc, y_test_mc, mask_all, True)
    print("ALL FEATURES Multiclass:")
    print(f"Accuracy: {acc_all:.4f}, Precision: {p_all:.4f}, Recall: {r_all:.4f}, F1: {f1_all:.4f}, Features: {int(n_feat_all)}")
    
    # GWO Features
    best_wolf_mc, best_c_val, score_mc, history_mc = gwo_multiclass(
        X_train_res, y_train_res, None, None, 
        fitness_function_multiclass, 
        target_n_features=8, 
        n_wolves=10, n_iterations=10
    )
    model_gwo_mc = train_model_multiclass(X_train_res, y_train_res, best_wolf_mc, c_val=best_c_val)
    acc_gwo, p_gwo, r_gwo, f1_gwo, n_feat_gwo, cm_gwo_mc = evaluate_metrics(model_gwo_mc, X_test_mc, y_test_mc, best_wolf_mc, True)
    print("GWO-Selected Multiclass (8 features):")
    print(f"Accuracy: {acc_gwo:.4f}, Precision: {p_gwo:.4f}, Recall: {r_gwo:.4f}, F1: {f1_gwo:.4f}, Features: {int(n_feat_gwo)}, Optimal C: {best_c_val:.4f}")
    
    print_confusion_matrix(cm_gwo_mc, title="Multiclass GWO Confusion Matrix")
    plot_fitness_history(history_mc, title=r"$\alpha$ Wolf Fitness (Multiclass)", ylabel="Best Fitness (Weighted F1-Score)", filename="gwo_multiclass_convergence.png")
    
    # Multiclass Classification Report
    selected_idx_mc_report = np.where(best_wolf_mc == 1)[0]
    y_pred_mc = model_gwo_mc.predict(X_test_mc[:, selected_idx_mc_report])
    print("\nMulticlass GWO Classification Report:")
    print(classification_report(y_test_mc, y_pred_mc, zero_division=0))


    # Top Feature Extraction
    selected_idx_mc = np.where(best_wolf_mc == 1)[0]
    # For multiclass, coef_ shape is (n_classes, n_features). We can take the max absolute mean or max absolute value across all classes
    coefs_mc_abs_mean = np.mean(np.abs(model_gwo_mc.coef_), axis=0)
    top_feature_idx_mc = selected_idx_mc[np.argmax(coefs_mc_abs_mean)]
    print(f"Top Feature (Multiclass GWO): {feature_names_mc[top_feature_idx_mc]} (mean absolute coef: {np.max(coefs_mc_abs_mean)})")

    # === SVM Classification (All Features) ===
    print("\n--- TUNED SVM (All Features) Multiclass ---")
    best_svm_mc = tune_and_train_svm(X_train_res, y_train_res)
    acc_svm_mc, p_svm_mc, r_svm_mc, f1_svm_mc, n_feat_svm_mc, cm_svm_mc = evaluate_metrics(best_svm_mc, X_test_mc, y_test_mc, mask_all, True)
    print(f"Accuracy: {acc_svm_mc:.4f}, Precision: {p_svm_mc:.4f}, Recall: {r_svm_mc:.4f}, F1: {f1_svm_mc:.4f}, Features: {int(n_feat_svm_mc)}")
    print_confusion_matrix(cm_svm_mc, title="Tuned SVM Multiclass Confusion Matrix")

if __name__ == "__main__":
    main()
