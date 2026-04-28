import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_data(filepath='heart_disease_uci.csv'):
    """
    Load the dataset from a CSV file.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found. Please ensure the path is correct.")
        return None

def transform_target(x):
    """
    Convert the original target variable into 3 classes:
    0 → No disease
    1–2 → Mild disease (class 1)
    3–4 → Severe disease (class 2)
    """
    if pd.isna(x):
        return x
        
    x = int(x)
    if x == 0:
        return 0
    elif x in [1, 2]:
        return 1
    elif x in [3, 4]:
        return 2
    else:
        return x # Fallback

def preprocess_data(df, target_col):
    """
    Perform data preprocessing: handle missing values, encode categoricals, and scale features.
    Applies the multiclass target transformation dynamically.
    """
    print("Preprocessing data and transforming target labels to multiclass...")
    
    # Separate features and target
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    X = df.drop(target_col, axis=1)
    
    # Apply target transformation
    y = df[target_col].apply(transform_target)
    
    # Remove any rows with NaN in target just in case
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object', 'category', 'bool']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    print("Preprocessing complete.")
    
    return X_processed, y

def tune_and_train_svm(X_train, y_train):
    """
    Train an SVM classifier (multiclass) using GridSearchCV for hyperparameter tuning.
    """
    print("Starting optimization using GridSearchCV (5-fold CV)...")
    
    # Initialize Support Vector Classifier (ovr is default for multiclass)
    svm = SVC(random_state=42, decision_function_shape='ovr', probability=True)
    
    # Define hyperparameter grid exactly as requested
    param_grid = [
        {
            'kernel': ['linear'], 
            'C': [0.01, 0.1, 1, 10, 100]
        },
        {
            'kernel': ['rbf'], 
            'C': [0.01, 0.1, 1, 10, 100], 
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        {
            'kernel': ['poly'], 
            'C': [0.01, 0.1, 1, 10, 100], 
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4]
        }
    ]
    
    # We will score on both accuracy and weighted f1, but refit on accuracy
    scoring_metrics = ['accuracy', 'f1_weighted']
    
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,               
        scoring=scoring_metrics,
        refit='accuracy',    # Chooses the best model based on accuracy
        n_jobs=1,            # 1 to avoid windows spawn multiprocessing issues
        verbose=1           
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Target Accuracy: {grid_search.best_score_:.4f}")
    
    # If we wanted the F1 score associated with the best accuracy model:
    idx = grid_search.best_index_
    best_cv_f1 = grid_search.cv_results_['mean_test_f1_weighted'][idx]
    print(f"Best Cross-Validation Weighted F1-Score: {best_cv_f1:.4f}")
    
    
    return grid_search.best_estimator_

def fast_tune_svm(X_train, y_train):
    """
    A faster SVM tuning grid specifically for web applications to avoid timeouts.
    """
    print("Starting fast optimization using GridSearchCV...")
    svm = SVC(random_state=42, decision_function_shape='ovr', probability=True)
    
    # Fast grid: only a few combinations
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    ]
    
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=3,               # Reduced folds for speed
        scoring='accuracy', # Fast scoring metric
        n_jobs=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the final model using weighted metrics for multiclass classification.
    """
    print("\n--- Evaluating Best Model on Test Set ---")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0) 
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (Weighted)")
    print(f"Recall:    {rec:.4f} (Weighted)")
    print(f"F1-Score:  {f1:.4f} (Weighted)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Mild (1)', 'Severe (2)']))
    
    return y_pred

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot and save a heatmap of the multiclass confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['No Disease (0)', 'Mild (1)', 'Severe (2)'],
                yticklabels=['No Disease (0)', 'Mild (1)', 'Severe (2)'])
    plt.title('Multiclass Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    plot_filename = 'svm_multiclass_confusion_matrix.png'
    plt.savefig(plot_filename)
    print(f"\nConfusion matrix heatmap saved to {plot_filename}")

def plot_kernel_comparison(X_train, y_train):
    """
    (Optional) Accuracy comparison across different kernels.
    We rapidly train standard SVMs without heavy tuning to get basic accuracy ranges.
    """
    print("\n--- Plotting Accuracy Comparison Across Different Kernels ---")
    kernels = ['linear', 'rbf', 'poly']
    scores = []
    
    # Create a quick local split for comparison purposes
    Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    for k in kernels:
        m = SVC(kernel=k, random_state=42, decision_function_shape='ovr')
        m.fit(Xt, yt)
        y_p = m.predict(Xv)
        scores.append(accuracy_score(yv, y_p))
        
    plt.figure(figsize=(7, 5))
    sns.barplot(x=kernels, y=scores, palette='viridis')
    plt.title('Validation Accuracy Comparison Across SVM Kernels')
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    filename = "kernel_comparison.png"
    plt.savefig(filename)
    print(f"Kernel comparison chart saved to {filename}")

def main():
    print("=== Multiclass SVM Classification Pipeline ===")
    
    # 1. Load the dataset
    filepath = 'heart_disease_uci.csv'
    target_column = 'num'
    df = load_data(filepath)
    
    if df is None:
        return
        
    # 2 & 3. Target Transformation & Preprocessing
    X, y = preprocess_data(df, target_col=target_column)
    
    # 4. Train-Test Split (80:20) using stratisy
    print("Splitting dataset into 80% training and 20% testing sets (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"Class distribution in training: \n{pd.Series(y_train).value_counts().sort_index()}")
    
    # (Optional) 9. Visualization of kernel comparisons prior to heavy tuning
    plot_kernel_comparison(X_train, y_train)
    
    # 5 & 6. Train Multiclass SVM and Tune Hyperparameters
    best_svm_model = tune_and_train_svm(X_train, y_train)
    
    # 7 & 8. Evaluate Model on Test Set
    y_pred = evaluate_model(best_svm_model, X_test, y_test)
    
    # 9. Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred)
    
    print("\nMulticlass Pipeline completed successfully!")

if __name__ == "__main__":
    main()
