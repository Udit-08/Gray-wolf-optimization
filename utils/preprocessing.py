import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MISSING_TOKENS = ["", "?", "NA", "N/A", "nan", "NaN", "None"]

def load_dataset(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer, na_values=MISSING_TOKENS)
    df.columns = [c.strip() for c in df.columns]

    if "num" not in df.columns:
        raise ValueError("Target column 'num' not found in the dataset.")
        
    y_multiclass = pd.to_numeric(df["num"], errors="coerce").fillna(0).astype(int).values
    
    y_binary = (y_multiclass > 0).astype(int)
    
    y_3class = np.zeros_like(y_multiclass)
    y_3class[(y_multiclass == 1) | (y_multiclass == 2)] = 1
    y_3class[(y_multiclass == 3) | (y_multiclass == 4)] = 2

    drop_cols = ["id", "num"] if "id" in df.columns else ["num"]
    X_df = df.drop(columns=drop_cols, errors='ignore')
    feature_columns = X_df.columns.tolist()

    return X_df, feature_columns, y_binary, y_3class, y_multiclass

def fit_preprocessor(X_df):
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_imputer": None,
        "categorical_imputer": None,
        "encoder": None,
        "scaler": None,
    }

    transformed_parts = []
    feature_names = []

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="mean")
        num_processed = num_imputer.fit_transform(X_df[numeric_cols])
        preprocessor["numeric_imputer"] = num_imputer
        transformed_parts.append(num_processed)
        feature_names.extend(numeric_cols)

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        cat_filled = cat_imputer.fit_transform(X_df[categorical_cols].astype(str))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_processed = encoder.fit_transform(cat_filled)
        
        preprocessor["categorical_imputer"] = cat_imputer
        preprocessor["encoder"] = encoder
        transformed_parts.append(cat_processed)
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())

    X = np.hstack(transformed_parts)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    preprocessor["scaler"] = scaler

    return preprocessor, X_scaled, feature_names

def transform_features(X_df, preprocessor):
    processed_parts = []
    
    if preprocessor["numeric_cols"]:
        missing_num_cols = [col for col in preprocessor["numeric_cols"] if col not in X_df.columns]
        for col in missing_num_cols:
            X_df[col] = np.nan
        num_data = X_df[preprocessor["numeric_cols"]].apply(pd.to_numeric, errors='coerce')
        num_processed = preprocessor["numeric_imputer"].transform(num_data)
        processed_parts.append(num_processed)
        
    if preprocessor["categorical_cols"]:
        missing_cat_cols = [col for col in preprocessor["categorical_cols"] if col not in X_df.columns]
        for col in missing_cat_cols:
            X_df[col] = np.nan
        cat_data = X_df[preprocessor["categorical_cols"]].astype(str).replace("nan", np.nan)
        cat_filled = preprocessor["categorical_imputer"].transform(cat_data)
        cat_processed = preprocessor["encoder"].transform(cat_filled)
        processed_parts.append(cat_processed)
        
    X = np.hstack(processed_parts)
    return preprocessor["scaler"].transform(X)
