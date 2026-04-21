import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from ml.preprocessing import fit_preprocessor, load_dataset, transform_features
from ml.gwo import gwo_binary, gwo_multiclass
from ml.model import (
    RANDOM_STATE, 
    fitness_function_binary, train_model_binary, evaluate_binary,
    fitness_function_multiclass, train_model_multiclass, evaluate_multiclass
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class State:
    df = None
    feature_columns = []
    y_binary = None
    y_3class = None
    preprocessor = None
    X_test = None
    y_test_target = None
    trained_model = None
    feature_mask = None
    selected_feature_names = []
    best_c_val = None
    classification_mode = "binary"
    
state = State()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        file_stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        X_df, feature_columns, y_binary, y_3class, y_multiclass = load_dataset(file_stream)
        
        state.df = X_df
        state.feature_columns = feature_columns
        state.y_binary = y_binary
        state.y_3class = y_3class
        
        return jsonify({
            "message": "Dataset loaded successfully.",
            "total_features": len(feature_columns),
            "features": feature_columns
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    if state.df is None:
        return jsonify({"error": "Please upload a dataset first."}), 400
        
    data = request.json or {}
    n_features = data.get("n_features", None)
    classification_mode = data.get("classification_mode", "binary")
    state.classification_mode = classification_mode

    if n_features is not None:
        try:
            n_features = int(n_features)
            if n_features < 7:
                return jsonify({"error": "Minimum 7 features required."}), 400
        except ValueError:
            return jsonify({"error": "Invalid n_features."}), 400

    try:
        active_target = state.y_binary if classification_mode == "binary" else state.y_3class

        indices = np.arange(len(state.df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=active_target, random_state=RANDOM_STATE
        )

        train_df = state.df.iloc[train_idx].copy()
        test_df = state.df.iloc[test_idx].copy()
        y_train_target = active_target[train_idx]
        y_test_target = active_target[test_idx]

        preprocessor, X_train, feature_names = fit_preprocessor(train_df)
        X_test = transform_features(test_df, preprocessor)

        n_iters = 30
        best_c_val = None

        if classification_mode == "binary":
            X_gwo_train, X_gwo_valid, y_gwo_train, y_gwo_valid = train_test_split(
                X_train, y_train_target, test_size=0.2, stratify=y_train_target, random_state=RANDOM_STATE
            )
            best_wolf, score, _ = gwo_binary(
                X_gwo_train, y_gwo_train, X_gwo_valid, y_gwo_valid, 
                fitness_function_binary, 
                target_n_features=n_features, 
                n_wolves=20, n_iterations=n_iters
            )
            model = train_model_binary(X_train, y_train_target, best_wolf)
            message = "Binary Model trained successfully."
        else:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train_target)

            best_wolf, best_c_val, score, _ = gwo_multiclass(
                X_train_res, y_train_res, None, None, 
                fitness_function_multiclass, 
                target_n_features=n_features, 
                n_wolves=20, n_iterations=n_iters
            )
            model = train_model_multiclass(X_train_res, y_train_res, best_wolf, c_val=best_c_val)
            message = f"Multiclass Model trained successfully. (Optimal C: {best_c_val:.4f})"

        selected_indices = np.where(best_wolf == 1)[0]
        selected_names = [feature_names[i] for i in selected_indices]

        state.preprocessor = preprocessor
        state.X_test = X_test
        state.y_test_target = y_test_target
        state.trained_model = model
        state.feature_mask = best_wolf
        state.selected_feature_names = selected_names
        state.best_c_val = best_c_val

        base_features = set()
        for f in selected_names:
             if f in state.feature_columns:
                  base_features.add(f)
             else:
                  for orig_col in state.preprocessor.get("categorical_cols", []):
                      if f.startswith(orig_col + "_"):
                          base_features.add(orig_col)
                          break
                  else:
                      base_features.add(f)

        return jsonify({
            "message": message,
            "selected_features": list(base_features)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if state.trained_model is None:
        return jsonify({"error": "Model not trained yet."}), 400
    
    try:
        if state.classification_mode == "binary":
            metrics = evaluate_binary(state.trained_model, state.X_test, state.y_test_target, state.feature_mask)
        else:
            metrics = evaluate_multiclass(state.trained_model, state.X_test, state.y_test_target, state.feature_mask)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if state.trained_model is None:
        return jsonify({"error": "Model not trained yet."}), 400
        
    user_inputs = request.json
    if not user_inputs:
        return jsonify({"error": "No input data provided."}), 400
        
    try:
        row_dict = {}
        for col in state.feature_columns:
            row_dict[col] = user_inputs.get(col, np.nan)
            
        X_df = pd.DataFrame([row_dict])
        X_custom = transform_features(X_df, state.preprocessor)
        
        selected_idx = np.where(state.feature_mask == 1)[0]
        prediction = state.trained_model.predict(X_custom[:, selected_idx])
        probabilities = state.trained_model.predict_proba(X_custom[:, selected_idx])
        
        return jsonify({
            "class": int(prediction[0]),
            "probabilities": probabilities[0].tolist(),
            "mode": state.classification_mode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
