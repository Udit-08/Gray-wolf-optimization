import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split

from ml.preprocessing import fit_preprocessor, load_dataset, transform_features
from ml.gwo import gwo
from ml.model import RANDOM_STATE, fitness_function_stage1, train_model, evaluate_stage1_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

class State:
    df = None
    feature_columns = []
    y_binary = None
    preprocessor = None
    X_test = None
    y_test_binary = None
    trained_model = None
    feature_mask = None
    selected_feature_names = []
    
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
        X_df, feature_columns, y_binary, y_multiclass = load_dataset(file_stream)
        
        state.df = X_df
        state.feature_columns = feature_columns
        state.y_binary = y_binary
        
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
    if n_features is not None:
        try:
            n_features = int(n_features)
            if n_features < 7:
                return jsonify({"error": "Minimum 7 features required."}), 400
        except ValueError:
            return jsonify({"error": "Invalid n_features."}), 400

    try:
        indices = np.arange(len(state.df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=state.y_binary, random_state=RANDOM_STATE
        )

        train_df = state.df.iloc[train_idx].copy()
        test_df = state.df.iloc[test_idx].copy()
        y_train_binary = state.y_binary[train_idx]
        y_test_binary = state.y_binary[test_idx]

        preprocessor, X_train, feature_names = fit_preprocessor(train_df)
        X_test = transform_features(test_df, preprocessor)

        X_gwo_train, X_gwo_valid, y_gwo_train, y_gwo_valid = train_test_split(
            X_train, y_train_binary, test_size=0.2, stratify=y_train_binary, random_state=RANDOM_STATE
        )

        n_iters = 10 if n_features else 15
        best_wolf, score, _ = gwo(
            X_gwo_train, y_gwo_train, X_gwo_valid, y_gwo_valid, 
            fitness_function_stage1, 
            target_n_features=n_features, 
            n_wolves=10, n_iterations=n_iters
        )

        model = train_model(X_train, y_train_binary, best_wolf)

        selected_indices = np.where(best_wolf == 1)[0]
        selected_names = [feature_names[i] for i in selected_indices]

        state.preprocessor = preprocessor
        state.X_test = X_test
        state.y_test_binary = y_test_binary
        state.trained_model = model
        state.feature_mask = best_wolf
        state.selected_feature_names = selected_names

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
            "message": "Model trained successfully.",
            "selected_features": list(base_features)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if state.trained_model is None:
        return jsonify({"error": "Model not trained yet."}), 400
    
    try:
        metrics = evaluate_stage1_model(state.trained_model, state.X_test, state.y_test_binary, state.feature_mask)
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
            "probability": float(probabilities[0][1])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
