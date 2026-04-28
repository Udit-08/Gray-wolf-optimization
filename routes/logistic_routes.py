from flask import Blueprint, render_template, request, jsonify
import numpy as np
from sklearn.model_selection import train_test_split
from state import app_state
from utils.preprocessing import transform_features
from ml.logistic.gwo import gwo_binary, gwo_multiclass
from ml.logistic.model import (
    fitness_function_binary, train_model_binary, evaluate_binary,
    fitness_function_multiclass, train_model_multiclass, evaluate_multiclass
)

lr_bp = Blueprint('lr', __name__, url_prefix='/lr')

RANDOM_STATE = 42

@lr_bp.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("lr_predict.html")

@lr_bp.route("/analysis_page", methods=["GET"])
def analysis_page():
    return render_template("lr_analysis.html")

@lr_bp.route("/train", methods=["POST"])
def train():
    if app_state.X_processed is None:
        return jsonify({"error": "Please upload a dataset first on the Home page."}), 400
        
    data = request.json
    n_features = int(data.get("n_features", 7))
    classification_mode = data.get("classification_mode", "binary")
    n_iters = 5  # Quick for demo, could be configurable
    
    app_state.lr_mode = classification_mode
    
    try:
        from imblearn.over_sampling import SMOTE
        y_target = app_state.y_binary if classification_mode == "binary" else app_state.y_3class
        
        X_train, X_test, y_train_target, y_test_target = train_test_split(
            app_state.X_processed, y_target, test_size=0.2, stratify=y_target, random_state=RANDOM_STATE
        )
        
        if classification_mode == "binary":
            X_gwo_train, X_gwo_valid, y_gwo_train, y_gwo_valid = train_test_split(
                X_train, y_train_target, test_size=0.2, stratify=y_train_target, random_state=RANDOM_STATE
            )
            best_wolf, score, fitness_history = gwo_binary(
                X_gwo_train, y_gwo_train, X_gwo_valid, y_gwo_valid, 
                fitness_function_binary, 
                target_n_features=n_features, 
                n_wolves=20, n_iterations=n_iters
            )
            model = train_model_binary(X_train, y_train_target, best_wolf)
            message = "Binary LR Model trained successfully."
        else:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train_target)

            best_wolf, best_c_val, score, fitness_history = gwo_multiclass(
                X_train_res, y_train_res, None, None, 
                fitness_function_multiclass, 
                target_n_features=n_features, 
                n_wolves=20, n_iterations=n_iters
            )
            model = train_model_multiclass(X_train_res, y_train_res, best_wolf, c_val=best_c_val)
            message = f"Multiclass LR Model trained successfully. (Optimal C: {best_c_val:.4f})"

        selected_indices = np.where(best_wolf == 1)[0]
        selected_names = [app_state.feature_names[i] for i in selected_indices]

        # Save to state
        app_state.lr_model = model
        app_state.lr_feature_mask = best_wolf
        app_state.lr_selected_names = selected_names
        app_state.lr_fitness_history = fitness_history
        app_state.lr_X_test = X_test
        app_state.lr_y_test = y_test_target
        
        # Calculate metrics immediately
        if classification_mode == "binary":
            app_state.lr_metrics = evaluate_binary(model, X_test, y_test_target, best_wolf)
            
            # ROC logic could go here or we can just fetch probas
            y_proba = model.predict_proba(X_test[:, selected_indices])[:, 1]
            app_state.lr_metrics['y_test'] = y_test_target.tolist()
            app_state.lr_metrics['y_proba'] = y_proba.tolist()
        else:
            app_state.lr_metrics = evaluate_multiclass(model, X_test, y_test_target, best_wolf)

        # Feature Importance for LR
        if hasattr(model, "coef_"):
            # For multiclass, coef_ is shape (n_classes, n_features), average them or take max
            importances = np.abs(model.coef_).mean(axis=0)
            app_state.lr_metrics["feature_importances"] = {
                "names": selected_names,
                "values": importances.tolist()
            }

        return jsonify({
            "message": message,
            "selected_features": selected_names
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@lr_bp.route("/predict", methods=["POST"])
def predict():
    if app_state.lr_model is None:
        return jsonify({"error": "LR Model not trained yet."}), 400
    
    try:
        data = request.json
        import pandas as pd
        X_df = pd.DataFrame([data])
        X_custom = transform_features(X_df, app_state.preprocessor)
        
        selected_idx = np.where(app_state.lr_feature_mask == 1)[0]
        lr_prediction = app_state.lr_model.predict(X_custom[:, selected_idx])
        lr_probabilities = app_state.lr_model.predict_proba(X_custom[:, selected_idx])
        
        return jsonify({
            "class": int(lr_prediction[0]),
            "probabilities": lr_probabilities[0].tolist(),
            "mode": app_state.lr_mode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@lr_bp.route("/metrics", methods=["GET"])
def metrics():
    if app_state.lr_model is None or app_state.lr_metrics is None:
        return jsonify({"error": "LR Model not trained yet."}), 400
    
    return jsonify({
        "metrics": app_state.lr_metrics,
        "fitness_history": app_state.lr_fitness_history,
        "mode": app_state.lr_mode
    })
