from flask import Blueprint, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from state import app_state
from utils.preprocessing import transform_features
from ml.svm.model import tune_and_train_svm
import numpy as np

svm_bp = Blueprint('svm', __name__, url_prefix='/svm')

RANDOM_STATE = 42

@svm_bp.route("/predict_page", methods=["GET"])
def predict_page():
    return render_template("svm_predict.html")

@svm_bp.route("/analysis_page", methods=["GET"])
def analysis_page():
    return render_template("svm_analysis.html")

@svm_bp.route("/train", methods=["POST"])
def train():
    if app_state.X_processed is None:
        return jsonify({"error": "Please upload a dataset first on the Home page."}), 400
        
    data = request.json
    classification_mode = data.get("classification_mode", "binary")
    
    app_state.svm_mode = classification_mode
    
    try:
        from imblearn.over_sampling import SMOTE
        y_target = app_state.y_binary if classification_mode == "binary" else app_state.y_3class
        
        X_train, X_test, y_train_target, y_test_target = train_test_split(
            app_state.X_processed, y_target, test_size=0.2, stratify=y_target, random_state=RANDOM_STATE
        )
        
        # Resampling is typically good for both
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train_target)
        
        # Exhaustive tuning (will take time)
        model = tune_and_train_svm(X_train_res, y_train_res)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test_target, y_pred)
        f1 = f1_score(y_test_target, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test_target, y_pred).tolist()
        
        # For ROC curve
        y_proba = model.predict_proba(X_test)
        roc_data = None
        if classification_mode == "binary":
            y_proba_for_roc = y_proba[:, 1].tolist()
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test_target, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        else:
            y_proba_for_roc = y_proba.tolist()

        app_state.svm_model = model
        app_state.svm_X_test = X_test
        app_state.svm_y_test = y_test_target
        
        app_state.svm_metrics = {
            "accuracy": acc,
            "f1": f1,
            "confusion_matrix": cm,
            "y_test": y_test_target.tolist(),
            "y_proba": y_proba_for_roc,
            "roc": roc_data
        }

        return jsonify({
            "message": f"{classification_mode.capitalize()} SVM Model trained successfully using full GridSearchCV.",
            "all_features": app_state.feature_names
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@svm_bp.route("/predict", methods=["POST"])
def predict():
    if app_state.svm_model is None:
        return jsonify({"error": "SVM Model not trained yet."}), 400
    
    try:
        data = request.json
        import pandas as pd
        X_df = pd.DataFrame([data])
        X_custom = transform_features(X_df, app_state.preprocessor)
        
        prediction = app_state.svm_model.predict(X_custom)
        probabilities = app_state.svm_model.predict_proba(X_custom)
        
        return jsonify({
            "class": int(prediction[0]),
            "probabilities": probabilities[0].tolist(),
            "mode": app_state.svm_mode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@svm_bp.route("/metrics", methods=["GET"])
def metrics():
    if app_state.svm_model is None or app_state.svm_metrics is None:
        return jsonify({"error": "SVM Model not trained yet."}), 400
    
    return jsonify({
        "metrics": app_state.svm_metrics,
        "mode": app_state.svm_mode
    })
