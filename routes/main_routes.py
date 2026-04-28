from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from state import app_state
from utils.preprocessing import load_dataset, fit_preprocessor

main_bp = Blueprint('main', __name__)

@main_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@main_bp.route("/dataset_info", methods=["GET"])
def dataset_info():
    return jsonify({
        "raw_feature_count": len(app_state.feature_columns),
        "processed_feature_count": len(app_state.feature_names)
    })

@main_bp.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed."}), 400

    try:
        X_df, feature_columns, y_binary, y_3class, y_multiclass = load_dataset(file)
        
        # Fit preprocessor globally once
        preprocessor, X_processed, feature_names = fit_preprocessor(X_df)
        
        # Update global state
        app_state.df = X_df
        app_state.feature_columns = feature_columns
        app_state.y_binary = y_binary
        app_state.y_3class = y_3class
        
        app_state.preprocessor = preprocessor
        app_state.X_processed = X_processed
        app_state.feature_names = feature_names
        
        # Reset ML models since dataset changed
        app_state.lr_model = None
        app_state.svm_model = None
        
        return jsonify({
            "message": "Dataset uploaded successfully.",
            "raw_feature_count": len(feature_columns),
            "processed_feature_count": len(feature_names)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
