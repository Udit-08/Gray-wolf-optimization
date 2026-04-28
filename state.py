class AppState:
    def __init__(self):
        # Global Dataset State
        self.df = None
        self.feature_columns = []
        self.y_binary = None
        self.y_3class = None
        
        # Shared Preprocessor (fit once per dataset upload ideally, or per model)
        # We will fit it when uploading
        self.preprocessor = None
        self.X_processed = None
        self.feature_names = []
        
        # --- Logistic Regression (GWO) State ---
        self.lr_model = None
        self.lr_mode = "binary"
        self.lr_feature_mask = None
        self.lr_selected_names = []
        self.lr_fitness_history = []
        self.lr_metrics = None
        
        # --- Support Vector Machine State ---
        self.svm_model = None
        self.svm_mode = "binary"
        self.svm_metrics = None

# Singleton instance
app_state = AppState()
