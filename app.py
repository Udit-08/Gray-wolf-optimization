from flask import Flask
from routes.main_routes import main_bp
from routes.logistic_routes import lr_bp
from routes.svm_routes import svm_bp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Register modular blueprints
app.register_blueprint(main_bp)
app.register_blueprint(lr_bp)
app.register_blueprint(svm_bp)

if __name__ == '__main__':
    app.run(debug=True)
