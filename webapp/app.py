"""
Flask web application for meme sentiment analysis.
Upload a meme → get sentiment, humor, and sarcasm predictions.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_meme
from config.config import Config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Global model (loaded once at startup)
_model = None
_config = None


def get_model():
    """Lazy-load the model on first request."""
    global _model, _config
    if _model is None:
        import torch
        from models.multimodal_model import MultimodalMemeAnalyzer
        _config = Config()
        _model = MultimodalMemeAnalyzer(_config)
        
        checkpoint_path = os.path.join(_config.checkpoints_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=_config.device, weights_only=False)
            _model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from {checkpoint_path}")
        else:
            print("⚠ No trained model found — using untrained model")
        
        _model = _model.to(_config.device)
        _model.eval()
    return _model, _config


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle meme upload and return predictions."""
    # Check for file upload
    if 'file' not in request.files:
        # Check for text-only analysis
        text = request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No file or text provided'}), 400
        
        # Text-only mode: create a blank image
        from PIL import Image
        import tempfile
        blank_path = os.path.join(app.config['UPLOAD_FOLDER'], '_blank.png')
        Image.new('RGB', (224, 224), (0, 0, 0)).save(blank_path)
        
        model, config = get_model()
        results = predict_meme(blank_path, model=model, config=config, text_override=text)
        return jsonify({'success': True, 'results': results})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        model, config = get_model()
        
        # Get optional manual text override
        text_override = request.form.get('text', '').strip() or None
        
        results = predict_meme(filepath, model=model, config=config, text_override=text_override)
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath) and filename != '_blank.png':
            os.remove(filepath)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


if __name__ == '__main__':
    print("\n🚀 Starting Meme Sentiment Analysis Web App...")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
