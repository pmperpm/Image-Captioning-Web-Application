from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import json
import uuid
import shutil
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'gallery-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
DB_FILE = 'gallery_db.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model globals
model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global processor, model
    try:
        logger.info("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        processor = model = None
        raise


def generate_caption(image_path):
    global model, processor
    if model is None or processor is None:
        return "Model not loaded"
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption or "No caption generated"
    except Exception as e:
        logger.error(f"Caption error: {e}")
        return f"Error: {e}"


# Simple JSON database
def load_db():
    if not os.path.exists(DB_FILE):
        return {"albums": {}}
    with open(DB_FILE, 'r') as f:
        return json.load(f)


def save_db(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Albums
@app.route('/api/albums', methods=['GET'])
def get_albums():
    db = load_db()
    albums_list = []
    for aid, album in db['albums'].items():
        cover = album['images'][0]['thumbnail'] if album['images'] else None
        albums_list.append({
            'id': aid,
            'name': album['name'],
            'description': album.get('description', ''),
            'created_at': album['created_at'],
            'image_count': len(album['images']),
            'cover': cover
        })
    albums_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(albums_list)


@app.route('/api/albums', methods=['POST'])
def create_album():
    data = request.get_json()
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Album name required'}), 400
    db = load_db()
    aid = str(uuid.uuid4())
    db['albums'][aid] = {
        'name': name,
        'description': data.get('description', ''),
        'created_at': datetime.now().isoformat(),
        'images': []
    }
    save_db(db)
    return jsonify({'id': aid, 'name': name})


@app.route('/api/albums/<album_id>', methods=['GET'])
def get_album(album_id):
    db = load_db()
    album = db['albums'].get(album_id)
    if not album:
        return jsonify({'error': 'Album not found'}), 404
    return jsonify({'id': album_id, **album})


@app.route('/api/albums/<album_id>', methods=['DELETE'])
def delete_album(album_id):
    db = load_db()
    if album_id not in db['albums']:
        return jsonify({'error': 'Album not found'}), 404
    album_folder = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    if os.path.exists(album_folder):
        shutil.rmtree(album_folder)
    del db['albums'][album_id]
    save_db(db)
    return jsonify({'success': True})


# Images
@app.route('/api/albums/<album_id>/images', methods=['POST'])
def upload_images(album_id):
    db = load_db()
    if album_id not in db['albums']:
        return jsonify({'error': 'Album not found'}), 404

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    album_folder = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    os.makedirs(album_folder, exist_ok=True)

    results = []
    for file in files:
        if file and allowed_file(file.filename):
            img_id = str(uuid.uuid4())
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{img_id}.{ext}"
            filepath = os.path.join(album_folder, filename)
            file.save(filepath)

            # Create thumbnail
            try:
                with Image.open(filepath) as img:
                    img.thumbnail((400, 400))
                    thumb_name = f"{img_id}_thumb.{ext}"
                    thumb_path = os.path.join(album_folder, thumb_name)
                    img.save(thumb_path)
                    thumb_url = f"/uploads/{album_id}/{thumb_name}"
            except Exception:
                thumb_url = f"/uploads/{album_id}/{filename}"

            # Generate caption
            caption = generate_caption(filepath)

            image_data = {
                'id': img_id,
                'filename': filename,
                'original_name': secure_filename(file.filename),
                'url': f"/uploads/{album_id}/{filename}",
                'thumbnail': thumb_url,
                'caption': caption,
                'uploaded_at': datetime.now().isoformat()
            }
            db['albums'][album_id]['images'].append(image_data)
            results.append(image_data)

    save_db(db)
    return jsonify({'uploaded': results, 'count': len(results)})


@app.route('/api/albums/<album_id>/images/<image_id>', methods=['DELETE'])
def delete_image(album_id, image_id):
    db = load_db()
    album = db['albums'].get(album_id)
    if not album:
        return jsonify({'error': 'Album not found'}), 404

    image = next((img for img in album['images'] if img['id'] == image_id), None)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    # Delete files
    album_folder = os.path.join(app.config['UPLOAD_FOLDER'], album_id)
    for f in os.listdir(album_folder):
        if f.startswith(image_id):
            os.remove(os.path.join(album_folder, f))

    album['images'] = [img for img in album['images'] if img['id'] != image_id]
    save_db(db)
    return jsonify({'success': True})


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'cuda': torch.cuda.is_available()
    })


if __name__ == '__main__':
    try:
        load_model()
        print("🚀 Personal Gallery starting...")
        print("🌐 Open http://localhost:5003")
        app.run(debug=True, host='0.0.0.0', port=5003)
    except Exception as e:
        print(f"❌ Failed: {e}")
