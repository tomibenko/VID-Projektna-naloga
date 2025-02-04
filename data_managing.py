import os
import cv2
import numpy as np
import random
import subprocess
import shutil
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model  
from image_compression import decompress_color_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
KNOWN_FOLDER = 'test/known'  # Path for saving all processed images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # Set max upload size to 1GB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(KNOWN_FOLDER):
    os.makedirs(KNOWN_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'status': 'failure', 'message': 'No video part'})
    video = request.files['video']
    user_id = request.form['user_id']
    if video.filename == '':
        return jsonify({'status': 'failure', 'message': 'No selected file'})
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    process_video(video_path,user_id)

    return jsonify({'status': 'success', 'message': 'Video uploaded successfully', 'path': video_path})

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'frame_{count}.jpg')
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()

def random_augment_image(img):
    def flip(img):
        return cv2.flip(img, 1)

    def rotate_image(img):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
        return cv2.warpAffine(img, M, (cols, rows))

    def adjust_brightness(img):
        return cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    def add_gaussian_noise(img):
        gauss = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, gauss)

    def crop_and_resize(img):
        rows, cols = img.shape[:2]
        start_row, start_col = int(rows * .1), int(cols * .1)
        end_row, end_col = int(rows * .9), int(cols * .9)
        cropped = img[start_row:end_row, start_col:end_col]
        return cv2.resize(cropped, (cols, rows))

    def add_salt_and_pepper_noise(img):
        salt_pepper_prob = 0.02
        noisy = np.copy(img)
        num_salt = np.ceil(salt_pepper_prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 1

        num_pepper = np.ceil(salt_pepper_prob * img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy

    def adjust_contrast(img):
        return cv2.convertScaleAbs(img, alpha=2.0, beta=0)

    def random_erasing(img):
        x, y = random.randint(0, img.shape[1] - 50), random.randint(0, img.shape[0] - 50)
        w, h = random.randint(20, 50), random.randint(20, 50)
        img[y:y+h, x:x+w] = 0
        return img

    augmentation_functions = [flip, rotate_image, adjust_brightness, add_gaussian_noise, crop_and_resize, add_salt_and_pepper_noise, adjust_contrast, random_erasing]
    chosen_functions = random.sample(augmentation_functions, k=random.randint(1, len(augmentation_functions)))

    for func in chosen_functions:
        img = func(img)

    return img

def process_and_save_frames(frames_dir):
    frames = os.listdir(frames_dir)
    for frame in frames:
        img = cv2.imread(os.path.join(frames_dir, frame))
        img = random_augment_image(img)
        cv2.imwrite(os.path.join(KNOWN_FOLDER, frame), img)

@app.route('/process_video', methods=['POST'])
def process_video(video_path,user_id):
    frames_dir = 'extracted_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    extract_frames(video_path, frames_dir)
    process_and_save_frames(frames_dir)

    subprocess.run(['python', 'projekt_1.py', '--user_id', user_id])

    # Izbrise vse slike iz test/known in uploads
    for folder in [KNOWN_FOLDER, UPLOAD_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    return jsonify({'status': 'success', 'message': 'Video processed and frames saved'})

def user_has_model(user_id):
    user_model = [f for f in os.listdir('.') if f.startswith(user_id)]
    return len(user_model) > 0

# 2FA login
@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('user_id')
    if not user_id or 'image' not in request.files:
        return jsonify({'status': 'failure', 'message': 'User ID or image missing'})

    # 1) Save compressed file
    compressed_file = request.files['image']
    compressed_path = os.path.join(UPLOAD_FOLDER, compressed_file.filename)
    compressed_file.save(compressed_path)

    # 2) Decompress it into a real image (e.g. "decompressed.jpg")
    decompressed_path = os.path.join(UPLOAD_FOLDER, "decompressed.jpg")
    try:
        decompress_color_file(compressed_path, decompressed_path)
    except Exception as e:
        return jsonify({'status': 'failure', 'message': f'Error decompressing: {str(e)}'})

    # 3) Now read the decompressed image with OpenCV
    img = cv2.imread(decompressed_path)
    if img is None:
        return jsonify({'status': 'failure', 'message': 'Decompressed image is invalid'})

    # Resize, batch dimension, etc.
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # 4) Load your user-specific model and make a prediction
    model_path = os.path.join('models', f'{user_id}.h5')
    if not os.path.exists(model_path):
        return jsonify({'status': 'failure', 'message': 'Model not found'})

    model = load_model(model_path)
    prediction = model.predict(img)

    # Suppose > 0.7 is a "match"
    if prediction[0][0] > 0.7:
        return jsonify({'status': 'success', 'match': True})
    else:
        return jsonify({'status': 'success', 'match': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
