import os
import os
import cv2
import numpy as np
import random
import requests
from pymongo import MongoClient
from flask import Flask, request, jsonify

# Konfiguracija za MongoDB
client = MongoClient('mongodb+srv://zanluka:g1NmZuoD4MHnACDp@razvojapkzainternet.tb9k65s.mongodb.net/')
db = client['face_recognition']
verification_codes_col = db['verification_codes']

# FCM konfiguracija
FCM_SERVER_KEY = 'YOUR_FCM_SERVER_KEY'

app = Flask(__name__)

# 1. Zajemanje slik
def capture_images(output_dir='captured_images', num_images=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(output_dir, f'image_{count}.jpg')
            cv2.imwrite(img_path, frame)
            count += 1
            print(f'Captured {count}/{num_images}')
    cap.release()
    cv2.destroyAllWindows()

# 2. Predobdelava slik
def preprocess_images(input_dir='captured_images', output_dir='preprocessed_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f'Preprocessed {img_name}')

# 3. Augmentacija podatkov
def augment_dataset(input_dir='preprocessed_images', output_dir='augmented_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        augmented_images = augment_image(img)
        for i, aug_img in enumerate(augmented_images):
            aug_img_path = os.path.join(output_dir, f'{img_name}_aug_{i}.jpg')
            cv2.imwrite(aug_img_path, aug_img)
            print(f'Augmented {img_name} as {img_name}_aug_{i}.jpg')

def augment_image(img):
    augmented_images = []
    rows, cols = img.shape[:2]

    # Flip horizontal
    augmented_images.append(cv2.flip(img, 1))

    # Rotate
    M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    augmented_images.append(cv2.warpAffine(img, M_rotate, (cols, rows)))

    # Brightness adjustment
    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.5, beta=0))

    # Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    augmented_images.append(cv2.add(img, noise))

    # Custom augmentations
    # 1. Crop
    start_row, start_col = int(rows * .1), int(cols * .1)
    end_row, end_col = int(rows * .9), int(cols * .9)
    cropped_img = img[start_row:end_row, start_col:end_col]
    resized_cropped_img = cv2.resize(cropped_img, (cols, rows))
    augmented_images.append(resized_cropped_img)

    # 2. Salt and pepper noise
    salt_pepper_noise = img.copy()
    salt_pepper_prob = 0.02
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = random.random()
            if rand < salt_pepper_prob:
                salt_pepper_noise[i][j] = 0
            elif rand > 1 - salt_pepper_prob:
                salt_pepper_noise[i][j] = 255
    augmented_images.append(salt_pepper_noise)

    # 3. Contrast adjustment
    contrast_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
    augmented_images.append(contrast_img)

    # 4. Random erasing
    erasing_img = img.copy()
    x, y = random.randint(0, cols - 50), random.randint(0, rows - 50)
    w, h = random.randint(20, 50), random.randint(20, 50)
    erasing_img[y:y+h, x:x+w] = 0
    augmented_images.append(erasing_img)

    return augmented_images

# 4. Implementacija 2FA z uporabo Flask in FCM
def send_push_notification(token, message):
    url = 'https://fcm.googleapis.com/fcm/send'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'key={FCM_SERVER_KEY}',
    }
    payload = {
        'to': token,
        'notification': {
            'title': '2FA Verification',
            'body': message,
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


@app.route('/send_verification', methods=['POST'])

# Pošiljanje verifikacijske kode
def send_verification():
    data = request.json
    user_id = data['user_id']
    device_token = data['device_token']
    
    code = random.randint(100000, 999999)
    verification_data = {
        "user_id": user_id,
        "code": code
    }
    
    # Shranjevanje verifikacijske kode v MongoDB
    verification_codes_col.update_one(
        {"user_id": user_id}, 
        {"$set": verification_data}, 
        upsert=True
    )
    
    message = f"Your verification code is {code}"
    send_push_notification(device_token, message)
    return jsonify({"message": "Verification code sent."}), 200

@app.route('/verify_code', methods=['POST'])

# Verifikacija kode
def verify_code():
    data = request.json
    user_id = data['user_id']
    code = int(data['code'])
    
    # Preverjanje kode v MongoDB
    verification_entry = verification_codes_col.find_one({"user_id": user_id})
    
    if verification_entry and verification_entry['code'] == code:
        return jsonify({"message": "Verification successful."}), 200
    else:
        return jsonify({"message": "Verification failed."}), 401