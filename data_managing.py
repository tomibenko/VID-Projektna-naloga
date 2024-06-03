import os
import cv2
from pymongo import MongoClient
from flask import Flask

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