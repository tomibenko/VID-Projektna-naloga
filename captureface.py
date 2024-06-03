import cv2
import os
import time

def capture_images_from_video(output_dir='captured_images', duration=10, fps=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    
    # Calculate the interval between frames
    interval = 1 / fps
    start_time = time.time()
    frame_count = 0
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_path = os.path.join(output_dir, f'image_{frame_count}.jpg')
        cv2.imwrite(img_path, frame)
        frame_count += 1
        print(f'Captured {frame_count} images')
        
        time.sleep(interval)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing images. {frame_count} images saved to {output_dir}")

if __name__ == "__main__":
    capture_images_from_video(output_dir='captured_images', duration=10, fps=100)
