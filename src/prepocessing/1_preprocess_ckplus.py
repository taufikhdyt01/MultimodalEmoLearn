import os
import cv2
import numpy as np
from pathlib import Path

def preprocess_ckplus_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocessing gambar CK+ untuk konsistensi ukuran
    """
    emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprised']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for emotion in emotions:
        input_emotion_dir = os.path.join(input_dir, emotion)
        output_emotion_dir = os.path.join(output_dir, emotion)
        
        if not os.path.exists(input_emotion_dir):
            continue
            
        os.makedirs(output_emotion_dir, exist_ok=True)
        
        processed_count = 0
        for img_file in os.listdir(input_emotion_dir):
            if img_file.endswith('.png'):
                img_path = os.path.join(input_emotion_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Resize ke ukuran target
                    img_resized = cv2.resize(img, target_size)
                    
                    # Simpan gambar yang sudah diresize
                    output_path = os.path.join(output_emotion_dir, img_file)
                    cv2.imwrite(output_path, img_resized)
                    processed_count += 1
        
        print(f"Processed {processed_count} images for {emotion}")

if __name__ == "__main__":
    input_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ck+"
    output_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ck+_processed"
    
    preprocess_ckplus_images(input_dir, output_dir)