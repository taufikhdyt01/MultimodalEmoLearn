"""
Pipeline lengkap untuk mempersiapkan dataset CK+ untuk training
"""
import sys
import os

# Tambahkan path proyek ke sys.path
sys.path.append('D:/research/2025_iris_taufik/MultimodalEmoLearn')

from scripts.preprocess_ckplus import preprocess_ckplus_images
from scripts.extract_landmarks_ckplus import extract_landmarks_ckplus
from scripts.prepare_ckplus_data import prepare_ckplus_for_training
from split_dataset import split_dataset
from prepare_landmark import prepare_landmark_data_for_lstm

def run_complete_ckplus_pipeline():
    """Jalankan pipeline lengkap untuk CK+"""
    
    print("=== STARTING CK+ DATA PIPELINE ===")
    
    # Paths
    base_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data"
    ckplus_raw = os.path.join(base_path, "ckplus")
    ckplus_processed = os.path.join(base_path, "ckplus_processed")
    landmarks_raw = os.path.join(base_path, "Emotion_Labels", "ckplus_landmarks_raw.xlsx")
    labeled_data = os.path.join(base_path, "Emotion_Labels", "ckplus_labeled_data.xlsx")
    split_dir = os.path.join(base_path, "ckplus_split")
    landmarks_dir = os.path.join(base_path, "ckplus_landmarks")
    
    # Step 1: Preprocess images
    print("\n1. Preprocessing images...")
    preprocess_ckplus_images(ckplus_raw, ckplus_processed)
    
    # Step 2: Extract landmarks
    print("\n2. Extracting landmarks...")
    extract_landmarks_ckplus(ckplus_processed, landmarks_raw)
    
    # Step 3: Prepare data format
    print("\n3. Preparing data format...")
    prepare_ckplus_for_training(landmarks_raw, labeled_data)
    
    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    split_dataset(labeled_data, split_dir, train_size=0.8, val_size=0.1, test_size=0.1)
    
    # Step 5: Prepare landmark data for LSTM
    print("\n5. Preparing landmark data for LSTM...")
    prepare_landmark_data_for_lstm(split_dir, landmarks_dir)
    
    print("\n=== CK+ PIPELINE COMPLETE ===")
    print(f"Data ready for training!")
    print(f"Landmark data location: {landmarks_dir}")

if __name__ == "__main__":
    run_complete_ckplus_pipeline()