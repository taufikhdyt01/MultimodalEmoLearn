import dlib
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def extract_landmarks_from_image(image_path, predictor_path="shape_predictor_68_face_landmarks.dat"):
    """
    Extract facial landmarks dari gambar menggunakan dlib
    """
    
    # Initialize dlib detectors
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Detect faces
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    # Use the first detected face
    face = faces[0]
    
    # Get landmarks
    landmarks = predictor(gray, face)
    
    # Convert to numpy array (68 points * 2 coordinates = 136 features)
    landmark_points = []
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmark_points.extend([x, y])
    
    return np.array(landmark_points)

def prepare_ckplus_kaggle_landmarks(split_dir, output_dir, predictor_path="shape_predictor_68_face_landmarks.dat"):
    """
    Extract landmarks dari semua gambar CK+ Kaggle dalam splits
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dlib predictor exists
    if not os.path.exists(predictor_path):
        print(f"Error: dlib predictor not found at {predictor_path}")
        print("Please download shape_predictor_68_face_landmarks.dat from dlib website")
        print("You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    split_files = {
        'train': os.path.join(split_dir, 'train_data.xlsx'),
        'val': os.path.join(split_dir, 'val_data.xlsx'),
        'test': os.path.join(split_dir, 'test_data.xlsx')
    }
    
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        
        print(f"Processing {split_name} landmarks...")
        df = pd.read_excel(split_file)
        
        X_landmarks = []
        y_emotions = []
        metadata = []
        
        successful = 0
        skipped = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name} landmarks"):
            try:
                img_path = row['image_path']
                emotion = row['Dominant_Emotion']
                
                # Extract landmarks
                landmarks = extract_landmarks_from_image(img_path, predictor_path)
                
                if landmarks is None:
                    skipped += 1
                    continue
                
                X_landmarks.append(landmarks)
                y_emotions.append(emotion)
                metadata.append({
                    'image_path': img_path,
                    'subject': row['subject'],
                    'sequence': row['sequence'],
                    'frame': row['frame'],
                    'emotion': emotion,
                    'original_emotion': row['original_emotion']
                })
                successful += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                skipped += 1
        
        if not X_landmarks:
            print(f"No valid landmark data found for {split_name}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_landmarks)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_landmarks.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
        
        # Save metadata
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(output_dir, f"{split_name}_metadata.csv"), index=False)
        
        print(f"  - Successfully processed: {successful} samples")
        print(f"  - Skipped: {skipped} samples")
        print(f"  - Shape of X_{split_name}_landmarks: {X.shape}")
        print(f"  - Shape of y_{split_name}: {y.shape}")
        
        # Print emotion distribution
        emotions, counts = np.unique(y, return_counts=True)
        print(f"  - Emotion distribution:")
        for emotion, count in zip(emotions, counts):
            percent = count / len(y) * 100
            print(f"    {emotion}: {count} ({percent:.2f}%)")