import cv2
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

def prepare_ckplus_kaggle_images_for_cnn(split_dir, output_dir, img_size=(48, 48)):
    """
    Memproses gambar CK+ Kaggle untuk training CNN
    
    Args:
        split_dir: Direktori berisi file split (train_data.xlsx, val_data.xlsx, test_data.xlsx)
        output_dir: Direktori output untuk menyimpan data yang sudah diproses
        img_size: Ukuran target untuk resize gambar
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # File splits
    split_files = {
        'train': os.path.join(split_dir, 'train_data.xlsx'),
        'val': os.path.join(split_dir, 'val_data.xlsx'),
        'test': os.path.join(split_dir, 'test_data.xlsx')
    }
    
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        
        print(f"Processing {split_name} images...")
        df = pd.read_excel(split_file)
        
        # Create emotion directories
        split_img_dir = os.path.join(output_dir, f"{split_name}_images")
        os.makedirs(split_img_dir, exist_ok=True)
        
        for emotion in df['Dominant_Emotion'].unique():
            os.makedirs(os.path.join(split_img_dir, str(emotion)), exist_ok=True)
        
        # Process images
        X_images = []
        y_emotions = []
        frame_paths = []
        
        found = 0
        not_found = 0
        skipped = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                img_path = row['image_path']
                emotion = row['Dominant_Emotion']
                
                # Check if image exists
                if not os.path.exists(img_path):
                    not_found += 1
                    continue
                
                # Copy image to organized structure
                frame_name = f"{row['subject']}_{row['sequence']}_{row['frame']}.png"
                dest_path = os.path.join(split_img_dir, str(emotion), frame_name)
                shutil.copy2(img_path, dest_path)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                
                # Convert grayscale to RGB if needed (CK+ images are usually grayscale)
                if len(img.shape) == 3 and img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 2:  # Pure grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize and normalize
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize to [0, 1]
                
                X_images.append(img)
                y_emotions.append(emotion)
                frame_paths.append(dest_path)
                found += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                skipped += 1
        
        if not X_images:
            print(f"No valid image data found for {split_name}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_images)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_images.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}_images.npy"), y)
        
        # Save frame paths for reference
        paths_df = pd.DataFrame({
            'path': frame_paths,
            'emotion': y_emotions,
            'subject': df['subject'][:len(frame_paths)],
            'sequence': df['sequence'][:len(frame_paths)]
        })
        paths_df.to_csv(os.path.join(output_dir, f"{split_name}_image_paths.csv"), index=False)
        
        # Print statistics
        print(f"  - Found images: {found}")
        print(f"  - Images not found: {not_found}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Shape of X_{split_name}_images: {X.shape}")
        
        # Print emotion distribution
        emotions, counts = np.unique(y, return_counts=True)
        print(f"  - Emotion distribution:")
        for emotion, count in zip(emotions, counts):
            percent = count / len(y) * 100
            print(f"    {emotion}: {count} ({percent:.2f}%)")