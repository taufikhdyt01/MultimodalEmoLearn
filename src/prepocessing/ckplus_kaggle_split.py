import pandas as pd
import numpy as np
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

def create_ckplus_kaggle_dataframe(ckplus_kaggle_path):
    """
    Membuat DataFrame dari struktur dataset CK+ Kaggle yang sudah organized by emotion folders
    
    Args:
        ckplus_kaggle_path: Path ke root folder CK+ dari Kaggle
        
    Returns:
        DataFrame dengan kolom: image_path, subject, sequence, frame, emotion_label
    """
    
    # Emotion folders dalam dataset Kaggle CK+
    emotion_folders = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    # Mapping ke emotion labels yang sesuai dengan model existing Anda
    # Sesuaikan dengan label yang digunakan di model primer Anda
    emotion_mapping = {
        'anger': 'angry',
        'contempt': 'disgusted',  # Map contempt ke disgusted
        'disgust': 'disgusted',
        'fear': 'sad',  # Map fear ke sad (jika model Anda tidak punya fear)
        'happy': 'happy',
        'sadness': 'sad',
        'surprise': 'surprised'
    }
    
    data_list = []
    
    print("Scanning CK+ Kaggle dataset...")
    
    for emotion_folder in emotion_folders:
        emotion_path = os.path.join(ckplus_kaggle_path, emotion_folder)
        
        if not os.path.exists(emotion_path):
            print(f"Warning: Emotion folder {emotion_folder} not found")
            continue
        
        # Get all PNG files in this emotion folder
        image_files = glob.glob(os.path.join(emotion_path, "*.png"))
        
        print(f"Found {len(image_files)} images in {emotion_folder} folder")
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Parse filename: S010_004_00000017.png
            # Format: S{subject}_{sequence}_{frame}.png
            try:
                # Remove .png extension
                name_without_ext = filename.replace('.png', '')
                parts = name_without_ext.split('_')
                
                if len(parts) >= 3:
                    subject = parts[0]  # S010
                    sequence = parts[1]  # 004
                    frame = parts[2]    # 00000017
                    
                    # Map emotion ke target emotion
                    target_emotion = emotion_mapping.get(emotion_folder, emotion_folder)
                    
                    data_list.append({
                        'image_path': img_path,
                        'subject': subject,
                        'sequence': sequence,
                        'frame': frame,
                        'Dominant_Emotion': target_emotion,
                        'filename': filename,
                        'original_emotion': emotion_folder
                    })
                else:
                    print(f"Warning: Cannot parse filename {filename}")
                    
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
    
    df = pd.DataFrame(data_list)
    
    if len(df) > 0:
        print(f"\nDataset Summary:")
        print(f"Total images: {len(df)}")
        print(f"Unique subjects: {df['subject'].nunique()}")
        print(f"Emotion distribution:")
        print(df['Dominant_Emotion'].value_counts())
        print(f"\nOriginal emotion distribution:")
        print(df['original_emotion'].value_counts())
    
    return df

def split_ckplus_kaggle_dataset(ckplus_kaggle_path, output_dir, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Split CK+ Kaggle dataset menjadi train/val/test berdasarkan subject
    """
    
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Buat DataFrame dari CK+ Kaggle dataset
    df = create_ckplus_kaggle_dataframe(ckplus_kaggle_path)
    
    if len(df) == 0:
        print("Error: No data found in CK+ Kaggle dataset.")
        return None, None, None
    
    print(f"\nTotal CK+ samples found: {len(df)}")
    
    # Split berdasarkan subject untuk menghindari data leakage
    subjects = df['subject'].unique()
    print(f"Total unique subjects: {len(subjects)}")
    
    # Split subjects first
    train_subjects, temp_subjects = train_test_split(
        subjects, 
        train_size=train_size, 
        random_state=42
    )
    
    relative_val_size = val_size / (val_size + test_size)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        train_size=relative_val_size,
        random_state=42
    )
    
    # Create splits based on subjects
    train_df = df[df['subject'].isin(train_subjects)]
    val_df = df[df['subject'].isin(val_subjects)]
    test_df = df[df['subject'].isin(test_subjects)]
    
    # Save splits
    train_df.to_excel(os.path.join(output_dir, "train_data.xlsx"), index=False)
    val_df.to_excel(os.path.join(output_dir, "val_data.xlsx"), index=False)
    test_df.to_excel(os.path.join(output_dir, "test_data.xlsx"), index=False)
    
    # Print detailed statistics
    print(f"\n" + "="*60)
    print("DATASET SPLIT RESULTS")
    print("="*60)
    print(f"Training set: {len(train_df)} samples ({len(train_subjects)} subjects)")
    print(f"Validation set: {len(val_df)} samples ({len(val_subjects)} subjects)")
    print(f"Test set: {len(test_df)} samples ({len(test_subjects)} subjects)")
    
    # Print emotion distribution for each split
    for split_name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{split_name} emotion distribution:")
        emotion_counts = split_df['Dominant_Emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percent = count / len(split_df) * 100
            print(f"  {emotion}: {count} ({percent:.1f}%)")
    
    return train_df, val_df, test_df