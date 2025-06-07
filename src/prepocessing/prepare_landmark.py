import pandas as pd
import numpy as np
import json
import os
import re

def extract_landmarks_with_regex(landmark_str):
    """
    Extract landmarks using regex directly from the string.
    """
    if not isinstance(landmark_str, str):
        return None
    
    # Pattern untuk mencari koordinat x dan y
    x_pattern = r'\"_x\":([0-9.]+)'
    y_pattern = r'\"_y\":([0-9.]+)'
    
    # Temukan semua nilai x dan y
    x_matches = re.findall(x_pattern, landmark_str)
    y_matches = re.findall(y_pattern, landmark_str)
    
    # Pastikan jumlah x dan y sama
    if len(x_matches) != len(y_matches):
        return None
    
    # Gabungkan nilai x dan y dalam array 1D
    flattened = []
    for x, y in zip(x_matches, y_matches):
        try:
            flattened.append(float(x))
            flattened.append(float(y))
        except ValueError:
            # Skip jika tidak bisa dikonversi ke float
            continue
    
    return flattened

def prepare_landmark_data_for_lstm(split_dir, output_dir):
    """
    Mempersiapkan data landmark untuk input LSTM.
    Versi yang disederhanakan tanpa filtering berdasarkan challenge.
    
    Args:
        split_dir: Direktori yang berisi file split data (train, val, test)
        output_dir: Direktori untuk menyimpan data yang telah diformat untuk LSTM
    """
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Define split files
    split_files = {
        'train': os.path.join(split_dir, 'train_data.xlsx'),
        'val': os.path.join(split_dir, 'val_data.xlsx'),
        'test': os.path.join(split_dir, 'test_data.xlsx')
    }
    
    # Process each split
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        
        print(f"Processing {split_name} data...")
        df = pd.read_excel(split_file)
        
        # Check if required columns exist
        if 'landmarks' not in df.columns:
            print(f"Error: 'landmarks' column not found in {split_file}. Skipping.")
            continue
            
        if 'Dominant_Emotion' not in df.columns:
            print(f"Error: 'Dominant_Emotion' column not found in {split_file}. Skipping.")
            continue
        
        # Create lists to store processed data
        X_landmarks = []
        y_emotions = []
        metadata = []
        
        # Process each row
        skipped = 0
        successful = 0
        
        for idx, row in df.iterrows():
            try:
                # Extract landmarks from JSON string
                landmarks_str = row['landmarks']
                if pd.isna(landmarks_str) or landmarks_str == "":
                    skipped += 1
                    continue
                
                # Try parsing with regex first (more robust)
                flattened_landmarks = extract_landmarks_with_regex(landmarks_str)
                
                # If regex fails, try JSON parsing
                if flattened_landmarks is None or len(flattened_landmarks) == 0:
                    try:
                        # Clean the JSON string if necessary
                        if isinstance(landmarks_str, str):
                            landmarks_str = landmarks_str.replace('\\"', '"')
                            if not landmarks_str.startswith('{'):
                                landmarks_str = '{' + landmarks_str.split('{', 1)[1]
                            if not landmarks_str.endswith('}'):
                                landmarks_str = landmarks_str.rsplit('}', 1)[0] + '}'
                        
                        # Try direct json parsing
                        try:
                            landmarks_data = json.loads(landmarks_str)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, skip this row
                            skipped += 1
                            continue
                        
                        # Extract coordinates
                        flattened_landmarks = []
                        if isinstance(landmarks_data, list):
                            for point in landmarks_data:
                                if isinstance(point, dict) and '_x' in point and '_y' in point:
                                    flattened_landmarks.extend([float(point['_x']), float(point['_y'])])
                        elif isinstance(landmarks_data, dict):
                            # Handle different JSON structures
                            for key, value in landmarks_data.items():
                                if isinstance(value, dict) and '_x' in value and '_y' in value:
                                    flattened_landmarks.extend([float(value['_x']), float(value['_y'])])
                        
                    except Exception as e:
                        skipped += 1
                        continue
                
                # Ensure we have landmarks data
                if not flattened_landmarks or len(flattened_landmarks) == 0:
                    skipped += 1
                    continue
                
                # Normalize landmark data length
                expected_length = 136  # Typical number for 68 landmarks (68 x 2 coordinates)
                
                # Check if we have fewer than expected
                if len(flattened_landmarks) < expected_length:
                    # Pad with zeros
                    flattened_landmarks.extend([0.0] * (expected_length - len(flattened_landmarks)))
                elif len(flattened_landmarks) > expected_length:
                    # Truncate
                    flattened_landmarks = flattened_landmarks[:expected_length]
                
                X_landmarks.append(flattened_landmarks)
                y_emotions.append(row['Dominant_Emotion'])
                
                # Add metadata for reference (tanpa Challenge)
                meta = {
                    'user_id': row['user_id'] if 'user_id' in df.columns else None,
                    'timestamp': str(row['timestamp']) if 'timestamp' in df.columns else None,
                    'Confidence_Score': row['Confidence_Score'] if 'Confidence_Score' in df.columns else None
                }
                metadata.append(meta)
                successful += 1
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped += 1
        
        if not X_landmarks:
            print(f"No valid landmark data found in {split_file}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_landmarks)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_landmarks.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
        
        # Save metadata as CSV
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(output_dir, f"{split_name}_metadata.csv"), index=False)
        
        # Print statistics
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
    
    print(f"Landmark data preparation complete. Files saved to {output_dir}")

# Contoh penggunaan
if __name__ == "__main__":
    split_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/split"
    output_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/landmarks"
    prepare_landmark_data_for_lstm(split_dir, output_dir)