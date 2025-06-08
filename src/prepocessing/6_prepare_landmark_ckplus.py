"""
Mempersiapkan data landmark CK+ untuk input LSTM/Dense model
Script ini akan mengubah data landmark dari split menjadi numpy arrays
"""

import pandas as pd
import numpy as np
import json
import os
import re
import ast

def extract_landmarks_from_string(landmark_str):
    """
    Extract landmarks dari string dengan berbagai format
    Mendukung format dari MediaPipe, dlib, dan format lainnya
    """
    if not isinstance(landmark_str, str) or landmark_str.strip() == "":
        return None
    
    try:
        # Method 1: Direct evaluation (for Python list format)
        # Format: [x1, y1, x2, y2, ...]
        try:
            landmarks = ast.literal_eval(landmark_str)
            if isinstance(landmarks, list) and len(landmarks) > 0:
                # Convert all to float
                flattened = [float(x) for x in landmarks]
                return flattened
        except (ValueError, SyntaxError):
            pass
        
        # Method 2: JSON parsing
        # Format: {"landmarks": [{"x": val, "y": val}, ...]}
        try:
            landmarks_data = json.loads(landmark_str)
            
            flattened = []
            if isinstance(landmarks_data, list):
                # Direct list format
                for point in landmarks_data:
                    if isinstance(point, dict):
                        if 'x' in point and 'y' in point:
                            flattened.extend([float(point['x']), float(point['y'])])
                        elif '_x' in point and '_y' in point:
                            flattened.extend([float(point['_x']), float(point['_y'])])
            elif isinstance(landmarks_data, dict):
                # Dictionary format
                if 'landmarks' in landmarks_data:
                    points = landmarks_data['landmarks']
                    for point in points:
                        if isinstance(point, dict):
                            if 'x' in point and 'y' in point:
                                flattened.extend([float(point['x']), float(point['y'])])
                            elif '_x' in point and '_y' in point:
                                flattened.extend([float(point['_x']), float(point['_y'])])
            
            if len(flattened) > 0:
                return flattened
        except json.JSONDecodeError:
            pass
        
        # Method 3: Regex extraction
        # Pattern untuk mencari koordinat x dan y
        patterns = [
            (r'\"x\":([0-9.-]+)', r'\"y\":([0-9.-]+)'),
            (r'\"_x\":([0-9.-]+)', r'\"_y\":([0-9.-]+)'),
            (r'x=([0-9.-]+)', r'y=([0-9.-]+)'),
            (r'([0-9.-]+),\s*([0-9.-]+)')  # Simple x,y format
        ]
        
        for x_pattern, y_pattern in patterns:
            x_matches = re.findall(x_pattern, landmark_str)
            y_matches = re.findall(y_pattern, landmark_str)
            
            if len(x_matches) == len(y_matches) and len(x_matches) > 0:
                flattened = []
                for x, y in zip(x_matches, y_matches):
                    try:
                        flattened.extend([float(x), float(y)])
                    except ValueError:
                        continue
                
                if len(flattened) > 0:
                    return flattened
        
        # Method 4: Simple number extraction
        # Extract all numbers and treat as alternating x,y pairs
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', landmark_str)
        if len(numbers) >= 4 and len(numbers) % 2 == 0:  # At least 2 points
            flattened = [float(x) for x in numbers]
            return flattened
        
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
    
    return None

def normalize_landmark_dimensions(landmarks, target_length=136):
    """
    Normalize landmark dimensions to consistent length
    target_length=136 for dlib (68 points * 2 coordinates)
    target_length=936 for MediaPipe (468 points * 2 coordinates)
    """
    if landmarks is None or len(landmarks) == 0:
        return [0.0] * target_length
    
    if len(landmarks) == target_length:
        return landmarks
    elif len(landmarks) < target_length:
        # Pad with zeros
        padded = landmarks + [0.0] * (target_length - len(landmarks))
        return padded
    else:
        # Truncate to target length
        return landmarks[:target_length]

def prepare_landmark_data_for_lstm_ckplus(split_dir, output_dir, target_dim=136):
    """
    Mempersiapkan data landmark CK+ untuk input LSTM/Dense model
    
    Args:
        split_dir: Direktori yang berisi file split data (train, val, test)
        output_dir: Direktori untuk menyimpan data yang telah diformat
        target_dim: Target dimensi landmark (136 untuk dlib, 936 untuk MediaPipe)
    """
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Define split files
    split_files = {
        'train': os.path.join(split_dir, 'train_data.xlsx'),
        'val': os.path.join(split_dir, 'val_data.xlsx'),
        'test': os.path.join(split_dir, 'test_data.xlsx')
    }
    
    print(f"🎯 Preparing CK+ landmark data (target dimension: {target_dim})")
    
    # Process each split
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        
        print(f"\n📂 Processing {split_name} data...")
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
        invalid_landmarks = 0
        
        print(f"   Processing {len(df)} samples...")
        
        for idx, row in df.iterrows():
            try:
                # Extract landmarks from string
                landmarks_str = row['landmarks']
                emotion = row['Dominant_Emotion']
                
                if pd.isna(landmarks_str) or landmarks_str == "":
                    skipped += 1
                    continue
                
                # Extract landmarks using multiple methods
                flattened_landmarks = extract_landmarks_from_string(landmarks_str)
                
                if flattened_landmarks is None or len(flattened_landmarks) == 0:
                    invalid_landmarks += 1
                    skipped += 1
                    continue
                
                # Normalize dimensions
                normalized_landmarks = normalize_landmark_dimensions(flattened_landmarks, target_dim)
                
                # Validate that we have valid coordinates
                if all(x == 0.0 for x in normalized_landmarks):
                    invalid_landmarks += 1
                    skipped += 1
                    continue
                
                X_landmarks.append(normalized_landmarks)
                y_emotions.append(emotion)
                
                # Store metadata
                metadata.append({
                    'original_index': idx,
                    'filename': row.get('filename', ''),
                    'user_id': row.get('user_id', ''),
                    'emotion': emotion,
                    'landmark_count': len(flattened_landmarks) // 2,
                    'original_landmark_dim': len(flattened_landmarks)
                })
                
                successful += 1
                
            except Exception as e:
                print(f"   Error processing row {idx}: {e}")
                skipped += 1
                continue
        
        if successful == 0:
            print(f"   ❌ No valid landmark data found for {split_name}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_landmarks, dtype=np.float32)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_landmarks.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
        
        # Save metadata as CSV
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(output_dir, f"{split_name}_metadata.csv"), index=False)
        
        # Print statistics
        print(f"   ✅ Successfully processed: {successful} samples")
        print(f"   ⚠️ Skipped: {skipped} samples")
        print(f"   ❌ Invalid landmarks: {invalid_landmarks} samples")
        print(f"   📊 Shape of X_{split_name}_landmarks: {X.shape}")
        print(f"   📊 Shape of y_{split_name}: {y.shape}")
        print(f"   🎯 Feature dimensions: {target_dim}")
        
        # Print emotion distribution
        emotions, counts = np.unique(y, return_counts=True)
        print(f"   🎭 Emotion distribution:")
        for emotion, count in zip(emotions, counts):
            percent = count / len(y) * 100
            print(f"      {emotion}: {count} ({percent:.2f}%)")
            
        # Print landmark statistics
        print(f"   📈 Landmark statistics:")
        print(f"      Min value: {X.min():.4f}")
        print(f"      Max value: {X.max():.4f}")
        print(f"      Mean: {X.mean():.4f}")
        print(f"      Std: {X.std():.4f}")
    
    print(f"\n✅ Landmark data preparation complete!")
    print(f"📁 Files saved to: {output_dir}")

def validate_landmark_data(data_dir):
    """
    Validasi data landmark yang sudah diproses
    """
    print("\n=== VALIDATING LANDMARK DATA ===")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        try:
            X_path = os.path.join(data_dir, f"X_{split}_landmarks.npy")
            y_path = os.path.join(data_dir, f"y_{split}.npy")
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                X = np.load(X_path)
                y = np.load(y_path)
                
                print(f"\n{split.upper()} SET:")
                print(f"  - Landmarks shape: {X.shape}")
                print(f"  - Labels shape: {y.shape}")
                print(f"  - Landmark dtype: {X.dtype}")
                print(f"  - Feature range: [{X.min():.3f}, {X.max():.3f}]")
                print(f"  - Memory usage: {X.nbytes / (1024**2):.2f} MB")
                
                # Check for issues
                if np.isnan(X).any():
                    print(f"  ⚠️ WARNING: NaN values found")
                if np.isinf(X).any():
                    print(f"  ⚠️ WARNING: Infinite values found")
                if (X == 0).all():
                    print(f"  ⚠️ WARNING: All values are zero")
                    
                # Check label distribution
                unique_labels, counts = np.unique(y, return_counts=True)
                print(f"  - Unique emotions: {list(unique_labels)}")
                
            else:
                print(f"\n{split.upper()} SET: Files not found")
                
        except Exception as e:
            print(f"Error validating {split} data: {e}")
    
    print("\n=== VALIDATION COMPLETE ===")

def detect_landmark_format(sample_landmark_str):
    """
    Deteksi format landmark dari sample string
    """
    print("🔍 DETECTING LANDMARK FORMAT...")
    
    landmarks = extract_landmarks_from_string(sample_landmark_str)
    
    if landmarks:
        num_points = len(landmarks) // 2
        print(f"   Detected {num_points} landmark points")
        print(f"   Total features: {len(landmarks)}")
        
        if num_points == 68:
            print("   Format: Likely dlib (68 points)")
            return 136
        elif num_points == 468:
            print("   Format: Likely MediaPipe (468 points)")
            return 936
        else:
            print(f"   Format: Custom ({num_points} points)")
            return len(landmarks)
    else:
        print("   ❌ Could not detect format")
        return 136  # Default to dlib format

# Contoh penggunaan
if __name__ == "__main__":
    # Paths untuk CK+ dataset
    split_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/split"
    output_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/landmarks"
    
    # Auto-detect landmark format dari sample data
    try:
        sample_file = os.path.join(split_dir, 'train_data.xlsx')
        if os.path.exists(sample_file):
            df = pd.read_excel(sample_file)
            if 'landmarks' in df.columns and len(df) > 0:
                sample_landmark = df['landmarks'].iloc[0]
                target_dim = detect_landmark_format(sample_landmark)
            else:
                target_dim = 136  # Default dlib
        else:
            target_dim = 136  # Default dlib
    except:
        target_dim = 136  # Default dlib
    
    print(f"\n🚀 Using target dimension: {target_dim}")
    
    # Prepare data
    prepare_landmark_data_for_lstm_ckplus(split_dir, output_dir, target_dim)
    
    # Validate data
    validate_landmark_data(output_dir)
    
    print("\n✅ CK+ landmark data preparation complete!")
    print(f"📁 Files saved to: {output_dir}")
    print(f"🚀 Ready for landmark model training!")