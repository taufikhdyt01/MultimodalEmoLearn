"""
Mempersiapkan data gambar CK+ untuk input CNN
Script ini akan mengubah data split menjadi numpy arrays untuk training CNN
"""

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Load dan preprocess satu gambar
    
    Args:
        image_path: Path ke file gambar
        img_size: Ukuran target (width, height)
    
    Returns:
        numpy array yang sudah dinormalisasi
    """
    try:
        # Load gambar
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize ke ukuran target
        image = cv2.resize(image, img_size)
        
        # Convert BGR to RGB (OpenCV default adalah BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalisasi pixel values ke range [0, 1]
        image = image.astype('float32') / 255.0
        
        return image
    
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def prepare_image_data_for_cnn_ckplus(split_dir, output_dir, img_size=(224, 224)):
    """
    Mempersiapkan data gambar CK+ untuk input CNN
    
    Args:
        split_dir: Direktori yang berisi file split data (train, val, test)
        output_dir: Direktori untuk menyimpan data yang telah diformat untuk CNN
        img_size: Ukuran gambar setelah resize (default: 224x224)
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
        if 'image_path' not in df.columns:
            print(f"Error: 'image_path' column not found in {split_file}. Skipping.")
            continue
            
        if 'Dominant_Emotion' not in df.columns:
            print(f"Error: 'Dominant_Emotion' column not found in {split_file}. Skipping.")
            continue
        
        # Create lists to store processed data
        X_images = []
        y_emotions = []
        metadata = []
        
        # Process each row
        skipped = 0
        successful = 0
        
        for idx, row in df.iterrows():
            try:
                # Get image path
                img_path = row['image_path']
                emotion = row['Dominant_Emotion']
                
                # Check if image file exists
                if not os.path.exists(img_path):
                    skipped += 1
                    print(f"Image not found: {img_path}")
                    continue
                
                # Load and preprocess image
                processed_image = load_and_preprocess_image(img_path, img_size)
                
                if processed_image is not None:
                    X_images.append(processed_image)
                    y_emotions.append(emotion)
                    
                    # Store metadata
                    metadata.append({
                        'original_index': idx,
                        'filename': row.get('filename', ''),
                        'user_id': row.get('user_id', ''),
                        'emotion': emotion,
                        'image_path': img_path
                    })
                    
                    successful += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                skipped += 1
                continue
        
        if successful == 0:
            print(f"No valid data found for {split_name}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_images)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_images.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}_images.npy"), y)
        
        # Save metadata as CSV
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(output_dir, f"{split_name}_metadata.csv"), index=False)
        
        # Print statistics
        print(f"  - Successfully processed: {successful} samples")
        print(f"  - Skipped: {skipped} samples")
        print(f"  - Shape of X_{split_name}_images: {X.shape}")
        print(f"  - Shape of y_{split_name}_images: {y.shape}")
        print(f"  - Image dimensions: {img_size}")
        
        # Print emotion distribution
        emotions, counts = np.unique(y, return_counts=True)
        print(f"  - Emotion distribution:")
        for emotion, count in zip(emotions, counts):
            percent = count / len(y) * 100
            print(f"    {emotion}: {count} ({percent:.2f}%)")
    
    print(f"Image data preparation complete. Files saved to {output_dir}")

def visualize_sample_images(data_dir, num_samples=5):
    """
    Visualisasi beberapa sampel gambar untuk validasi
    
    Args:
        data_dir: Direktori yang berisi data numpy
        num_samples: Jumlah sampel untuk ditampilkan
    """
    try:
        # Load training data
        X_train = np.load(os.path.join(data_dir, "X_train_images.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train_images.npy"))
        
        # Get unique emotions
        unique_emotions = np.unique(y_train)
        
        # Create subplot
        fig, axes = plt.subplots(len(unique_emotions), num_samples, 
                                figsize=(num_samples * 3, len(unique_emotions) * 3))
        
        if len(unique_emotions) == 1:
            axes = axes.reshape(1, -1)
        
        for i, emotion in enumerate(unique_emotions):
            # Get indices for this emotion
            emotion_indices = np.where(y_train == emotion)[0]
            
            # Select random samples
            if len(emotion_indices) >= num_samples:
                selected_indices = np.random.choice(emotion_indices, num_samples, replace=False)
            else:
                selected_indices = emotion_indices
                
            for j in range(num_samples):
                if j < len(selected_indices):
                    img_idx = selected_indices[j]
                    img = X_train[img_idx]
                    
                    axes[i, j].imshow(img)
                    axes[i, j].set_title(f"{emotion}\nSample {j+1}")
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "sample_images_visualization.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {data_dir}/sample_images_visualization.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def validate_image_data(data_dir):
    """
    Validasi data gambar yang sudah diproses
    """
    print("=== VALIDATING IMAGE DATA ===")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        try:
            X_path = os.path.join(data_dir, f"X_{split}_images.npy")
            y_path = os.path.join(data_dir, f"y_{split}_images.npy")
            
            if os.path.exists(X_path) and os.path.exists(y_path):
                X = np.load(X_path)
                y = np.load(y_path)
                
                print(f"{split.upper()} SET:")
                print(f"  - Images shape: {X.shape}")
                print(f"  - Labels shape: {y.shape}")
                print(f"  - Image dtype: {X.dtype}")
                print(f"  - Pixel value range: [{X.min():.3f}, {X.max():.3f}]")
                print(f"  - Memory usage: {X.nbytes / (1024**2):.2f} MB")
                
                # Check for any NaN or invalid values
                if np.isnan(X).any():
                    print(f"  ⚠️ WARNING: NaN values found in {split} images")
                if np.isinf(X).any():
                    print(f"  ⚠️ WARNING: Infinite values found in {split} images")
                    
                print()
            else:
                print(f"{split.upper()} SET: Files not found")
                
        except Exception as e:
            print(f"Error validating {split} data: {e}")
    
    print("=== VALIDATION COMPLETE ===")

# Contoh penggunaan
if __name__ == "__main__":
    # Paths untuk CK+ dataset
    split_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/split"
    output_dir = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/images"
    
    print("🖼️ Preparing CK+ image data for CNN...")
    
    # Prepare data
    prepare_image_data_for_cnn_ckplus(split_dir, output_dir, img_size=(224, 224))
    
    # Validate data
    validate_image_data(output_dir)
    
    # Create visualization
    print("\n📊 Creating sample visualization...")
    visualize_sample_images(output_dir, num_samples=3)
    
    print("\n✅ Image data preparation complete!")
    print(f"📁 Files saved to: {output_dir}")
    print("\n🚀 Ready for CNN training!")