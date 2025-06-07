"""
Pipeline lengkap untuk mempersiapkan dataset CK+ menggunakan dlib
Alternative ke MediaPipe dengan dlib untuk landmark extraction
"""
import sys
import os

# Tambahkan path proyek ke sys.path
sys.path.append('D:/research/2025_iris_taufik/MultimodalEmoLearn')

def run_dlib_ckplus_pipeline():
    """Jalankan pipeline lengkap untuk CK+ dengan dlib"""
    
    print("=== STARTING CK+ PIPELINE WITH DLIB ===")
    
    # Paths
    base_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data"
    ckplus_raw = os.path.join(base_path, "ckplus")
    ckplus_processed = os.path.join(base_path, "ckplus_processed")
    landmarks_dlib = os.path.join(base_path, "Emotion_Labels", "ckplus_landmarks_dlib.xlsx")
    labeled_data = os.path.join(base_path, "Emotion_Labels", "ckplus_labeled_data_dlib.xlsx")
    split_dir = os.path.join(base_path, "ckplus_split_dlib")
    landmarks_dir = os.path.join(base_path, "ckplus_landmarks_dlib")
    
    # Step 1: Preprocess images (sama seperti sebelumnya)
    print("\n1. Preprocessing images...")
    from src.prepocessing.preprocess_ckplus import preprocess_ckplus_images
    preprocess_ckplus_images(ckplus_raw, ckplus_processed)
    
    # Step 2: Extract landmarks dengan dlib
    print("\n2. Extracting landmarks with dlib...")
    from src.prepocessing.extract_landmarks_ckplus_dlib import extract_landmarks_ckplus_dlib
    extract_landmarks_ckplus_dlib(ckplus_processed, landmarks_dlib)
    
    # Step 3: Prepare data format
    print("\n3. Preparing data format...")
    prepare_ckplus_for_training_dlib(landmarks_dlib, labeled_data)
    
    # Step 4: Split dataset
    print("\n4. Splitting dataset...")
    from split_dataset import split_dataset
    split_dataset(labeled_data, split_dir, train_size=0.8, val_size=0.1, test_size=0.1)
    
    # Step 5: Prepare landmark data for LSTM
    print("\n5. Preparing landmark data for LSTM...")
    from prepare_landmark import prepare_landmark_data_for_lstm
    prepare_landmark_data_for_lstm(split_dir, landmarks_dir)
    
    print("\n=== CK+ DLIB PIPELINE COMPLETE ===")
    print(f"Data ready for training!")
    print(f"Landmark data location: {landmarks_dir}")
    print(f"Feature dimensions: 136 (68 landmark points × 2)")

def prepare_ckplus_for_training_dlib(dlib_landmarks_file, output_path):
    """
    Mempersiapkan data CK+ dengan landmarks dlib untuk training
    """
    import pandas as pd
    
    print("📋 Loading dlib landmark data...")
    df = pd.read_excel(dlib_landmarks_file)
    
    # Mapping emosi ke format yang konsisten dengan model
    emotion_mapping = {
        'anger': 'angry',
        'contempt': 'contempt',  # Bisa disesuaikan
        'disgust': 'disgusted', 
        'fear': 'fear',
        'happy': 'happy',
        'sadness': 'sad',
        'surprise': 'surprised'
    }
    
    # Update kolom emosi
    df['Dominant_Emotion'] = df['emotion'].map(emotion_mapping)
    
    # Filter data yang valid
    df = df.dropna(subset=['landmarks', 'Dominant_Emotion'])
    
    # Tambahkan kolom yang diperlukan
    df['id'] = range(1, len(df) + 1)
    
    # Validasi landmark dimensions
    def validate_landmarks(landmark_str):
        try:
            landmarks = eval(landmark_str)
            return len(landmarks) == 136  # 68 points × 2 coordinates
        except:
            return False
    
    # Filter hanya landmarks yang valid
    valid_mask = df['landmarks'].apply(validate_landmarks)
    df = df[valid_mask]
    
    # Simpan sebagai labeled_data format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    
    print(f"✅ Data preparation complete. Saved to {output_path}")
    print(f"📊 Total samples: {len(df)}")
    print(f"🎭 Emotion distribution:")
    print(df['Dominant_Emotion'].value_counts())
    print(f"🎯 Feature dimensions: 136 (dlib 68 landmarks)")
    
    return df

def compare_mediapipe_vs_dlib():
    """
    Perbandingan MediaPipe vs dlib untuk referensi
    """
    print("\n" + "="*50)
    print("📊 MEDIAPIPE vs DLIB COMPARISON")
    print("="*50)
    
    comparison = {
        "Aspect": ["Landmark Points", "Feature Dimensions", "Installation", "Speed", "Accuracy", "GPU Support"],
        "MediaPipe": ["468 points", "936 features", "Easy (pip)", "Very Fast", "High", "Yes"],
        "dlib": ["68 points", "136 features", "Moderate", "Fast", "Very High", "No"]
    }
    
    import pandas as pd
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))
    
    print("\n💡 Recommendations:")
    print("   - Use MediaPipe for: Real-time applications, GPU acceleration")
    print("   - Use dlib for: Research reproducibility, precise landmarks")
    print("   - Both are good for emotion recognition research")

if __name__ == "__main__":
    # Tampilkan perbandingan
    compare_mediapipe_vs_dlib()
    
    # Tanya user mau pakai yang mana
    print("\n🤔 Which library would you like to use?")
    print("   1. MediaPipe (468 landmarks, faster)")
    print("   2. dlib (68 landmarks, more stable)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\n🚀 Running dlib pipeline...")
        run_dlib_ckplus_pipeline()
    else:
        print("\n🚀 Running MediaPipe pipeline...")
        print("   Please use: python src/prepocessing/run_ckplus_pipeline.py")
        print("   (Make sure MediaPipe is installed: pip install mediapipe)")