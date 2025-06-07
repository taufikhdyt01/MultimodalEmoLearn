import pandas as pd
import os

def prepare_ckplus_for_training(ckplus_landmarks_file, output_path):
    """
    Mempersiapkan data CK+ untuk training dengan format yang sesuai
    """
    df = pd.read_excel(ckplus_landmarks_file)
    
    # Mapping emosi ke format yang konsisten dengan model Anda
    emotion_mapping = {
        'anger': 'angry',
        'contempt': 'contempt',  # Tambahkan kategori baru atau map ke existing
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
    
    # Simpan sebagai labeled_data format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    
    print(f"Data preparation complete. Saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Emotion distribution:")
    print(df['Dominant_Emotion'].value_counts())
    
    return df

if __name__ == "__main__":
    landmarks_file = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/Emotion_Labels/ckplus_landmarks_dlib.xlsx"
    output_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/Emotion_Labels/ckplus_labeled_data.xlsx"
    
    prepare_ckplus_for_training(landmarks_file, output_path)