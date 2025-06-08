"""
Ekstraksi landmark facial menggunakan dlib untuk dataset CK+
Alternative implementation dengan dlib sebagai pengganti MediaPipe
"""

import dlib
import cv2
import pandas as pd
import numpy as np
import os
import urllib.request
from pathlib import Path

def download_dlib_model(model_path="shape_predictor_68_face_landmarks.dat"):
    """
    Download model dlib jika belum ada
    """
    if not os.path.exists(model_path):
        print("📥 Downloading dlib face landmark model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        # Download compressed file
        compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, compressed_file)
        
        # Extract bz2 file
        import bz2
        with bz2.BZ2File(compressed_file, 'rb') as fr, open(model_path, 'wb') as fw:
            fw.write(fr.read())
        
        # Remove compressed file
        os.remove(compressed_file)
        print(f"✅ Model downloaded: {model_path}")
    
    return model_path

def extract_landmarks_with_dlib(image_path, detector, predictor):
    """
    Ekstraksi landmark dari satu gambar menggunakan dlib
    
    Returns:
        list: Flattened landmarks [x1, y1, x2, y2, ...] atau None jika gagal
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Use the first face detected
        face = faces[0]
        
        # Get landmarks
        landmarks = predictor(gray, face)
        
        # Convert to coordinate list
        coords = []
        for i in range(68):  # dlib returns 68 landmark points
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            coords.extend([x, y])
        
        return coords
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_landmarks_ckplus_dlib(dataset_path, output_path, model_path=None):
    """
    Ekstraksi landmark facial dari dataset CK+ menggunakan dlib
    
    Args:
        dataset_path: Path ke dataset CK+ (yang sudah dipreprocess)
        output_path: Path untuk menyimpan hasil ekstraksi
        model_path: Path ke model dlib (akan didownload otomatis jika None)
    """
    
    # Download atau load model dlib
    if model_path is None:
        model_path = download_dlib_model()
    
    # Initialize dlib detector and predictor
    print("🔧 Initializing dlib models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    
    results_data = []
    emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprised']
    
    print("🎯 Starting landmark extraction...")
    
    total_processed = 0
    total_failed = 0
    
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"⚠️ Emotion folder not found: {emotion}")
            continue
        
        print(f"📂 Processing {emotion}...")
        emotion_processed = 0
        emotion_failed = 0
        
        image_files = [f for f in os.listdir(emotion_path) if f.endswith('.png')]
        
        for img_file in image_files:
            img_path = os.path.join(emotion_path, img_file)
            
            # Extract landmarks
            landmarks = extract_landmarks_with_dlib(img_path, detector, predictor)
            
            if landmarks is not None:
                # Extract metadata from filename
                # Format: S010_004_00000017.png -> Subject: S010, Challenge: 004
                parts = img_file.replace('.png', '').split('_')
                subject_id = parts[0] if len(parts) > 0 else 'unknown'
                challenge_id = parts[1] if len(parts) > 1 else 'unknown'
                
                results_data.append({
                    'filename': img_file,
                    'emotion': emotion,
                    'landmarks': str(landmarks),  # Save as string for Excel compatibility
                    'image_path': img_path,
                    'user_id': subject_id,
                    'challenge_id': challenge_id,
                    'landmark_count': len(landmarks) // 2  # Number of landmark points
                })
                
                emotion_processed += 1
                total_processed += 1
            else:
                emotion_failed += 1
                total_failed += 1
        
        print(f"   ✅ {emotion}: {emotion_processed} processed, {emotion_failed} failed")
    
    # Save results
    print(f"\n💾 Saving results to {output_path}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results_data)
    df.to_excel(output_path, index=False)
    
    # Print summary
    print(f"\n📊 Extraction Summary:")
    print(f"   Total processed: {total_processed}")
    print(f"   Total failed: {total_failed}")
    print(f"   Success rate: {total_processed/(total_processed+total_failed)*100:.2f}%")
    print(f"   Landmark points per face: 68")
    print(f"   Feature dimensions: {68*2} (136 features)")
    
    # Print emotion distribution
    if len(results_data) > 0:
        emotion_counts = df['emotion'].value_counts()
        print(f"\n🎭 Emotion Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count}")
    
    print(f"\n✅ Landmark extraction complete!")
    print(f"📁 Results saved to: {output_path}")
    
    return df

def validate_dlib_installation():
    """
    Validasi instalasi dlib
    """
    try:
        import dlib
        print(f"✅ dlib version: {dlib.__version__}")
        
        # Test detector
        detector = dlib.get_frontal_face_detector()
        print("✅ Face detector initialized")
        
        return True
    except ImportError:
        print("❌ dlib not installed")
        print("📦 Install with: pip install dlib")
        print("   or: conda install -c conda-forge dlib")
        return False
    except Exception as e:
        print(f"❌ dlib error: {e}")
        return False

if __name__ == "__main__":
    # Validate installation first
    if not validate_dlib_installation():
        print("\n🔧 Please install dlib first:")
        print("   pip install dlib")
        print("   or: conda install -c conda-forge dlib")
        exit(1)
    
    # Paths
    dataset_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ck+_processed"
    output_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/Emotion_Labels/ck+_landmarks_dlib.xlsx"
    
    # Extract landmarks
    df = extract_landmarks_ckplus_dlib(dataset_path, output_path)
    
    print("\n🚀 Ready for next step: prepare_ckplus_data.py")