import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def explore_ck_dataset(dataset_path):
    """
    Eksplorasi struktur dataset CK+ yang sudah di-download dari Kaggle
    """
    print("=== EKSPLORASI DATASET CK+ ===")
    
    # 1. Cek struktur direktori
    print("\n1. STRUKTUR DIREKTORI:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Tampilkan beberapa file sample
        if files and level < 3:  # Batasi depth untuk tidak terlalu verbose
            sub_indent = ' ' * 2 * (level + 1)
            for i, file in enumerate(files[:5]):  # Tampilkan max 5 file
                print(f"{sub_indent}{file}")
            if len(files) > 5:
                print(f"{sub_indent}... dan {len(files) - 5} file lainnya")
    
    # 2. Analisis file gambar
    print("\n2. ANALISIS GAMBAR:")
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    total_images = 0
    image_info = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_images += 1
                file_path = os.path.join(root, file)
                
                # Ambil info dari path
                path_parts = file_path.replace(dataset_path, '').split(os.sep)
                
                # Analisis pattern nama file dan folder
                image_info.append({
                    'path': file_path,
                    'filename': file,
                    'relative_path': os.path.join(*path_parts[1:]) if len(path_parts) > 1 else file,
                    'folder_structure': path_parts[1:-1] if len(path_parts) > 1 else []
                })
    
    print(f"Total gambar ditemukan: {total_images}")
    
    # 3. Analisis pattern penamaan
    print("\n3. PATTERN PENAMAAN:")
    if image_info:
        sample_paths = image_info[:10]
        for img in sample_paths:
            print(f"  {img['relative_path']}")
    
    # 4. Cek ukuran gambar
    print("\n4. ANALISIS UKURAN GAMBAR:")
    if image_info:
        sample_images = image_info[:20]  # Cek 20 gambar pertama
        sizes = []
        
        for img in sample_images:
            try:
                image = cv2.imread(img['path'])
                if image is not None:
                    h, w = image.shape[:2]
                    sizes.append((w, h))
            except:
                continue
        
        if sizes:
            unique_sizes = list(set(sizes))
            print(f"Ukuran gambar yang ditemukan:")
            for size in unique_sizes[:10]:  # Tampilkan max 10 ukuran berbeda
                count = sizes.count(size)
                print(f"  {size[0]}x{size[1]}: {count} gambar")
    
    # 5. Deteksi label emosi
    print("\n5. DETEKSI LABEL EMOSI:")
    
    # Cari file label atau CSV
    label_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.csv', '.txt', '.xlsx')):
                label_files.append(os.path.join(root, file))
    
    if label_files:
        print("File label yang ditemukan:")
        for lf in label_files:
            print(f"  {lf}")
            
        # Coba baca file CSV pertama
        for lf in label_files:
            if lf.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(lf)
                    print(f"\nIsi file {os.path.basename(lf)}:")
                    print(f"Shape: {df.shape}")
                    print(f"Kolom: {list(df.columns)}")
                    print(f"Sample data:")
                    print(df.head())
                    break
                except Exception as e:
                    print(f"Error membaca {lf}: {e}")
    else:
        # Coba deteksi dari struktur folder
        print("Tidak ada file label eksplisit ditemukan.")
        print("Mencoba deteksi label dari struktur folder...")
        
        # Analisis nama folder yang mungkin mengandung label emosi
        emotion_keywords = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']
        folder_emotions = set()
        
        for img in image_info:
            for folder in img['folder_structure']:
                folder_lower = folder.lower()
                if any(emotion in folder_lower for emotion in emotion_keywords):
                    folder_emotions.add(folder)
        
        if folder_emotions:
            print(f"Folder yang mungkin berisi label emosi: {list(folder_emotions)}")
    
    return image_info, label_files

def analyze_ck_labels(dataset_path):
    """
    Analisis khusus untuk label CK+ berdasarkan struktur standar
    """
    print("\n=== ANALISIS LABEL CK+ ===")
    
    # CK+ biasanya memiliki struktur:
    # - cohn-kanade-images/ untuk gambar
    # - Emotion/ untuk label emosi
    # - FACS/ untuk action units
    
    emotion_path = None
    facs_path = None
    images_path = None
    
    # Cari folder yang mengandung label
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root).lower()
        
        if 'emotion' in folder_name:
            emotion_path = root
        elif 'facs' in folder_name:
            facs_path = root
        elif any(x in folder_name for x in ['image', 'cohn', 'kanade']):
            images_path = root
    
    print(f"Emotion labels path: {emotion_path}")
    print(f"FACS labels path: {facs_path}")
    print(f"Images path: {images_path}")
    
    # Analisis file emotion
    if emotion_path:
        emotion_files = []
        for root, dirs, files in os.walk(emotion_path):
            for file in files:
                if file.endswith('.txt'):
                    emotion_files.append(os.path.join(root, file))
        
        print(f"\nTotal file emotion label: {len(emotion_files)}")
        
        if emotion_files:
            # Baca beberapa file sample
            emotion_labels = {}
            for i, ef in enumerate(emotion_files[:10]):
                try:
                    with open(ef, 'r') as f:
                        content = f.read().strip()
                        if content.isdigit():
                            emotion_code = int(content)
                            emotion_labels[ef] = emotion_code
                            
                except Exception as e:
                    print(f"Error reading {ef}: {e}")
            
            if emotion_labels:
                print("\nSample emotion labels:")
                for file, label in list(emotion_labels.items())[:5]:
                    print(f"  {os.path.basename(file)}: {label}")
                
                # Mapping emotion codes (standar CK+)
                ck_emotion_map = {
                    0: 'neutral',
                    1: 'anger', 
                    2: 'contempt',
                    3: 'disgust',
                    4: 'fear',
                    5: 'happy',
                    6: 'sadness',
                    7: 'surprise'
                }
                
                print(f"\nMapping emosi CK+:")
                for code, emotion in ck_emotion_map.items():
                    print(f"  {code}: {emotion}")

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path dataset CK+ Anda
    dataset_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ckplus"
    
    # Jalankan eksplorasi
    image_info, label_files = explore_ck_dataset(dataset_path)
    analyze_ck_labels(dataset_path)
