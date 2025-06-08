"""
Script untuk menjalankan preprocessing dataset JAFFE
Pastikan sudah mengatur path dengan benar sebelum menjalankan
"""

import os
import sys
from pathlib import Path

# Tambahkan path ke script preprocessing
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

from jaffe_preprocessing import JAFFEPreprocessor

def check_requirements():
    """Check apakah semua requirements sudah terinstall"""
    required_packages = [
        'opencv-python',
        'dlib', 
        'numpy',
        'scikit-learn',
        'pandas',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        
        if 'dlib' in missing_packages:
            print("\n⚠️ Special note for dlib installation:")
            print("   Windows: pip install dlib")
            print("   atau: conda install -c conda-forge dlib")
            print("   Linux: sudo apt-get install build-essential cmake")
            print("          pip install dlib")
        return False
    
    return True

def verify_jaffe_structure(jaffe_path):
    """Verify struktur dataset JAFFE"""
    jaffe_path = Path(jaffe_path)
    
    if not jaffe_path.exists():
        print(f"❌ JAFFE path tidak ditemukan: {jaffe_path}")
        return False
    
    # Check for .tiff or .tif files
    tiff_files = list(jaffe_path.glob("*.tiff"))
    tif_files = list(jaffe_path.glob("*.tif"))
    
    total_files = len(tiff_files) + len(tif_files)
    
    if total_files == 0:
        print(f"❌ Tidak ditemukan file .tiff atau .tif di {jaffe_path}")
        print("📁 Isi folder:")
        for item in jaffe_path.iterdir():
            print(f"   - {item.name}")
        return False
    
    print(f"✅ Ditemukan {total_files} file gambar JAFFE")
    
    # Check beberapa nama file untuk validasi format
    sample_files = (tiff_files + tif_files)[:5]
    print("📄 Sample filenames:")
    for f in sample_files:
        print(f"   - {f.name}")
    
    return True

def main():
    """Main function untuk menjalankan preprocessing"""
    
    print("🚀 Starting JAFFE Dataset Preprocessing")
    print("=" * 50)
    
    # Check requirements
    print("1️⃣ Checking requirements...")
    if not check_requirements():
        return False
    
    # Configure paths - SESUAIKAN DENGAN LOKASI ANDA
    jaffe_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/jaffe"  # ⚠️ UBAH PATH INI
    output_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data"  # ⚠️ UBAH PATH INI
    
    # Verify paths
    print("2️⃣ Verifying dataset structure...")
    if not verify_jaffe_structure(jaffe_path):
        print("\n💡 Petunjuk:")
        print("   - Pastikan path JAFFE benar")
        print("   - Ekstrak dataset JAFFE jika masih dalam format zip")
        print("   - File gambar harus berformat .tiff atau .tif")
        print("   - Contoh nama file: KM-HA1.tiff, YM-SA2.tiff")
        return False
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("3️⃣ Starting preprocessing...")
    print(f"📁 Input: {jaffe_path}")
    print(f"💾 Output: {output_path}")
    
    try:
        # Initialize preprocessor
        preprocessor = JAFFEPreprocessor(jaffe_path, output_path)
        
        # Process images and extract features
        print("\n🔄 Extracting images and landmarks...")
        images, landmarks, labels, metadata = preprocessor.preprocess_images()
        
        if len(images) == 0:
            print("❌ Gagal memproses gambar!")
            return False
        
        # Create label mapping
        print("🏷️ Creating label mapping...")
        label_map = preprocessor.create_label_mapping(labels)
        
        # Split data
        print("📊 Splitting data...")
        data_splits = preprocessor.split_data(images, landmarks, labels)
        
        # Save preprocessed data
        print("💾 Saving preprocessed data...")
        preprocessor.save_preprocessed_data(data_splits, label_map)
        
        # Success summary
        print("\n" + "=" * 50)
        print("🎉 PREPROCESSING BERHASIL!")
        print("=" * 50)
        
        print(f"📊 Total samples: {len(images)}")
        print(f"📈 Distribution per emotion:")
        for emotion, count in zip(*np.unique(labels, return_counts=True)):
            print(f"   {emotion}: {count} samples")
        
        print(f"\n📁 Data tersimpan di: {output_path}")
        print("🚀 Siap untuk training model!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np
    success = main()
    
    if success:
        print("\n✅ Preprocessing selesai. Lanjutkan dengan:")
        print("   1. Train CNN model: python optimized_train_cnn.py")
        print("   2. Train Landmark model: python optimized_train_landmark.py") 
        print("   3. Train Late Fusion: python optimized_train_late_fusion.py")
    else:
        print("\n❌ Preprocessing gagal. Periksa error di atas.")