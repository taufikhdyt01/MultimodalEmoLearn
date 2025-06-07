def main_ckplus_kaggle_preprocessing():
    """
    Main script untuk preprocessing dataset CK+ dari Kaggle
    """
    
    # Configuration - SESUAIKAN PATH INI DENGAN LOKASI DATASET ANDA
    CKPLUS_KAGGLE_PATH = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ckplus"
    
    if not os.path.exists(CKPLUS_KAGGLE_PATH):
        print(f"Error: CK+ dataset path not found: {CKPLUS_KAGGLE_PATH}")
        return
    
    # Output directories
    OUTPUT_BASE_DIR = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ckplus_kaggle"
    DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    
    split_dir = os.path.join(OUTPUT_BASE_DIR, "split")
    images_dir = os.path.join(OUTPUT_BASE_DIR, "images")
    landmarks_dir = os.path.join(OUTPUT_BASE_DIR, "landmarks")
    
    print("=" * 80)
    print("🎭 CK+ KAGGLE DATASET PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Input path: {CKPLUS_KAGGLE_PATH}")
    print(f"Output path: {OUTPUT_BASE_DIR}")
    
    # Step 1: Split dataset
    print("\n1️⃣ Splitting CK+ Kaggle dataset...")
    train_df, val_df, test_df = split_ckplus_kaggle_dataset(
        CKPLUS_KAGGLE_PATH, 
        split_dir,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )
    
    if train_df is None:
        print("Failed to split dataset. Exiting.")
        return
    
    # Step 2: Prepare images for CNN
    print("\n2️⃣ Preparing images for CNN...")
    prepare_ckplus_kaggle_images_for_cnn(split_dir, images_dir, img_size=(48, 48))
    
    # Step 3: Extract landmarks
    print("\n3️⃣ Extracting facial landmarks...")
    if os.path.exists(DLIB_PREDICTOR_PATH):
        prepare_ckplus_kaggle_landmarks(split_dir, landmarks_dir, DLIB_PREDICTOR_PATH)
    else:
        print(f"⚠️  Dlib predictor not found at {DLIB_PREDICTOR_PATH}")
        print("Please download it manually from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("For now, skipping landmark extraction...")
    
    print("\n" + "=" * 80)
    print("🎉 CK+ KAGGLE PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"📁 Data saved to: {OUTPUT_BASE_DIR}")
    print("\n📋 Next steps:")
    print("1. Update paths in training scripts:")
    print(f"   BASE_PATH = '{OUTPUT_BASE_DIR}/'")
    print(f"   MODEL_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn/models/ckplus_kaggle/'")
    print("2. Train CNN: python optimized_train_cnn.py")
    print("3. Train Landmark: python optimized_train_landmark.py") 
    print("4. Train Late Fusion: python optimized_train_late_fusion.py")

if __name__ == "__main__":
    main_ckplus_kaggle_preprocessing()