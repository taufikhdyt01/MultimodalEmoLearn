import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import dlib
import pickle
from pathlib import Path
import requests
import zipfile

class JAFFEPreprocessor:
    def __init__(self, jaffe_path, output_path):
        """
        JAFFE Dataset Preprocessor untuk Multimodal Emotion Recognition dengan dlib
        
        Args:
            jaffe_path: Path ke folder dataset JAFFE (berisi file .tiff)
            output_path: Path untuk menyimpan hasil preprocessing
        """
        self.jaffe_path = Path(jaffe_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup dlib untuk landmark detection
        self.detector = None
        self.predictor = None
        self._setup_dlib()
        
        # Mapping pose codes ke emotion labels
        self.emotion_mapping = {
            'NE': 'neutral',
            'HA': 'happy', 
            'SA': 'sad',
            'SU': 'surprise',
            'AN': 'angry',
            'DI': 'disgust',
            'FE': 'fear'
        }
        
        print("✅ JAFFE Preprocessor with dlib initialized")
        print(f"📁 Input path: {self.jaffe_path}")
        print(f"💾 Output path: {self.output_path}")
    
    def _setup_dlib(self):
        """Setup dlib face detector dan predictor"""
        try:
            # Initialize face detector
            self.detector = dlib.get_frontal_face_detector()
            
            # Download predictor model jika belum ada
            predictor_path = self._download_predictor_model()
            
            # Initialize landmark predictor
            self.predictor = dlib.shape_predictor(predictor_path)
            
            print("✅ dlib face detector dan predictor berhasil diinisialisasi")
            
        except Exception as e:
            print(f"❌ Error setting up dlib: {e}")
            raise
    
    def _download_predictor_model(self):
        """Download dlib facial landmark predictor model jika belum ada"""
        
        # Path untuk menyimpan model
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True)
        predictor_path = model_dir / "shape_predictor_68_face_landmarks.dat"
        
        if predictor_path.exists():
            print(f"✅ Model predictor sudah ada: {predictor_path}")
            return str(predictor_path)
        
        print("📥 Downloading dlib facial landmark predictor...")
        
        # URL untuk download model
        model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        
        try:
            # Download compressed model
            print("⏬ Downloading shape_predictor_68_face_landmarks.dat.bz2...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            compressed_path = model_dir / "shape_predictor_68_face_landmarks.dat.bz2"
            
            with open(compressed_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract bz2 file
            print("📦 Extracting model...")
            import bz2
            
            with bz2.open(compressed_path, 'rb') as src, open(predictor_path, 'wb') as dst:
                dst.write(src.read())
            
            # Remove compressed file
            compressed_path.unlink()
            
            print(f"✅ Model downloaded dan extracted: {predictor_path}")
            return str(predictor_path)
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("💡 Alternatif: Download manual dari http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print(f"   Extract dan simpan di: {predictor_path}")
            raise
    
    def parse_filename(self, filename):
        """
        Parse JAFFE filename untuk mendapatkan informasi
        Format bisa: 
        - Standard: XX-YY#.tiff (XX=poser, YY=emotion, #=pose_number)
        - Variant: XX.YY#.###.tiff (XX=poser, YY=emotion, #=pose_number, ###=file_number)
        
        Returns:
            dict: {'poser': str, 'emotion': str, 'pose_num': int}
        """
        # Remove .tiff extension
        name = filename.replace('.tiff', '').replace('.tif', '')
        
        # Try different parsing methods
        
        # Method 1: Standard format (XX-YY#)
        if '-' in name:
            parts = name.split('-')
            if len(parts) == 2:
                poser = parts[0]
                emotion_pose = parts[1]
                
                # Extract emotion (first 2 chars) and pose number (last char)
                if len(emotion_pose) >= 3:
                    emotion_code = emotion_pose[:2]
                    pose_num = int(emotion_pose[2:]) if emotion_pose[2:].isdigit() else 1
                else:
                    return None
        
        # Method 2: Dot-separated format (XX.YY#.###)
        elif '.' in name:
            parts = name.split('.')
            if len(parts) >= 2:
                poser = parts[0]
                emotion_pose = parts[1]
                
                # Extract emotion and pose number from emotion_pose
                if len(emotion_pose) >= 3:
                    emotion_code = emotion_pose[:2]  # First 2 chars (e.g., 'AN', 'HA')
                    pose_num_str = emotion_pose[2:]  # Remaining chars (e.g., '1', '2')
                    pose_num = int(pose_num_str) if pose_num_str.isdigit() else 1
                else:
                    return None
        else:
            return None
        
        # Map emotion code to emotion name
        emotion = self.emotion_mapping.get(emotion_code)
        
        if emotion is None:
            print(f"⚠️ Unknown emotion code '{emotion_code}' in file {filename}")
            return None
            
        return {
            'poser': poser,
            'emotion': emotion,
            'pose_num': pose_num,
            'filename': filename
        }
    
    def extract_landmarks_dlib(self, image):
        """
        Extract facial landmarks menggunakan dlib (68 points)
        
        Args:
            image: Grayscale image array
            
        Returns:
            landmarks: Normalized landmark coordinates (68 points x 2 = 136 features)
        """
        try:
            # Detect faces
            faces = self.detector(image)
            
            if len(faces) == 0:
                print("⚠️ No face detected, returning zero landmarks")
                return np.zeros(136)  # 68 landmarks * 2 coordinates
            
            # Use the first (and hopefully only) face
            face = faces[0]
            
            # Get landmarks
            landmarks = self.predictor(image, face)
            
            # Convert to numpy array dan normalize
            coords = []
            h, w = image.shape[:2]
            
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                # Normalize coordinates to [0,1]
                coords.extend([x/w, y/h])
            
            return np.array(coords)
            
        except Exception as e:
            print(f"⚠️ Error extracting landmarks: {e}")
            return np.zeros(136)
    
    def extract_landmarks_advanced(self, image):
        """
        Extract advanced facial landmarks dengan geometric features
        
        Args:
            image: Grayscale image array
            
        Returns:
            features: Extended feature vector (landmarks + geometric features)
        """
        # Basic landmarks (68 points = 136 features)
        basic_landmarks = self.extract_landmarks_dlib(image)
        
        if np.sum(basic_landmarks) == 0:  # No landmarks detected
            return np.zeros(200)  # Extended feature size
        
        # Reshape landmarks untuk processing
        landmarks_2d = basic_landmarks.reshape(68, 2)
        
        try:
            # Calculate geometric features
            geometric_features = self._calculate_geometric_features(landmarks_2d)
            
            # Combine basic landmarks dengan geometric features
            extended_features = np.concatenate([basic_landmarks, geometric_features])
            
            return extended_features
            
        except Exception as e:
            print(f"⚠️ Error calculating geometric features: {e}")
            # Return basic landmarks padded with zeros
            padding = np.zeros(200 - len(basic_landmarks))
            return np.concatenate([basic_landmarks, padding])
    
    def _calculate_geometric_features(self, landmarks_2d):
        """
        Calculate geometric features dari facial landmarks
        
        Args:
            landmarks_2d: Array of shape (68, 2) dengan landmark coordinates
            
        Returns:
            geometric_features: Array of geometric measurements
        """
        features = []
        
        # Eye aspect ratios
        left_eye = landmarks_2d[36:42]
        right_eye = landmarks_2d[42:48]
        
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        features.extend([left_ear, right_ear])
        
        # Mouth aspect ratio
        mouth = landmarks_2d[48:68]
        mar = self._mouth_aspect_ratio(mouth)
        features.append(mar)
        
        # Eyebrow heights (relative to eyes)
        left_eyebrow = landmarks_2d[17:22]
        right_eyebrow = landmarks_2d[22:27]
        
        left_eyebrow_height = np.mean(left_eyebrow[:, 1]) - np.mean(left_eye[:, 1])
        right_eyebrow_height = np.mean(right_eyebrow[:, 1]) - np.mean(right_eye[:, 1])
        features.extend([left_eyebrow_height, right_eyebrow_height])
        
        # Face width to height ratio
        face_width = landmarks_2d[16, 0] - landmarks_2d[0, 0]  # Jawline width
        face_height = landmarks_2d[8, 1] - landmarks_2d[19, 1]  # Chin to forehead
        face_ratio = face_width / (face_height + 1e-6)
        features.append(face_ratio)
        
        # Distance between eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        features.append(eye_distance)
        
        # Nose width
        nose_width = landmarks_2d[35, 0] - landmarks_2d[31, 0]
        features.append(nose_width)
        
        # Add more features to reach desired size
        while len(features) < 64:  # Total 136 + 64 = 200 features
            features.append(0.0)
        
        return np.array(features[:64])
    
    def _eye_aspect_ratio(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        # EAR calculation
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def _mouth_aspect_ratio(self, mouth_landmarks):
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Vertical distances
        A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])  # 50-58
        B = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])   # 52-56
        # Horizontal distance
        C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])   # 48-54
        # MAR calculation
        mar = (A + B) / (2.0 * C + 1e-6)
        return mar
    
    def preprocess_images(self, target_size=(224, 224), use_advanced_features=True):
        """
        Load dan preprocess semua gambar JAFFE
        
        Args:
            target_size: Target size untuk resize gambar
            use_advanced_features: Gunakan extended geometric features
        
        Returns:
            images: Array of preprocessed images
            landmarks: Array of facial landmarks  
            labels: Array of emotion labels
            metadata: List of file metadata
        """
        images = []
        landmarks = []
        labels = []
        metadata = []
        
        print("🔄 Processing JAFFE images dengan dlib...")
        
        # Get all .tiff files
        image_files = list(self.jaffe_path.glob("*.tiff"))
        
        if not image_files:
            # Try .tif extension as well
            image_files = list(self.jaffe_path.glob("*.tif"))
        
        print(f"📸 Found {len(image_files)} image files")
        
        processed_count = 0
        failed_count = 0
        no_face_count = 0
        
        for img_file in image_files:
            try:
                # Parse filename
                file_info = self.parse_filename(img_file.name)
                if file_info is None:
                    print(f"⚠️ Skipping {img_file.name} - invalid format")
                    failed_count += 1
                    continue
                
                # Load image
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"⚠️ Failed to load {img_file.name}")
                    failed_count += 1
                    continue
                
                # Extract landmarks menggunakan grayscale image
                if use_advanced_features:
                    landmark_features = self.extract_landmarks_advanced(image)
                else:
                    landmark_features = self.extract_landmarks_dlib(image)
                
                # Check if face was detected
                if np.sum(landmark_features) == 0:
                    print(f"⚠️ No face detected in {img_file.name}")
                    no_face_count += 1
                    # Skip image tanpa face atau tetap include dengan zero landmarks?
                    # Untuk JAFFE, sebaiknya skip karena semua image seharusnya ada face
                    continue
                
                # Convert to RGB untuk CNN processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Resize image
                image_resized = cv2.resize(image_rgb, target_size)
                
                # Normalize to [0,1]
                image_normalized = image_resized.astype(np.float32) / 255.0
                
                # Store results
                images.append(image_normalized)
                landmarks.append(landmark_features)
                labels.append(file_info['emotion'])
                metadata.append(file_info)
                
                processed_count += 1
                
                if processed_count % 20 == 0:
                    print(f"✅ Processed {processed_count}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
                failed_count += 1
                continue
        
        print(f"\n📊 Processing Summary:")
        print(f"✅ Successfully processed: {processed_count}")
        print(f"⚠️ No face detected: {no_face_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"🎯 Landmark features per sample: {landmark_features.shape[0] if processed_count > 0 else 'N/A'}")
        
        return np.array(images), np.array(landmarks), np.array(labels), metadata
    
    def create_label_mapping(self, labels):
        """Create label to integer mapping"""
        unique_labels = sorted(list(set(labels)))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return label_map
    
    def split_data(self, images, landmarks, labels, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train/validation/test sets
        
        Args:
            images: Image array
            landmarks: Landmark array  
            labels: Label array
            test_size: Fraction for test set
            val_size: Fraction for validation set (dari training set)
            random_state: Random seed
            
        Returns:
            Dictionary dengan train/val/test splits
        """
        print(f"📊 Splitting data: Train/Val/Test")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, land_temp, land_test = train_test_split(
            images, labels, landmarks, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val, land_train, land_val = train_test_split(
            X_temp, y_temp, land_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'landmarks_train': land_train,
            'X_val': X_val, 'y_val': y_val, 'landmarks_val': land_val,
            'X_test': X_test, 'y_test': y_test, 'landmarks_test': land_test
        }
    
    def save_preprocessed_data(self, data_splits, label_map):
        """Save preprocessed data untuk training"""
        
        # Create directories
        img_dir = self.output_path / 'images'
        landmark_dir = self.output_path / 'landmarks'
        img_dir.mkdir(exist_ok=True)
        landmark_dir.mkdir(exist_ok=True)
        
        print("💾 Saving preprocessed data...")
        
        # Save images
        np.save(img_dir / 'X_train_images.npy', data_splits['X_train'])
        np.save(img_dir / 'X_val_images.npy', data_splits['X_val'])
        np.save(img_dir / 'X_test_images.npy', data_splits['X_test'])
        np.save(img_dir / 'y_train_images.npy', data_splits['y_train'])
        np.save(img_dir / 'y_val_images.npy', data_splits['y_val'])
        np.save(img_dir / 'y_test_images.npy', data_splits['y_test'])
        
        # Save landmarks
        np.save(landmark_dir / 'X_train_landmarks.npy', data_splits['landmarks_train'])
        np.save(landmark_dir / 'X_val_landmarks.npy', data_splits['landmarks_val'])
        np.save(landmark_dir / 'X_test_landmarks.npy', data_splits['landmarks_test'])
        np.save(landmark_dir / 'y_train_landmarks.npy', data_splits['y_train'])
        np.save(landmark_dir / 'y_val_landmarks.npy', data_splits['y_val'])
        np.save(landmark_dir / 'y_test_landmarks.npy', data_splits['y_test'])
        
        # Save label mapping
        with open(self.output_path / 'label_map.pkl', 'wb') as f:
            pickle.dump(label_map, f)
        
        print(f"✅ Data saved to {self.output_path}")
        
        # Print data info
        print(f"\n📊 Dataset Information:")
        print(f"Image shape: {data_splits['X_train'].shape[1:]}")
        print(f"Landmark features: {data_splits['landmarks_train'].shape[1]}")
        print(f"Classes: {list(label_map.keys())}")
        print(f"Number of classes: {len(label_map)}")

def main():
    """Main preprocessing function"""
    
    # Paths - sesuaikan dengan lokasi dataset JAFFE Anda
    jaffe_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/jaffe"  # Folder berisi file .tiff
    output_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data"
    
    # Initialize preprocessor
    preprocessor = JAFFEPreprocessor(jaffe_path, output_path)
    
    # Process images and extract features
    # use_advanced_features=True untuk geometric features tambahan
    images, landmarks, labels, metadata = preprocessor.preprocess_images(
        target_size=(224, 224),
        use_advanced_features=True
    )
    
    if len(images) == 0:
        print("❌ No images processed! Check your JAFFE path and file format.")
        return
    
    # Create label mapping
    label_map = preprocessor.create_label_mapping(labels)
    
    # Split data
    data_splits = preprocessor.split_data(images, landmarks, labels)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(data_splits, label_map)
    
    # Print summary
    print(f"\n🎉 JAFFE preprocessing completed!")
    print(f"📈 Distribution per emotion:")
    for emotion, count in zip(*np.unique(labels, return_counts=True)):
        print(f"   {emotion}: {count} samples")

if __name__ == "__main__":
    main()