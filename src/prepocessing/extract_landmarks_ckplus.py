import mediapipe as mp
import pandas as pd
import cv2
import os

def extract_landmarks_ckplus(dataset_path, output_path):
    """
    Ekstraksi landmark facial dari dataset CK+
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    results_data = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        
        for emotion in emotions:
            emotion_path = os.path.join(dataset_path, emotion)
            if not os.path.exists(emotion_path):
                continue
                
            print(f"Processing {emotion}...")
            count = 0
            
            for img_file in os.listdir(emotion_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(emotion_path, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is not None:
                        # Convert BGR to RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Proses landmark
                        results = face_mesh.process(rgb_image)
                        
                        if results.multi_face_landmarks:
                            landmarks = []
                            for face_landmarks in results.multi_face_landmarks:
                                for landmark in face_landmarks.landmark:
                                    landmarks.extend([landmark.x, landmark.y])
                            
                            results_data.append({
                                'filename': img_file,
                                'emotion': emotion,
                                'landmarks': str(landmarks),
                                'image_path': img_path,
                                'user_id': img_file.split('_')[0],  # Extract subject ID
                                'challenge_id': img_file.split('_')[1]  # Extract challenge ID
                            })
                            count += 1
            
            print(f"  Extracted landmarks from {count} images")
    
    # Simpan ke Excel
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results_data)
    df.to_excel(output_path, index=False)
    print(f"Landmark extraction complete. Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    dataset_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/ckplus_processed"
    output_path = "D:/research/2025_iris_taufik/MultimodalEmoLearn/data/Emotion_Labels/ckplus_landmarks_raw.xlsx"
    
    extract_landmarks_ckplus(dataset_path, output_path)