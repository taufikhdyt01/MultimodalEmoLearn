import os
import numpy as np
import torch
import pickle
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ViTEmotionClassifier:
    """
    ViT-based Emotion Classifier yang terintegrasi dengan pipeline existing
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize ViT classifier
        
        Args:
            model_path (str): Path ke fine-tuned model
            device (str): Device untuk inference ('cuda' atau 'cpu')
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model dan processor
        self.model = None
        self.processor = None
        self.label_map = None
        self.id_to_label = None
        
        self._load_model()
    
    def _load_model(self):
        """Load fine-tuned ViT model"""
        try:
            print(f"Loading ViT model from {self.model_path}...")
            
            # Load model
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load processor
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            
            # Create label mappings
            self.id_to_label = self.model.config.id2label
            self.label_map = self.model.config.label2id
            
            print(f"✅ Model loaded successfully on {self.device}")
            print(f"Classes: {list(self.label_map.keys())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_single_image(self, image):
        """
        Predict emotion untuk single image
        
        Args:
            image: numpy array (H, W, 3) atau PIL Image
            
        Returns:
            dict: prediction results
        """
        # Convert ke PIL jika perlu
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Process image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities.max().item()
        
        # Convert probabilities ke numpy
        probs_np = probabilities.cpu().numpy().flatten()
        
        return {
            'predicted_emotion': self.id_to_label[predicted_class_id],
            'confidence': confidence,
            'probabilities': {
                emotion: float(probs_np[class_id]) 
                for emotion, class_id in self.label_map.items()
            },
            'predicted_class_id': predicted_class_id
        }
    
    def predict_batch(self, images, batch_size=16):
        """
        Predict emotions untuk batch images
        
        Args:
            images: list of numpy arrays atau PIL Images
            batch_size: batch size untuk processing
            
        Returns:
            list: prediction results untuk setiap image
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                result = self.predict_single_image(image)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_probabilities_array(self, images):
        """
        Get probability arrays untuk fusion dengan landmark model
        
        Args:
            images: numpy array (N, H, W, 3)
            
        Returns:
            numpy array (N, num_classes): probability matrix
        """
        prob_matrix = []
        
        for image in images:
            result = self.predict_single_image(image)
            probs = [result['probabilities'][emotion] for emotion in sorted(self.label_map.keys())]
            prob_matrix.append(probs)
        
        return np.array(prob_matrix)


def evaluate_vit_model(model_path, test_images, test_labels, target_names):
    """
    Evaluate ViT model dan compare dengan baseline
    
    Args:
        model_path: Path ke fine-tuned ViT model
        test_images: Test images array
        test_labels: True labels
        target_names: List of emotion names
    """
    print("🧪 Evaluating ViT Fine-tuned Model...")
    
    # Initialize classifier
    vit_classifier = ViTEmotionClassifier(model_path)
    
    # Get predictions
    print("Getting predictions...")
    predictions = []
    probabilities = []
    
    for i, image in enumerate(test_images):
        if i % 50 == 0:
            print(f"Processing {i}/{len(test_images)}...")
        
        result = vit_classifier.predict_single_image(image)
        predictions.append(result['predicted_class_id'])
        
        # Convert probabilities to array
        probs = [result['probabilities'][emotion] for emotion in target_names]
        probabilities.append(probs)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"\n📊 ViT Model Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=target_names, yticklabels=target_names)
    plt.title(f'ViT Fine-tuned Model - Confusion Matrix\nAccuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': cm
    }


def compare_with_existing_models(vit_results, model_results_path):
    """
    Compare ViT results dengan existing models (CNN, Landmark, Fusion)
    
    Args:
        vit_results: Results dari ViT evaluation
        model_results_path: Path ke results file dari existing models
    """
    print("🔄 Comparing dengan existing models...")
    
    try:
        # Load existing results
        with open(model_results_path, 'rb') as f:
            existing_results = pickle.load(f)
        
        print("\n📊 Model Comparison:")
        print("=" * 50)
        
        # Extract accuracies
        models = {}
        
        if 'model_rankings' in existing_results:
            for model_info in existing_results['model_rankings']:
                models[model_info['Model']] = model_info['Accuracy']
        
        # Add ViT results
        models['ViT Fine-tuned'] = vit_results['accuracy']
        
        # Sort by accuracy
        sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
        
        print(f"🏆 RANKING BY ACCURACY:")
        for i, (model_name, accuracy) in enumerate(sorted_models, 1):
            print(f"{i}. {model_name}: {accuracy:.4f}")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        model_names = [item[0] for item in sorted_models]
        accuracies = [item[1] for item in sorted_models]
        
        colors = ['gold' if 'ViT' in name else 'skyblue' for name in model_names]
        bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.ylim(0, max(accuracies) + 0.05)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate improvements
        best_existing = max([acc for name, acc in models.items() if 'ViT' not in name])
        vit_improvement = vit_results['accuracy'] - best_existing
        
        print(f"\n📈 ViT vs Best Existing Model:")
        print(f"Best Existing: {best_existing:.4f}")
        print(f"ViT Fine-tuned: {vit_results['accuracy']:.4f}")
        print(f"Improvement: {vit_improvement:.4f} ({vit_improvement*100:.2f}%)")
        
        if vit_improvement > 0:
            print("🎉 ViT outperforms all existing models!")
        else:
            print("📝 Existing models still competitive")
        
        return sorted_models
        
    except FileNotFoundError:
        print("⚠️ Existing model results not found")
        return None


def late_fusion_with_vit(vit_probs, landmark_probs, true_labels, target_names):
    """
    Perform late fusion between ViT dan Landmark model
    
    Args:
        vit_probs: ViT probability predictions
        landmark_probs: Landmark model probability predictions
        true_labels: True labels
        target_names: List of emotion names
    """
    print("🔗 Performing Late Fusion: ViT + Landmark...")
    
    best_accuracy = 0
    best_weight = 0
    fusion_results = []
    
    # Test different weight combinations
    for vit_weight in np.arange(0.0, 1.1, 0.1):
        landmark_weight = 1 - vit_weight
        
        # Weighted fusion
        fused_probs = (vit_weight * vit_probs) + (landmark_weight * landmark_probs)
        fused_preds = np.argmax(fused_probs, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, fused_preds)
        fusion_results.append((vit_weight, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weight = vit_weight
        
        print(f"ViT Weight: {vit_weight:.1f}, Accuracy: {accuracy:.4f}")
    
    print(f"\n🏆 Best Fusion Configuration:")
    print(f"ViT Weight: {best_weight:.1f}")
    print(f"Landmark Weight: {1-best_weight:.1f}")
    print(f"Fusion Accuracy: {best_accuracy:.4f}")
    
    # Plot weight optimization
    weights, accuracies = zip(*fusion_results)
    plt.figure(figsize=(10, 6))
    plt.plot(weights, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.scatter([best_weight], [best_accuracy], color='red', s=200, marker='*', 
               label=f'Best ({best_weight:.1f}, {best_accuracy:.4f})', zorder=5)
    plt.xlabel('ViT Weight')
    plt.ylabel('Fusion Accuracy')
    plt.title('Late Fusion: ViT + Landmark Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'best_vit_weight': best_weight,
        'best_accuracy': best_accuracy,
        'fusion_results': fusion_results
    }


def create_ensemble_classifier(vit_model_path, landmark_model_path, fusion_weights):
    """
    Create ensemble classifier yang menggabungkan ViT dan Landmark model
    
    Args:
        vit_model_path: Path ke ViT model
        landmark_model_path: Path ke Landmark model
        fusion_weights: Dict dengan ViT dan Landmark weights
    """
    
    class EnsembleEmotionClassifier:
        def __init__(self, vit_path, landmark_path, weights):
            # Load ViT model
            self.vit_classifier = ViTEmotionClassifier(vit_path)
            
            # Load Landmark model
            import tensorflow as tf
            self.landmark_model = tf.keras.models.load_model(landmark_path)
            
            # Fusion weights
            self.vit_weight = weights['vit']
            self.landmark_weight = weights['landmark']
            
            print(f"✅ Ensemble classifier initialized")
            print(f"ViT weight: {self.vit_weight:.2f}")
            print(f"Landmark weight: {self.landmark_weight:.2f}")
        
        def predict(self, image, landmarks):
            """
            Predict emotion menggunakan ensemble
            
            Args:
                image: Input image
                landmarks: Landmark features
            
            Returns:
                dict: Ensemble prediction results
            """
            # Get ViT prediction
            vit_result = self.vit_classifier.predict_single_image(image)
            vit_probs = np.array([vit_result['probabilities'][emotion] 
                                for emotion in sorted(vit_result['probabilities'].keys())])
            
            # Get Landmark prediction
            landmarks_reshaped = landmarks.reshape(1, -1)
            landmark_probs = self.landmark_model.predict(landmarks_reshaped, verbose=0)[0]
            
            # Fusion
            fused_probs = (self.vit_weight * vit_probs) + (self.landmark_weight * landmark_probs)
            predicted_class = np.argmax(fused_probs)
            confidence = fused_probs[predicted_class]
            
            emotions = sorted(vit_result['probabilities'].keys())
            
            return {
                'predicted_emotion': emotions[predicted_class],
                'confidence': float(confidence),
                'ensemble_probabilities': {
                    emotion: float(fused_probs[i]) for i, emotion in enumerate(emotions)
                },
                'vit_probabilities': vit_result['probabilities'],
                'landmark_probabilities': {
                    emotion: float(landmark_probs[i]) for i, emotion in enumerate(emotions)
                }
            }
    
    return EnsembleEmotionClassifier(
        vit_model_path, 
        landmark_model_path, 
        {'vit': fusion_weights['vit'], 'landmark': fusion_weights['landmark']}
    )


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/"
    
    # Path configurations
    vit_model_path = os.path.join(MODEL_PATH, "vit-emotion-final")
    existing_results_path = os.path.join(MODEL_PATH, "comprehensive_model_comparison.pkl")
    
    print("🚀 ViT Integration Script")
    print("=" * 40)
    
    # Load test data
    BASE_PATH = "D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/"
    X_test_images = np.load(BASE_PATH + 'images/X_test_images.npy')
    y_test = np.load(BASE_PATH + 'images/y_test_images.npy')
    
    # Create label mapping
    unique_labels = np.unique(y_test)
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    y_test_ids = np.array([label_to_id[label] for label in y_test])
    
    # Evaluate ViT model
    vit_results = evaluate_vit_model(vit_model_path, X_test_images, y_test_ids, unique_labels)
    
    # Compare dengan existing models
    comparison_results = compare_with_existing_models(vit_results, existing_results_path)
    
    print("\n✅ Integration completed!")
    print("💡 ViT model is now ready untuk production use atau further fusion")
