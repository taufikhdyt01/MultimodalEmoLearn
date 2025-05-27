"""
ViT Emotion Recognition - Production Deployment Script
Optimized untuk AMD RX 6600 LE dan real-time inference
"""

import os
import time
import numpy as np
import torch
import cv2
import threading
import queue
from pathlib import Path
import logging
from datetime import datetime
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# HuggingFace dan ML imports
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import tensorflow as tf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vit_emotion_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration untuk deployment"""
    model_path: str
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    batch_size: int = 4
    max_image_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.5
    fps_target: int = 30
    enable_tensorrt: bool = False  # Untuk NVIDIA GPU
    enable_quantization: bool = False
    log_predictions: bool = True
    save_results: bool = True


class OptimizedViTClassifier:
    """
    Production-ready ViT classifier dengan optimasi untuk real-time inference
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        self.label_map = {}
        self.id_to_label = {}
        self.inference_times = []
        
        # Performance monitoring
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.start_time = time.time()
        
        self._load_model()
        self._optimize_model()
        
        logger.info(f"✅ OptimizedViTClassifier initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device untuk inference"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"🎮 Auto-selected GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logger.info("💻 Auto-selected CPU (no GPU available)")
        else:
            device = torch.device(self.config.device)
            logger.info(f"🔧 Manual device selection: {device}")
        
        # GPU memory optimization
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # Optimize untuk consistent input sizes
            
        return device
    
    def _load_model(self):
        """Load dan setup model"""
        try:
            logger.info(f"📥 Loading model from {self.config.model_path}")
            
            # Load model
            self.model = ViTForImageClassification.from_pretrained(self.config.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load processor
            self.processor = ViTImageProcessor.from_pretrained(self.config.model_path)
            
            # Setup label mappings
            self.id_to_label = self.model.config.id2label
            self.label_map = {v: k for k, v in self.id_to_label.items()}
            
            logger.info(f"✅ Model loaded with {len(self.label_map)} classes")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _optimize_model(self):
        """Apply various optimizations untuk production"""
        try:
            # Enable inference mode
            torch.inference_mode()
            
            # Quantization untuk CPU inference (opsional)
            if self.config.enable_quantization and self.device.type == 'cpu':
                logger.info("🔧 Applying dynamic quantization...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            # Compile model untuk PyTorch 2.0+ (jika available)
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(self.model)
                    logger.info("⚡ Model compiled for optimization")
                except:
                    logger.warning("⚠️ Model compilation failed, using standard model")
            
            # JIT tracing untuk consistent input sizes
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    traced_model = torch.jit.trace(self.model, dummy_input)
                    traced_model.eval()
                    self.model = traced_model
                    logger.info("🚀 Model traced for faster inference")
            except:
                logger.warning("⚠️ Model tracing failed, using eager execution")
            
        except Exception as e:
            logger.warning(f"⚠️ Some optimizations failed: {e}")
    
    @torch.inference_mode()
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Fast single image prediction
        
        Args:
            image: numpy array (H, W, 3) dalam format BGR atau RGB
            
        Returns:
            dict: prediction results dengan timing info
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Convert BGR ke RGB jika perlu (OpenCV default)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image)
            
            # Process dengan ViT processor
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            # Inference
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get results
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities.max().item()
            
            # Convert ke numpy untuk serialization
            probs_np = probabilities.cpu().numpy().flatten()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            self.total_predictions += 1
            
            # Keep only last 100 timing measurements
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            result = {
                'predicted_emotion': self.id_to_label[predicted_class_id],
                'confidence': float(confidence),
                'predicted_class_id': predicted_class_id,
                'probabilities': {
                    emotion: float(probs_np[class_id]) 
                    for emotion, class_id in self.label_map.items()
                },
                'inference_time_ms': inference_time * 1000,
                'timestamp': datetime.now().isoformat(),
                'above_threshold': confidence >= self.config.confidence_threshold
            }
            
            self.successful_predictions += 1
            
            # Log jika confidence rendah
            if confidence < self.config.confidence_threshold:
                logger.warning(f"⚠️ Low confidence prediction: {confidence:.3f} for {result['predicted_emotion']}")
            
            return result
            
        except Exception as e:
            self.failed_predictions += 1
            logger.error(f"❌ Prediction failed: {e}")
            return {
                'error': str(e),
                'inference_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Batch prediction untuk improved throughput
        
        Args:
            images: List of numpy arrays
            
        Returns:
            List of prediction results
        """
        batch_size = min(len(images), self.config.batch_size)
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                result = self.predict_single(image)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_predictions': self.total_predictions,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'success_rate': self.successful_predictions / max(self.total_predictions, 1),
            'average_inference_time_ms': avg_inference_time * 1000,
            'current_fps': fps,
            'target_fps': self.config.fps_target,
            'device': str(self.device)
        }


class RealTimeEmotionProcessor:
    """
    Real-time emotion processing untuk webcam atau video stream
    """
    
    def __init__(self, config: InferenceConfig, source=0):
        self.config = config
        self.classifier = OptimizedViTClassifier(config)
        self.source = source
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Setup video capture
        self.cap = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        logger.info("📹 RealTimeEmotionProcessor initialized")
    
    def start_capture(self):
        """Start video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                raise Exception("Cannot open video source")
            
            logger.info(f"✅ Video capture started: {self.source}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start capture: {e}")
            return False
    
    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Add frame ke queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame dan add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            else:
                logger.warning("⚠️ Failed to read frame")
                time.sleep(0.1)
    
    def process_frames(self):
        """Process frames in separate thread"""
        while self.is_running:
            try:
                # Get frame dari queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Face detection untuk crop ROI (opsional)
                processed_frame = self.preprocess_frame(frame)
                
                if processed_frame is not None:
                    # Emotion prediction
                    result = self.classifier.predict_single(processed_frame)
                    result['original_frame'] = frame
                    result['processed_frame'] = processed_frame
                    
                    # Add ke result queue
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # Remove oldest result
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess frame untuk emotion recognition
        Include face detection dan cropping
        """
        try:
            # Simple resize untuk testing
            # Dalam production, add face detection dengan OpenCV atau MediaPipe
            resized = cv2.resize(frame, self.config.max_image_size)
            return resized
            
        except Exception as e:
            logger.error(f"❌ Frame preprocessing failed: {e}")
            return None
    
    def start_real_time_processing(self, duration_seconds=None):
        """Start real-time processing"""
        if not self.start_capture():
            return False
        
        self.is_running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_frames, daemon=True)
        
        capture_thread.start()
        process_thread.start()
        
        logger.info("🚀 Real-time processing started")
        
        try:
            start_time = time.time()
            while self.is_running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Display results
                self.display_results()
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.01)  # Small delay
                
        except KeyboardInterrupt:
            logger.info("⏹️ Stopped by user")
        finally:
            self.stop_processing()
        
        return True
    
    def display_results(self):
        """Display real-time results"""
        try:
            result = self.result_queue.get_nowait()
            
            if 'error' not in result:
                frame = result['original_frame']
                emotion = result['predicted_emotion']
                confidence = result['confidence']
                
                # Draw emotion text on frame
                text = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Add FPS info
                self.fps_counter += 1
                if time.time() - self.fps_start_time > 1.0:
                    fps = self.fps_counter / (time.time() - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Show frame
                cv2.imshow('Real-time Emotion Recognition', frame)
            
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"❌ Display error: {e}")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        stats = self.classifier.get_performance_stats()
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total Predictions: {stats['total_predictions']}")
        logger.info(f"   Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"   Average FPS: {stats['current_fps']:.1f}")
        
        logger.info("⏹️ Real-time processing stopped")


def create_benchmark_suite(config: InferenceConfig):
    """
    Create comprehensive benchmark untuk testing performance
    """
    
    class PerformanceBenchmark:
        def __init__(self, classifier):
            self.classifier = classifier
            self.results = {}
        
        def benchmark_single_inference(self, num_iterations=100):
            """Benchmark single image inference"""
            logger.info(f"🔬 Benchmarking single inference ({num_iterations} iterations)...")
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            times = []
            for i in range(num_iterations):
                start = time.time()
                result = self.classifier.predict_single(dummy_image)
                end = time.time()
                
                if 'error' not in result:
                    times.append(end - start)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"   Progress: {i+1}/{num_iterations}")
            
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            fps = 1.0 / avg_time
            
            self.results['single_inference'] = {
                'average_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'fps': fps,
                'successful_predictions': len(times),
                'total_iterations': num_iterations
            }
            
            logger.info(f"✅ Single Inference Benchmark Results:")
            logger.info(f"   Average: {avg_time*1000:.2f}ms ({fps:.1f} FPS)")
            logger.info(f"   Range: {min_time*1000:.2f}-{max_time*1000:.2f}ms")
        
        def benchmark_batch_inference(self, batch_sizes=[1, 2, 4, 8, 16]):
            """Benchmark batch inference"""
            logger.info(f"🔬 Benchmarking batch inference...")
            
            batch_results = {}
            
            for batch_size in batch_sizes:
                logger.info(f"   Testing batch size: {batch_size}")
                
                # Create batch
                images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                         for _ in range(batch_size)]
                
                # Benchmark
                start = time.time()
                results = self.classifier.predict_batch(images)
                end = time.time()
                
                total_time = end - start
                time_per_image = total_time / batch_size
                fps = batch_size / total_time
                
                batch_results[batch_size] = {
                    'total_time_ms': total_time * 1000,
                    'time_per_image_ms': time_per_image * 1000,
                    'batch_fps': fps,
                    'successful_predictions': len([r for r in results if 'error' not in r])
                }
                
                logger.info(f"     Batch FPS: {fps:.1f}, Per image: {time_per_image*1000:.2f}ms")
            
            self.results['batch_inference'] = batch_results
        
        def memory_usage_test(self):
            """Test memory usage"""
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
                
                # Run inference
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                self.classifier.predict_single(dummy_image)
                
                peak_memory = torch.cuda.max_memory_allocated()
                current_memory = torch.cuda.memory_allocated()
                
                self.results['memory_usage'] = {
                    'initial_mb': initial_memory / 1024 / 1024,
                    'peak_mb': peak_memory / 1024 / 1024,
                    'current_mb': current_memory / 1024 / 1024,
                    'inference_overhead_mb': (peak_memory - initial_memory) / 1024 / 1024
                }
                
                logger.info(f"💾 Memory Usage:")
                logger.info(f"   Peak: {peak_memory/1024/1024:.1f}MB")
                logger.info(f"   Inference overhead: {(peak_memory-initial_memory)/1024/1024:.1f}MB")
        
        def save_benchmark_results(self, filepath):
            """Save benchmark results"""
            self.results['benchmark_config'] = {
                'device': str(self.classifier.device),
                'model_path': self.classifier.config.model_path,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"💾 Benchmark results saved: {filepath}")
    
    return PerformanceBenchmark


def main():
    """Main deployment script"""
    print("🚀 ViT Emotion Recognition - Production Deployment")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/vit-emotion-final"
    
    config = InferenceConfig(
        model_path=MODEL_PATH,
        device='auto',
        batch_size=4,
        confidence_threshold=0.7,
        fps_target=30,
        enable_quantization=False,  # Set True untuk CPU optimization
        log_predictions=True
    )
    
    try:
        # Initialize classifier
        logger.info("🔧 Initializing ViT classifier...")
        classifier = OptimizedViTClassifier(config)
        
        # Quick test
        logger.info("🧪 Running quick test...")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classifier.predict_single(test_image)
        
        if 'error' not in result:
            logger.info(f"✅ Test successful: {result['predicted_emotion']} ({result['confidence']:.3f})")
            logger.info(f"   Inference time: {result['inference_time_ms']:.2f}ms")
        else:
            logger.error(f"❌ Test failed: {result['error']}")
            return
        
        # Performance benchmark
        logger.info("📊 Running performance benchmark...")
        benchmark = create_benchmark_suite(config)(classifier)
        benchmark.benchmark_single_inference(50)
        benchmark.benchmark_batch_inference([1, 2, 4])
        benchmark.memory_usage_test()
        benchmark.save_benchmark_results("vit_benchmark_results.json")
        
        # Real-time processing option
        choice = input("\n🎥 Start real-time webcam processing? (y/n): ").lower()
        if choice == 'y':
            processor = RealTimeEmotionProcessor(config, source=0)
            processor.start_real_time_processing(duration_seconds=60)  # 1 minute test
        
        logger.info("✅ Deployment test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
