# 🚀 ViT Fine-Tuning Complete Setup Guide

## Panduan Lengkap untuk Fine-Tuning ViT Model pada AMD RX 6600 LE

### 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Fine-Tuning Process](#fine-tuning-process)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)
7. [Production Deployment](#production-deployment)
8. [Performance Optimization](#performance-optimization)

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, atau macOS
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: AMD RX 6600 LE (4GB VRAM) atau equivalent
- **Storage**: Minimum 10GB free space
- **Python**: 3.8 - 3.11

### GPU Requirements
```bash
# Check GPU status
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD (if ROCm installed)
```

---

## 🐍 Environment Setup

### Step 1: Create Conda Environment
```bash
# Create new environment
conda create -n emotion-finetune python=3.9 -y
conda activate emotion-finetune

# Update conda
conda update conda -y
```

### Step 2: Install PyTorch for AMD GPU
```bash
# Option 1: DirectML for AMD (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-directml

# Option 2: ROCm for AMD (Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Option 3: CPU only (fallback)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Required Packages
```bash
# Core ML packages
pip install transformers[torch]==4.35.0
pip install datasets==2.14.0
pip install accelerate==0.24.0
pip install evaluate==0.4.0

# Data processing
pip install scikit-learn==1.3.0
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install pillow==10.0.0

# Visualization
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Utilities
pip install tqdm==4.66.0
pip install psutil==5.9.5
pip install opencv-python==4.8.0.74

# Optional: Advanced optimization
pip install onnx==1.14.1
pip install onnxruntime-gpu==1.16.0  # For NVIDIA
# pip install onnxruntime==1.16.0    # For CPU
```

### Step 4: Verify Installation
```python
import torch
import transformers
import cv2
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Test DirectML (for AMD on Windows)
try:
    device = torch.device("dml" if hasattr(torch, "dml") else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
except:
    device = torch.device("cpu")
    print(f"Fallback to CPU")
```

---

## 📁 Data Preparation

### Data Structure
```
data/
├── images/
│   ├── X_train_images.npy
│   ├── X_val_images.npy
│   ├── X_test_images.npy
│   ├── y_train_images.npy
│   ├── y_val_images.npy
│   └── y_test_images.npy
└── landmarks/
    ├── X_train_landmarks.npy
    ├── X_val_landmarks.npy
    ├── X_test_landmarks.npy
    ├── y_train_landmarks.npy
    ├── y_val_landmarks.npy
    └── y_test_landmarks.npy
```

### Data Validation Script
```python
import numpy as np
import matplotlib.pyplot as plt

def validate_data(data_path):
    """Validate loaded data"""
    
    # Load data
    X_train = np.load(f"{data_path}/images/X_train_images.npy")
    y_train = np.load(f"{data_path}/images/y_train_images.npy")
    
    print(f"✅ Data loaded successfully")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Image shape: {X_train.shape[1:]}")
    print(f"   Data type: {X_train.dtype}")
    print(f"   Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📊 Class Distribution:")
    for emotion, count in zip(unique, counts):
        print(f"   {emotion}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, emotion in enumerate(unique[:10]):
        row, col = i // 5, i % 5
        
        # Find first sample of this emotion
        idx = np.where(y_train == emotion)[0][0]
        image = X_train[idx]
        
        # Normalize for display
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{emotion}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Run validation
validate_data("D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data")
```

---

## 🏋️ Fine-Tuning Process

### Step 1: Configure Training Parameters
```python
# training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "trpakov/vit-face-expression"
    num_epochs: int = 15
    
    # Hardware-specific settings for AMD RX 6600 LE
    batch_size: int = 6       # Conservative for 4GB VRAM
    eval_batch_size: int = 12
    gradient_accumulation_steps: int = 4  # Simulate larger batch
    
    # Optimization
    learning_rate: float = 1e-5  # Conservative for fine-tuning
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Memory optimization
    fp16: bool = True           # Enable mixed precision
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    
    # Monitoring
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    
    # Class imbalance handling
    use_weighted_loss: bool = True
    use_focal_loss: bool = False  # Set True for extreme imbalance
    use_weighted_sampling: bool = True
```

### Step 2: Launch Fine-Tuning
```bash
# Navigate to project directory
cd /path/to/your/project

# Activate environment
conda activate emotion-finetune

# Start fine-tuning (run the notebook or script)
jupyter notebook vit_finetune_notebook.ipynb

# Or run as script
python vit_finetune_script.py
```

### Step 3: Monitor Training Progress
```python
# Monitor GPU usage during training
import psutil
import time

def monitor_training():
    """Monitor system resources during training"""
    while True:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | "
              f"Available: {memory.available/1024**3:.1f}GB")
        
        # GPU (if available)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU Memory: {gpu_memory:.2f}GB")
        
        time.sleep(5)

# Run in separate terminal during training
# monitor_training()
```

---

## 🔧 Troubleshooting

### Common Issues dan Solutions

#### 1. CUDA Out of Memory
```
❌ Error: CUDA out of memory. Tried to allocate X.XXMiB (GPU 0; X.XXGB total capacity)
```

**Solutions:**
```python
# Reduce batch size
training_args.per_device_train_batch_size = 4
training_args.per_device_eval_batch_size = 8

# Increase gradient accumulation
training_args.gradient_accumulation_steps = 8

# Enable mixed precision
training_args.fp16 = True

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Slow Training Speed
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use multiple workers
training_args.dataloader_num_workers = 4

# Enable compilation (PyTorch 2.0+)
model = torch.compile(model)
```

#### 3. DirectML Issues (AMD on Windows)
```bash
# Reinstall DirectML
pip uninstall torch-directml
pip install torch-directml

# Fallback to CPU
device = torch.device("cpu")
training_args.fp16 = False
```

#### 4. Model Loading Errors
```python
# Check model path
import os
assert os.path.exists(model_path), f"Model path not found: {model_path}"

# Try loading step by step
from transformers import AutoConfig, AutoModel
config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
```

---

## 💡 Best Practices

### 1. **Memory Management**
```python
# Clear cache regularly
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Monitor memory usage
def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

### 2. **Training Stability**
```python
# Use learning rate scheduler
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)

# Gradient clipping
training_args.max_grad_norm = 1.0

# Save checkpoints frequently
training_args.save_steps = 200
```

### 3. **Data Handling**
```python
# Efficient data loading
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# Data augmentation for better generalization
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])
```

### 4. **Class Imbalance Handling**
```python
# Weighted sampling
from torch.utils.data import WeightedRandomSampler

class_counts = np.bincount(y_train_encoded)
class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train_encoded]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(y_train_encoded),
    replacement=True
)

# Focal Loss for extreme imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 🚀 Production Deployment

### Step 1: Model Optimization
```python
# Convert to ONNX for faster inference
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "vit_emotion_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# Quantization for CPU deployment
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Step 2: Create Production Pipeline
```python
# production_pipeline.py
class ProductionEmotionClassifier:
    def __init__(self, model_path, device='auto'):
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.processor = self._load_processor(model_path)
        
    def predict(self, image):
        """Production-ready prediction"""
        start_time = time.time()
        
        # Preprocess
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Post-process
        predicted_class = probabilities.argmax().item()
        confidence = probabilities.max().item()
        
        return {
            'emotion': self.id2label[predicted_class],
            'confidence': float(confidence),
            'inference_time_ms': (time.time() - start_time) * 1000
        }

# Usage
classifier = ProductionEmotionClassifier("./models/vit-emotion-final")
result = classifier.predict(your_image)
print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
```

### Step 3: API Deployment
```python
# api_server.py
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io

app = Flask(__name__)
classifier = ProductionEmotionClassifier("./models/vit-emotion-final")

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get image from request
        image_data = request.json['image']  # base64 encoded
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict
        result = classifier.predict(image)
        
        return jsonify({
            'success': True,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'inference_time_ms': result['inference_time_ms']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## ⚡ Performance Optimization

### 1. **Model Optimization**
```python
# JIT Compilation
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("vit_emotion_traced.pt")

# Mixed Precision Inference
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)
```

### 2. **Batch Processing**
```python
def batch_predict(images, batch_size=8):
    """Efficient batch prediction"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_tensors = torch.stack([
            processor(img, return_tensors="pt")['pixel_values'].squeeze()
            for img in batch
        ]).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensors)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        for j, probs in enumerate(probabilities):
            predicted_class = probs.argmax().item()
            confidence = probs.max().item()
            
            results.append({
                'emotion': id2label[predicted_class],
                'confidence': float(confidence)
            })
    
    return results
```

### 3. **Caching dan Preprocessing**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_preprocess(image_hash):
    """Cache preprocessed images"""
    # Implementation depends on your use case
    pass

def preprocess_with_cache(image):
    """Preprocess with caching"""
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    return cached_preprocess(image_hash)
```

---

## 📊 Monitoring dan Maintenance

### 1. **Performance Monitoring**
```python
# performance_monitor.py
class PerformanceMonitor:
    def __init__(self):
        self.inference_times = []
        self.confidence_scores = []
        self.predictions = []
    
    def log_prediction(self, result, inference_time):
        self.inference_times.append(inference_time)
        self.confidence_scores.append(result['confidence'])
        self.predictions.append(result['emotion'])
    
    def get_stats(self):
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'avg_confidence': np.mean(self.confidence_scores),
            'total_predictions': len(self.predictions),
            'emotion_distribution': dict(pd.Series(self.predictions).value_counts())
        }
```

### 2. **Health Checks**
```python
# health_check.py
def health_check():
    """Automated health check"""
    checks = {}
    
    try:
        # Model loading check
        classifier = ProductionEmotionClassifier(MODEL_PATH)
        checks['model_loading'] = 'PASS'
        
        # Inference check
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classifier.predict(test_image)
        checks['inference'] = 'PASS' if result['confidence'] > 0 else 'FAIL'
        
        # Performance check
        checks['performance'] = 'PASS' if result['inference_time_ms'] < 500 else 'WARNING'
        
    except Exception as e:
        checks['error'] = str(e)
    
    return checks
```

### 3. **Automated Retraining**
```python
# retrain_trigger.py
def should_retrain(performance_metrics):
    """Determine if model needs retraining"""
    
    # Check for performance degradation
    if performance_metrics['avg_confidence'] < 0.6:
        return True, "Low confidence scores detected"
    
    # Check for data drift
    current_distribution = performance_metrics['emotion_distribution']
    # Compare with training distribution
    
    # Check inference time degradation
    if performance_metrics['avg_inference_time_ms'] > 1000:
        return True, "Inference time too slow"
    
    return False, "Model performing well"
```

---

## 🎯 Next Steps

### 1. **Integration dengan Landmark Model**
- Implement late fusion dengan landmark features
- Test different fusion strategies (weighted average, learned fusion)
- Optimize ensemble inference speed

### 2. **Advanced Techniques**
- Test different ViT architectures (ViT-Large, DeiT, Swin Transformer)
- Implement knowledge distillation
- Explore self-supervised pre-training

### 3. **Production Enhancements**
- Add A/B testing framework
- Implement model versioning
- Create automated deployment pipeline

### 4. **Research Extensions**
- Multi-modal fusion with audio
- Temporal emotion recognition (video sequences)
- Real-time emotion tracking

---

## 📚 Additional Resources

### Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [DirectML Documentation](https://docs.microsoft.com/en-us/windows/ai/directml/)

### Tools
- [Weights & Biases](https://wandb.ai/) - Training monitoring
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Visualization
- [Gradio](https://gradio.app/) - Quick demo interfaces

### Community
- [HuggingFace Community](https://huggingface.co/spaces)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)

---

## ✅ Checklist

Sebelum deployment, pastikan:

- [ ] Environment setup correctly
- [ ] Data validated dan preprocessed
- [ ] Model fine-tuned successfully
- [ ] Performance benchmarks met
- [ ] Health checks implemented
- [ ] Production pipeline tested
- [ ] Monitoring system ready
- [ ] Documentation complete

---

**Good luck dengan fine-tuning ViT model! 🚀**

Jika ada pertanyaan atau issues, silakan refer ke troubleshooting guide atau buat issue di repository project.