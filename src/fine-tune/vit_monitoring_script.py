"""
ViT Emotion Recognition - Monitoring & Troubleshooting Script
Comprehensive monitoring, debugging, dan maintenance tools
"""

import os
import sys
import psutil
import platform
import subprocess
import json
import time
import numpy as np
import torch
import cv2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor system resources dan GPU performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.gpu_available = torch.cuda.is_available()
        self.measurements = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_memory_used': [],
            'gpu_utilization': [],
            'timestamps': []
        }
    
    def check_system_requirements(self) -> Dict:
        """Check system requirements untuk ViT deployment"""
        logger.info("🔍 Checking system requirements...")
        
        requirements = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if self.gpu_available:
            requirements.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'cuda_version': torch.version.cuda
            })
        
        # Check minimum requirements
        issues = []
        if requirements['total_memory_gb'] < 8:
            issues.append("⚠️ RAM < 8GB may cause performance issues")
        
        if requirements['cpu_count'] < 4:
            issues.append("⚠️ CPU cores < 4 may cause slow processing")
        
        if self.gpu_available and requirements['gpu_memory_gb'] < 4:
            issues.append("⚠️ GPU memory < 4GB may limit batch size")
        
        requirements['issues'] = issues
        requirements['status'] = 'OK' if not issues else 'WARNING'
        
        return requirements
    
    def monitor_resources(self, duration_seconds: int = 60):
        """Monitor system resources dalam real-time"""
        logger.info(f"📊 Monitoring resources for {duration_seconds} seconds...")
        
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            timestamp = datetime.now()
            
            # CPU dan Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            self.measurements['cpu_percent'].append(cpu_percent)
            self.measurements['memory_percent'].append(memory_info.percent)
            self.measurements['timestamps'].append(timestamp)
            
            # GPU monitoring jika available
            if self.gpu_available:
                try:
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    self.measurements['gpu_memory_used'].append(gpu_memory)
                    
                    # GPU utilization (require nvidia-ml-py jika available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.measurements['gpu_utilization'].append(gpu_util.gpu)
                    except:
                        self.measurements['gpu_utilization'].append(0)
                        
                except:
                    self.measurements['gpu_memory_used'].append(0)
                    self.measurements['gpu_utilization'].append(0)
            
            print(f"\rCPU: {cpu_percent:5.1f}% | RAM: {memory_info.percent:5.1f}% | "
                  f"GPU Mem: {self.measurements['gpu_memory_used'][-1]:4.1f}GB", end='')
        
        print()  # New line
        logger.info("✅ Resource monitoring completed")
    
    def generate_resource_report(self, save_path: str = "resource_report.html"):
        """Generate HTML report untuk resource usage"""
        if not self.measurements['timestamps']:
            logger.warning("⚠️ No measurements available")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Resource Monitoring Report', fontsize=16)
        
        timestamps = self.measurements['timestamps']
        
        # CPU Usage
        axes[0, 0].plot(timestamps, self.measurements['cpu_percent'], 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='High Usage (80%)')
        axes[0, 0].legend()
        
        # Memory Usage
        axes[0, 1].plot(timestamps, self.measurements['memory_percent'], 'g-', linewidth=2)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=90, color='r', linestyle='--', alpha=0.7, label='High Usage (90%)')
        axes[0, 1].legend()
        
        if self.gpu_available and self.measurements['gpu_memory_used']:
            # GPU Memory
            axes[1, 0].plot(timestamps, self.measurements['gpu_memory_used'], 'r-', linewidth=2)
            axes[1, 0].set_title('GPU Memory Usage (GB)')
            axes[1, 0].set_ylabel('GPU Memory (GB)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # GPU Utilization
            axes[1, 1].plot(timestamps, self.measurements['gpu_utilization'], 'm-', linewidth=2)
            axes[1, 1].set_title('GPU Utilization (%)')
            axes[1, 1].set_ylabel('GPU %')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 1].text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary statistics
        stats = {
            'monitoring_duration_minutes': len(timestamps),
            'cpu_avg': np.mean(self.measurements['cpu_percent']),
            'cpu_max': np.max(self.measurements['cpu_percent']),
            'memory_avg': np.mean(self.measurements['memory_percent']),
            'memory_max': np.max(self.measurements['memory_percent']),
        }
        
        if self.gpu_available and self.measurements['gpu_memory_used']:
            stats.update({
                'gpu_memory_avg_gb': np.mean(self.measurements['gpu_memory_used']),
                'gpu_memory_max_gb': np.max(self.measurements['gpu_memory_used']),
                'gpu_utilization_avg': np.mean(self.measurements['gpu_utilization']),
                'gpu_utilization_max': np.max(self.measurements['gpu_utilization'])
            })
        
        logger.info(f"📊 Resource Statistics:")
        logger.info(f"   CPU: avg={stats['cpu_avg']:.1f}%, max={stats['cpu_max']:.1f}%")
        logger.info(f"   Memory: avg={stats['memory_avg']:.1f}%, max={stats['memory_max']:.1f}%")
        
        return stats


class ModelDiagnostics:
    """Diagnostic tools untuk ViT model troubleshooting"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.load_status = self._load_model()
    
    def _load_model(self) -> Dict:
        """Load model dengan error handling"""
        status = {'success': False, 'errors': [], 'warnings': []}
        
        try:
            # Check if path exists
            if not os.path.exists(self.model_path):
                status['errors'].append(f"Model path tidak ditemukan: {self.model_path}")
                return status
            
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            from transformers import ViTForImageClassification, ViTImageProcessor
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            status['success'] = True
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            status['errors'].append(f"Model loading failed: {str(e)}")
            logger.error(f"❌ Model loading failed: {e}")
        
        return status
    
    def check_model_integrity(self) -> Dict:
        """Check model file integrity dan configuration"""
        logger.info("🔍 Checking model integrity...")
        
        integrity_check = {
            'model_files_exist': False,
            'config_valid': False,
            'weights_loadable': False,
            'processor_valid': False,
            'issues': []
        }
        
        if not self.load_status['success']:
            integrity_check['issues'].extend(self.load_status['errors'])
            return integrity_check
        
        try:
            # Check required files
            required_files = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']
            missing_files = []
            
            for file in required_files:
                file_path = os.path.join(self.model_path, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if missing_files:
                integrity_check['issues'].append(f"Missing files: {missing_files}")
            else:
                integrity_check['model_files_exist'] = True
            
            # Check config
            config_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                required_config_keys = ['num_labels', 'id2label', 'label2id']
                missing_keys = [key for key in required_config_keys if key not in config]
                
                if missing_keys:
                    integrity_check['issues'].append(f"Missing config keys: {missing_keys}")
                else:
                    integrity_check['config_valid'] = True
                    integrity_check['num_classes'] = config['num_labels']
                    integrity_check['class_names'] = list(config['label2id'].keys())
            
            # Test model inference
            if self.model is not None:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    output = self.model(dummy_input)
                    if hasattr(output, 'logits'):
                        integrity_check['weights_loadable'] = True
                        integrity_check['output_shape'] = output.logits.shape
            
            # Test processor
            if self.processor is not None:
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image
                pil_image = Image.fromarray(dummy_image)
                processed = self.processor(pil_image, return_tensors="pt")
                if 'pixel_values' in processed:
                    integrity_check['processor_valid'] = True
            
        except Exception as e:
            integrity_check['issues'].append(f"Integrity check failed: {str(e)}")
        
        # Summary
        all_checks_passed = all([
            integrity_check['model_files_exist'],
            integrity_check['config_valid'],
            integrity_check['weights_loadable'],
            integrity_check['processor_valid']
        ])
        
        integrity_check['overall_status'] = 'PASS' if all_checks_passed else 'FAIL'
        
        return integrity_check
    
    def test_inference_performance(self, num_iterations: int = 20) -> Dict:
        """Test inference performance dan stability"""
        logger.info(f"⚡ Testing inference performance ({num_iterations} iterations)...")
        
        if not self.load_status['success']:
            return {'error': 'Model not loaded'}
        
        performance_data = {
            'inference_times': [],
            'memory_usage': [],
            'successful_inferences': 0,
            'failed_inferences': 0,
            'errors': []
        }
        
        for i in range(num_iterations):
            try:
                # Create random input
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image
                pil_image = Image.fromarray(dummy_image)
                
                # Memory before inference
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()
                
                # Inference timing
                start_time = time.time()
                
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Memory after inference
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage = memory_after - memory_before
                    performance_data['memory_usage'].append(memory_usage)
                
                performance_data['inference_times'].append(inference_time)
                performance_data['successful_inferences'] += 1
                
            except Exception as e:
                performance_data['failed_inferences'] += 1
                performance_data['errors'].append(str(e))
        
        # Calculate statistics
        if performance_data['inference_times']:
            performance_data['avg_inference_time_ms'] = np.mean(performance_data['inference_times']) * 1000
            performance_data['min_inference_time_ms'] = np.min(performance_data['inference_times']) * 1000
            performance_data['max_inference_time_ms'] = np.max(performance_data['inference_times']) * 1000
            performance_data['fps'] = 1.0 / np.mean(performance_data['inference_times'])
            
            if performance_data['memory_usage']:
                performance_data['avg_memory_usage_mb'] = np.mean(performance_data['memory_usage']) / (1024**2)
        
        performance_data['success_rate'] = performance_data['successful_inferences'] / num_iterations
        
        return performance_data


class TroubleshootingGuide:
    """Interactive troubleshooting guide"""
    
    def __init__(self):
        self.common_issues = {
            'cuda_out_of_memory': {
                'symptoms': ['CUDA out of memory', 'RuntimeError: out of memory'],
                'solutions': [
                    '1. Reduce batch size',
                    '2. Enable gradient checkpointing',
                    '3. Use mixed precision (fp16)',
                    '4. Clear CUDA cache: torch.cuda.empty_cache()',
                    '5. Use CPU inference instead'
                ]
            },
            'model_not_found': {
                'symptoms': ['FileNotFoundError', 'model path not found', 'No such file'],
                'solutions': [
                    '1. Check model path spelling',
                    '2. Verify model files exist',
                    '3. Re-download or re-train model',
                    '4. Check file permissions'
                ]
            },
            'slow_inference': {
                'symptoms': ['inference > 100ms', 'low FPS', 'slow prediction'],
                'solutions': [
                    '1. Use GPU if available',
                    '2. Enable model optimization',
                    '3. Use batch processing',
                    '4. Apply quantization',
                    '5. Reduce image resolution'
                ]
            },
            'poor_accuracy': {
                'symptoms': ['low accuracy', 'wrong predictions', 'confidence < 0.5'],
                'solutions': [
                    '1. Check input preprocessing',
                    '2. Verify label mappings',
                    '3. Use proper image normalization',
                    '4. Check for data distribution shift',
                    '5. Consider model fine-tuning'
                ]
            },
            'dependency_issues': {
                'symptoms': ['ImportError', 'ModuleNotFoundError', 'version conflicts'],
                'solutions': [
                    '1. Install missing packages',
                    '2. Update package versions',
                    '3. Create fresh conda environment',
                    '4. Check CUDA compatibility',
                    '5. Reinstall PyTorch with correct CUDA version'
                ]
            }
        }
    
    def diagnose_issue(self, error_message: str) -> List[str]:
        """Auto-diagnose issue based on error message"""
        error_lower = error_message.lower()
        suggested_solutions = []
        
        for issue_type, issue_info in self.common_issues.items():
            for symptom in issue_info['symptoms']:
                if symptom.lower() in error_lower:
                    suggested_solutions.extend(issue_info['solutions'])
                    break
        
        return suggested_solutions
    
    def interactive_troubleshoot(self):
        """Interactive troubleshooting session"""
        print("\n🔧 Interactive Troubleshooting Guide")
        print("=" * 40)
        
        print("\nWhat issue are you experiencing?")
        print("1. CUDA/GPU related errors")
        print("2. Model loading issues")
        print("3. Slow inference performance")
        print("4. Poor accuracy/predictions")
        print("5. Dependency/import errors")
        print("6. Custom error message")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            issue_map = {
                '1': 'cuda_out_of_memory',
                '2': 'model_not_found',
                '3': 'slow_inference',
                '4': 'poor_accuracy',
                '5': 'dependency_issues'
            }
            
            if choice in issue_map:
                issue_type = issue_map[choice]
                solutions = self.common_issues[issue_type]['solutions']
                
                print(f"\n💡 Suggested solutions for {issue_type.replace('_', ' ').title()}:")
                for solution in solutions:
                    print(f"   {solution}")
                    
            elif choice == '6':
                error_msg = input("\nPaste your error message: ").strip()
                solutions = self.diagnose_issue(error_msg)
                
                if solutions:
                    print(f"\n💡 Suggested solutions:")
                    for solution in solutions:
                        print(f"   {solution}")
                else:
                    print("\n❓ No specific solutions found. Please check:")
                    print("   1. Error message formatting")
                    print("   2. Dependencies installation")
                    print("   3. System requirements")
                    
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Troubleshooting session ended")


def create_health_check_script(model_path: str) -> str:
    """Generate automated health check script"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Automated Health Check untuk ViT Emotion Recognition Model
Generated by monitoring script
"""

import sys
import time
import json
from datetime import datetime

# Add your project path
sys.path.append('.')

def health_check():
    """Comprehensive health check"""
    results = {{
        'timestamp': datetime.now().isoformat(),
        'model_path': '{model_path}',
        'checks': {{}}
    }}
    
    try:
        # 1. Model Loading Test
        print("🔍 Testing model loading...")
        from vit_deployment_script import OptimizedViTClassifier, InferenceConfig
        
        config = InferenceConfig(model_path='{model_path}')
        classifier = OptimizedViTClassifier(config)
        results['checks']['model_loading'] = 'PASS'
        
        # 2. Inference Test
        print("🧪 Testing inference...")
        import numpy as np
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        start_time = time.time()
        result = classifier.predict_single(test_image)
        inference_time = time.time() - start_time
        
        if 'error' not in result:
            results['checks']['inference'] = 'PASS'
            results['inference_time_ms'] = inference_time * 1000
            results['prediction'] = result['predicted_emotion']
            results['confidence'] = result['confidence']
        else:
            results['checks']['inference'] = 'FAIL'
            results['error'] = result['error']
        
        # 3. Performance Check
        if inference_time > 0.5:  # 500ms threshold
            results['checks']['performance'] = 'WARNING - Slow inference'
        else:
            results['checks']['performance'] = 'PASS'
        
        # 4. Memory Check
        print("💾 Checking memory usage...")
        import psutil
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 90:
            results['checks']['memory'] = 'WARNING - High memory usage'
        else:
            results['checks']['memory'] = 'PASS'
        
        results['memory_usage_percent'] = memory_percent
        
        # Overall status
        failed_checks = [k for k, v in results['checks'].items() if 'FAIL' in v]
        warning_checks = [k for k, v in results['checks'].items() if 'WARNING' in v]
        
        if failed_checks:
            results['overall_status'] = 'FAIL'
        elif warning_checks:
            results['overall_status'] = 'WARNING'
        else:
            results['overall_status'] = 'HEALTHY'
        
        print(f"\\n📊 Health Check Results:")
        print(f"   Overall Status: {{results['overall_status']}}")
        for check, status in results['checks'].items():
            print(f"   {{check.title()}}: {{status}}")
        
        return results
        
    except Exception as e:
        results['checks']['error'] = f'FAIL - {{str(e)}}'
        results['overall_status'] = 'FAIL'
        print(f"❌ Health check failed: {{e}}")
        return results

def save_results(results, filepath='health_check_results.json'):
    """Save health check results"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {{filepath}}")

if __name__ == "__main__":
    print("🏥 ViT Model Health Check")
    print("=" * 30)
    
    results = health_check()
    save_results(results)
    
    # Exit code for automation
    if results['overall_status'] == 'HEALTHY':
        sys.exit(0)
    elif results['overall_status'] == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(2)
'''
    
    return script_content


def main():
    """Main monitoring script"""
    print("🔧 ViT Emotion Recognition - Monitoring & Troubleshooting")
    print("=" * 60)
    
    MODEL_PATH = "D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/vit-emotion-final"
    
    print("\nWhat would you like to do?")
    print("1. System requirements check")
    print("2. Model diagnostics")
    print("3. Performance monitoring")
    print("4. Interactive troubleshooting")
    print("5. Generate health check script")
    print("6. Full comprehensive check")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            print("\n🔍 Checking system requirements...")
            monitor = SystemMonitor()
            requirements = monitor.check_system_requirements()
            
            print(f"\n📋 System Information:")
            print(f"   Platform: {requirements['platform']}")
            print(f"   Python: {requirements['python_version']}")
            print(f"   CPU Cores: {requirements['cpu_count']}")
            print(f"   Total RAM: {requirements['total_memory_gb']:.1f} GB")
            print(f"   Available RAM: {requirements['available_memory_gb']:.1f} GB")
            print(f"   PyTorch: {requirements['pytorch_version']}")
            print(f"   CUDA Available: {requirements['cuda_available']}")
            
            if requirements['cuda_available']:
                print(f"   GPU: {requirements['gpu_name']}")
                print(f"   GPU Memory: {requirements['gpu_memory_gb']:.1f} GB")
            
            if requirements['issues']:
                print(f"\n⚠️ Issues Found:")
                for issue in requirements['issues']:
                    print(f"   {issue}")
            else:
                print(f"\n✅ All requirements satisfied!")
        
        elif choice == '2':
            print("\n🔍 Running model diagnostics...")
            diagnostics = ModelDiagnostics(MODEL_PATH)
            
            # Integrity check
            integrity = diagnostics.check_model_integrity()
            print(f"\n📋 Model Integrity: {integrity['overall_status']}")
            
            if integrity['issues']:
                print("❌ Issues found:")
                for issue in integrity['issues']:
                    print(f"   {issue}")
            else:
                print("✅ No integrity issues found")
            
            # Performance test
            if integrity['overall_status'] == 'PASS':
                performance = diagnostics.test_inference_performance()
                
                if 'error' not in performance:
                    print(f"\n⚡ Performance Results:")
                    print(f"   Success Rate: {performance['success_rate']:.1%}")
                    print(f"   Average Inference: {performance['avg_inference_time_ms']:.2f}ms")
                    print(f"   FPS: {performance['fps']:.1f}")
                    
                    if performance.get('avg_memory_usage_mb'):
                        print(f"   Memory per inference: {performance['avg_memory_usage_mb']:.1f}MB")
        
        elif choice == '3':
            print("\n📊 Starting performance monitoring...")
            duration = int(input("Enter monitoring duration (seconds, default 30): ") or "30")
            
            monitor = SystemMonitor()
            monitor.monitor_resources(duration)
            stats = monitor.generate_resource_report()
            
            print(f"\n📊 Performance Summary:")
            print(f"   CPU Average: {stats['cpu_avg']:.1f}%")
            print(f"   Memory Average: {stats['memory_avg']:.1f}%")
            
        elif choice == '4':
            troubleshooter = TroubleshootingGuide()
            troubleshooter.interactive_troubleshoot()
        
        elif choice == '5':
            print("\n⚙️ Generating health check script...")
            script_content = create_health_check_script(MODEL_PATH)
            
            with open('automated_health_check.py', 'w') as f:
                f.write(script_content)
            
            print("✅ Health check script generated: automated_health_check.py")
            print("💡 Run with: python automated_health_check.py")
        
        elif choice == '6':
            print("\n🔍 Running comprehensive check...")
            
            # System check
            monitor = SystemMonitor()
            requirements = monitor.check_system_requirements()
            print(f"✅ System Status: {requirements['status']}")
            
            # Model diagnostics
            diagnostics = ModelDiagnostics(MODEL_PATH)
            integrity = diagnostics.check_model_integrity()
            print(f"✅ Model Status: {integrity['overall_status']}")
            
            # Quick performance test
            if integrity['overall_status'] == 'PASS':
                performance = diagnostics.test_inference_performance(5)  # Quick test
                if 'error' not in performance:
                    print(f"✅ Performance: {performance['fps']:.1f} FPS")
            
            # Generate health check script
            script_content = create_health_check_script(MODEL_PATH)
            with open('automated_health_check.py', 'w') as f:
                f.write(script_content)
            print("✅ Health check script generated")
            
            print(f"\n🎉 Comprehensive check completed!")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n👋 Monitoring session ended")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
