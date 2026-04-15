# Panduan Training di Linux (NVIDIA T4 - Biznet Gio VPS)

> **Server:** Biznet Gio VPS dengan NVIDIA T4
> **Akses:** MobaXterm (SSH + file transfer)

## 1. Transfer Project ke VPS

### Option A: Git clone di VPS (RECOMMENDED)

Buka MobaXterm → SSH ke VPS → jalankan:
```bash
git clone https://github.com/taufikhdyt01/MultimodalEmoLearn.git
cd MultimodalEmoLearn
```

### Option B: Upload via MobaXterm file browser

MobaXterm punya panel file browser di sebelah kiri saat SSH session aktif.
Cukup **drag & drop** folder project dari Windows ke panel tersebut.

## 2. Transfer Data ke VPS

Data tidak ada di git (di-gitignore, terlalu besar ~10GB). Harus transfer manual.

### Langkah 1: Compress data di Windows

Buka Git Bash / terminal di folder project:
```bash
cd D:/MultimodalEmoLearn
tar -czf dataset.tar.gz data/dataset data/dataset_augmented
# Hasilnya: dataset.tar.gz (~3-4 GB setelah compress)
```

### Langkah 2: Upload ke VPS

**Cara A: Drag & drop via MobaXterm (paling mudah)**
1. Buka MobaXterm → SSH ke VPS
2. Di panel kiri (file browser), navigasi ke `/home/USER/MultimodalEmoLearn/`
3. Drag file `dataset.tar.gz` dari Windows Explorer ke panel kiri MobaXterm
4. Tunggu upload selesai

**Cara B: Via terminal MobaXterm**

MobaXterm punya terminal lokal bawaan yang sudah support SCP:
```bash
scp D:/MultimodalEmoLearn/dataset.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/
```

> **Tips:** Upload ~3-4 GB bisa memakan waktu tergantung kecepatan internet.

### Langkah 3: Extract di VPS

Di terminal SSH MobaXterm:
```bash
cd MultimodalEmoLearn
tar -xzf dataset.tar.gz

# Verifikasi
ls data/dataset/*.npy | wc -l           # harus 9 files
ls data/dataset_augmented/*.npy | wc -l  # harus 9 files

# Hapus file compress (opsional, hemat disk)
rm dataset.tar.gz
```

### File yang dibutuhkan:
```
data/
├── dataset/
│   ├── X_train_images.npy      (~4.2 GB)
│   ├── X_train_landmarks.npy   (~3.8 MB)
│   ├── y_train.npy             (~28 KB)
│   ├── X_val_images.npy        (~0.7 GB)
│   ├── X_val_landmarks.npy     (~0.6 MB)
│   ├── y_val.npy               (~5 KB)
│   ├── X_test_images.npy       (~1.0 GB)
│   ├── X_test_landmarks.npy    (~0.9 MB)
│   ├── y_test.npy              (~7 KB)
│   ├── class_weights.json
│   ├── dataset_info.json
│   └── label_map.json
└── dataset_augmented/
    ├── X_train_images.npy      (~4.5 GB)
    ├── X_train_landmarks.npy   (~4.1 MB)
    ├── y_train.npy             (~30 KB)
    ├── class_weights.json
    └── ... (val/test di-copy dari dataset/)
```

## 3. Setup Environment di VPS

Semua perintah di bawah dijalankan di VPS (setelah `ssh USER@IP_VPS`):

```bash
# 1. Install Miniconda (jika belum ada)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Ikuti instruksi, jawab "yes" untuk init conda
source ~/.bashrc

# 2. Buat conda environment
conda create -n emotrain python=3.10 -y
conda activate emotrain

# 3. Install PyTorch dengan CUDA (untuk NVIDIA T4)
# Cek dulu versi CUDA di VPS:
nvidia-smi
# Lihat "CUDA Version" di pojok kanan atas output

# Untuk CUDA 12.x:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Untuk CUDA 11.x:
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install dependencies lainnya
pip install numpy scikit-learn matplotlib seaborn jupyter openpyxl

# 5. Verifikasi GPU terdeteksi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected output:
# CUDA: True
# GPU: Tesla T4
```

## 4. Verifikasi Setup

```bash
cd MultimodalEmoLearn

# Cek data ada
python -c "
import numpy as np
from pathlib import Path

for name in ['dataset', 'dataset_augmented']:
    d = Path(f'data/{name}')
    for f in sorted(d.glob('*.npy')):
        arr = np.load(f)
        print(f'{name}/{f.name}: shape={arr.shape}, dtype={arr.dtype}')
"

# Cek model bisa dibuat
python -c "
import sys; sys.path.insert(0, 'src')
from training.models import EmotionCNN, EmotionFCNN, IntermediateFusion
import torch

device = torch.device('cuda')
m1 = EmotionCNN(7).to(device)
m2 = EmotionFCNN(136, 7).to(device)
m3 = IntermediateFusion(7, 136).to(device)

# Test forward pass
img = torch.randn(2, 3, 224, 224).to(device)
lm = torch.randn(2, 136).to(device)

print('CNN:', m1(img).shape)          # [2, 7]
print('FCNN:', m2(lm).shape)          # [2, 7]
print('Fusion:', m3(img, lm).shape)   # [2, 7]
print('All models OK!')
"
```

## 5. Jalankan Training

### Opsi A: Notebook interaktif via MobaXterm (RECOMMENDED)

Jalankan Jupyter di VPS, akses dari browser di laptop:

```bash
# Di MobaXterm, buka SSH session ke VPS, lalu jalankan:
conda activate emotrain
cd MultimodalEmoLearn
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

MobaXterm otomatis membuat SSH tunnel. Buka browser di laptop: **http://localhost:8888**

> Jika port 8888 tidak ter-tunnel otomatis, buat manual:
> MobaXterm → menu Tunneling → New SSH tunnel → Local port 8888 → Remote port 8888

Jalankan notebook secara berurutan:
- **7-class:** `01` → `02` → `03` → `04` → `05`
- **4-class:** `06` → `07` → `08` → `09` → `10`

### Opsi B: Jalankan semua di background (RECOMMENDED kalau mau ditinggal)

Koneksi SSH bisa putus kapan saja tanpa mempengaruhi training:

```bash
# Di MobaXterm SSH session:

# Pakai tmux supaya proses tidak mati kalau koneksi putus
tmux new -s training

# Di dalam tmux:
conda activate emotrain
cd MultimodalEmoLearn
bash scripts/run_all.sh

# Setelah jalan, DETACH dari tmux:
# Tekan: Ctrl+B, lalu tekan D
# Sekarang bisa tutup MobaXterm, training tetap jalan di VPS

# Nanti kalau mau cek progress, buka MobaXterm lagi:
tmux attach -t training
```

### Opsi C: Jalankan satu-satu manual

```bash
conda activate emotrain
cd MultimodalEmoLearn

# Jalankan per notebook:
jupyter nbconvert --to notebook --execute notebooks/01_train_cnn.ipynb \
    --output 01_train_cnn_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200
# Ulangi untuk 02-10
```

## 6. Estimasi Waktu Training 7-Class (NVIDIA T4)

| Notebook | Model | Estimasi per Skenario | Total (3 skenario) |
|----------|-------|-----------------------|---------------------|
| 01_train_cnn | CNN | ~15-30 menit | ~45-90 menit |
| 02_train_fcnn | FCNN | ~2-5 menit | ~6-15 menit |
| 03_late_fusion | Late Fusion | ~1-2 menit (inference only) | ~3-6 menit |
| 04_intermediate_fusion | Intermediate | ~20-40 menit | ~60-120 menit |
| 05_comparison | - | ~1 menit | ~1 menit |
| **Total** | | | **~2-4 jam** |

## 7. Setelah Training Selesai

### Transfer hasil dari VPS ke laptop Windows:

**Cara A: Drag & drop via MobaXterm**
1. Di MobaXterm SSH session, navigasi panel kiri ke `MultimodalEmoLearn/models/`
2. Select semua folder → drag ke Windows Explorer

**Cara B: Compress lalu download**

Di VPS:
```bash
cd MultimodalEmoLearn
tar -czf results.tar.gz models/ notebooks/results/
```

Lalu download `results.tar.gz` via panel kiri MobaXterm, atau dari Git Bash:
```bash
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn
tar -xzf results.tar.gz
```

### File hasil yang dihasilkan:
```
models/
├── cnn/
│   ├── cnn_b1_baseline.pth
│   ├── cnn_b2_weighted.pth
│   ├── cnn_b3_augmented.pth
│   └── cnn_results.json
├── fcnn/
│   ├── fcnn_b1_baseline.pth
│   ├── fcnn_b2_weighted.pth
│   ├── fcnn_b3_augmented.pth
│   └── fcnn_results.json
├── late_fusion/
│   └── late_fusion_results.json
├── intermediate_fusion/
│   ├── intermediate_b1_baseline.pth
│   ├── intermediate_b2_weighted.pth
│   ├── intermediate_b3_augmented.pth
│   └── intermediate_fusion_results.json
└── experiment_summary.json
```

---

## 8. Menjalankan Eksperimen 4-Class (Lanjutan)

> **Prasyarat:** Notebook 01-05 (7-class) sudah selesai dijalankan.

Setelah hasil 7-class dianalisis, langkah selanjutnya adalah menjalankan eksperimen dengan 4 kelas emosi sebagai perbandingan:
- **neutral, happy, sad, negative** (angry+fearful+disgusted+surprised digabung)

### Langkah 1: Pull kode terbaru di VPS

```bash
ssh USER@IP_VPS   # atau buka MobaXterm
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Generate dataset 4-class

Dataset 4-class dibuat dari dataset 7-class yang sudah ada (hanya remap label, tidak perlu transfer data baru):

```bash
conda activate emotrain
python src/preprocessing/prepare_dataset_4class.py
```

### Langkah 3: Jalankan training 4-class

```bash
# Pakai tmux supaya aman kalau koneksi putus
tmux new -s training4class

conda activate emotrain
bash scripts/run_4class.sh

# Detach: Ctrl+B lalu D
# Re-attach: tmux attach -t training4class
```

Atau jalankan interaktif via Jupyter (notebook 06 → 07 → 08 → 09 → 10).

### Estimasi waktu 4-class (NVIDIA T4):

| Notebook | Model | Total (3 skenario) |
|----------|-------|---------------------|
| 06_train_cnn_4class | CNN | ~45-90 menit |
| 07_train_fcnn_4class | FCNN | ~6-15 menit |
| 08_late_fusion_4class | Late Fusion | ~3-6 menit |
| 09_intermediate_fusion_4class | Intermediate | ~60-120 menit |
| 10_comparison_4class | Comparison | ~1 menit |
| **Total** | | **~2-4 jam** |

### Langkah 4: Transfer hasil 4-class

```bash
# Di VPS:
cd MultimodalEmoLearn
tar -czf results_4class.tar.gz models/4class/ notebooks/results/
```

Download via MobaXterm (drag & drop) atau:
```bash
# Di laptop:
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results_4class.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn && tar -xzf results_4class.tar.gz
```

### File hasil 4-class:
```
models/4class/
├── cnn/
│   ├── cnn_4c_b1_baseline.pth
│   ├── cnn_4c_b2_weighted.pth
│   ├── cnn_4c_b3_augmented.pth
│   └── cnn_4class_results.json
├── fcnn/
│   ├── fcnn_4c_b1_baseline.pth
│   ├── fcnn_4c_b2_weighted.pth
│   ├── fcnn_4c_b3_augmented.pth
│   └── fcnn_4class_results.json
├── late_fusion/
│   └── late_fusion_4class_results.json
├── intermediate_fusion/
│   ├── intermediate_4c_b1_baseline.pth
│   ├── intermediate_4c_b2_weighted.pth
│   ├── intermediate_4c_b3_augmented.pth
│   └── intermediate_fusion_4class_results.json
└── experiment_summary_4class.json
```

---

## 9. Menjalankan Transfer Learning (Lanjutan)

> **Prasyarat:** Notebook 01-10 (from scratch) sudah selesai dijalankan.

Transfer Learning menggunakan ResNet18 pretrained ImageNet untuk menggantikan CNN from scratch.

### Langkah 1: Pull kode terbaru

```bash
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Jalankan

```bash
tmux new -s transfer

conda activate emotrain
bash scripts/run_transfer.sh

# Detach: Ctrl+B lalu D
```

Notebook yang dijalankan: `11` → `12` → `13` → `14` → `15` → `16` → `17`

### Estimasi waktu Transfer Learning (NVIDIA T4):

| Notebook | Model | Total (3 skenario) |
|----------|-------|---------------------|
| 11_train_cnn_transfer | CNN TL 7-class | ~30-60 menit |
| 12_late_fusion_transfer | Late Fusion TL 7-class | ~3-6 menit |
| 13_intermediate_fusion_transfer | Intermediate TL 7-class | ~40-80 menit |
| 14_train_cnn_transfer_4class | CNN TL 4-class | ~30-60 menit |
| 15_late_fusion_transfer_4class | Late Fusion TL 4-class | ~3-6 menit |
| 16_intermediate_fusion_transfer_4class | Intermediate TL 4-class | ~40-80 menit |
| 17_comparison_all | Final comparison | ~1 menit |
| **Total** | | **~2.5-5 jam** |

---

## 10. Menjalankan Eksperimen Front-Only (Lanjutan)

> **Prasyarat:** Notebook 01-17 (front+side) sudah selesai dijalankan.

Eksperimen front-only menggunakan **hanya data sudut depan** dari kedua batch, untuk konsistensi karena batch 1 hanya memiliki sudut depan sedangkan batch 2 memiliki depan+samping.

### Langkah 1: Pull kode terbaru

```bash
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Transfer dataset front-only ke VPS

Dataset front-only sudah di-generate di laptop. Cukup upload **base dataset saja** (~4 GB), sisanya di-generate di VPS.

**Di laptop (Git Bash / terminal):**
```bash
cd D:/MultimodalEmoLearn
tar -czf dataset_frontonly.tar.gz data/dataset_frontonly/
# Hasilnya: dataset_frontonly.tar.gz (~2-3 GB setelah compress)
```

**Upload ke VPS** via MobaXterm (drag & drop) atau:
```bash
scp dataset_frontonly.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/
```

**Di VPS — extract dan generate dataset turunan:**
```bash
cd MultimodalEmoLearn
tar -xzf dataset_frontonly.tar.gz
rm dataset_frontonly.tar.gz  # hemat disk

# Verify
ls data/dataset_frontonly/*.npy | wc -l  # harus 9+ files

# Generate augmented + 4-class (dari base dataset, ~5 menit)
conda activate emotrain
python scripts/prepare_frontonly_all.py
```

Ini akan menghasilkan 4 dataset:
```
data/
├── dataset_frontonly/              # 7-class front-only (7,091 sampel) — dari upload
├── dataset_frontonly_augmented/    # 7-class + augmentasi kelas minoritas — di-generate
├── dataset_frontonly_4class/       # 4-class front-only — di-generate
└── dataset_frontonly_4class_augmented/  # 4-class + augmentasi — di-generate
```

Ini akan menghasilkan 4 dataset:
```
data/
├── dataset_frontonly/              # 7-class front-only (7,091 sampel)
├── dataset_frontonly_augmented/    # 7-class + augmentasi kelas minoritas
├── dataset_frontonly_4class/       # 4-class front-only
└── dataset_frontonly_4class_augmented/  # 4-class + augmentasi
```

### Langkah 3: Jalankan training front-only

```bash
tmux new -s frontonly

conda activate emotrain

# From scratch 7-class (notebook 18-21)
bash scripts/run_frontonly_7class.sh

# From scratch 4-class (notebook 22-25)
bash scripts/run_frontonly_4class.sh

# Transfer Learning (notebook 26-31)
bash scripts/run_frontonly_transfer.sh

# Comparison (notebook 32)
jupyter nbconvert --to notebook --execute notebooks/32_comparison_frontonly_vs_original.ipynb \
    --output 32_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200

# Detach: Ctrl+B lalu D
```

Atau jalankan interaktif via Jupyter (notebook 18 → 19 → ... → 32).

### Notebook front-only:

| No | Notebook | Model | Kelas |
|----|----------|-------|-------|
| **From Scratch** |||
| 18 | `cnn_frontonly_7class` | CNN | 7 |
| 19 | `fcnn_frontonly_7class` | FCNN | 7 |
| 20 | `late_fusion_frontonly_7class` | Late Fusion | 7 |
| 21 | `intermediate_frontonly_7class` | Intermediate | 7 |
| 22 | `cnn_frontonly_4class` | CNN | 4 |
| 23 | `fcnn_frontonly_4class` | FCNN | 4 |
| 24 | `late_fusion_frontonly_4class` | Late Fusion | 4 |
| 25 | `intermediate_frontonly_4class` | Intermediate | 4 |
| **Transfer Learning** |||
| 26 | `cnn_tl_frontonly_7class` | CNN TL (ResNet18) | 7 |
| 27 | `late_fusion_tl_frontonly_7class` | Late Fusion TL | 7 |
| 28 | `intermediate_tl_frontonly_7class` | Intermediate TL | 7 |
| 29 | `cnn_tl_frontonly_4class` | CNN TL (ResNet18) | 4 |
| 30 | `late_fusion_tl_frontonly_4class` | Late Fusion TL | 4 |
| 31 | `intermediate_tl_frontonly_4class` | Intermediate TL | 4 |
| **Perbandingan** |||
| 32 | `comparison_frontonly_vs_original` | Semua | - |

### Estimasi waktu front-only (NVIDIA T4):

| Tahap | Notebook | Estimasi |
|-------|----------|----------|
| From scratch 7-class | 18-21 | ~2-4 jam |
| From scratch 4-class | 22-25 | ~2-4 jam |
| Transfer Learning | 26-31 | ~2.5-5 jam |
| Comparison | 32 | ~1 menit |
| **Total** | | **~6.5-13 jam** |

### Langkah 4: Transfer hasil

```bash
# Di VPS:
cd MultimodalEmoLearn
tar -czf results_frontonly.tar.gz models/frontonly/ notebooks/results/
```

Download via MobaXterm atau:
```bash
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results_frontonly.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn && tar -xzf results_frontonly.tar.gz
```

### File hasil front-only:
```
models/frontonly/
├── 7class/
│   ├── cnn_b1.pth, cnn_b2.pth, cnn_b3.pth
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
│   ├── intermediate_b1.pth, intermediate_b2.pth, intermediate_b3.pth
│   └── *_results.json
├── 4class/
│   └── (sama seperti 7class)
├── 7class_tl/
│   ├── cnn_tl_b1.pth, cnn_tl_b2.pth, cnn_tl_b3.pth
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
│   ├── intermediate_tl_b1.pth, ...
│   └── *_results.json
├── 4class_tl/
│   └── (sama seperti 7class_tl)
└── results_transfer_frontonly.json
```

---

## 11. LOSO Cross-Validation (Lanjutan)

LOSO (Leave-One-Subject-Out) mengevaluasi robustness model — setiap user jadi test set 1x.

### Prerequisite: Generate user_ids (jika belum ada)

```bash
# Jalankan di LAPTOP (butuh data/processed & data/final), lalu upload hasilnya
python scripts/generate_user_ids.py
# Upload file-file kecil ini ke VPS:
#   data/dataset_frontonly/user_ids_all.npy (~28 KB)
#   data/dataset_frontonly/y_all.npy (~28 KB)
#   data/dataset_frontonly/user_ids_train.npy
#   data/dataset_frontonly/user_ids_val.npy
#   data/dataset_frontonly/user_ids_test.npy
```

### Jalankan LOSO

```bash
tmux new -s loso
conda activate emotrain
cd MultimodalEmoLearn

# Jalankan via shell script (otomatis execute notebook 33)
bash scripts/run_loso.sh
```

**Estimasi:** ~8-15 jam untuk 3 model × 37 fold.

> **Status (10 Apr 2026):** Baru 1 model (intermediate_tl) yang selesai LOSO (34/37 fold).
> Late fusion dan FCNN belum dijalankan karena waktu terbatas.
> Hasil intermediate_tl sudah tersimpan di `models/frontonly/loso/loso_intermediate_tl_4class.json`.
> Untuk melanjutkan, jalankan ulang `bash scripts/run_loso.sh` — model yang sudah selesai akan di-overwrite,
> atau modifikasi notebook 33 agar skip model yang sudah ada.

### Output LOSO:
```
models/frontonly/loso/
├── loso_intermediate_tl_4class.json   # ✅ Selesai (34/37 fold)
├── loso_late_fusion_4class.json       # ❌ Belum dijalankan
├── loso_fcnn_4class.json              # ❌ Belum dijalankan
└── loso_comparison.png                # Bar chart per fold

notebooks/results/
└── 33_loso_frontonly_executed.ipynb    # Executed notebook dengan chart
```

---

## 12. 5-Fold Cross-Validation (Lanjutan)

5-Fold CV subject-wise — user dikelompokkan ke 5 grup, rotasi test set.
Lebih cepat dari LOSO (~2-4 jam vs ~8-15 jam).

```bash
tmux new -s crossval
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_crossval.sh
```

### Output 5-Fold CV:
```
models/frontonly/crossval/
├── cv5_intermediate_tl_4class.json
├── cv5_late_fusion_4class.json
├── cv5_fcnn_4class.json
└── cv5_comparison.png

notebooks/results/
└── 34_crossval_frontonly_executed.ipynb
```

---

## 13. Random Split — Baseline Comparison (Lanjutan)

Random split sebagai baseline untuk menunjukkan efek data leakage.
Sampel diacak tanpa memperhatikan user → user yang sama bisa di train & test.

```bash
tmux new -s randomsplit
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_randomsplit.sh
```

**Estimasi:** ~30-60 menit (5 repeats × 3 model).

### Output Random Split:
```
models/frontonly/randomsplit/
├── random_intermediate_tl_4class.json
├── random_late_fusion_4class.json
├── random_fcnn_4class.json
└── split_strategy_comparison.png      # Grouped bar chart semua strategi

notebooks/results/
└── 35_randomsplit_frontonly_executed.ipynb
```

### Ringkasan Semua Strategi Split:

| No | Notebook | Strategi | Fold/Repeat | Estimasi |
|----|----------|----------|:-----------:|----------|
| 33 | `loso_frontonly` | LOSO (37 fold) | 37 × 3 model | ~8-15 jam |
| 34 | `crossval_frontonly` | 5-Fold CV | 5 × 3 model | ~2-4 jam |
| 35 | `randomsplit_frontonly` | Random Split | 5 × 3 model | ~30-60 menit |

---

## 14. Benchmark JAFFE & CK+ (Lanjutan)

Benchmark menggunakan dataset standar JAFFE dan CK+ dengan skenario yang sama.
Hanya B1 (Baseline) karena kedua dataset relatif seimbang.

### Prerequisite: Upload dataset benchmark ke VPS

```bash
# Di laptop: compress
cd D:/MultimodalEmoLearn
tar -czf benchmark_data.tar.gz data/benchmark/jaffe_7class data/benchmark/jaffe_4class \
    data/benchmark/ckplus_7class data/benchmark/ckplus_4class data/benchmark/ckplus_4class_contempt

# Upload ke VPS via SCP
scp benchmark_data.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/

# Di VPS: extract
cd MultimodalEmoLearn
tar -xzf benchmark_data.tar.gz
rm benchmark_data.tar.gz
```

### Jalankan Benchmark

```bash
tmux new -s benchmark
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_benchmark.sh
```

**Estimasi:** ~1-2 jam total (12 eksperimen per dataset × 2 dataset).

### Eksperimen per dataset:

| Model | Kelas | Total |
|-------|:-----:|:-----:|
| CNN | 7 + 4 | 2 |
| FCNN | 7 + 4 | 2 |
| Late Fusion | 7 + 4 | 2 |
| Intermediate | 7 + 4 | 2 |
| CNN TL | 7 + 4 | 2 |
| Intermediate TL | 7 + 4 | 2 |
| **Total per dataset** | | **12** |

### Output Benchmark:
```
models/benchmark/
├── jaffe/
│   ├── jaffe_7c_results.json
│   └── jaffe_4c_results.json
└── ckplus/
    ├── ckplus_7c_results.json
    └── ckplus_4c_results.json

notebooks/results/
├── 36_benchmark_jaffe_executed.ipynb
└── 37_benchmark_ckplus_executed.ipynb
```

### Benchmark LOSO & 10-Fold CV

Setelah single split selesai, jalankan LOSO (JAFFE) dan 10-fold CV (CK+):

```bash
tmux new -s benchmark_cv
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_benchmark_cv.sh
```

**Estimasi:** ~8-13 jam total (120 training runs per dataset).

### Output:
```
models/benchmark/
├── jaffe_loso/
│   ├── jaffe_7c_loso_results.json
│   └── jaffe_4c_loso_results.json
└── ckplus_cv10/
    ├── ckplus_7c_cv10_results.json
    └── ckplus_4c_cv10_results.json

notebooks/results/
├── 38_benchmark_jaffe_loso_executed.ipynb
└── 39_benchmark_ckplus_cv10_executed.ipynb
```

---

## 15. Improved Experiments: Focal Loss + FER2013 Pre-Training

### Step 1: Download & Prepare FER2013 di VPS

```bash
# Install kaggle CLI (jika belum)
pip install kaggle

# Copy kaggle.json dari laptop ke VPS
# Di laptop:
scp C:/Users/grinv/.kaggle/kaggle.json USER@IP_VPS:~/.kaggle/

# Di VPS:
chmod 600 ~/.kaggle/kaggle.json
cd MultimodalEmoLearn

# Download FER2013
python -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.dataset_download_files('msambare/fer2013', path='data/benchmark/fer2013/', unzip=True)
print('Done!')
"

# Prepare (resize 48->224, grayscale->RGB)
python scripts/prepare_fer2013.py
```

### Step 2: Run Improved Experiments

```bash
tmux new -s improved
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_improved.sh
```

**Estimasi:** ~3-5 jam (pre-train FER2013 + 8 experiments).

### Output:
```
models/pretrained/resnet18_fer2013.pth    # FER2013 pre-trained weights
models/frontonly/improved/
├── CNN_TL_FocalLoss.pth
├── CNN_FER2013_Focal.pth
├── IntermediateTL_FER2013_Focal.pth
├── ...
└── improved_results.json
```

---

## 16. Undersampling Neutral (Lanjutan)

Mengurangi dominasi neutral (78% train) untuk meningkatkan deteksi kelas minoritas.

### Generate dataset undersampled (jika belum)
```bash
python scripts/prepare_undersampled.py
```

Menghasilkan 3 variasi:
```
data/dataset_frontonly_under_660_4class/   # neutral=660 (rasio 5.8:1)
data/dataset_frontonly_under_382_4class/   # neutral=382 (rasio 3.4:1)
data/dataset_frontonly_under_114_4class/   # neutral=114 (rasio 1:1)
```

### Jalankan eksperimen
```bash
tmux new -s undersample
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_undersampled.sh
```

**Estimasi:** ~2-3 jam (3 model x 4 dataset = 12 eksperimen).

### Output:
```
models/frontonly/undersampled/
├── IntTL_*.pth, FCNN_*.pth, LateFusion_*.pth
├── undersampled_results.json        # per-class F1 scores
└── undersampled_perclass.png        # comparison chart

notebooks/results/
└── 42_frontonly_undersampled_executed.ipynb
```

---

## 17. Troubleshooting

### CUDA Out of Memory
```python
# Kurangi batch size di notebook:
BATCH_SIZE = 16  # atau 8 untuk intermediate fusion
```

### Training terlalu lama
```python
# Kurangi epochs dan patience:
EPOCHS = 30
PATIENCE = 10
```

### Notebook error saat import
```bash
# Pastikan working directory benar:
cd MultimodalEmoLearn
# Pastikan src/ ada di path:
ls src/training/models.py  # harus ada
```
