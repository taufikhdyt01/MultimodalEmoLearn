# Bahan Bimbingan Tesis - Progress & Konsultasi

**Nama:** Taufik Hidayat  
**NIM:** 256150117111007  
**Judul:** Integrasi Multimodal CNN dan Facial Landmark untuk Pengenalan Emosi dalam Konteks Pembelajaran Pemrograman  
**Tanggal:** April 2025

---

## SLIDE 1: Agenda

> **Poin yang disampaikan:**  
> "Pak/Bu, hari ini saya ingin melaporkan progress preprocessing data dan meminta konsultasi terkait beberapa keputusan teknis sebelum masuk ke tahap training."

1. Progress pengumpulan dan preprocessing data
2. Statistik dataset yang dihasilkan
3. Konsultasi: Validasi ahli psikologi
4. Konsultasi: Strategi penanganan class imbalance
5. Rencana selanjutnya

---

## SLIDE 2: Data yang Terkumpul

### Total 37 Mahasiswa

| Batch | Jumlah | Sudut Kamera | Periode |
|-------|--------|-------------|---------|
| Batch 1 (lama) | 20 mahasiswa | Depan saja | April-Mei 2025 |
| Batch 2 (baru) | 17 mahasiswa | Depan + Samping | November 2025 |
| **Total** | **37 mahasiswa** | | |

> **Penjelasan lisan:**  
> "Dari target 38 mahasiswa di proposal, saya berhasil mengumpulkan 37. Satu mahasiswa (ID 204) datanya tidak lengkap sehingga tidak bisa digunakan."
>
> "Perbedaan utama batch 2 adalah penambahan sudut kamera samping (side view), sehingga satu mahasiswa menghasilkan dua set frame: depan dan samping. Ini untuk menambah variasi data dan menguji apakah side view bisa membantu pengenalan emosi."

---

## SLIDE 3: Pipeline Preprocessing

```
Video Rekaman (.mp4/.mkv)
        |
        v
[1] Ekstraksi Frame (berdasarkan timestamp emosi di database)
        |
        v
[2] Face Detection & Cropping (MediaPipe → 224x224 px)
        |
        v
[3] Landmark Extraction (68 titik → 136 fitur koordinat)
        |
        v
[4] Dataset Preparation (matching label + split train/val/test)
```

> **Penjelasan lisan:**  
> "Berbeda dengan preprocessing sebelumnya yang mengekstrak frame setiap 5 detik secara blind, sekarang saya hanya mengekstrak frame pada timestamp yang tepat sesuai data emosi yang tercatat di database. Jadi setiap frame pasti memiliki label emosi yang berkorespondensi."
>
> "Untuk face detection, saya menggunakan MediaPipe (bukan dlib seperti di proposal) karena MediaPipe mampu mendeteksi wajah dari sudut samping, sedangkan dlib hanya efektif untuk wajah frontal. Hasil landmark tetap di-map ke 68 titik standar yang setara dengan dlib, sehingga tidak mengubah desain penelitian di proposal. Hasilnya 136 fitur (68 titik × 2 koordinat x,y)."

---

## SLIDE 4: Hasil Preprocessing

### Statistik Ekstraksi Frame

| Tahap | Batch 1 | Batch 2 | Total |
|-------|---------|---------|-------|
| Frame diekstrak | 3,849 | 6,553 | 10,402 |
| Face terdeteksi | 3,824 (99.4%) | 6,070 (92.6%) | 9,894 |
| Gagal deteksi | 25 | 483 | 508 |

> **Penjelasan lisan:**  
> "Dari 10,402 frame yang diekstrak, 9,894 berhasil dideteksi wajahnya (95%). Yang gagal deteksi umumnya frame di mana mahasiswa sedang menunduk, keluar frame, atau wajah tertutup tangan. Angka ini normal dan tidak mempengaruhi kualitas dataset."

### Detail Batch 2 (dengan side view)

| Angle | Frame | Matched ke label |
|-------|-------|-----------------|
| Front | 3,275 | 3,035 |
| Side  | 3,278 | 3,035 |
| **Total** | **6,553** | **6,070** |

---

## SLIDE 5: Distribusi Emosi

| Emosi | Jumlah | Persentase |
|-------|--------|-----------|
| Neutral | 8,356 | **84.5%** |
| Happy | 783 | 7.9% |
| Sad | 576 | 5.8% |
| Surprised | 79 | 0.8% |
| Angry | 63 | 0.6% |
| Disgusted | 24 | 0.2% |
| Fearful | 13 | 0.1% |
| **Total** | **9,894** | **100%** |

> **Penjelasan lisan:**  
> "Distribusi emosi sangat imbalanced dengan neutral mendominasi 84.5%. Ini konsisten dengan literatur yang disebutkan di proposal bahwa mahasiswa cenderung menampilkan ekspresi netral saat fokus pemrograman (Coto et al., 2022 melaporkan dominasi ekspresi tertentu hingga 65%)."
>
> "Emosi fearful (13) dan disgusted (24) sangat sedikit. Ini menjadi tantangan tersendiri yang perlu kita diskusikan apakah kelas-kelas ini tetap dipertahankan atau digabung."

---

## SLIDE 6: Penanganan Class Imbalance

### Masalah

Jika model ditraining tanpa penanganan, model akan cenderung selalu memprediksi "neutral" karena itu pilihan yang "paling aman" (benar 84.5% dari waktu). Emosi langka seperti fearful, disgusted, dan angry hampir tidak akan pernah diprediksi.

### Solusi yang Sudah Disiapkan: Class Weights

Menggunakan metode **Class-Balanced Loss** (Cui et al., 2019) — memberikan bobot (penalty) berbeda pada loss function saat training:

| Emosi | Jumlah di Train | Weight | Artinya |
|-------|----------------|--------|---------|
| Neutral | 5,678 | **1.0x** | Baseline penalty |
| Happy | 751 | **1.9x** | Salah prediksi 2x lebih mahal |
| Sad | 490 | **2.6x** | Salah prediksi 3x lebih mahal |
| Surprised | 70 | **14.7x** | Salah prediksi 15x lebih mahal |
| Angry | 48 | **21.3x** | Salah prediksi 21x lebih mahal |
| Disgusted | 19 | **52.9x** | Salah prediksi 53x lebih mahal |
| Fearful | 8 | **125.0x** | Salah prediksi 125x lebih mahal |

> **Penjelasan lisan:**  
> "Cara kerjanya seperti ini: tanpa weights, kalau model salah prediksi neutral atau salah prediksi fearful, penalty-nya sama. Akibatnya model tidak termotivasi untuk belajar emosi langka. Dengan weights, kalau model salah prediksi fearful, penalty-nya 125 kali lebih besar daripada salah prediksi neutral. Ini memaksa model untuk serius belajar emosi langka juga."
>
> "Metode ini dari paper Cui et al. (2019) - 'Class-Balanced Loss Based on Effective Number of Samples' yang dipublikasikan di CVPR. Jadi secara referensi sudah kuat."

### Ilustrasi (untuk di slide):

```
Tanpa Class Weights:
  Prediksi neutral padahal fearful → penalty 1x
  Prediksi neutral padahal neutral → benar!
  → Model "malas" → prediksi neutral terus → accuracy tinggi tapi tidak berguna

Dengan Class Weights:
  Prediksi neutral padahal fearful → penalty 125x !!!
  Prediksi neutral padahal neutral → benar!
  → Model "dipaksa" belajar semua emosi → lebih balanced
```

### Status Implementasi

| Tahap | Status |
|-------|--------|
| Deteksi masalah imbalance | Selesai |
| Hitung class weights (Cui et al., 2019) | Selesai, tersimpan di `class_weights.json` |
| Apply ke training (weighted cross-entropy) | Belum, diterapkan saat training nanti |
| Metrik evaluasi: Macro F1-Score | Belum, diterapkan saat evaluasi nanti |

> **Penjelasan lisan:**  
> "Weights sudah dihitung dan siap pakai. Nanti saat training, weights ini akan dimasukkan ke loss function (weighted cross-entropy). Selain itu, untuk evaluasi saya akan menggunakan Macro F1-Score, bukan accuracy. Accuracy tidak cocok untuk data imbalanced karena model yang selalu prediksi neutral saja sudah dapat accuracy 84.5%, tapi sebenarnya tidak berguna."

### **(KONSULTASI 1)** Strategi Class Imbalance

> **Yang perlu ditanyakan:**  
> "Pak/Bu, untuk penanganan class imbalance saya sudah menyiapkan 3 skenario yang akan dibandingkan hasilnya saat training nanti."

**Skenario B1: Tanpa penanganan (baseline)**
- Training biasa tanpa class weights
- Sebagai baseline untuk dibandingkan
- Dataset: `data/dataset/` (7,064 train samples)

**Skenario B2: Dengan class weights saja**
- Weighted cross-entropy loss (Cui et al., 2019)
- Tidak mengubah data, hanya mengubah loss function
- Dataset: `data/dataset/` + `class_weights.json`

**Skenario B3: Dengan class weights + augmentasi data (sudah disiapkan)**
- Augmentasi kelas minoritas: horizontal flip, rotasi (-15/+15 derajat), brightness adjustment, dan kombinasinya
- Hanya kelas < 150 sample yang diaugmentasi, target minimal 150 per kelas
- **Hanya train set yang diaugmentasi** — val/test tetap original (supaya evaluasi fair)
- Dataset: `data/dataset_augmented/` (7,519 train samples, +455 augmented)

| Emosi | Original | Setelah Augmentasi | Ditambah |
|-------|----------|-------------------|----------|
| Neutral | 5,678 | 5,678 | - |
| Happy | 751 | 751 | - |
| Sad | 490 | 490 | - |
| Angry | 48 | **150** | +102 |
| Fearful | 8 | **150** | +142 |
| Disgusted | 19 | **150** | +131 |
| Surprised | 70 | **150** | +80 |
| **Total** | **7,064** | **7,519** | **+455** |

Evaluasi semua skenario menggunakan **Macro F1-Score** (bukan accuracy, karena accuracy bias ke kelas mayoritas).

> **Penjelasan lisan:**  
> "Saya menyiapkan 3 skenario untuk dibandingkan: tanpa penanganan sebagai baseline, dengan class weights saja, dan dengan class weights plus augmentasi. Ketiga skenario ini akan dijalankan untuk setiap model (CNN, FCNN, Late Fusion, Intermediate Fusion), sehingga kita bisa lihat mana yang paling efektif."
>
> "Augmentasi hanya dilakukan pada training set — validation dan test set tidak disentuh supaya evaluasinya fair dan merepresentasikan kondisi data real."
>
> "Apakah Bapak/Ibu setuju dengan pendekatan perbandingan 3 skenario ini?"

---

## SLIDE 7: Split Dataset

**Strategi: Split by User + Smart Rare-Emotion Distribution**

| Split | Samples | Users | Persentase |
|-------|---------|-------|-----------|
| Train | 7,064 | 29 users | 71.4% |
| Validation | 1,174 | 3 users | 11.9% |
| Test | 1,656 | 5 users | 16.7% |

### Kenapa rasionya bukan tepat 80/10/10?

Di proposal tertulis 80/10/10, namun karena split dilakukan **berdasarkan user** (bukan random per-sample), rasio tidak bisa tepat. Rasio aktual 71/12/17 masih dalam rentang standar yang lazim di penelitian:

| Rasio | Digunakan di |
|-------|-------------|
| 80/10/10 | Paling umum (split random) |
| **70/15/15** | **Umum untuk user-based split** |
| 70/10/20 | Dataset dengan test set lebih besar |
| 60/20/20 | Dataset kecil |

> **Penjelasan lisan:**  
> "Di proposal saya tulis 80/10/10, tapi karena split dilakukan per-user untuk mencegah data leaking, rasionya menjadi 71/12/17. Ini masih dalam rentang standar 70/10-15/15-20 yang lazim di penelitian deep learning. Lebih baik sedikit bergeser rasio daripada mencampur data user yang sama di train dan test, karena itu bisa menyebabkan model menghafal wajah, bukan belajar mengenali emosi."
>
> "Di tesis nanti akan saya tuliskan: 'Pembagian data dilakukan berdasarkan user (user-level split) untuk mencegah data leaking, menghasilkan rasio train/validation/test sebesar 71.4%/11.9%/16.7% dari total 9,894 sampel.'"

### Distribusi emosi per split (semua 7 emosi ada di semua split):

| Emosi | Train | Val | Test |
|-------|-------|-----|------|
| Neutral | 5,678 | 1,090 | 1,588 |
| Happy | 575 | 22 | 10 |
| Sad | 402 | 48 | 38 |
| Angry | 43 | 2 | 13 |
| Fearful | 7 | 4 | 1 |
| Disgusted | 19 | 2 | 3 |
| Surprised | 39 | 6 | 3 |

> **Penjelasan lisan (lanjutan):**  
> "Tantangan utama: karena split by user, emosi langka (fearful hanya 13 total, disgusted hanya 24 total) bisa saja tidak muncul di salah satu split. Untuk mengatasinya, saya menggunakan algoritma smart split yang memastikan user yang memiliki emosi langka tersebar ke semua split. Hasilnya semua 7 emosi terwakili di train, validation, dan test."
>
> "Ini penting karena kalau misalnya fearful tidak ada di test set, kita tidak bisa mengevaluasi kemampuan model mengenali emosi fearful."

---

## SLIDE 7: Validasi Ahli Psikologi

> **Penjelasan lisan:**  
> "Sesuai proposal, label emosi saat ini sepenuhnya dari auto-detection (Face API). Untuk dual-validation, kita perlu ahli psikologi untuk memvalidasi sebagian sample."

### **(KONSULTASI 2)** Berapa Sample untuk Validasi?

Saya sudah menyiapkan 3 opsi:

| Opsi | Total Sample | Strategi | Beban Ahli |
|------|-------------|----------|-----------|
| **A** | 1,938 | Semua non-neutral + 400 neutral | Berat (~8 jam) |
| **B** | 1,067 | Stratified 10% per kelas | Sedang (~4 jam) |
| **C** | 583 | Stratified 5% per kelas | Ringan (~2 jam) |

#### Detail per opsi:

**Opsi A (1,938 sample):**
- Semua emosi non-neutral divalidasi 100%
- Paling kuat secara metodologi
- Justifikasi: "Seluruh sample non-neutral divalidasi oleh ahli psikologi"

**Opsi B (1,067 sample, 10% stratified):**
- Proporsional 10% dari tiap kelas, minimum 30 per kelas
- Distribusi validasi merepresentasikan distribusi dataset
- Justifikasi: "10% stratified random sampling dengan minimum 30 sampel per kelas"

**Opsi C (583 sample, 5% stratified):**
- Proporsional 5%, minimum 30 per kelas
- Masih memenuhi syarat statistik untuk Cohen's Kappa
- Justifikasi: "5% stratified random sampling, memenuhi minimum statistical requirement"

> **Yang perlu ditanyakan:**  
> 1. "Opsi mana yang paling tepat untuk level tesis S2?"
> 2. "Apakah 1 ahli cukup, atau perlu 2 ahli untuk inter-rater reliability?"
> 3. "Apakah Bapak/Ibu punya rekomendasi ahli psikologi yang bisa dihubungi?"

### Format Validasi yang Sudah Disiapkan:
- File Excel dengan kolom: gambar wajah, label otomatis, confidence score, 7 skor emosi
- Kolom kosong "Expert Label" dan "Expert Notes" untuk diisi ahli
- Folder berisi 224x224 cropped face images
- Petunjuk pengisian sudah disertakan

---

## SLIDE 8: Perubahan dari Proposal

| Aspek | Di Proposal | Implementasi | Alasan |
|-------|------------|-------------|--------|
| Face detection | dlib | **MediaPipe** | Dlib gagal deteksi side view, MediaPipe support multi-angle |
| Landmark | 68 titik (dlib) | **68 titik (mapped dari MediaPipe)** | Setara, output tetap 136 fitur |
| Jumlah mahasiswa | 38 | **37** | 1 mahasiswa data tidak lengkap |
| Fusion strategy | Late + Intermediate | Late + Intermediate (Hybrid) | Tetap sesuai proposal |

> **Penjelasan lisan:**  
> "Perubahan utama hanya pada tool face detection dari dlib ke MediaPipe. Ini karena data baru memiliki side view yang tidak bisa dideteksi oleh dlib. Output landmark tetap 68 titik yang di-map dari 478 titik MediaPipe, sehingga secara substansi tidak mengubah desain penelitian."

### **(KONSULTASI 3)** Apakah Perubahan Ini Perlu Direvisi di Proposal?

> "Pak/Bu, apakah perubahan dari dlib ke MediaPipe ini perlu direvisi di dokumen proposal, atau cukup dijelaskan di BAB 4 (Implementasi) saja?"

---

## SLIDE 9: Rencana Selanjutnya

| No | Tahap | Status | Target |
|----|-------|--------|--------|
| 1 | Pengumpulan data | Selesai | - |
| 2 | Preprocessing (frame, crop, landmark) | Selesai | - |
| 3 | Prepare dataset (numpy arrays) | Selesai | - |
| 4 | **Validasi ahli psikologi** | **Menunggu keputusan** | Setelah bimbingan ini |
| 5 | Training CNN (fitur penampilan) | Belum | Setelah validasi |
| 6 | Training FCNN (fitur landmark) | Belum | Setelah validasi |
| 7 | Late Fusion | Belum | Setelah unimodal |
| 8 | Intermediate Fusion | Belum | Setelah unimodal |
| 9 | Evaluasi & perbandingan | Belum | Setelah semua training |

> **Penjelasan lisan:**  
> "Preprocessing sudah selesai semua. Yang menentukan langkah selanjutnya adalah keputusan validasi ahli. Namun sambil menunggu, saya bisa mulai training untuk melihat baseline performance."
>
> "Apakah Bapak/Ibu setuju saya mulai training dulu dengan label auto-detection, lalu nanti update model setelah validasi ahli selesai?"

---

## Ringkasan Poin Konsultasi

| No | Topik | Pertanyaan | Opsi Rekomendasi |
|----|-------|-----------|------------------|
| 1 | Class imbalance | Cukup class weights saja, atau perlu augmentasi/gabung kelas? | Class weights (Opsi A) |
| 2 | Validasi ahli | Berapa sample? (583 / 1,067 / 1,938) dan berapa ahli? | Tergantung ketersediaan ahli |
| 3 | Perubahan tool | Perlu revisi proposal untuk perubahan dlib → MediaPipe? | Cukup di BAB 4 |
| 4 | Timeline | Boleh training dulu sambil menunggu validasi? | Ya, sebagai baseline |

---

## Lampiran A: Struktur Project Saat Ini

```
MultimodalEmoLearn/
├── src/
│   ├── preprocessing/
│   │   ├── prepare_dataset.py          # Gabung data + split + numpy
│   │   └── generate_validation_set.py  # Generate set validasi ahli
│   └── utils/
│       ├── batch_video_processor.py    # Ekstrak frame dari video
│       ├── face_crop_landmark.py       # Face crop + 68 landmark
│       └── generate_emotion_label.py   # Generate label emosi
├── data/
│   ├── final/                          # Cropped faces 224x224 + landmarks
│   │   ├── old/{user_id}/front/        # 20 user lama
│   │   └── new/{user_id}/{front,side}/ # 17 user baru
│   ├── dataset/                        # Numpy arrays siap training
│   │   ├── X_train_images.npy (4.25 GB)
│   │   ├── X_train_landmarks.npy
│   │   ├── y_train.npy
│   │   ├── class_weights.json          # Bobot per kelas emosi
│   │   ├── dataset_info.json           # Metadata dataset
│   │   └── ... (val + test)
│   ├── validation_full_1938/           # Opsi A validasi
│   ├── validation_stratified_10pct/    # Opsi B validasi
│   └── validation_stratified_5pct/     # Opsi C validasi
└── docs/
    └── bimbingan_progress.md           # File ini
```

---

## Lampiran B: Referensi Metode yang Digunakan

| Metode | Referensi | Digunakan untuk |
|--------|-----------|----------------|
| Class-Balanced Loss | Cui et al. (2019), CVPR | Penanganan class imbalance via weighted loss |
| MediaPipe Face Mesh | Lugaresi et al. (2019), Google | Face detection + 468 landmark extraction |
| 68-point landmark mapping | Sagonas et al. (2016) | Standar landmark untuk FER (di-map dari MediaPipe) |
| Macro F1-Score | Sokolova & Lapalme (2009) | Evaluasi yang fair untuk data imbalanced |
| User-based split | - | Mencegah data leaking antar split |
