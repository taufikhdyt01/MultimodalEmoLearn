# Bahan Bimbingan Tesis - Progress & Konsultasi

**Nama:** Taufik Hidayat  
**NIM:** 256150117111007  
**Judul:** Integrasi Multimodal Citra Wajah dan Facial Landmark untuk Pengenalan Emosi dalam Konteks Pembelajaran Pemrograman  
**Pembimbing 1:** Dr.Eng. Fitra Abdurrachman Bachtiar, S.T., M.Eng.  
**Pembimbing 2:** Dr.Eng. Budi Darma Setiawan, S.Kom., M.Cs.  
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

| Batch | Direkam | Tersedia | Sudut Kamera | Periode |
|-------|---------|----------|-------------|---------|
| Batch 1 (lama) | 20 mahasiswa | 20 mahasiswa | Depan saja | April-Mei 2025 |
| Batch 2 (baru) | 20 mahasiswa | **17 mahasiswa** | Depan + Samping | November 2025 |
| **Total** | **40** | **37 mahasiswa** | | |

**Catatan Batch 2:** Awalnya batch 2 merekam 20 mahasiswa, namun 3 data rekaman tidak ditemukan di hardisk PC perekaman. Setelah ditelusuri dan hardisk dibuka, file rekaman 3 mahasiswa tersebut tidak tersimpan. Sehingga data batch 2 yang tersedia hanya 17 mahasiswa.

> **Penjelasan lisan:**  
> "Untuk batch 2, awalnya saya merekam 20 mahasiswa, sama seperti batch 1. Namun saat akan memproses datanya, 3 file rekaman tidak ditemukan di hardisk PC yang digunakan untuk perekaman. Setelah saya cek langsung ke hardisk-nya — karena CPU-nya sempat tidak menyala — ternyata file rekamannya memang tidak tersimpan. Kemungkinan gagal saat proses recording. Jadi data batch 2 yang bisa digunakan hanya 17 mahasiswa."
>
> "Total data yang tersedia menjadi 37 mahasiswa dari target 38 di proposal."
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

### Tool Validasi yang Sudah Disiapkan:

Saya sudah membuat **web tool interaktif** (Streamlit) untuk memudahkan proses validasi:

**Fitur utama:**
- Tampilan gambar wajah besar + label otomatis + confidence score + bar chart 7 emosi
- Ahli tinggal klik **"Setuju"** atau pilih emosi yang benar — tidak perlu ketik manual
- **Multi-validator** — support 1-2 ahli, masing-masing punya progress terpisah
- **3 opsi set validasi** tersedia dalam 1 tool — tinggal pilih saat login
- Auto-save setiap klik, ada progress tracker
- Halaman ringkasan otomatis menghitung **Cohen's Kappa** dan **inter-rater agreement**
- Download hasil ke CSV

**Cara akses:**
- Tool akan di-deploy online (Streamlit Cloud) — ahli cukup klik link, tidak perlu install apapun
- Data wajah mahasiswa sudah mendapat informed consent
- Repo bersifat **private** (tidak publik)

**Demo** *(bisa ditunjukkan saat bimbingan jika diminta):*

```
Alur validasi ahli:
1. Buka link → Login nama → Pilih set validasi
2. Lihat gambar wajah → Lihat label otomatis
3. Klik "Setuju" atau pilih emosi yang benar
4. Otomatis lanjut ke sample berikutnya
5. Setelah selesai → Download hasil CSV
```

> **Penjelasan lisan:**  
> "Untuk memudahkan ahli psikologi, saya sudah menyiapkan web tool interaktif. Ahli cukup buka link di browser, lalu klik-klik untuk validasi — tidak perlu buka Excel dan folder gambar satu-satu. Tool ini juga otomatis menghitung Cohen's Kappa setelah validasi selesai."
>
> "Kalau validatornya 2 orang, tool juga bisa menghitung inter-rater agreement antara kedua ahli secara otomatis."

> **Yang perlu ditanyakan:**  
> 1. "Opsi mana yang paling tepat untuk level tesis S2?"
> 2. "Apakah 1 ahli cukup, atau perlu 2 ahli untuk inter-rater reliability?"
> 3. "Apakah Bapak/Ibu punya rekomendasi ahli psikologi yang bisa dihubungi?"
> 4. "Apakah perlu diberikan honorarium untuk validator?"

---

## SLIDE 9: Perubahan dari Proposal

| Aspek | Di Proposal | Implementasi | Alasan |
|-------|------------|-------------|--------|
| Face detection | dlib | **MediaPipe** | Dlib gagal deteksi side view, MediaPipe support multi-angle |
| Landmark | 68 titik (dlib) | **68 titik (mapped dari MediaPipe)** | Setara, output tetap 136 fitur |
| Jumlah mahasiswa | 38 | **37** | 1 mahasiswa data tidak lengkap |
| Fusion strategy | Late + Intermediate | Late + Intermediate | Tetap sesuai proposal |

> **Penjelasan lisan:**  
> "Perubahan utama hanya pada tool face detection dari dlib ke MediaPipe. Ini karena data baru memiliki side view yang tidak bisa dideteksi oleh dlib. Output landmark tetap 68 titik yang di-map dari 478 titik MediaPipe, sehingga secara substansi tidak mengubah desain penelitian."

### **(KONSULTASI 3)** Apakah Perubahan Ini Perlu Direvisi di Proposal?

> "Pak/Bu, apakah perubahan dari dlib ke MediaPipe ini perlu direvisi di dokumen proposal, atau cukup dijelaskan di BAB 4 (Implementasi) saja?"

---

## SLIDE 10: Rencana Selanjutnya

| No | Tahap | Status | Target |
|----|-------|--------|--------|
| 1 | Pengumpulan data | Selesai | - |
| 2 | Preprocessing (frame, crop, landmark) | Selesai | - |
| 3 | Prepare dataset (numpy arrays) | Selesai | - |
| 4 | Class weights + augmentasi | Selesai | - |
| 5 | Tool validasi ahli (web app) | Selesai, siap deploy | - |
| 6 | Deploy tool + kirim ke ahli | **Menunggu keputusan** | Setelah bimbingan |
| 7 | Training CNN | **Selesai** | - |
| 8 | Training FCNN | **Selesai** | - |
| 9 | Late Fusion | **Selesai** | - |
| 10 | Intermediate Fusion | **Selesai** | - |
| 11 | Evaluasi & perbandingan | **Selesai** | - |
| 12 | **Perbaikan model** | **Perlu diskusi** | Setelah bimbingan |

---

## SLIDE 11: Hasil Training (4 Model × 3 Skenario = 12 Eksperimen)

Training dilakukan di VPS Biznet Gio (NVIDIA T4, 16GB VRAM) menggunakan PyTorch.

### Tabel Hasil Lengkap (diurutkan berdasarkan Macro F1)

| Rank | Model | Skenario | Accuracy | Macro F1 | Weighted F1 |
|------|-------|----------|----------|----------|-------------|
| 1 | **FCNN** | **B1 Baseline** | **95.8%** | **0.234** | 0.952 |
| 2 | Late Fusion | B1 Baseline | 95.8% | 0.230 | 0.951 |
| 3 | FCNN | B2 Weights | 89.1% | 0.189 | 0.912 |
| 4 | Late Fusion | B2 Weights | 89.7% | 0.189 | 0.909 |
| 5 | Late Fusion | B3 Aug | 92.5% | 0.182 | 0.929 |
| 6 | FCNN | B3 Aug | 92.3% | 0.182 | 0.927 |
| 7 | Intermediate | B2 Weights | 84.5% | 0.140 | 0.881 |
| 8 | Intermediate | B3 Aug | 81.6% | 0.134 | 0.866 |
| 9 | CNN | B2 Weights | 82.8% | 0.134 | 0.872 |
| 10 | CNN | B1 Baseline | 84.2% | 0.133 | 0.881 |
| 11 | CNN | B3 Aug | 64.9% | 0.119 | 0.766 |
| 12 | Intermediate | B1 Baseline | 63.3% | 0.111 | 0.744 |

### Kombinasi Terbaik: **FCNN + B1 Baseline** (Macro F1: 0.234)

> **Penjelasan lisan:**  
> "Training sudah selesai untuk semua 12 kombinasi. Hasilnya, model FCNN (berbasis landmark) tanpa class weights justru memberikan hasil terbaik dengan accuracy 95.8% dan Macro F1 0.234."

---

## SLIDE 12: Analisis Hasil

### Temuan 1: FCNN (Landmark) > CNN (Image)

| Model | Best Macro F1 |
|-------|--------------|
| **FCNN** | **0.234** |
| CNN | 0.134 |

> **Penjelasan lisan:**  
> "Fitur geometrik dari facial landmark ternyata lebih efektif dari fitur penampilan (citra wajah) untuk dataset ini. Ini konsisten dengan temuan Bachtiar et al. (2024) yang menunjukkan bahwa facial landmark bisa superior untuk ekspresi halus. Kemungkinan karena landmark lebih robust terhadap variasi pencahayaan dan sudut wajah yang terjadi selama sesi pemrograman."

### Temuan 2: Fusion tidak lebih baik dari FCNN saja

| Model | Best Macro F1 |
|-------|--------------|
| FCNN | **0.234** |
| Late Fusion (10% CNN + 90% FCNN) | 0.230 |
| Intermediate Fusion | 0.140 |

> **Penjelasan lisan:**  
> "Late Fusion optimal pada bobot CNN hanya 10% dan FCNN 90% — artinya CNN hampir tidak berkontribusi. Intermediate Fusion justru lebih buruk karena fitur CNN yang noisy mengganggu fitur landmark yang sudah bagus. Ini menunjukkan bahwa untuk dataset pembelajaran pemrograman ini, pendekatan unimodal FCNN sudah cukup optimal."

### Temuan 3: Class weights dan augmentasi TIDAK membantu

| Skenario | FCNN Macro F1 | CNN Macro F1 |
|----------|--------------|-------------|
| B1 Baseline | **0.234** | 0.133 |
| B2 Class Weights | 0.189 | 0.134 |
| B3 Weights + Aug | 0.182 | 0.119 |

> **Penjelasan lisan:**  
> "Yang mengejutkan, class weights dan augmentasi justru menurunkan performa. Ini kemungkinan karena memberikan bobot terlalu besar pada kelas yang sangat sedikit (fearful: 8 sample, weight 125x) membuat model tidak stabil. Baseline tanpa penanganan justru lebih baik."

### Temuan 4: Masalah utama — kelas minoritas tidak terdeteksi

Dari classification report model terbaik (FCNN B1):

| Emosi | Support | Recall | F1 |
|-------|---------|--------|-----|
| Neutral | 1,588 | 99% | 0.98 |
| Happy | 10 | 0% | 0.00 |
| Sad | 38 | 32% | 0.41 |
| Angry | 13 | 0% | 0.00 |
| Fearful | 1 | 0% | 0.00 |
| Disgusted | 3 | 0% | 0.00 |
| Surprised | 3 | 0% | 0.00 |

> **Penjelasan lisan:**  
> "Meskipun accuracy tinggi (95.8%), model hanya bisa mengenali neutral dan sedikit sad. Kelas lain seperti happy, angry, fearful, disgusted, dan surprised tidak terdeteksi sama sekali. Ini karena jumlah sampelnya terlalu sedikit di test set (1-13 sample) dan model belum cukup belajar dari data training yang juga sangat sedikit untuk kelas-kelas ini."
>
> "Accuracy 95.8% sebenarnya misleading — model bisa dapat accuracy segitu hanya dengan memprediksi semua sebagai neutral. Itulah kenapa kita pakai Macro F1 (0.234) sebagai metrik utama, yang menunjukkan performa sebenarnya."

---

## SLIDE 13: Hasil Training 4-Class (Analisis Tambahan)

Sebagai analisis tambahan, dilakukan eksperimen dengan **4 kelas emosi**:
- neutral, happy, sad, **negative** (gabungan angry + fearful + disgusted + surprised)

### Tabel Hasil 4-Class (diurutkan berdasarkan Macro F1)

| Rank | Model | Skenario | Accuracy | Macro F1 | Weighted F1 |
|------|-------|----------|----------|----------|-------------|
| 1 | **FCNN** | **B3 Aug** | **94.4%** | **0.394** | 0.943 |
| 2 | Late Fusion | B3 Aug | 94.6% | 0.385 | 0.943 |
| 3 | FCNN | B1 Baseline | 95.8% | 0.330 | 0.943 |
| 4 | Late Fusion | B1 Baseline | 95.8% | 0.330 | 0.943 |
| 5 | FCNN | B2 Weights | 92.9% | 0.327 | 0.929 |
| 6 | Late Fusion | B2 Weights | 92.9% | 0.327 | 0.929 |
| 7 | CNN | B2 Weights | 92.8% | 0.296 | 0.929 |
| 8 | CNN | B1 Baseline | 92.8% | 0.282 | 0.932 |
| 9 | Intermediate | B2 Weights | 87.3% | 0.258 | 0.895 |
| 10 | Intermediate | B1 Baseline | 91.9% | 0.245 | 0.919 |
| 11 | Intermediate | B3 Aug | 83.0% | 0.238 | 0.875 |
| 12 | CNN | B3 Aug | 79.2% | 0.238 | 0.853 |

### Kombinasi Terbaik 4-Class: **FCNN + B3 Augmented** (Macro F1: 0.394)

> **Penjelasan lisan:**  
> "Dengan 4 kelas, performa meningkat signifikan di semua model. FCNN dengan augmentasi memberikan Macro F1 terbaik 0.394, naik 68% dari 7-kelas. Ini menunjukkan bahwa penggabungan kelas langka menjadi 'negative' efektif meningkatkan kemampuan model."

---

## SLIDE 14: Perbandingan 7-Class vs 4-Class

### Best per Model

| Model | 7-Class (Macro F1) | 4-Class (Macro F1) | Peningkatan |
|-------|-------------------|-------------------|-------------|
| **FCNN** | 0.234 (B1) | **0.394 (B3)** | **+68%** |
| Late Fusion | 0.230 (B1) | 0.385 (B3) | +67% |
| CNN | 0.134 (B2) | 0.296 (B2) | +121% |
| Intermediate | 0.140 (B2) | 0.258 (B2) | +84% |

### Temuan baru dari 4-class:

**Temuan 5: Penggabungan kelas signifikan meningkatkan Macro F1**

> **Penjelasan lisan:**  
> "Dengan menggabungkan 4 emosi langka (angry, fearful, disgusted, surprised) menjadi satu kelas 'negative', Macro F1 meningkat 68% pada model terbaik. Ini menunjukkan bahwa masalah utama di 7-kelas bukan pada modelnya, tapi pada jumlah data kelas minoritas yang terlalu sedikit."

**Temuan 6: Augmentasi efektif untuk 4-class (berbeda dengan 7-class)**

> **Penjelasan lisan:**  
> "Menariknya, di 4-kelas skenario B3 (augmentasi) justru terbaik, sedangkan di 7-kelas B1 (baseline) yang terbaik. Ini karena di 4-kelas, kelas 'negative' sudah punya 145 sample asli — cukup besar untuk mendapat manfaat dari augmentasi. Di 7-kelas, kelas terkecil hanya 8 sample — terlalu sedikit sehingga augmentasi hanya menghasilkan variasi dari data yang sangat terbatas."

**Temuan 7: FCNN tetap konsisten terbaik di kedua konfigurasi**

> **Penjelasan lisan:**  
> "Baik di 7-kelas maupun 4-kelas, FCNN (fitur landmark) selalu unggul dari CNN (fitur citra wajah). Ini memperkuat temuan bahwa fitur geometrik lebih efektif dari fitur penampilan untuk dataset pembelajaran pemrograman ini."

---

## SLIDE 15: Diskusi dan Kesimpulan Sementara

### Jawaban terhadap Rumusan Masalah:

**RQ1 (Performa CNN):**
Model CNN menghasilkan Macro F1 0.134 (7-kelas) dan 0.296 (4-kelas). Performa relatif rendah dibanding FCNN, kemungkinan karena variasi pencahayaan dan sudut wajah selama sesi pemrograman yang mengganggu fitur penampilan.

**RQ2 (Performa FCNN):**
Model FCNN menghasilkan Macro F1 **0.234** (7-kelas) dan **0.394** (4-kelas). Fitur geometrik dari 68 facial landmark terbukti lebih robust dan efektif untuk konteks pembelajaran pemrograman.

**RQ3 (Perbandingan Fusion):**
Late Fusion memberikan sedikit perbaikan dari FCNN saja (0.385 vs 0.394 di 4-kelas). Intermediate Fusion justru lebih buruk. Ini menunjukkan bahwa untuk dataset ini, pendekatan unimodal FCNN sudah cukup optimal — menambah modalitas CNN justru memperkenalkan noise.

### **(KONSULTASI 4)** Pertanyaan untuk Pembimbing

> "Pak/Bu, hasil eksperimen sudah lengkap. Beberapa hal yang perlu didiskusikan:"

1. **7-kelas vs 4-kelas:** "Apakah eksperimen 4-kelas cukup sebagai analisis tambahan di BAB Pembahasan, atau perlu dijadikan eksperimen utama?"

2. **FCNN > Fusion:** "Temuan bahwa unimodal FCNN lebih baik dari multimodal fusion agak bertentangan dengan hipotesis di proposal. Bagaimana sebaiknya menyikapi ini di tesis?"

3. **Macro F1 masih rendah:** "Meskipun 4-kelas lebih baik (0.394), nilainya masih di bawah 0.5. Apakah ini masih acceptable untuk tesis, atau perlu eksplorasi arsitektur/metode lain?"

---

## Ringkasan Poin Konsultasi

| No | Topik | Pertanyaan | Opsi Rekomendasi |
|----|-------|-----------|------------------|
| 1 | 7-kelas vs 4-kelas | Analisis tambahan atau eksperimen utama? | Analisis tambahan di BAB 5 |
| 2 | FCNN > Fusion | Bagaimana menyikapi di tesis? | Report sebagai temuan, bahas alasannya |
| 3 | Macro F1 rendah | Acceptable atau perlu metode lain? | Diskusikan dengan pembimbing |
| 4 | Validasi ahli - jumlah sample | 583 (5%) / 1,067 (10%) / 1,938 (full non-neutral)? | Tergantung ketersediaan ahli |
| 5 | Validasi ahli - jumlah ahli | 1 ahli atau 2 ahli (untuk inter-rater reliability)? | 2 ahli jika memungkinkan |
| 6 | Validasi ahli - honorarium | Perlu diberikan fee untuk validator? | Tergantung kebijakan prodi |
| 7 | Perubahan tool | Perlu revisi proposal untuk perubahan dlib → MediaPipe? | Cukup di BAB 4 |

---

## Lampiran A: Struktur Project Saat Ini

```
MultimodalEmoLearn/
├── src/
│   ├── preprocessing/
│   │   ├── prepare_dataset.py          # Gabung data + split + numpy
│   │   ├── augment_minority.py         # Augmentasi kelas minoritas
│   │   └── generate_validation_set.py  # Generate set validasi ahli
│   ├── training/
│   │   ├── models.py                   # Arsitektur CNN, FCNN, IntermediateFusion
│   │   └── utils.py                    # Training loop, evaluasi, visualisasi
│   ├── tools/
│   │   ├── validation_app.py           # Streamlit web tool validasi
│   │   └── process_validation_results.py # Proses hasil validasi + Cohen's Kappa
│   └── utils/
│       ├── batch_video_processor.py    # Ekstrak frame dari video
│       ├── face_crop_landmark.py       # Face crop + 68 landmark
│       └── generate_emotion_label.py   # Generate label emosi
├── notebooks/
│   ├── 01-05                           # Training 7-class (CNN, FCNN, Fusion, Comparison)
│   ├── 06-10                           # Training 4-class (CNN, FCNN, Fusion, Comparison)
│   └── results/                        # Executed notebooks dari VPS
├── models/
│   ├── cnn/, fcnn/, late_fusion/,      # Hasil 7-class
│   │   intermediate_fusion/
│   ├── 4class/                         # Hasil 4-class
│   │   ├── cnn/, fcnn/, late_fusion/,
│   │   │   intermediate_fusion/
│   │   └── experiment_summary_4class.json
│   └── experiment_summary.json         # Ringkasan 7-class
├── data/
│   ├── dataset/                        # Numpy arrays 7-class
│   ├── dataset_augmented/              # Numpy arrays 7-class augmented
│   ├── dataset_4class/                 # Numpy arrays 4-class
│   ├── dataset_4class_augmented/       # Numpy arrays 4-class augmented
│   └── validation_*/                   # Set validasi ahli (3 opsi)
├── scripts/
│   ├── run_all.sh                      # Jalankan semua (7-class + 4-class)
│   └── run_4class.sh                   # Jalankan hanya 4-class
└── docs/
    ├── bimbingan_progress.md           # File ini
    └── linux_training_guide.md         # Panduan training di VPS
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
| Intermediate Fusion | Boulahia et al. (2021) | Feature-level fusion CNN + FCNN |
| Late Fusion | Boulahia et al. (2021) | Decision-level weighted averaging |

---

## Lampiran C: Konfigurasi Training

| Parameter | CNN | FCNN | Intermediate Fusion |
|-----------|-----|------|---------------------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 0.0001 | 0.0001 | 0.0001 |
| Batch Size | 32 | 128 | 16 |
| Max Epochs | 50 | 100 | 80 |
| Early Stopping Patience | 15 | 20 | 25 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) | ReduceLROnPlateau (factor=0.5, patience=8) | ReduceLROnPlateau (factor=0.5, patience=8) |
| Best Model Criteria | Val Macro F1 | Val Macro F1 | Val Macro F1 |
| GPU | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) |
| Framework | PyTorch | PyTorch | PyTorch |
