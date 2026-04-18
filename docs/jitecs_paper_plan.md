# JITeCS Paper — Rencana Penyusunan

> **Judul kerja:** *Multimodal Fusion of Facial Image and Landmark Features with Transfer Learning for Emotion Recognition in Programming Learning Context*
>
> **Target venue:** JITeCS (Journal of Information Technology and Computer Science), SINTA 2
>
> **Format:** IEEE, 8–12 halaman, 20–25 referensi
>
> **Scope:** Fokus pada eksperimen **dataset primer conf60** dengan studi komparatif 5 arsitektur fusion × transfer learning.

---

## 1. Struktur Paper (Final)

| Section | Perkiraan Halaman |
|---------|:-----------------:|
| Abstract | ~1 paragraph |
| 1. Introduction | ~1 |
| 2. Related Work | ~1 |
| 3. Proposed Method | ~2 |
| 4. Experimental Results | ~2–3 |
| 5. Discussion | ~1 |
| 6. Conclusion | ~0.5 |
| References (20–25 sitasi) | — |

### Daftar Isi Detail

**Abstract**
- Problem statement (FER di konteks programming natural)
- Method summary (5 arsitektur × fusion × transfer learning)
- Data description (6,795 samples dari 37 mahasiswa, conf60)
- Best result (Late Fusion TL 4c B3 = Macro F1 0.567)
- Key insight

**1. Introduction**
- Motivasi: emosi → learning outcome programming
- Gap: FER existing fokus data lab (posed), bukan natural
- Rumusan masalah & tujuan penelitian
- Kontribusi: dataset baru + studi sistematis fusion × TL
- Struktur paper

**2. Related Work**
- 2.1 Deep Learning for FER
- 2.2 Multimodal Fusion of Image and Landmark Features
- 2.3 Transfer Learning for FER
- 2.4 Affective Computing in Education

**3. Proposed Method**
- 3.1 Dataset
- 3.2 Multimodal Architecture (5 varian)
- 3.3 Training Setup
- 3.4 Experimental Design (B1/B2/B3 × 7c/4c × metrics)

**4. Experimental Results**
- 4.1 Overall Performance (60 configs)
- 4.2 Effect of Transfer Learning
- 4.3 Effect of Fusion Strategy
- 4.4 Effect of Class Granularity
- 4.5 Per-Class Analysis

**5. Discussion**
- 5.1 Multimodal Fusion vs Single-Modality (jawab RQ1)
- 5.2 Fusion Strategy Comparison: Intermediate vs Late (jawab RQ2)
- 5.3 Transfer Learning Effectiveness (jawab RQ3)
- 5.4 Limitations
- 5.5 Implications for Learning Analytics

**6. Conclusion**
- Ringkasan kontribusi & best result
- Future work

---

## 2. Research Questions (Formal)

Tiga RQ yang akan dijawab di paper:

**RQ1**: *Apakah fusi multimodal antara citra wajah dan facial landmark memberikan kinerja yang lebih baik dibandingkan pendekatan single-modality untuk pengenalan emosi pada konteks pembelajaran pemrograman?*

**RQ2**: *Strategi fusion manakah — Early Fusion, Intermediate Fusion, atau Late Fusion — yang lebih efektif dalam menangani data ekspresi wajah yang natural dan imbalanced?*

**RQ3**: *Bagaimana pengaruh transfer learning (ResNet18 pretrained ImageNet) terhadap kinerja model pengenalan emosi wajah multimodal pada dataset kecil dengan ekspresi natural?*

---

## 3. Dataset untuk Paper

### Primer Dataset (satu-satunya)

| Aspek | Detail |
|-------|--------|
| Sumber | Akuisisi sendiri, sesi pembelajaran pemrograman |
| Total samples | **6,795** (setelah filter confidence ≥ 60%) |
| Jumlah user | 37 mahasiswa (2 batch) |
| Emosi | 7 kelas (neutral, happy, sad, angry, fearful, disgusted, surprised) |
| Split | User-wise 80/10/10: **train 5,287 / val 579 / test 929** |
| Preprocessing | Face crop 224×224, MediaPipe 68 landmarks → 136-dim |
| Konfigurasi kelas | 7-class (original) + 4-class remap (neutral/happy/sad/negative) |
| Lokasi | `data/dataset_frontonly_conf60/` |

### Statistik Imbalance

Distribusi train set (conf60, 7-class):

| Emosi | Jumlah | % |
|-------|:------:|:-:|
| neutral | 5,691 | 83.8% |
| happy | 651 | 9.6% |
| sad | 361 | 5.3% |
| angry | 32 | 0.5% |
| fearful | 5 | 0.1% |
| disgusted | 16 | 0.2% |
| surprised | 39 | 0.6% |

→ **Ratio 1:1138** (fearful vs neutral). Kelas minoritas sangat langka.

---

## 4. Arsitektur Model (5 Varian)

| # | Model | Input | Fusion Point | Backbone |
|---|-------|-------|:------------:|----------|
| 1 | **CNN** | Citra 224×224×3 | — (single modal) | Scratch / ResNet18 TL |
| 2 | **FCNN** | Landmark 136-dim | — (single modal) | Scratch (no TL for landmarks) |
| 3 | **Early Fusion** | Citra + heatmap 224×224×4 | **Input level (0%)** | Scratch / ResNet18 TL (4-ch first conv) |
| 4 | **Intermediate Fusion** | Citra + Landmark | Feature level (~50%) | Scratch / ResNet18 TL |
| 5 | **Late Fusion** | Citra + Landmark | Decision level (~95%) | Scratch / ResNet18 TL |

**Transfer Learning Backbone:**
- ResNet18 pretrained ImageNet (1000-class)
- Fine-tune seluruh network, learning rate kecil (5×10⁻⁵)
- Early Fusion TL: first `Conv2d` dimodifikasi dari 3→4 channel. Weight RGB di-copy, weight heatmap di-init dari mean(RGB).
- Referensi Early Fusion: Wu et al. (MMM 2020) — HAE-Net.

---

## 5. Matriks Eksperimen

**Total: 60 konfigurasi** = 5 arsitektur × 2 backbone × 3 skenario × 2 kelas

⚠️ Pengecualian: FCNN tidak punya TL variant (landmark tidak punya pretrained). Jadi effective: 9 model-backbone × 3 skenario × 2 kelas = 54 configs praktis.

### Tiga Skenario Handling Imbalance

| Skenario | Keterangan |
|----------|-----------|
| **B1** — Baseline | Standard cross-entropy, no intervention |
| **B2** — Class Weights | Weighted CE: `weight_c ∝ 1 / freq(c)` |
| **B3** — Weights + Aug | B2 + data augmentation (rotation, flip, brightness) untuk kelas minoritas |

### Dua Konfigurasi Kelas

| Kelas | Labels |
|-------|--------|
| **7-class** | neutral, happy, sad, angry, fearful, disgusted, surprised |
| **4-class (remap)** | neutral, happy, sad, **negative** (= angry+fearful+disgusted+surprised digabung) |

### Metrik Evaluasi

- **Macro F1** (utama) — rata-rata F1 per kelas, unbiased terhadap imbalance
- **Micro F1** = Accuracy (dalam multi-class single-label)
- **Weighted F1** — rata-rata F1 bobot support, didominasi kelas mayoritas
- **Accuracy**

---

## 6. Hasil Eksperimen (Macro F1 per Config)

Untuk referensi penulisan Section 4 (Results) dan Section 5 (Discussion).

### 7-Class

| Model | B1 | B2 | B3 |
|-------|:--:|:--:|:--:|
| CNN | 0.277 | 0.240 | 0.253 |
| FCNN | 0.232 | 0.244 | 0.222 |
| Early Fusion | *(pending nb 64)* | *(pending)* | *(pending)* |
| Intermediate | 0.261 | 0.247 | 0.229 |
| Late Fusion | 0.288 | 0.266 | 0.260 |
| CNN TL | 0.273 | 0.243 | 0.241 |
| Early Fusion TL | *(pending)* | *(pending)* | *(pending)* |
| Intermediate TL | 0.277 | 0.283 | 0.292 |
| Late Fusion TL | 0.301 | 0.264 | 0.260 |

### 4-Class

| Model | B1 | B2 | B3 |
|-------|:--:|:--:|:--:|
| CNN | 0.438 | 0.448 | 0.432 |
| FCNN | 0.422 | 0.459 | 0.421 |
| Early Fusion | *(pending)* | *(pending)* | *(pending)* |
| Intermediate | 0.445 | 0.416 | 0.382 |
| Late Fusion | 0.482 | 0.503 | 0.463 |
| CNN TL | 0.456 | 0.447 | 0.507 |
| Early Fusion TL | *(pending)* | *(pending)* | *(pending)* |
| Intermediate TL | 0.489 | 0.508 | 0.521 |
| **Late Fusion TL** | 0.513 | 0.519 | **0.567** ⭐ |

---

## 7. Best Results (Untuk Abstract & Discussion)

### Overall Best: Late Fusion TL 4-class B3 = Macro F1 0.567

| Model | Scenario | Kelas | Macro F1 | Note |
|-------|----------|:-----:|:--------:|------|
| **Late Fusion TL** | **B3** | **4** | **0.567** | ⭐ Best overall |
| Intermediate TL | B3 | 4 | 0.521 | |
| Late Fusion TL | B2 | 4 | 0.519 | |
| CNN TL | B3 | 4 | 0.507 | |
| Late Fusion TL | B1 | 4 | 0.513 | |

### Best per Section

| Section | Finding |
|---------|---------|
| **Fusion strategies** | Late Fusion TL > Intermediate TL > Single-modal |
| **TL effect** | TL +0.05-0.10 Macro F1 konsisten di semua arch |
| **Class granularity** | 4-class F1 ≈ 2× lebih tinggi dari 7-class (0.567 vs 0.292) |
| **Imbalance handling** | B3 (aug) > B2 (weights) > B1 (baseline) di TL variants |

---

## 8. Gambar & Tabel yang Perlu Disiapkan

### Gambar (Figures)
- **Fig 1**: Architecture diagram (5 varian fusion strategies, visualizing fusion points)
- **Fig 2**: Dataset sample images (front-facing programming sessions) — contoh per kelas emosi
- **Fig 3**: Landmark heatmap generation illustration (untuk Early Fusion)
- **Fig 4**: Macro F1 bar chart — 5 arsitektur × 2 kelas (best scenario per model)
- **Fig 5**: Confusion matrix best model (Late Fusion TL 4c B3)

### Tabel (Tables)
- **Tab 1**: Dataset distribution (7-class vs 4-class, train/val/test)
- **Tab 2**: Main results — 50 configurations, rank by Macro F1
- **Tab 3**: Top 5 configurations detailed (all 4 metrics)
- **Tab 4**: Per-class F1 for best model
- **Tab 5**: Ablation — effect of TL (scratch vs TL per architecture)

---

## 9. Referensi Target (Starter List)

### FER with Deep Learning
- Dada et al. (2023) — CNN-10
- Li et al. (2024) — AA-DCN
- Khan et al. (2023) — ResNet50 TL for FER2013
- Zhang et al. (2022) — Late Fusion CK+/JAFFE

### Multimodal Fusion (Image + Landmark)
- **Wu et al. (2020) — HAE-Net: Emotion Recognition with Facial Landmark Heatmaps** ⭐ (Early Fusion reference)
- Chen et al. (2024) — β-skeleton + CNN hybrid
- Zhang et al. (2024) — GhostNet Multimodal

### Transfer Learning & Attention
- He et al. (2016) — ResNet (for backbone reference)
- Boulahia et al. (2021) — Early/Intermediate/Late fusion strategies

### Affective Computing in Education
- Picard (1997) — Affective Computing (foundational)
- D'Mello et al. — AutoTutor / affect-aware systems
- Sharma et al. — MOOC engagement via FER

### FER Datasets
- Lucey et al. (2010) — CK+
- Lyons et al. (1998) — JAFFE
- Li et al. (2017) — RAF-DB
- Lundqvist et al. (1998) — KDEF

---

## 10. Catatan untuk Methodology Section

**Hyperparameters (untuk disebutkan di Section 3.3 Training Setup):**
- Optimizer: Adam
- Learning rate: 1×10⁻⁴ (scratch), 5×10⁻⁵ (TL)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1×10⁻⁷)
- Batch size: 32
- Max epochs: 50
- Early stopping: patience=15 (monitor Macro F1 on val set)
- Loss: CrossEntropyLoss (B1), Weighted CE (B2, B3)

**Random seed note**:
- `np.random.RandomState(42)` untuk split (deterministic)
- `torch.manual_seed` / cuDNN deterministic **tidak** diset → variance antar-run ±0.01-0.05 Macro F1 (normal di DL)
- **Single run per config** di paper (bukan mean±std). Bisa disebutkan sebagai limitation / future work.

**Pembobotan class weights (B2, B3)**:
- Inverse-frequency: `w_c ∝ 1 / freq(c)`, dinormalisasi supaya `sum(w) = num_classes`
