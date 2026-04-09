# Emotion Validation Tool

Tool interaktif untuk ahli psikologi memvalidasi label emosi otomatis dari sistem pengenalan ekspresi wajah.

## Deploy ke Streamlit Cloud

### 1. Buat repo private di GitHub

```bash
cd deploy/emotion-validation
git init
git add .
git commit -m "Initial commit - emotion validation tool"
git remote add origin https://github.com/USERNAME/emotion-validation.git
git branch -M main
git push -u origin main
```

### 2. Deploy di Streamlit Cloud

1. Buka https://share.streamlit.io
2. Login dengan akun GitHub
3. Klik "New app"
4. Pilih repo: `USERNAME/emotion-validation`
5. Branch: `main`
6. Main file: `app.py`
7. Klik "Deploy"

### 3. Kirim link ke ahli

Setelah deploy, akan mendapat link seperti:
`https://emotion-validation-username.streamlit.app`

Kirim link ini ke ahli psikologi.

### 4. Setelah validasi selesai

Download hasil dari halaman "Ringkasan" di app, atau ambil file `data/expert_results.json` dari repo.
