# Deteksi Transaksi Penipuan (Credit Card Fraud Detection)

Proyek bootcamp AI Engineering: model klasifikasi untuk menandai transaksi kartu
kredit yang berpotensi penipuan, lengkap dengan aplikasi web Streamlit. Hasil
aplikasi adalah rekomendasi untuk membantu analis, bukan keputusan pemblokiran
transaksi.

## Data

- Dataset: Credit Card Fraud Detection (ULB Machine Learning Group, tersedia di Kaggle).
- 284.807 transaksi, 492 di antaranya penipuan (sekitar 0,17%), sehingga data sangat tidak seimbang.
- Fitur: Time, V1 sampai V28 (hasil PCA yang dianonimkan), dan Amount. Target: Class (0 = normal, 1 = penipuan).
- 1.081 baris duplikat dibuang, menyisakan 283.726 baris dengan 473 penipuan.
- Data dibagi 80/20 dengan stratify (226.980 latih, 56.746 uji).

## Hasil (data uji)

- Model: Random Forest hasil tuning, ambang probabilitas 0,34 (dipilih lewat cross-validation pada data latih).
- PR-AUC: 0,8158
- Precision (penipuan): 0,9459
- Recall (penipuan): 0,7368
- F1 (penipuan): 0,8284
- Akurasi tidak dijadikan ukuran utama karena menyesatkan pada data tidak seimbang.

## Menjalankan aplikasi

1. Pastikan Python sudah terpasang.
2. Buka terminal di folder proyek.
3. Buat dan aktifkan virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

   Di Mac/Linux: `source .venv/bin/activate`

4. Pasang dependensi:

   ```
   pip install -r requirements.txt
   ```

5. Jalankan aplikasi:

   ```
   streamlit run app_fraud.py
   ```

6. Buka `http://localhost:8501` di browser.

## Menggunakan aplikasi

- **Unggah CSV:** unggah file berisi kolom Time, V1 sampai V28, dan Amount. Aplikasi menampilkan jumlah transaksi, daftar transaksi yang dicurigai (diurutkan dari probabilitas tertinggi), dan tombol unduh hasil. Contoh file ada di `data/contoh_transaksi.csv`.
- **Contoh transaksi:** ambil satu transaksi acak dari `data/contoh_transaksi_berlabel.csv` dan lihat prediksi model beserta label aslinya.

## Membuat ulang model

Jalankan seluruh cell di `fraud_eksperimen.ipynb`. Cell pertama mengunduh data ke
`data/creditcard.csv`. Model dan metrik disimpan di folder `artifacts/`.

## Melihat eksperimen di MLflow

1. Aktifkan virtual environment, lalu jalankan dari folder proyek:

   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

2. Buka `http://127.0.0.1:5000`, pilih tab **Model training**, lalu eksperimen **fraud-detection**.

## Struktur folder

- `data/` : dataset dan contoh transaksi untuk demo
- `artifacts/` : model (`fraud_model.joblib`), metrik, dan perbandingan model
- `fraud_eksperimen.ipynb` : pembersihan data, EDA, pemodelan, dan evaluasi
- `app_fraud.py` : aplikasi Streamlit

## Catatan dan keterbatasan

- PR-AUC pada data latih (0,99) jauh lebih tinggi daripada data uji (0,82): model cenderung menghafal data latih. Performa yang bisa diharapkan pada transaksi baru adalah sekitar 0,82.
- Data uji hanya berisi 95 penipuan, jadi metrik bisa bergeser beberapa poin hanya karena satu atau dua kasus.
- Sebagian penipuan masih lolos (recall 0,74).
- Fitur V1 sampai V28 dianonimkan, sehingga alasan di balik prediksi tidak bisa ditafsirkan.
- Proyek ini simulasi untuk pembelajaran, bukan untuk keputusan operasional nyata.