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
- File `data/creditcard.csv` (sekitar 144 MB) tidak disertakan di repositori. Cell pertama
  `fraud_eksperimen.ipynb` mengunduhnya otomatis.

## Hasil (data uji)

- Model: Random Forest hasil tuning, ambang probabilitas 0,34 (dipilih lewat cross-validation pada data latih).
- PR-AUC: 0,8158
- Precision (penipuan): 0,9459
- Recall (penipuan): 0,7368
- F1 (penipuan): 0,8284
- Dari 95 penipuan di data uji, 70 terdeteksi dan 25 lolos. Hanya 4 dari 56.651 transaksi normal yang salah ditandai.
- Akurasi (99,95%) tidak dijadikan ukuran utama karena menyesatkan pada data tidak seimbang.

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

Aplikasi memiliki tiga tab dan sebuah sidebar.

**Tiga tingkat risiko.** Setiap transaksi dikelompokkan berdasarkan probabilitas penipuan:

| Tingkat | Aturan |
| --- | --- |
| Curigai | probabilitas ≥ ambang (0,34) |
| Perlu dicek | di antara batas aman dan ambang |
| Aman | probabilitas < batas aman (bawaan 0,01) |

**Tab Unggah CSV.** Unggah file berisi kolom Time, V1 sampai V28, dan Amount. Aplikasi
menampilkan jumlah transaksi per tingkat dan daftar transaksi yang bisa dipilih per
tingkat (Curigai, Perlu dicek, Aman, atau Semua), serta tombol unduh hasil. Jika file
memiliki kolom Class, aplikasi juga menampilkan tabel rincian terhadap label asli.
Contoh file: `data/contoh_transaksi.csv`, `data/dummy_100.csv` (tanpa label), dan
`data/dummy_100_berlabel.csv` (dengan label).

**Tab Contoh transaksi.** Mengambil satu transaksi acak dari
`data/contoh_transaksi_berlabel.csv`, lalu menampilkan prediksi beserta label aslinya.

**Tab Evaluasi model.** Menampilkan Precision, Recall, F1, PR-AUC, dan confusion matrix
pada data uji untuk ambang yang dipilih. Tab ini butuh `data/creditcard.csv`.

**Sidebar.**
- Ambang probabilitas dan batas aman (slider).
- Pilihan model: model resmi atau model hasil latih ulang.
- Latih ulang model (eksplorasi): atur persentase penipuan di data latih, jumlah
  pohon, kedalaman, minimum sampel per daun, dan class_weight. Data uji selalu
  tetap asli (penipuan 0,17%). Fitur ini butuh `data/creditcard.csv`.

Catatan: mengubah pengaturan sambil melihat skor data uji membuat skor itu terlalu
optimis, jadi fitur latih ulang hanya untuk eksplorasi. Angka resmi proyek adalah
yang ada di bagian Hasil di atas.

## Membuat ulang model

Jalankan seluruh cell di `fraud_eksperimen.ipynb`. Cell pertama mengunduh data ke
`data/creditcard.csv`. Model, metrik, dan laporan evaluasi disimpan di folder `artifacts/`.

## Melihat eksperimen di MLflow

1. Aktifkan virtual environment, lalu jalankan dari folder proyek:

   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

2. Buka `http://127.0.0.1:5000`, pilih tab **Model training**, lalu eksperimen **fraud-detection**.

Database `mlflow.db` dan folder `mlruns/` tidak disertakan di repositori. Jalankan
notebook untuk membuatnya.

## Struktur folder

- `data/` : data contoh untuk demo (dataset lengkap diunduh oleh notebook)
- `artifacts/` : model (`fraud_model.joblib`), metrik, perbandingan model, laporan evaluasi
- `fraud_eksperimen.ipynb` : pembersihan data, EDA, pemodelan, dan evaluasi
- `app_fraud.py` : aplikasi Streamlit

## Catatan dan keterbatasan

- PR-AUC pada data latih (0,99) jauh lebih tinggi daripada data uji (0,82): model cenderung menghafal data latih. Performa yang bisa diharapkan pada transaksi baru adalah sekitar 0,82.
- Data uji hanya berisi 95 penipuan, jadi metrik bisa bergeser beberapa poin hanya karena satu atau dua kasus.
- Sebagian penipuan masih lolos (recall 0,74). Penipuan yang lolos cenderung bernilai kecil.
- Kategori Aman berarti risiko sangat rendah, bukan jaminan. Pada validasi silang di data latih, kategori ini (batas 0,01) berisi 98,3% transaksi dengan sekitar 1,8 penipuan per 10.000 (rata-rata keseluruhan sekitar 16,7), tetapi 40 dari 378 penipuan (10,6%) tetap berada di dalamnya.
- Eksperimen latih ulang dengan 10% penipuan di data latih menaikkan recall (0,84) tetapi menurunkan precision (0,27) dan menambah alarm palsu dari 4 menjadi 221. Ini hasil eksplorasi, bukan hasil resmi.
- Fitur V1 sampai V28 dianonimkan, sehingga alasan di balik prediksi tidak bisa ditafsirkan.
- Proyek ini simulasi untuk pembelajaran, bukan untuk keputusan operasional nyata.