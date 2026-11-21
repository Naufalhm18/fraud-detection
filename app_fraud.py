import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Deteksi Transaksi Penipuan", page_icon="🛡️")

FITUR = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_PATH = "data/creditcard.csv"
ADA_DATA = os.path.exists(DATA_PATH)


@st.cache_resource
def load_resmi():
    model = joblib.load("artifacts/fraud_model.joblib")
    with open("artifacts/metrics.json") as f:
        ambang = json.load(f)["threshold"]
    return model, ambang


@st.cache_resource(show_spinner="Memuat data...")
def load_split():
    df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
    X = df[FITUR]
    y = df["Class"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@st.cache_resource(show_spinner="Menghitung prediksi data uji...")
def proba_uji_resmi():
    _, X_test, _, _ = load_split()
    return load_resmi()[0].predict_proba(X_test)[:, 1]


def kurangi_normal(X_train, y_train, persen):
    """Kurangi transaksi normal agar penipuan = persen % dari data latih."""
    idx_fraud = y_train[y_train == 1].index.values
    idx_normal = y_train[y_train == 0].index.values
    n_normal = int(len(idx_fraud) * (100 - persen) / persen)
    if n_normal >= len(idx_normal):
        return X_train, y_train
    pilih = np.random.RandomState(42).choice(idx_normal, n_normal, replace=False)
    idx = np.concatenate([idx_fraud, pilih])
    return X_train.loc[idx], y_train.loc[idx]


def latih(persen, n_pohon, kedalaman, min_daun, seimbang):
    X_train, X_test, y_train, _ = load_split()
    if persen is not None:
        X_train, y_train = kurangi_normal(X_train, y_train, persen)
    model = Pipeline([
        ("prep", ColumnTransformer(
            [("scale", RobustScaler(), ["Time", "Amount"])],
            remainder="passthrough")),
        ("model", RandomForestClassifier(
            n_estimators=n_pohon,
            max_depth=kedalaman,
            min_samples_leaf=min_daun,
            max_features=0.3,
            class_weight="balanced_subsample" if seimbang else None,
            n_jobs=-1,
            random_state=42)),
    ])
    model.fit(X_train, y_train)
    return {
        "model": model,
        "proba_uji": model.predict_proba(X_test)[:, 1],
        "n_latih": int(len(y_train)),
        "n_fraud": int(y_train.sum()),
    }


MODEL_RESMI, AMBANG_RESMI = load_resmi()

# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
sumber = st.sidebar.radio("Model yang dipakai",
                          ["Model resmi", "Model hasil latih ulang"])
ambang = st.sidebar.slider("Ambang probabilitas", 0.05, 0.95,
                           float(round(AMBANG_RESMI, 2)), 0.01)
st.sidebar.caption("Ambang lebih rendah: lebih banyak penipuan tertangkap, "
                   "tetapi alarm palsu bertambah.")
batas_aman = st.sidebar.slider("Batas aman", 0.0, float(ambang),
                               float(min(0.01, ambang)), 0.01)
st.sidebar.caption("Probabilitas di bawah batas aman = Aman. Di antara batas "
                   "aman dan ambang = Perlu dicek.")

with st.sidebar.expander("Latih ulang model (eksplorasi)"):
    if not ADA_DATA:
        st.info("Butuh file data/creditcard.csv. "
                "Jalankan notebook untuk mengunduhnya.")
    asli = st.checkbox("Pakai proporsi penipuan asli (sekitar 0,17%)", value=True)
    persen = None
    if not asli:
        persen = st.slider("Persentase penipuan di data latih (%)",
                           1.0, 50.0, 10.0, 0.5)
    n_pohon = st.slider("Jumlah pohon", 50, 300, 100, 25)
    kedalaman = st.slider("Kedalaman maksimum (0 = tanpa batas)", 0, 40, 20)
    min_daun = st.slider("Minimum sampel per daun", 1, 10, 2)
    seimbang = st.checkbox("Pakai class_weight seimbang", value=True)
    if st.button("Latih ulang", disabled=not ADA_DATA):
        with st.spinner("Melatih model, mohon tunggu..."):
            st.session_state["baru"] = latih(
                persen, n_pohon, kedalaman or None, min_daun, seimbang)
        st.success("Selesai. Pilih 'Model hasil latih ulang' di atas.")

pakai_baru = sumber == "Model hasil latih ulang" and "baru" in st.session_state
if sumber == "Model hasil latih ulang" and not pakai_baru:
    st.sidebar.warning("Belum ada model hasil latih ulang, memakai model resmi.")
model = st.session_state["baru"]["model"] if pakai_baru else MODEL_RESMI


def tingkat(p):
    if p >= ambang:
        return "Curigai"
    if p >= batas_aman:
        return "Perlu dicek"
    return "Aman"


def prediksi(df):
    proba = model.predict_proba(df[FITUR])[:, 1]
    out = df.copy()
    out["probabilitas_penipuan"] = proba.round(4)
    out["rekomendasi"] = [tingkat(p) for p in proba]
    return out


# ---------------- Halaman utama ----------------
st.title("🛡️ Deteksi Transaksi Penipuan")
st.caption("Hasil model hanya rekomendasi untuk membantu analis. "
           "Keputusan memblokir transaksi tetap oleh petugas.")
nama_model = "hasil latih ulang (eksplorasi)" if pakai_baru else "resmi"
st.write(f"Model aktif: **{nama_model}** | Ambang: **{ambang:.2f}** | "
         f"Batas aman: **{batas_aman:.2f}**")

tab1, tab2, tab3 = st.tabs(["Unggah CSV", "Contoh transaksi", "Evaluasi model"])

with tab1:
    st.write("Unggah file CSV dengan kolom berikut:")
    st.code(", ".join(FITUR))
    file = st.file_uploader("File CSV", type="csv")

    if file is not None:
        try:
            batch = pd.read_csv(file)
            batch.columns = batch.columns.str.strip()
            kurang = [c for c in FITUR if c not in batch.columns]
            if kurang:
                st.error(f"Kolom berikut tidak ada di file: {', '.join(kurang)}")
            else:
                hasil = prediksi(batch)
                n = len(hasil)
                jml = hasil["rekomendasi"].value_counts()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Diproses", n)
                c2.metric("Curigai", int(jml.get("Curigai", 0)))
                c3.metric("Perlu dicek", int(jml.get("Perlu dicek", 0)))
                c4.metric("Aman", int(jml.get("Aman", 0)))

                kolom = ["Time", "Amount", "probabilitas_penipuan", "rekomendasi"]
                if "Class" in hasil.columns:
                    kolom.append("Class")
                    st.write("Rincian terhadap label asli di file:")
                    rinci = pd.crosstab(
                        hasil["rekomendasi"],
                        hasil["Class"].map({0: "Normal", 1: "Penipuan"}))
                    rinci = rinci.reindex(["Curigai", "Perlu dicek", "Aman"]) \
                        .fillna(0).astype(int)
                    st.dataframe(rinci)

                tampil = st.radio("Tampilkan transaksi",
                                  ["Curigai", "Perlu dicek", "Aman", "Semua"],
                                  horizontal=True)
                if tampil == "Semua":
                    bagian = hasil
                else:
                    bagian = hasil[hasil["rekomendasi"] == tampil]
                bagian = bagian.sort_values("probabilitas_penipuan",
                                            ascending=(tampil == "Aman"))
                if len(bagian):
                    st.dataframe(bagian[kolom])
                else:
                    st.info("Tidak ada transaksi pada kategori ini.")
                st.caption("Aman berarti probabilitas penipuan di bawah batas "
                           "aman, bukan jaminan. Model masih bisa melewatkan "
                           "penipuan bernilai kecil.")

                st.download_button(
                    "Unduh semua hasil (CSV)",
                    hasil.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_deteksi.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"File tidak bisa diproses: {e}")

with tab2:
    st.write("Ambil satu transaksi acak dari data contoh, lalu lihat prediksi model.")
    if st.button("Ambil transaksi acak"):
        contoh = pd.read_csv("data/contoh_transaksi_berlabel.csv")
        baris = contoh.sample(1)
        hasil = prediksi(baris).iloc[0]
        st.write(f"Waktu: {hasil['Time']:.0f} | Nilai transaksi: {hasil['Amount']:.2f}")
        st.write(f"Probabilitas penipuan: **{hasil['probabilitas_penipuan']:.1%}**")
        if hasil["rekomendasi"] == "Curigai":
            st.error("Rekomendasi: CURIGAI (kemungkinan penipuan)")
        elif hasil["rekomendasi"] == "Perlu dicek":
            st.warning("Rekomendasi: PERLU DICEK (probabilitas menengah)")
        else:
            st.success("Rekomendasi: AMAN (probabilitas sangat rendah)")
        label = "Penipuan" if hasil["Class"] == 1 else "Normal"
        st.caption(f"Label asli di data: {label}")

with tab3:
    if not ADA_DATA:
        st.info("Evaluasi memerlukan data/creditcard.csv. "
                "Jalankan notebook untuk mengunduhnya.")
    else:
        _, X_test, _, y_test = load_split()
        if pakai_baru:
            proba = st.session_state["baru"]["proba_uji"]
        else:
            proba = proba_uji_resmi()
        pred = (proba >= ambang).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        st.write(f"Data uji: {len(y_test):,} transaksi, "
                 f"{int(y_test.sum())} di antaranya penipuan.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
        c2.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
        c3.metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
        c4.metric("PR-AUC", f"{average_precision_score(y_test, proba):.3f}")

        st.dataframe(pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Sebenarnya Normal", "Sebenarnya Penipuan"],
            columns=["Diprediksi Normal", "Diprediksi Penipuan"]))

        if pakai_baru:
            b = st.session_state["baru"]
            st.caption(f"Model dilatih dengan {b['n_latih']:,} transaksi, "
                       f"{b['n_fraud']} penipuan ({b['n_fraud'] / b['n_latih']:.2%}).")
        st.warning("Bagian ini untuk eksplorasi. Mengubah pengaturan sambil "
                   "melihat skor data uji membuat skor itu terlalu optimis. "
                   "Angka resmi proyek ada di laporan evaluasi.")