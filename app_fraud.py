import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deteksi Transaksi Penipuan", page_icon="🛡️")

FITUR = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_all():
    model = joblib.load("artifacts/fraud_model.joblib")
    with open("artifacts/metrics.json") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold


model, THRESHOLD = load_all()


def prediksi(df):
    proba = model.predict_proba(df[FITUR])[:, 1]
    out = df.copy()
    out["probabilitas_penipuan"] = proba.round(4)
    out["rekomendasi"] = ["Curigai" if p >= THRESHOLD else "Normal" for p in proba]
    return out


st.title("🛡️ Deteksi Transaksi Penipuan")
st.caption("Hasil model hanya rekomendasi untuk membantu analis. "
           "Keputusan memblokir transaksi tetap oleh petugas.")
st.write(f"Ambang probabilitas yang dipakai: **{THRESHOLD:.2f}**")

tab1, tab2 = st.tabs(["Unggah CSV", "Contoh transaksi"])

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
                n_curiga = int((hasil["rekomendasi"] == "Curigai").sum())

                c1, c2 = st.columns(2)
                c1.metric("Transaksi diproses", n)
                c2.metric("Ditandai mencurigakan", n_curiga)

                kolom = ["Time", "Amount", "probabilitas_penipuan", "rekomendasi"]
                if "Class" in hasil.columns:
                    kolom.append("Class")
                    asli = int((hasil["Class"] == 1).sum())
                    tertangkap = int(((hasil["Class"] == 1) &
                                      (hasil["rekomendasi"] == "Curigai")).sum())
                    st.info(f"Penipuan sebenarnya di file: {asli}, "
                            f"yang berhasil ditandai: {tertangkap}.")

                curiga = hasil[hasil["rekomendasi"] == "Curigai"] \
                    .sort_values("probabilitas_penipuan", ascending=False)
                if n_curiga:
                    st.subheader("Transaksi yang dicurigai")
                    st.dataframe(curiga[kolom])
                else:
                    st.success("Tidak ada transaksi yang dicurigai.")

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
        else:
            st.success("Rekomendasi: NORMAL")
        label = "Penipuan" if hasil["Class"] == 1 else "Normal"
        st.caption(f"Label asli di data: {label}")