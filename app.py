"""IoT Compressive Sensing Web Dashboard.

Run via:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.cs.encoder import CompressiveEncoder
from src.cs.decoder import build_dct_basis, omp_reconstruct, compute_reconstruction_metrics
from run_demo import generate_synthetic_ecg_window

st.set_page_config(page_title="IoT Compressive Sensing", page_icon="📶", layout="wide")

st.title("📶 Low-Power IoT Biomedical Compressive Sensing & Telemetry")
st.markdown(
    "**Sub-Nyquist Wireless Transmission for Wearable Health Monitors** | "
    "*[GitHub Repository](https://github.com/hosseinnosrati/iot-compressive-sensing)*"
)
st.caption("Drastically extends battery life by reducing on-air wireless transmission volume by 50% to 80%.")

st.sidebar.header("⚙️ Compression Parameters")
cr_slider = st.sidebar.slider("Compression Ratio (% Data Reduction):", 40, 80, 70, 5) / 100.0
omp_iters = st.sidebar.slider("OMP Gateway Iterations:", 10, 50, 30, 5)

N = 256
encoder = CompressiveEncoder(block_size=N, compression_ratio=cr_slider)
Psi = build_dct_basis(N)
x_orig = generate_synthetic_ecg_window(N=N)

# Compress & Reconstruct
y = encoder.encode_block(x_orig)
x_rec = omp_reconstruct(y, encoder.Phi, Psi, max_iter=omp_iters)
metrics = compute_reconstruction_metrics(x_orig, x_rec)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Transmitted", f"{len(y)} / {N} samples", delta=f"-{cr_slider*100:.0f}% reduction")
col2.metric("Clinical PRD Error", f"{metrics['prd_percent']:.2f}%", delta="Normal: < 2.0%", delta_color="inverse")
col3.metric("Reconstruction SNR", f"{metrics['snr_db']:.1f} dB")
col4.metric("Estimated Battery Multiplier", f"~{1.0 / (1.0 - cr_slider + 0.1):.1f}× longer")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
plt.subplots_adjust(hspace=0.3)

ax1.plot(x_orig, label="Original Signal (Sensor Node)", color="#292b2c", lw=1.5)
ax1.plot(x_rec, "--", label="Reconstructed Signal (Gateway)", color="#0275d8", lw=1.4)
ax1.set_title(f"Original vs OMP Reconstruction (CR={cr_slider*100:.0f}%, PRD={metrics['prd_percent']:.2f}%)", fontweight="bold")
ax1.set_ylabel("Amplitude (mV)")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right")

residual = x_orig - x_rec
ax2.plot(residual, color="#d9534f", lw=1.1, label="Error Residual")
ax2.set_title("Reconstruction Residual Error", fontweight="bold")
ax2.set_xlabel("Sample Index")
ax2.set_ylabel("Error (mV)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right")

st.pyplot(fig)

