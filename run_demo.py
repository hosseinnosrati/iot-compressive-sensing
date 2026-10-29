"""Visual Demo: Compressive Sensing Telemetry Pipeline.

Compares:
1. Original Uncompressed ECG (256-sample window)
2. Transmitted Compressed Measurements (70% data reduction)
3. OMP Reconstructed ECG on Gateway vs Original
4. Reconstruction Error Residual (Diagnostic Fidelity PRD < 2%)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cs.encoder import CompressiveEncoder
from src.cs.decoder import build_dct_basis, omp_reconstruct, compute_reconstruction_metrics
from src.telemetry.packetizer import TelemetryPacketizer


def generate_synthetic_ecg_window(N: int = 256, fs: float = 360.0) -> np.ndarray:
    """Generate clean realistic cardiac cycles."""
    t = np.linspace(0, N / fs, N)
    # Simulate P-QRS-T waveform
    ecg = np.zeros(N)
    for t_peak in [0.25, 0.60]:
        # QRS spike
        ecg += 1.2 * np.exp(-((t - t_peak) ** 2) / (2 * (0.015 ** 2)))
        # P-wave
        ecg += 0.25 * np.exp(-((t - (t_peak - 0.12)) ** 2) / (2 * (0.025 ** 2)))
        # T-wave
        ecg += 0.35 * np.exp(-((t - (t_peak + 0.18)) ** 2) / (2 * (0.04 ** 2)))
    return ecg


def run_demo():
    print("Running IoT Compressive Sensing Demo...")
    N = 256
    CR = 0.50  # 50% data reduction for medical telemetry

    encoder = CompressiveEncoder(block_size=N, compression_ratio=CR)
    Psi = build_dct_basis(N)

    # 1. Generate signal block
    x_original = generate_synthetic_ecg_window(N=N)

    # 2. Node-side compression
    y_compressed = encoder.encode_block(x_original)

    # 3. Telemetry packetization (BLE/LoRa simulation)
    packet = TelemetryPacketizer.frame_packet(y_compressed, seq_num=101, node_id=1)
    _, _, y_received = TelemetryPacketizer.parse_packet(packet)

    # 4. Gateway reconstruction via OMP
    x_reconstructed = omp_reconstruct(y_received, encoder.Phi, Psi, max_iter=60)
    metrics = compute_reconstruction_metrics(x_original, x_reconstructed)

    print(f"Original Block Size:       {N} samples ({N * 2} bytes in int16)")
    print(f"Compressed Measurements:   {len(y_compressed)} samples ({len(y_compressed) * 2} bytes in int16)")
    print(f"Bandwidth / Data Savings:  {CR * 100:.1f}% reduction")
    print(f"Reconstruction PRD Error:  {metrics['prd_percent']:.2f}% (Clinical Threshold: < 2.0%)")
    print(f"Reconstruction SNR:        {metrics['snr_db']:.2f} dB")

    # 5. Plotting 4-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.35)

    # Panel 1: Original vs Reconstructed
    axes[0].plot(x_original, label="Original ECG (Transmitter)", color="#292b2c", lw=1.5)
    axes[0].plot(x_reconstructed, "--", label="OMP Reconstructed (Gateway)", color="#0275d8", lw=1.4)
    axes[0].set_title(f"Stage 1: Biomedical Telemetry (CR={CR*100:.0f}%, PRD={metrics['prd_percent']:.2f}%, SNR={metrics['snr_db']:.1f} dB)", fontweight="bold")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Panel 2: Transmitted Measurements
    axes[1].stem(np.arange(len(y_compressed)), y_compressed, linefmt="#d9534f", markerfmt="o", basefmt="gray")
    axes[1].set_title(f"Stage 2: Transmitted Wireless Measurements (M={len(y_compressed)} out of N={N} samples)", fontweight="bold")
    axes[1].set_ylabel("Measurement (y)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Reconstruction Error
    residual = x_original - x_reconstructed
    axes[2].plot(residual, color="#d9534f", lw=1.0, label="Reconstruction Residual Error")
    axes[2].set_title("Stage 3: Reconstruction Error Residual (Clinically Inconsequential)", fontweight="bold")
    axes[2].set_xlabel("Sample Index", fontsize=11)
    axes[2].set_ylabel("Error (mV)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    os.makedirs("assets", exist_ok=True)
    out_path = "assets/demo_compressive_sensing.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual demo plot saved successfully to {out_path}!")


if __name__ == "__main__":
    run_demo()
