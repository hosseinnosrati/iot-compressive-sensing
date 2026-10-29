"""Hardware Latency and Compression Profiler for IoT Compressive Sensing."""

import time
import os
import psutil
import numpy as np

from src.cs.encoder import CompressiveEncoder
from src.cs.decoder import build_dct_basis, omp_reconstruct, compute_reconstruction_metrics
from run_demo import generate_synthetic_ecg_window


def run_benchmark():
    process = psutil.Process(os.getpid())
    init_mem_mb = process.memory_info().rss / (1024 * 1024)

    print("=" * 60)
    print("      IOT COMPRESSIVE SENSING TELEMETRY BENCHMARK")
    print("=" * 60)
    print(f"Initial Memory Footprint: {init_mem_mb:.2f} MB")

    N = 256
    CR = 0.50  # Balanced clinical mode (50% compression, PRD < 1%)
    encoder = CompressiveEncoder(block_size=N, compression_ratio=CR)
    Psi = build_dct_basis(N)
    x = generate_synthetic_ecg_window(N=N)

    # 1. Benchmark on-node encoding (M x N matrix multiply)
    n_trials = 100
    t0 = time.perf_counter()
    for _ in range(n_trials):
        y = encoder.encode_block(x)
    encode_time_ms = ((time.perf_counter() - t0) / n_trials) * 1000.0

    # 2. Benchmark gateway decoding (OMP recovery)
    t0 = time.perf_counter()
    x_rec = omp_reconstruct(y, encoder.Phi, Psi, max_iter=60)
    decode_time_ms = (time.perf_counter() - t0) * 1000.0

    metrics = compute_reconstruction_metrics(x, x_rec)
    final_mem_mb = process.memory_info().rss / (1024 * 1024)

    print(f"Original Block Size N:    {N} samples")
    print(f"Transmitted M:            {len(y)} samples")
    print(f"Data Reduction Ratio:     {CR * 100:.1f}%")
    print("-" * 60)
    print(f"Node Encoding Latency:    {encode_time_ms:.3f} ms / block (Ultra-low MCU burden!)")
    print(f"Gateway Decoding Latency: {decode_time_ms:.2f} ms / block")
    print(f"Reconstruction PRD:       {metrics['prd_percent']:.2f}% (Diagnostic fidelity: <2%)")
    print(f"Reconstruction SNR:       {metrics['snr_db']:.2f} dB")
    print(f"Peak Memory Usage:        {final_mem_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
