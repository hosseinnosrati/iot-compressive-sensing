"""IoT Sensor Node: Compressive Sensing Encoder.

Performs sub-Nyquist linear dimensionality reduction y = Phi * x on low-power sensor nodes.
Requires only simple matrix-vector multiply O(M*N), minimizing on-node CPU & energy consumption.
"""

from typing import Tuple
import numpy as np


class CompressiveEncoder:
    """Low-complexity sensing encoder for IoT microcontroller deployment."""

    def __init__(self, block_size: int = 256, compression_ratio: float = 0.70, seed: int = 42):
        """Parameters

        ----------
        block_size : int
            Length of the uncompressed signal block N (e.g., 256 samples).
        compression_ratio : float
            Fraction of data reduction (e.g. 0.70 means transmitting only 30% of samples).
        seed : int
            Fixed pseudo-random seed shared between node and gateway.
        """
        self.N = block_size
        self.CR = compression_ratio
        # M is the number of compressed measurements to transmit
        self.M = int(np.round(self.N * (1.0 - self.CR)))

        # Construct deterministic sensing matrix Phi (normalized Gaussian / Bernoulli)
        rng = np.random.RandomState(seed)
        # Bernoulli matrix (+1 / -1) allows integer/addition-only computation on microcontrollers
        raw_phi = rng.choice([-1.0, 1.0], size=(self.M, self.N))
        self.Phi = raw_phi / np.sqrt(self.M)

    def encode_block(self, signal_block: np.ndarray) -> np.ndarray:
        """Compress block of N samples into M measurements.

        Parameters
        ----------
        signal_block : np.ndarray (N,)

        Returns
        -------
        y : np.ndarray (M,) compressed measurements.
        """
        assert len(signal_block) == self.N, f"Expected block size {self.N}, got {len(signal_block)}"
        return np.dot(self.Phi, signal_block)
