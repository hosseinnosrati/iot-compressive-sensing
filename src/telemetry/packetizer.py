"""Wireless IoT Telemetry Packetizer (BLE / LoRa Framing).

Encapsulates compressed measurements into framed packets with sequence tracking and CRC16 checksums.
"""

import struct
from typing import Tuple
import numpy as np


def compute_crc16(data: bytes) -> int:
    """Standard CCITT CRC-16 (poly 0x1021)."""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class TelemetryPacketizer:
    PREAMBLE = 0xAA55

    @classmethod
    def frame_packet(cls, measurements: np.ndarray, seq_num: int, node_id: int = 1) -> bytes:
        # Quantize float measurements to int16 for wireless transmission
        quantized = np.clip(np.round(measurements * 1000.0), -32768, 32767).astype(np.int16)
        payload = quantized.tobytes()

        # Header: Preamble (2B) + Node ID (1B) + Seq Num (2B) + Payload Length (2B)
        header = struct.pack(">HBH H", cls.PREAMBLE, node_id, seq_num & 0xFFFF, len(payload))
        crc = compute_crc16(header + payload)
        return header + payload + struct.pack(">H", crc)

    @classmethod
    def parse_packet(cls, packet: bytes) -> Tuple[int, int, np.ndarray]:
        assert len(packet) >= 9, "Packet too short"
        preamble, node_id, seq_num, length = struct.unpack(">HBH H", packet[:7])
        assert preamble == cls.PREAMBLE, "Invalid preamble"

        payload = packet[7 : 7 + length]
        received_crc = struct.unpack(">H", packet[7 + length : 9 + length])[0]
        assert compute_crc16(packet[: 7 + length]) == received_crc, "CRC mismatch"

        quantized = np.frombuffer(payload, dtype=np.int16)
        measurements = quantized.astype(np.float32) / 1000.0
        return node_id, seq_num, measurements
