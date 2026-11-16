"""Generate minimal 32x32 icon.ico for the Windows resource embed (stdlib only).

Rebuild: python scripts/gen_icon.py   (output: src-tauri/icons/icon.ico)
"""
import struct
from pathlib import Path

W = H = 32
BGRA = bytes((0xE0, 0x7B, 0x20, 0xFF))  # brand blue, opaque


def bmp_data() -> bytes:
    pixels = BGRA * (W * H)                      # bottom-up irrelevant for solid color
    and_mask = b"\x00" * (H * 4)                 # 32 bits/row, all opaque
    header = struct.pack("<IiiHHIIiiII", 40, W, H * 2, 1, 32, 0,
                         len(pixels) + len(and_mask), 0, 0, 0, 0)
    return header + pixels + and_mask


def main() -> None:
    bmp = bmp_data()
    ico = (struct.pack("<HHH", 0, 1, 1)
           + struct.pack("<BBBBHHII", W, H, 0, 0, 1, 32, len(bmp), 22)
           + bmp)
    out = Path(__file__).resolve().parent.parent / "src-tauri" / "icons" / "icon.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ico)
    print(f"written: {out} ({len(ico)} bytes)")


if __name__ == "__main__":
    main()
