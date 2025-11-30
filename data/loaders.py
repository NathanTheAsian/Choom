import struct
import numpy as np

def read_gnt(path):
    samples = []

    with open(path, "rb") as f:
        while True:
            # ---- 1. Read sample length (4 bytes, unsigned int, little-endian)
            bytes4 = f.read(4)
            if not bytes4:
                break  # EOF

            sample_size = struct.unpack("<I", bytes4)[0]

            # ---- 2. Read character tag code (2 bytes)
            tag_bytes = f.read(2)
            tag_code = tag_bytes.hex()  # stored little-endian

            # ---- 3. Width (2 bytes unsigned short)
            width = struct.unpack("<H", f.read(2))[0]

            # ---- 4. Height (2 bytes unsigned short)
            height = struct.unpack("<H", f.read(2))[0]

            # ---- 5. Bitmap (width * height bytes)
            bitmap_data = f.read(width * height)

            bitmap = np.frombuffer(bitmap_data, dtype=np.uint8).reshape((height, width))

            samples.append({
                "tag_code": tag_code,         # GB2312 LABEL
                "width": width,
                "height": height,
                "bitmap": bitmap
            })

    return samples

samples = read_gnt("C:/Users/natha/OneDrive/Documents/Desktop/Choom/Choom/data/raw/001-f.gnt")


for s in samples[:5]:
    print(f"Character: {s['tag_code']}, Size: {s['width']}x{s['height']}")



