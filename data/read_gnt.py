import os
import struct
import numpy as np
from PIL import Image

# Set your output directory
output_dir = "C:/Users/natha/OneDrive/Documents/Desktop/Choom/Choom/output"
os.makedirs(output_dir, exist_ok=True)

def read_gnt(path):
    samples = []
    index = 0  # unique filenames

    with open(path, "rb") as f:
        while True:
            # ---- 1. Length (4 bytes, little-endian)
            length_bytes = f.read(4)
            if not length_bytes:
                break
            sample_size = struct.unpack("<I", length_bytes)[0]

            # ---- 2. Tag code (2 bytes, BIG-ENDIAN)
            tag_bytes = f.read(2)
            if len(tag_bytes) < 2:
                break

            # Correct GB2312 hex code representation (hi-lo)
            tag_code = f"{tag_bytes[0]:02x}{tag_bytes[1]:02x}"

            # ---- 3. Width (2 bytes, little-endian)
            width = struct.unpack("<H", f.read(2))[0]

            # ---- 4. Height (2 bytes, little-endian)
            height = struct.unpack("<H", f.read(2))[0]

            # ---- 5. Bitmap
            bitmap_data = f.read(width * height)
            if len(bitmap_data) < width * height:
                break

            bitmap = np.frombuffer(bitmap_data, dtype=np.uint8).reshape((height, width))

            # ---- 6. Save PNG
            img = Image.fromarray(bitmap).convert("L")
            png_path = os.path.join(output_dir, f"{tag_code}_{index}.png")
            img.save(png_path)

            # ---- 7. Decode character

            try:
                char = tag_bytes.decode("gb2312")
            except:
                char = "?"

            # ---- 8. Store record
            samples.append({
                "tag_code": tag_code,
                "char": char,
                "width": width,
                "height": height,
                "bitmap": bitmap,
                "fileName": png_path
            })

            # ---- 9. Print the mapping
            print(f"GB2312: {tag_code}, Char: {char}, Size: {width}x{height}, File: {png_path}")

            index += 1

    return samples


# Run it
samples = read_gnt("C:/Users/natha/OneDrive/Documents/Desktop/Choom/Choom/data/raw/001-f.gnt")
