#!/usr/bin/env python3
import os
import argparse
from PIL import Image

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

def main():
    parser = argparse.ArgumentParser(description="Select and convert left*.png to sequential JPEGs with stride and max frames.")
    parser.add_argument('--input', type=str, required=True, help='Input directory containing left*.png images')
    parser.add_argument('--stride', type=int, default=1, help='Select one every N frames (default: 1)')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum number of frames to process (default: 0, meaning no limit)')
    args = parser.parse_args()

    input_dir = args.input
    stride = max(1, args.stride)
    max_frames = args.max_frames

    png_files = sorted([f for f in os.listdir(input_dir) if f.startswith('left') and f.endswith('.png')])
    selected = png_files[::stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    n_selected = len(selected)
    print(f"Selected {n_selected} frames out of {len(png_files)} (stride={stride}, max_frames={max_frames}).")
    if n_selected == 0:
        print("No images to process. Exiting.")
        return
    print(f"Starting conversion to JPEG...")

    iterator = tqdm(enumerate(selected), total=n_selected, desc="Converting") if use_tqdm else enumerate(selected)
    for idx, fname in iterator:
        in_path = os.path.join(input_dir, fname)
        # Compute the original frame index
        orig_idx = idx * stride
        out_name = f"{orig_idx:05d}.jpg"
        out_path = os.path.join(input_dir, out_name)
        if not use_tqdm:
            print(f"[{idx+1}/{n_selected}] Converting {fname} -> {out_name}")
        with Image.open(in_path) as im:
            im = im.convert('RGB')
            im.save(out_path, 'JPEG', quality=95)
        os.remove(in_path)
    print("Conversion complete.")

if __name__ == '__main__':
    main() 