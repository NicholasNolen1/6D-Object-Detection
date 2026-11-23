#!/usr/bin/env python3
"""
Targeted Sequence Downloader for YCBMultiTrack-Real
Streams and extracts specific sequence clips directly from the remote Hugging Face dataset
archive using HTTP Range requests without downloading the entire 9GB archive.
"""

import os
import io
import sys
import argparse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

REMOTE_ZIP_URL = "https://huggingface.co/datasets/tzuyuanlin/YCBMultiTrack/resolve/main/YCBMultiTrack-Real.zip"


class RangeCacheIO(io.RawIOBase):
    """File-like reader that serves reads from cached byte ranges."""
    def __init__(self, total_length, ranges_dict):
        self.length = total_length
        self.ranges = ranges_dict  # sorted list of (start, end, data_bytes)
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        
        target_end = self.pos + size
        for start, end, data in self.ranges:
            if start <= self.pos < end:
                offset_in_buf = self.pos - start
                available = end - self.pos
                to_read = min(size, available)
                self.pos += to_read
                return data[offset_in_buf : offset_in_buf + to_read]

        # If not in cache, fallback to on-demand range request
        req = urllib.request.Request(
            REMOTE_ZIP_URL,
            headers={"Range": f"bytes={self.pos}-{target_end - 1}"}
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def get_archive_length():
    req = urllib.request.Request(REMOTE_ZIP_URL, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers["Content-Length"])


def fetch_byte_range(start, end):
    req = urllib.request.Request(
        REMOTE_ZIP_URL,
        headers={"Range": f"bytes={start}-{end - 1}"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser(description="Download targeted clip from YCBMultiTrack-Real")
    parser.add_argument("--sequence", default="006_mustard_bottle", help="Sequence name")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of frames to extract")
    parser.add_argument("--output-dir", default="data/real", help="Target output directory")
    args = parser.parse_args()

    end_frame = args.start_frame + args.num_frames
    target_frames = set(f"{i:06d}" for i in range(args.start_frame, end_frame))
    seq_prefix = f"YCBMultiTrack-Real/{args.sequence}/"

    print("=" * 70)
    print(f"📥 Fetching targeted clip: {args.sequence} (frames {args.start_frame} to {end_frame - 1})")
    print("=" * 70)

    # 1. Read central directory from end of remote zip
    total_len = get_archive_length()
    print(f"Remote archive total size: {total_len / (1024**3):.2f} GB")

    # Read the last 20MB of the zip to cover the central directory of 105k files
    tail_size = 22 * 1024 * 1024
    tail_start = max(0, total_len - tail_size)
    print(f"Reading central directory (~22MB)...")
    tail_data = fetch_byte_range(tail_start, total_len)

    cache = [(tail_start, total_len, tail_data)]
    bio = RangeCacheIO(total_len, cache)
    zf = zipfile.ZipFile(bio)

    # 2. Identify target files for this clip
    all_infos = zf.infolist()
    needed = []
    for info in all_infos:
        if not info.filename.startswith(seq_prefix):
            continue
        fname = info.filename[len(seq_prefix):]
        # Include metadata files (intrinsics, labels)
        if any(fname.endswith(ext) for ext in [".txt", ".npy", ".json"]) and not "/" in fname:
            needed.append(info)
            continue
        if "annotated_poses" in fname or "masks" in fname or "rgb" in fname or "depth" in fname:
            stem = os.path.splitext(os.path.basename(fname))[0]
            if stem in target_frames or fname.endswith(".npy"):
                needed.append(info)

    if not needed:
        print(f"Error: No matching files found for sequence '{args.sequence}'")
        sys.exit(1)

    print(f"Selected {len(needed)} files covering {args.num_frames} frames.")
    needed.sort(key=lambda x: x.header_offset)

    # 3. Calculate merged byte chunks
    chunks = []
    curr_start = needed[0].header_offset
    curr_end = curr_start + needed[0].compress_size + 256
    for f in needed[1:]:
        f_start = f.header_offset
        f_end = f_start + f.compress_size + 256
        if f_start - curr_end < 512 * 1024:  # merge gaps < 512 KB
            curr_end = max(curr_end, f_end)
        else:
            chunks.append((curr_start, curr_end))
            curr_start = f_start
            curr_end = f_end
    chunks.append((curr_start, curr_end))

    total_download_mb = sum(end - start for start, end in chunks) / (1024**2)
    print(f"Downloading {len(chunks)} merged byte chunk(s) totaling {total_download_mb:.2f} MB...")

    # 4. Download chunks in parallel
    def download_chunk(item):
        idx, (start, end) = item
        data = fetch_byte_range(start, end)
        return (start, end, data)

    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as ex:
        downloaded = list(ex.map(download_chunk, enumerate(chunks)))

    print("Download finished. Extracting files...")
    # Add downloaded chunks to cache
    all_ranges = sorted(cache + downloaded, key=lambda x: x[0])
    cached_io = RangeCacheIO(total_len, all_ranges)
    extract_zf = zipfile.ZipFile(cached_io)

    out_base = os.path.join(args.output_dir, args.sequence)
    os.makedirs(out_base, exist_ok=True)

    for info in needed:
        # Strip the top-level 'YCBMultiTrack-Real/<sequence>/' directory
        rel_path = info.filename[len(seq_prefix):]
        if not rel_path:
            continue
        target_path = os.path.join(out_base, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with extract_zf.open(info) as src, open(target_path, "wb") as dst:
            dst.write(src.read())

    print(f"Successfully extracted {len(needed)} files to '{out_base}'.")


if __name__ == "__main__":
    main()
