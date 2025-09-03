# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:04:55 2025

@author: Erfan
"""

# scripts/download_transcripts.py

import os
import gdown

# Google Drive folder URL (public, provided by you)
DRIVE_URL = "https://drive.google.com/drive/folders/1FuAi9ULYW3tJqSe92TIHPgm73lObMrTW?usp=sharing"

def main():
    out_dir = "data/transcripts"
    os.makedirs(out_dir, exist_ok=True)

    print(f"📥 Downloading transcripts from: {DRIVE_URL}")
    print(f"📂 Saving into: {out_dir}")

    gdown.download_folder(
        url=DRIVE_URL,
        output=out_dir,
        quiet=False,
        use_cookies=False
    )

    print("✅ Download complete.")

if __name__ == "__main__":
    main()
