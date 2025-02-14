#!/usr/bin/env python3
"""
This script downloads and extracts the MNIST dataset files:
  - train-images.idx3-ubyte
  - train-labels.idx1-ubyte
  - t10k-images.idx3-ubyte
  - t10k-labels.idx1-ubyte

It uses a reliable mirror hosted on Google Cloud Storage. After running this script,
the MNIST files will be available in the working directory, and you can immediately run
cnn_optimized.cu.
"""

import os
import urllib.request
import gzip
import shutil

def download_and_extract(url, dest_filename):
    if os.path.exists(dest_filename):
        print(f"{dest_filename} already exists. Skipping download.")
        return

    gz_filename = dest_filename + ".gz"
    print(f"Downloading {url} to {gz_filename} ...")
    urllib.request.urlretrieve(url, gz_filename)
    
    print(f"Extracting {gz_filename} to {dest_filename} ...")
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(dest_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Removing {gz_filename} ...")
    os.remove(gz_filename)
    print(f"{dest_filename} is ready.\n")

def main():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        ("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")
    ]
    
    for gz_file, filename in files:
        url = base_url + gz_file
        download_and_extract(url, filename)
    
    print("All MNIST files have been downloaded and extracted.")

if __name__ == "__main__":
    main()
