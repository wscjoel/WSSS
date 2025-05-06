import os
import shutil
import gzip


def decompress_nii_gz(file_path):
    with gzip.open(file_path, 'rb') as f_in:
        output_path = os.path.splitext(file_path)[0]
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def batch(folder_path):
    for root, dir, files in os.walk(folder_path):
        for file in files:
            if file.endswith('nii.gz'):
                file_path = os.path.join(root, file)
                decompress_nii_gz(file_path)



batch('./labels')