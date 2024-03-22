
import glob
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

def check_data(train_folder, label_folder):

    train_images = sorted(glob.glob(os.path.join(train_folder, '*.png')))
    label_images = sorted(glob.glob(os.path.join(label_folder, '*.png')))

    image_data = []
    size_set = set()

    for train_img_path, label_img_path in zip(train_images, label_images):
        train_img = Image.open(train_img_path)
        label_img = Image.open(label_img_path)

        train_img_size = train_img.size
        label_img_size = label_img.size

        size_set.add(train_img.size)
        size_set.add(label_img.size)

        image_data.append({
            'train_image_path': train_img_path,
            'label_image_path': label_img_path,
            'train_image_size': train_img_size,
            'label_image_size': label_img_size
        })

    # Create a Pandas DataFrame from the image_data
    image_df = pd.DataFrame(image_data)

    # Store the DataFrame in a CSV file
    image_df.to_csv('image_details.csv', index=False)

    print(size_set)

import nibabel as nib
import csv
# custom
import sys
sys.path.insert(0, os.path.realpath((__file__) + "/../utils"))
from data_utilities import *
def extract_data_info(file_path):
    img_nib = nib.load(file_path)
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    header = img_iso.header
    data_info = {
        'File Name': os.path.basename(file_path),
        'Size of the data': header.get_data_shape(),
        'Scale of the data': header.get_zooms(),   
        # 'Orientation of the data': header.get_best_affine().tolist(),
        'Axes units': header.get_xyzt_units(),
        'Axes codes': nib.orientations.aff2axcodes(header.get_best_affine())
    }
    return data_info

def search_and_extract_data_info(directory):
    data_info_list = []
    for root, subdirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                data_info = extract_data_info(file_path)
                print(data_info)
                data_info_list.append(data_info)
    return data_info_list

def save_to_csv(data_info_list, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        fieldnames = ['File Name', 'Size of the data', 'Scale of the data', 'Orientation of the data', 'Axes units', 'Axes codes']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data_info in data_info_list:
            writer.writerow(data_info)


if __name__ == "__main__":
    
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process training and label image folders.')

    # Add arguments for train and label folders
    parser.add_argument('--data_dir', type=str, default="/workspace/third/mmseg2/mmsegmentation/data/extract", help='Path to the training image folder')
    parser.add_argument('--phase', type=str, default="val", help='Path to the label image folder')
    parser.add_argument('--scan_nii', action='store_true', help="")

    # Parse the arguments
    args = parser.parse_args()

    if args.scan_nii:
        data_info = search_and_extract_data_info(args.data_dir)
        save_to_csv(data_info, "nii_info.csv")
        exit(0)

    image_dir = os.path.join(args.data_dir, "img_dir" ,args.phase)
    label_dir = os.path.join(args.data_dir, "ann_dir" ,args.phase)

    check_data(image_dir, label_dir)


