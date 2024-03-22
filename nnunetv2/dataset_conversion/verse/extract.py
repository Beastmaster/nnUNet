

import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

# custom
import sys
sys.path.insert(0, os.path.realpath((__file__) + "/../utils"))
print(sys.path)
from data_utilities import *
from pathlib import Path
from PIL import Image
import cv2

def print_header(header):
    print("Size of the data:", header.get_data_shape())
    # Extract and print the scale of the data
    print("Scale of the data:", header.get_zooms())
    # Extract and print the orientation of the data
    print("Orientation of the data:", header.get_best_affine())
    print("Axes units:", header.get_xyzt_units())
    axescode = nib.orientations.aff2axcodes(header.get_best_affine())
    print("Axes codes:", axescode)

class NiiSample:
    """
    1-7: cervical spine: C1-C7
    8-19: thoracic spine: T1-T12
    20-25: lumbar spine: L1-L6
    26: sacrum - not labeled in this dataset
    27: cocygis - not labeled in this dataset
    28: additional 13th thoracic vertebra, T13
    """
    def __init__(self, img_name, mask_name):
        self.img_np, self.img_header = self._extract_nii(img_name, 3)
        self.mask_np, self.mask_header = self._extract_nii(mask_name, 0)
        self.shape = self.img_np.shape
        self.name = Path(img_name).stem.split('.')[0]

        print(self.img_np.shape)
        print_header(self.img_header)

    def extract_sag(self):
        """
        inter_order = 0 for mask
        """
        for i in range(self.shape[2]):
            img, label = self.fix_channel(self.img_np[:,:,i], self.mask_np[:, :, i])
            yield (i, img, label)
    
    def extract_cor(self):
        for i in range(self.shape[2]):
            img, label = self.fix_channel(self.img_np[:,i, :], self.mask_np[:, i, :])
            yield (i, img, label)

    def _extract_nii(self, img_name, inter_order = 3):
        img_nib = nib.load(img_name)
        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=inter_order)
        img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
        return img_iso.get_fdata(), img_iso.header
    
    def _extract_niix(self, nii_path, inter_order = 3):
        img_nib = nib.load(nii_path)
        img_nib = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=inter_order)
        return img_nib.get_fdata(), img_nib.header

    def fix_channel(self, img, label):
        img = Image.fromarray(img).convert('RGB')

        label = Image.fromarray(label).convert('L')
        np_label = np.array(label)
        minmax = np.min(np_label), np.max(np_label)
        return img, label



def SaveSample(sample, save_root, phase):
    """
    ### output format:
        # ├── data
        #      ├── my_dataset
        #           ├── img_dir
        #                ├── train
        #                     ├── xxx{img_suffix}
        #                     ├── yyy{img_suffix}
        #                     ├── zzz{img_suffix}
        #                ├── val
        #           ├── ann_dir
        #                ├── train
        #                     ├── xxx{seg_map_suffix}
        #                     ├── yyy{seg_map_suffix}
        #                     ├── zzz{seg_map_suffix}
        #                ├── val

    """
    print(f"saving sample: {sample.name}")
    name = sample.name
    img_dir = Path(save_root) / "img_dir" / phase
    ann_dir = Path(save_root) / "ann_dir" / phase

    img_dir.mkdir(exist_ok = True, parents=True)
    ann_dir.mkdir(exist_ok = True, parents=True)

    for i, sag_img, sag_mask in sample.extract_sag():
        img_name = img_dir / f"{sample.name}_{i}_sag.png"
        sag_img.save(str(img_name))
        # cv2.imwrite(, sag_img)
        mask_name = ann_dir / f"{sample.name}_{i}_sag_mask.png"
        sag_mask.save(str(mask_name))
        # cv2.imwrite(str(mask_name), sag_mask)

def process_fn(sub_dir, mask_path, save_root, phase):
    sub_name = sub_dir.name
    # print(sub_name)
    nii_files = list(sub_dir.glob("*.nii.gz"))
    if len(nii_files) < 1:
        return
    img_nii = nii_files[0]
    print(img_nii.name.split('.')[0])
    mask_files = list((mask_path / sub_name).glob("*.nii.gz"))
    if len(mask_files) < 1:
        return
    mask_nii = mask_files[0]
    data_sample = NiiSample(img_nii, mask_nii)
    SaveSample(data_sample, save_root, phase)


if __name__ == "__main__":
    import argparse
    from multiprocessing import Pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str, default="verse/data/dataset-verse19validation", help="")
    parser.add_argument("--save_root", type=str, default="verse/data/extract", help="")
    parser.add_argument("--phase", type=str, default="train", help="train/test/val")
    args = parser.parse_args()

    raw_data_path = Path(args.source_root) / "rawdata"
    mask_path = Path(args.source_root) / "derivatives"
    
    # iterate dirs
    # for sub_dir in raw_data_path.iterdir():
    #     process_fn(sub_dir, mask_path, args.save_root, args.phase)

    rets = []
    with Pool(5) as p:
        for sub_dir in raw_data_path.iterdir():
            ret = p.apply_async(process_fn, args=(sub_dir, mask_path, args.save_root, args.phase))
            rets.append(ret)
        
        [ret.get() for ret in rets]
            
        