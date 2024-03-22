

import os
# os.environ["nnUNet_raw"] = "F:\\Data\\dataset_verseg\\dataset-verse19training\\raw"
# os.environ["nnUNet_preprocessed"] = "F:\\Data\\dataset_verseg\\dataset-verse19training\\preprocess"
# os.environ["nnUNet_results"] = "F:\\Data\\dataset_verseg\\dataset-verse19training\\results"

# nnUNet_raw = os.environ.get('nnUNet_raw')
# nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
# nnUNet_results = os.environ.get('nnUNet_results')

import argparse
import multiprocessing
import shutil
from multiprocessing import Pool
from typing import Optional
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.configuration import default_num_processes
import numpy as np
from pathlib import Path
from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.verse.utils.data_utilities import resample_nib, reorient_to
import nibabel as nib
import nibabel.orientations as nio

v_dict = {
    0: 'bg',
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

def seg_label_map_fn(verse_label):
    if verse_label == 0:
        return 0
    if verse_label in [26, 27]:
        return 2
    elif verse_label in v_dict:
        return 1
    else:
        return 0

def copy_Verse_segment_label_to_nnUnet(in_file: str, out_file: str)-> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in v_dict:
            raise RuntimeError(f'unexpected label: {u}')
    seg_new = np.zeros_like(img_npy)
    for k, _ in v_dict.items():
        seg_new[img_npy == k] = seg_label_map_fn(k)
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def copy_Verse_segment_label_to_nnUnet2(in_file: str, out_file: str)-> None:

    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img_nib = nib.load(in_file)
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=0)
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    img_npy = img_iso.get_fdata()
    unique_labels = np.unique(img_npy)

    seg_new = np.zeros_like(img_npy)
    for ul in unique_labels:
        if ul not in v_dict:
            print(f'{in_file}: unexpected label: {ul}')
            seg_new[img_npy == ul] = 0
            continue
        seg_new[img_npy == ul] = seg_label_map_fn(ul)

    # Create a new NIfTI image with the modified data but original affine and header
    modified_img = nib.Nifti1Image(seg_new, img_iso.affine, img_iso.header)
    # Save the modified NIfTI file
    nib.save(modified_img, out_file)

def copy_Verse_ori_image_to_nnUnet2(in_file: str, out_file: str)-> None:
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
    img_nib = nib.load(in_file)
    img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
    img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
    # Save the modified NIfTI file
    nib.save(img_iso, out_file)

def copy_sample_to_preprocess(img_nii, mask_nii, out_base):
    print(img_nii)
    out_base = Path(out_base)
    case_id = Path(img_nii.stem).stem

    img_out_dir = out_base / "imagesTr"
    print(img_out_dir)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir = out_base / "labelsTr"
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    img_out_nii = img_out_dir / f"{case_id}_0000.nii.gz"
    mask_out_nii = mask_out_dir/ f"{case_id}.nii.gz" 

    if False:
        shutil.copy(img_nii, img_out_nii)
        copy_Verse_segment_label_to_nnUnet(mask_nii, str(mask_out_nii))

    copy_Verse_ori_image_to_nnUnet2(img_nii, str(img_out_nii))
    copy_Verse_segment_label_to_nnUnet2(mask_nii, str(mask_out_nii))

    return case_id
    

def convert_Verse_dataset(source_folder: str, overwrite_target_id: Optional[int] = None,
                        num_processes: int = default_num_processes) -> None:
    # check if target dataset id is taken
    target_id = overwrite_target_id
    existing_datasets = find_candidate_datasets(target_id)
    assert len(existing_datasets) == 0, f"Target dataset id {target_id} is already taken, please consider changing " \
                                        f"it using overwrite_target_id. Conflicting dataset: {existing_datasets} (check nnUNet_results, nnUNet_preprocessed and nnUNet_raw!)"

    target_dataset_name = f"Dataset{target_id:03d}_verse"
    target_folder = join(nnUNet_raw, target_dataset_name)
  
    ignore = ["sub-verse525", "sub-verse577", "sub-verse641", "sub-verse642", "sub-verse593", "sub-verse833"]

    case_ids = []
    # scan source dir
    raw_data_path = Path(source_folder) / "rawdata"
    mask_path = Path(source_folder) / "derivatives"
    # iterate dirs
    for sub_dir in raw_data_path.iterdir():
        skip = False
        for i in ignore:
            if i in str(sub_dir):
                skip = True
                break
        if skip:
            continue
        
        sub_name = sub_dir.name
        # print(sub_name)
        img_nii_files = list(sub_dir.glob("*.nii.gz"))
        mask_files = list((mask_path / sub_name).glob("*.nii.gz"))
        if len(img_nii_files) < 1 or len(mask_files) < 1:
            continue
        assert len(img_nii_files) == len(mask_files)

        for img_nii, mask_nii in zip(img_nii_files, mask_files):
            # copy
            case_id = copy_sample_to_preprocess(img_nii, mask_nii, target_folder)
            case_ids.append(case_id)
        # if len(case_ids) > 5:
        #     break
    
    generate_dataset_json(target_folder,
                          channel_names={0: 'CT', },
                          labels={
                              'background': 0,
                              'vertebra': 1,
                              'others': 2,
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')


def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder. CANNOT be nnUNetv1 dataset! Example: '
                             '/home/fabian/Downloads/Task05_Prostate')
    parser.add_argument('-overwrite_id', type=int, required=False, default=None,
                        help='Overwrite the dataset id. If not set we use the id of the MSD task (inferred from '
                             'folder name). Only use this if you already have an equivalently numbered dataset!')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'Number of processes used. Default: {default_num_processes}')
    args = parser.parse_args()
    convert_Verse_dataset(args.i, args.overwrite_id, args.np)




if __name__ == '__main__':
    print("----")
    convert_Verse_dataset('/workspace/third/mmseg2/mmsegmentation/data/dataset-01training/', overwrite_target_id=999)
    

    # step2:
    # nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
    # python nnunetv2\experiment_planning\plan_and_preprocess_entrypoints.py  -d 999 --verify_dataset_integrit

    # nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
    