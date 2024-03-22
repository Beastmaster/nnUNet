



export nnUNet_raw=/workspace/third/mmseg2/mmsegmentation/data/dataset-01training/
export nnUNet_preprocessed=/workspace/third/mmseg2/mmsegmentation/data/nnunet
export nnUNet_results=/workspace/third/mmseg2/mmsegmentation/data/nnunet

function convert() {
  python nnunetv2/dataset_conversion/convert_VerSe_dataset.py
}

function preprocess() {
  python nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py  \
	-d 999 --verify_dataset_integrity
}

function train() {
  python  nnunetv2/run/run_training.py  999  3d_fullres  1 --c
}

#convert
#preprocess
train




