

python verse/extract.py  \
    --source_root  /workspace/third/mmseg2/mmsegmentation/data/dataset-01training \
    --save_root  /workspace/third/mmseg2/mmsegmentation/data/extract2 \
    --phase  train

python verse/extract.py  \
    --source_root  /workspace/third/mmseg2/mmsegmentation/data/dataset-02validation \
    --save_root  /workspace/third/mmseg2/mmsegmentation/data/extract2 \
    --phase  val

python verse/extract.py  \
    --source_root  /workspace/third/mmseg2/mmsegmentation/data/dataset-03test \
    --save_root  /workspace/third/mmseg2/mmsegmentation/data/extract2 \
    --phase  test


