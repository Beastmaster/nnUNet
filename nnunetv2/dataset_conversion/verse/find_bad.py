import os
from PIL import Image
from tqdm import tqdm

def find_corrupted_images(directory):
    corrupted_images = []
    for root, dirs, files in tqdm(os.walk(directory)):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                corrupted_images.append(file_path)
    return corrupted_images

directory_to_search = '/workspace/third/mmseg2/mmsegmentation/data/extract'
corrupted_files = find_corrupted_images(directory_to_search)
print("Corrupted image files:")
for file in corrupted_files:
    print(file)