import fiftyone as fo
import os
from PIL import Image
import numpy as np
import shutil
from fiftyone.utils import openimages


TARGET_RESIZE = (224,224)

def expand_mask_to_full_image(detection):
    cropped_mask = detection.mask 
    
    x, y, w, h = detection.bounding_box
    
    x_px = int(x * TARGET_RESIZE[0])
    y_px = int(y * TARGET_RESIZE[1])
    w_px = int(w * TARGET_RESIZE[0])
    h_px = int(h * TARGET_RESIZE[1])
    
    mask_img = Image.fromarray(cropped_mask)
    resized_mask = mask_img.resize((w_px, h_px), Image.NEAREST) 
    
    full_mask = np.zeros(TARGET_RESIZE[::-1], dtype=np.uint8) 
    
    full_mask[y_px:y_px+h_px, x_px:x_px+w_px] = np.array(resized_mask) * 255
    
    return full_mask

def createDataset(className, typeOfData, size):
    return fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split=typeOfData,
        max_samples=size,
        classes=[className],
        label_types=["segmentations"],
        only_matching=True,
        overwrite = True
    )

def createDsAndExport(className, typeOfData, size):
    dataset = createDataset(className, typeOfData, size)

    for i, sample in enumerate(dataset):
        sample_dir = ""

        if i <= 266:
            sample_dir = f"./samples/{typeOfData}/{className}/{className}{i}"
        else:
            sample_dir = f"./samples/validation/{className}/{className}{i-267}"

        os.makedirs(sample_dir, exist_ok=True)

        img = Image.open(sample.filepath)
        img = img.resize(TARGET_RESIZE, Image.BILINEAR)
        img.save(f"{sample_dir}/image.jpg")

        correct_masks = [
            det for det in sample["ground_truth"].detections 
            if det.label == className
        ]
    
        for j, det in enumerate(correct_masks):
            full_mask = expand_mask_to_full_image(det)
            Image.fromarray((full_mask)).save(f"{sample_dir}/mask_{j}.png")
    
    fo.delete_dataset(dataset.name)

if __name__ == "__main__":

    output_dirs = [
        "./samples/train/Cello",
        "./samples/train/Piano",
        "./samples/train/Pizza",
        "./samples/validation/Cello",
        "./samples/validation/Piano",
        "./samples/validation/Pizza"
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path) 
        os.makedirs(dir_path)

    createDsAndExport("Cello", "train", 344)
    createDsAndExport("Piano", "train", 344)
    createDsAndExport("Pizza", "train", 344)

    # print(np.stack(openimages.get_segmentation_classes(version='v7', dataset_dir=None)))