import random
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as TF



class SegmentationDataset(Dataset):
    def __init__(self, samplesPaths, labels, transformImage=None, transformMask=None, isTraining=True):
        self.labels = labels
        self.transformImage = transformImage
        self.transformMask = transformMask
        self.samples = samplesPaths
        self.isTraining = isTraining
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, mask_paths = self.samples[index]

        merged_mask = np.zeros((224, 224), dtype=np.uint8)
        class_index = self.labels[Path(img_path).parents[1].name]
        
        for mask_file in mask_paths:
            mask = Image.open(mask_file).convert('L')
            mask_array = np.array(mask)
            binary_mask = (mask_array > 0).astype(np.uint8)
            merged_mask = np.maximum(merged_mask, binary_mask * class_index)

        mask_pil = Image.fromarray(merged_mask)
        image = Image.open(img_path).convert('RGB')

        rand_flip = random.random() > 0.5
        rand_rotate = random.uniform(-30, 30)

        if self.isTraining and rand_flip:
            mask_pil = TF.hflip(mask_pil)
            image = TF.hflip(image)
        if self.isTraining and abs(rand_rotate) > 1:
            mask_pil = TF.rotate(mask_pil, angle=rand_rotate, interpolation=TF.InterpolationMode.NEAREST)
            image = TF.rotate(image, angle=rand_rotate, interpolation=TF.InterpolationMode.BILINEAR)

        if self.transformImage:
            image = self.transformImage(image)
        if self.transformMask:
            mask = self.transformMask(mask_pil)

        return image, mask