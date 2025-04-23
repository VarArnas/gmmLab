from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torch
from dataset import SegmentationDataset 
from model import SegmentationModel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torchmetrics import F1Score
from torchmetrics.segmentation import DiceScore



NUM_WORKERS = 2
BATCH_SIZE = 24
PREFETCH_FAC = 1
TRAIN_PATH = "./samples/train"
VALIDATION_PATH = "./samples/validation"
LABELS = {"Cello" : 1, "Piano" : 2, "Pizza" : 3}
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel()
model = model.to(device)
scaler = torch.amp.GradScaler(
    'cuda',
    init_scale=32768.0,  
    growth_factor=1.5, 
    backoff_factor=0.5,
    growth_interval=1000
)

def convertMaskTensorToOther(maskTensor):
    return maskTensor.squeeze().long()

transformImage = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3),
    transforms.RandomPosterize(bits=5, p=0.5),
    transforms.RandomInvert(p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transformMask = transforms.Compose([
    transforms.PILToTensor(),  
    convertMaskTensorToOther
])

transformImageValid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def collectPaths(samplesPath):
    samples = []
    classes = os.listdir(samplesPath)

    for clas in classes:
            classSamples = os.listdir(os.path.join(samplesPath, clas))

            for sample in classSamples:
                dirPath = os.path.join(samplesPath, clas, sample) 
                mask_files = [os.path.join(dirPath, f) for f in os.listdir(dirPath) 
                              if f.startswith('mask')]
                samples.append((os.path.join(dirPath, "image.jpg"), mask_files))

    return samples

def train(trainDL, validateDL):
    weights = torch.tensor([1.0, 3.0, 3.0, 3.0]).to(device)
    lossFunc = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.7, min_lr=1e-4)
    torch.cuda.empty_cache()

    best_total = 0.0
    if os.path.exists("./modelData/best_dice.pth"):
        checkpoint = torch.load("./modelData/best_dice.pth")
        best_total = checkpoint['dice'] + checkpoint['f1_macro']

    for epoch in range(NUM_EPOCHS):
        model.train()
        acc_loss = 0.0

        for images, masks in trainDL:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = lossFunc(outputs, masks)
              
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scaler.get_scale() < 1e-4:
                print("Gradient scaling too low!")
                break

            acc_loss += loss.item()

        acc_loss = acc_loss / len(trainDL)
        metrics = inference(validateDL)

        log_text = (
            f"Epoch: {epoch+1}\n"
            f"Loss: {acc_loss:.4f}\n"
            f"Dice_score: {metrics['dice']:.4f}\n"
            f"F1_Macro: {metrics['f1_macro']:.4f}\n"
            f"F1_Micro: {metrics['f1_micro']:.4f}\n"
            f"Dice_single:\n"
            f"   Background: {metrics['dice_single'][0]}\n"
            f"   Cello: {metrics['dice_single'][1]}\n"
            f"   Piano: {metrics['dice_single'][2]}\n"
            f"   Pizza: {metrics['dice_single'][3]}\n"
            f"Last learning rate: {scheduler.get_last_lr()}\n\n"
        )
        print(log_text)
        with open('./modelData/logs.txt', 'a') as log_file:
            log_file.write(log_text)

        current_total = metrics['dice'] + metrics['f1_macro']
        if current_total > best_total and epoch > 20:
            best_total = current_total
            state = {
                'loss': acc_loss,
                'model_weights': model.state_dict(),
                'dice': metrics['dice'],
                'f1_macro': metrics['f1_macro'],
                'f1_micro': metrics['f1_micro'],
                'epoch': epoch+1
            }
            torch.save(state, "./modelData/best_dice.pth")
            
        scheduler.step(metrics['dice'])

        if epoch % 5 == 0:  
            torch.cuda.empty_cache()

def inference(validateDL, model):
    model.eval()

    dice_metric = DiceScore(num_classes=4, average='macro', include_background=False, input_format='index').to('cpu')
    f1_macro = F1Score(task='multiclass', num_classes=4, average='macro').to('cpu')
    f1_micro = F1Score(task='multiclass', num_classes=4, average='micro').to('cpu')
    dice_metric_none = DiceScore(num_classes=4, average='none', input_format='index').to('cpu')

    with torch.inference_mode():
        for images, masks in validateDL:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            predictions = predictions.cpu()
            masks = masks.cpu()

            dice_metric.update(predictions, masks)
            f1_macro.update(predictions, masks)
            f1_micro.update(predictions, masks)
            dice_metric_none.update(predictions, masks)
        
    metrics = {
        'dice' : dice_metric.compute(),
        'f1_macro' : f1_macro.compute(),
        'f1_micro' : f1_micro.compute(),
        'dice_single' : dice_metric_none.compute()
    }

    dice_metric.reset()
    f1_macro.reset()
    f1_micro.reset()
    dice_metric_none.reset()

    return metrics

if __name__ == "__main__":
    print(device)
    trainSamples = collectPaths(TRAIN_PATH)
    validationSamples = collectPaths(VALIDATION_PATH)

    trainDataset = SegmentationDataset(trainSamples, LABELS, transformImage, transformMask)
    validationDataset = SegmentationDataset(validationSamples, LABELS, transformImageValid, transformMask, isTraining=False)

    trainDL = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FAC, pin_memory=True, persistent_workers=True)
    validationDL = DataLoader(validationDataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FAC, pin_memory=True, persistent_workers=True)

    train(trainDL, validationDL)