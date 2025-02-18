# Arnas Variakojis
# LSP: 2213811
# Variantas: resnet50 [Goose, Jellyfish, Snail] 


import torch
import torchvision.models as models
from PIL import Image
import os
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache

# initialize global variables
dataset_path = "./images"
batch_size = 32
num_workers = 4
thresholds = [0.5, 0.5, 0.5] # order in which images folder is set, i.e. : [Goose, Jellyfish, Snail]
prefetch_factor = 2
task = "multilabel"
num_labels = len(thresholds)

# initialize metrics
accuracy_metric = Accuracy(task=task, num_labels=num_labels, average="macro")
precision_metric = Precision(task=task, num_labels=num_labels, average="macro")
recall_metric = Recall(task=task, num_labels=num_labels, average="macro")
f1_metric = F1Score(task=task, num_labels=num_labels, average="macro")

# load the model and set it to be using CUDA cors of gpu if available
def load_resnet():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    model = model.to(device)
    return model, device, weights

# custom dataset class for dataloader
class ImagesDataset(Dataset):
    
    # called once upon creation, to initialize and create objects fields
    def __init__(self, images, labels, num_classes, transform):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.transform = transform

    # used by the dataloader to know how long the dataset is
    def __len__(self):
        return len(self.images)
    
    # cache to store transformed images, and to reuse them, if wanting to run evaluation again, without needing to transform
    @lru_cache(maxsize=256) 
    def load_and_transform(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
    # used by dataloader workers to retrieve transformed images from the dataset into batch
    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image_transformed = self.load_and_transform(image_path)

        # from scalar label value create a vector which is tensor([num_classes]) made up of zeros
        # at the index of the label add 1
        label_one_hot = torch.zeros(self.num_classes)
        label_one_hot[label] = 1

        return image_transformed, label_one_hot


# loads the openImage images from a folder and creates labels for the different classes
def load_images(image_folder):

    image_paths = []
    labels = []
    
    class_names = os.listdir(image_folder)

    # creation of each picture's unique path variable and label
    for id, class_name in enumerate(class_names):
        class_path = os.path.join(image_folder, class_name, "images") 
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(id)

    return image_paths, labels, class_names


# feeding the images to the model, getting logits, transforming to probabilities and applying the thresholds
def predict_image(data_loader, model, device, indices):

    # inference_mode decorator to make calculations more efficient, by removing gradient calculations and flags from tensors 
    with torch.inference_mode():

        # taking batches instead of single images from data_loader
        for images_batch, labels in data_loader:

            # add images and labels to the GPU device
            images_batch = images_batch.to(device)
            labels = labels.to(device)

            # retrieve a batch of logits
            logits = model(images_batch)

            # apply sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(logits)

            # find the probabilities of the relevant indices of the 3 chosen classes
            selected_probs = probabilities[:, indices]

            # apply thresholds to the batch
            thresholds = evaluate_threshold(selected_probs)

            # Add batches to the metrics, inside of which TP, TN, FP, FN will be retrieved from predictions and labels
            accuracy_metric.update(thresholds.cpu(), labels.cpu())
            precision_metric.update(thresholds.cpu(), labels.cpu())
            recall_metric.update(thresholds.cpu(), labels.cpu())
            f1_metric.update(thresholds.cpu(), labels.cpu())


# apply the threshold to a batch of probabilties
def evaluate_threshold(output_probs):

    # transform threshold array to a tensor with the correct shape, for comparison operations
    thresholds_tensor = torch.tensor(thresholds, device=output_probs.device).view(1, -1)
    return (output_probs > thresholds_tensor).int()


# caclulate metrics
def compute_metrics():
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    return accuracy.item(), precision.item(), recall.item(), f1.item()


def main():

    model, device, weights = load_resnet()

    images, labels, class_names = load_images(dataset_path)

    # retrieve all of the classes from resnet50 dataset
    # find the relevant (the ones chosen from openImages) classes indices
    resnet_classes = weights.meta["categories"][:]
    indices = [resnet_classes.index(name) for name in class_names if name in resnet_classes]

    # intializes custom dataset, and pass transform function of resnet50
    dataset = ImagesDataset(images, labels, len(set(labels)), weights.transforms())

    # intialize dataloader with batch size, workers and prefetch factor
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch_factor)

    predict_image(data_loader, model, device, indices)

    accuracy, precision, recall, f1 = compute_metrics()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()