import torch
from torch import optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from PIL import Image
import os
from sklearn.model_selection import train_test_split

task = "multiclass"
num_classes = 3
image_folder = "../Inference_RESNET/images"
batch_size = 32
learningRate = 0.001

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),      
    transforms.RandomRotation(degrees=15),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.conv_stack(x)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNetwork()
model = model.to(device)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]
        image = self.transform(image)
        return image, label

def load_images(image_folder):

    image_paths = []
    labels = []
    
    class_names = os.listdir(image_folder)

    for id, class_name in enumerate(class_names):
        class_path = os.path.join(image_folder, class_name, "images") 
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(id)

    return image_paths, labels, class_names

def train_model(dataloader):
    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        acc_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lossFunc(outputs, labels)
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()
        print(f'Epoch: {epoch+1} , loss: {acc_loss / len(dataloader)}')

def evaluate_model(dataloader, trainedModel):
    trainedModel.eval()

    accuracy_metric = Accuracy(task=task, num_classes=num_classes, average='weighted')
    precision_metric = Precision(task=task, num_classes=num_classes, average='weighted')
    recall_metric = Recall(task=task, num_classes=num_classes, average='weighted')
    f1_metric = F1Score(task=task, num_classes=num_classes, average='weighted')
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = trainedModel(images)
            _, prediction = torch.max(outputs, 1)

            accuracy_metric.update(prediction.cpu(), labels.cpu())
            precision_metric.update(prediction.cpu(), labels.cpu())
            recall_metric.update(prediction.cpu(), labels.cpu())
            f1_metric.update(prediction.cpu(), labels.cpu())
            confmat_metric.update(prediction.cpu(), labels.cpu())
    
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    f1 = f1_metric.compute()
    confmat = confmat_metric.compute()

    print(f"Accuracy: {accuracy.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}")
    print(f"Confusion matrix\n: {confmat.numpy()}")
    return accuracy, precision, recall, f1, confmat


if __name__ == "__main__":
    print(device)
    image_paths, labels, class_names = load_images(image_folder)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels
    )

    train_dataset = CustomDataset(train_paths, train_labels, transform_train)
    test_dataset = CustomDataset(test_paths, test_labels, transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, prefetch_factor=2)

    train_model(train_loader)
    torch.save(model.state_dict(), 'modelWeights/model_weights.pth')
    evaluate_model(test_loader, model)
