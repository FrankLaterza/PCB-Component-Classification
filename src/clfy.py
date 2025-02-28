import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

MODEL_NAME = "component_classification"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO gabe: function will return str of all the components
def get_dataset_component_path(root_dir):
    # example of how to get
    paths = glob.glob(os.path.join(root_dir, "*.*"))
    return paths

class ImageLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths= get_dataset_component_path(root_dir)
        self.transform = transform
        self.labels = list(set([os.path.basename(p).split('_')[0] for p in self.image_paths]))
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = os.path.basename(img_path).split('_')[0]
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.label_to_idx[label]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageLabelDataset(root_dir='./dataset/mega', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(dataset.labels)
model = CNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
        
        if acc > best_acc:
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_mapping': dataset.label_to_idx
            }, f'{MODEL_NAME}.pth')
            best_acc = acc

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    checkpoint = torch.load(f'{MODEL_NAME}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    idx_to_label = {v: k for k, v in checkpoint['label_mapping'].items()}
    
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return idx_to_label[predicted.item()]
