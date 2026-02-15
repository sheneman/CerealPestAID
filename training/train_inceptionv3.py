#
# FACT AI - Classifier
#
# Luke Sheneman
# sheneman@uidaho.edu
#

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 12 bytes but only got 2.")
warnings.filterwarnings("ignore", message="Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.")

TRAIN_DATA_PATH = "./data/labels/train"
VAL_DATA_PATH   = "./data/labels/val"
MODEL_SAVE_PATH = "./models"
MODEL_STATE_PATH = os.path.join(MODEL_SAVE_PATH, 'best_factai_inceptionv3.pth')

LEARNING_RATE = 1e-4
LR_FACTOR     = 0.9
LR_PATIENCE   = 5
EPOCHS        = 1000
BATCH_SIZE    = 8
NUM_WORKERS   = 20

# Create the models directory if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Define transformations for training and validation
transform_train = transforms.Compose([
    transforms.Resize(572),
    transforms.RandomResizedCrop(528, scale=(0.60, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(572),
    transforms.CenterCrop(528),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_train)
val_dataset   = datasets.ImageFolder(root=VAL_DATA_PATH, transform=transform_val)

# Calculate class counts and weights for training
class_counts_train = np.bincount([label for _, label in train_dataset.imgs])
class_weights_train = 1.0 / class_counts_train
samples_weights_train = [class_weights_train[label] for _, label in train_dataset.imgs]

# Calculate class counts and weights for validation
class_counts_val = np.bincount([label for _, label in val_dataset.imgs])
class_weights_val = 1.0 / class_counts_val
samples_weights_val = [class_weights_val[label] for _, label in val_dataset.imgs]

# Weighted samplers
sampler_train = WeightedRandomSampler(samples_weights_train, num_samples=len(samples_weights_train), replacement=True)
sampler_val = WeightedRandomSampler(samples_weights_val, num_samples=len(samples_weights_val), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler_train, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=sampler_val, num_workers=NUM_WORKERS)

# Load the InceptionV3 model
model = models.inception_v3(pretrained=True, aux_logits=True)  # Enable aux_logits

# Modify the classifier layer to match the number of classes
if os.path.exists(MODEL_STATE_PATH):
    state_dict = torch.load(MODEL_STATE_PATH, map_location=torch.device('cpu'))
    num_classes = state_dict['fc.weight'].shape[0]
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load the saved model weights
    model.load_state_dict(state_dict)
    print("Loaded saved model weights.")
else:
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print("Using default pretrained weights.")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)

# Initialize training parameters -- SHENEMAN
best_val_loss = float('inf')
start_epoch = 0

# Check for additional state information
if os.path.exists(MODEL_STATE_PATH):
    state = torch.load(MODEL_STATE_PATH, map_location=torch.device('cpu'))
    if 'epoch' in state:
        start_epoch = state['epoch'] + 1
        print(f"Starting from epoch {start_epoch}.")
    else:
        print(f"No epoch information found. Starting from epoch {start_epoch}.")
    if 'best_val_loss' in state:
        best_val_loss = state['best_val_loss']
        print(f"Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"No best validation loss information found. Using default best validation loss {best_val_loss:.4f}.")
    if 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print("Loaded optimizer state dict.")
    else:
        print("optimizer_state_dict not found in the state file.")
    if 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
        print("Loaded scheduler state dict.")
    else:
        print("scheduler_state_dict not found in the state file.")

# Training and validation loop
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch_num, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Check if aux_logits are present in training mode
        if isinstance(outputs, tuple):
            outputs, aux_outputs = outputs
            loss1 = criterion(outputs, labels)  # Main loss
            loss2 = criterion(aux_outputs, labels)  # Auxiliary loss
            loss = loss1 + 0.4 * loss2  # Combine the losses with weight for auxiliary loss
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)  # Use main outputs for predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_num % 100 == 0:
            batch_loss = running_loss / (batch_num * BATCH_SIZE)
            batch_acc = running_corrects.double() / (batch_num * BATCH_SIZE)
            print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_num}/{len(train_loader)}, Loss: {batch_loss:.4f}, Accuracy: {batch_acc:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs, _ = outputs
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)  # Use main outputs for predictions
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    # Adjust the learning rate based on the validation loss
    scheduler.step(val_loss)

    # Save the model if validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': epoch_loss,
            'val_loss': val_loss
        }, MODEL_STATE_PATH)
        print('Model saved!')

print('Training complete.')

