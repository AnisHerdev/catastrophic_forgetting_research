# with k-fold validation
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset # Added Subset
from tqdm import tqdm
# Imports for K-Fold
from sklearn.model_selection import KFold
import numpy as np

# --- Configuration for Baseline Experiments ---
TRANSFER_LEARNING_EPOCHS = 5 # Reduced epochs for faster CV demo
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.2
NUM_CLASSES = 100

# --- Configuration for K-Fold ---
K_FOLDS = 5 # Define the number of folds

# --- Environment Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
print("Loading the full CIFAR-100 dataset...")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
# We use the full train_dataset for CV splitting
train_dataset = datasets.CIFAR100(root=".", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root=".", train=False, download=True, transform=transform)
# Note: We won't use train_loader/test_loader directly, they will be created per fold.
# We keep test_dataset loaded but unused during CV.
print(f"Data loaded: {len(train_dataset)} training images, {len(test_dataset)} test images.")


# --- Model Definition ---
class VGG16Adapted(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_RATE):
        super().__init__()
        # Ensure weights are loaded if using transfer learning
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features  # Pre-trained convolutional layers

        # ***************************************************************
        # All feature layers are unfrozen (trainable)
        # ***************************************************************

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Global average pooling equivalent for VGG output (7x7 -> 1x1 average pooling)
        # Since CIFAR images are 32x32, the final feature map size is 1x1, not 7x7.
        # We ensure it's flattened correctly.
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

print("✅ VGG16 Adapted model class defined (with unfrozen features).")


# --- Training and Evaluation Functions ---
def train_model(model, dataloader, criterion, optimizer, epochs):
    """Trains the model for a specified number of epochs."""
    model.train()
    for epoch in range(epochs):
        # We remove epoch logging from tqdm to keep the k-fold output cleaner
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # print(f"Finished training for {epochs} epochs.") # Commented out for CV loop clarity

def evaluate_model(model, dataloader):
    """Evaluates the model on a given dataset and returns accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

# --- K-Fold Implementation ---

def run_kfold_cv(dataset, num_folds=K_FOLDS, epochs=TRANSFER_LEARNING_EPOCHS):
    """
    Performs K-Fold Cross-Validation on the provided dataset.
    Weights are unfrozen and trained from scratch for each fold.
    """
    
    # Initialize KFold splitter
    # Setting a random_state ensures reproducibility of the splits
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = {}

    print(f"\n--- Starting {num_folds}-Fold Cross Validation ---")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n----- FOLD {fold+1}/{num_folds} -----")

        # 1. Create Subsets
        train_subset = Subset(dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        # 2. Create DataLoaders
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # 3. Initialize Model, Loss, Optimizer (must be fresh for each fold)
        # This ensures that each fold starts training with the pre-trained ImageNet weights
        # and not the weights trained in previous folds.
        model = VGG16Adapted(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 4. Train the model
        print(f"Training model for {epochs} epochs on Fold {fold+1}...")
        # tqdm is used here just to show overall training progress of the fold
        train_model(
            model, 
            tqdm(train_loader, desc=f"Fold {fold+1} Training"), 
            criterion, 
            optimizer, 
            epochs=epochs
        )

        # 5. Evaluate on Validation Set
        print(f"Evaluating model on Fold {fold+1} validation set...")
        accuracy = evaluate_model(model, val_loader)
        
        print(f"FOLD {fold+1} Validation Accuracy: {accuracy:.2f}%")
        fold_results[fold + 1] = accuracy
        
    # 6. Summarize Results
    print("\n" + "*"*60)
    print("      K-Fold Cross Validation Summary")
    print("*"*60)
    
    accuracies = list(fold_results.values())
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    for fold, acc in fold_results.items():
        print(f"  Fold {fold}: Accuracy = {acc:.2f}%")

    print("---------------------------------------")
    print(f"  Average CV Accuracy: {mean_accuracy:.2f}%")
    print(f"  Standard Deviation: (+/- {std_accuracy:.2f}%)")
    print("---------------------------------------")
    
    return mean_accuracy

# --- Main Execution ---
if __name__ == "__main__":
    
    # The training process now runs K-Fold CV on the original train_dataset
    run_kfold_cv(
        dataset=train_dataset, 
        num_folds=K_FOLDS, 
        epochs=TRANSFER_LEARNING_EPOCHS
    )
    
    # Note: If you wanted a final evaluation on the isolated test_dataset, 
    # you would retrain a final model using the optimal hyperparameters 
    # (or the standard parameters found here) on the entire train_dataset, 
    # and then evaluate on test_dataset.
    
    print("\nK-Fold process completed.")