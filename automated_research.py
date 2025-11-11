import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import kagglehub
import csv

# --- Configuration ---
PHASE1_EPOCHS_LIST = [5, 10]
PHASE2_EPOCHS_LIST = [10, 15]  # changed: allow multiple choices for Phase 2
CLASSES_PER_SUPERCLASS_OPTIONS = [2, 3]
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
RESULTS_CSV_FILE = "fdl_results_part_2.csv"

# --- Environment Setup ---
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---
print("Downloading CIFAR-100 dataset...")
kagglehub.dataset_download("fedesoriano/cifar100") # Downloads to the current directory

CIFAR100_CLASSES = datasets.CIFAR100(root=".", download=True).classes

# Define superclasses and their constituent classes
SUPERCLASSES = {
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
}

ALL_CHOSEN_CLASSES = [cls for group in SUPERCLASSES.values() for cls in group]
ALL_CHOSEN_INDICES = [CIFAR100_CLASSES.index(cls) for cls in ALL_CHOSEN_CLASSES]

print(f"Total chosen classes ({len(ALL_CHOSEN_CLASSES)}): {ALL_CHOSEN_CLASSES}")

# ImageNet normalization transforms
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_full_cifar = datasets.CIFAR100(root=".", train=True, download=True, transform=transform)
test_full_cifar = datasets.CIFAR100(root=".", train=False, download=True, transform=transform)

def remap_labels_and_filter_dataset(dataset, allowed_indices):
    """Filters a dataset to include only specified classes and remaps their labels to 0..N-1."""
    mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(allowed_indices)}
    images, labels = [], []
    for img, lbl in dataset:
        if lbl in allowed_indices:
            images.append(img)
            labels.append(mapping[lbl])
    return TensorDataset(torch.stack(images), torch.tensor(labels))

# --- Model Definition ---
class VGG16Adapted(nn.Module):
    def __init__(self, num_classes, dropout=DROPOUT_RATE):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16.features  # Pre-trained convolutional layers

        # Custom classifier head for CIFAR-100 (32x32 input leads to 512x1x1 features)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the classifier
        x = self.classifier(x)
        return x

print("✅ VGG16 Adapted model class defined.")

# --- Training and Evaluation Functions ---
def train_model(model, dataloader, criterion, optimizer, epochs):
    """Trains the model for a specified number of epochs."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

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

# --- Experiment Runner ---
print("\nStarting experimental runs...")
print(f"Phase 1 epochs options: {PHASE1_EPOCHS_LIST}")
print(f"Phase 2 epochs options: {PHASE2_EPOCHS_LIST}")
print(f"Classes-per-superclass options: {CLASSES_PER_SUPERCLASS_OPTIONS}")

# Prepare CSV header
with open(RESULTS_CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "Ratio", "Phase1 Accuracy", "Phase1 Epochs",
        "Final Phase2 Accuracy", "Phase2 Epochs",
        "Phase2 Accuracy on Phase1 Dataset", "Difference (H-D)"
    ])

# Loop over all combinations of classes_per_superclass and phase1_epochs
for classes_per_superclass in CLASSES_PER_SUPERCLASS_OPTIONS:
    # Define phase 1 classes for the current iteration
    current_phase1_classes = [
        cls for group in SUPERCLASSES.values() for cls in group[:classes_per_superclass]
    ]
    current_phase1_indices = [CIFAR100_CLASSES.index(cls) for cls in current_phase1_classes]

    # Prepare datasets for current phase 1 and the full phase 2
    train_phase1_ds = remap_labels_and_filter_dataset(train_full_cifar, current_phase1_indices)
    test_phase1_ds = remap_labels_and_filter_dataset(test_full_cifar, current_phase1_indices)
    train_phase2_ds = remap_labels_and_filter_dataset(train_full_cifar, ALL_CHOSEN_INDICES)
    test_phase2_ds = remap_labels_and_filter_dataset(test_full_cifar, ALL_CHOSEN_INDICES)

    # Dataloaders for Phase 2 (constant across phase1_epochs for this classes_per_superclass)
    train_loader_p2 = DataLoader(train_phase2_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_p2 = DataLoader(test_phase2_ds, batch_size=BATCH_SIZE, shuffle=False)

    for phase1_epochs in PHASE1_EPOCHS_LIST:
        for phase2_epochs in PHASE2_EPOCHS_LIST:
            print("\n" + "="*50)
            print(f"Running combo: Ratio={classes_per_superclass}:5, Phase1 Epochs={phase1_epochs}, Phase2 Epochs={phase2_epochs}")
            print("="*50)

            # Dataloaders for Phase 1 (specific to this classes_per_superclass)
            train_loader_p1 = DataLoader(train_phase1_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader_p1 = DataLoader(test_phase1_ds, batch_size=BATCH_SIZE, shuffle=False)

            # --- Phase 1 Training ---
            print("\n--- Phase 1 Training ---")
            model_p1 = VGG16Adapted(num_classes=len(current_phase1_indices)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_p1 = optim.Adam(model_p1.parameters(), lr=LEARNING_RATE)
            train_model(model_p1, train_loader_p1, criterion, optimizer_p1, epochs=phase1_epochs)

            phase1_acc = evaluate_model(model_p1, test_loader_p1)
            print(f"Phase 1 Accuracy on its own test set: {phase1_acc:.2f}%")

            # --- Phase 2 Training ---
            print("\n--- Phase 2 Training ---")
            model_p2 = VGG16Adapted(num_classes=len(ALL_CHOSEN_INDICES)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_p2 = optim.Adam(model_p2.parameters(), lr=LEARNING_RATE)
            train_model(model_p2, train_loader_p2, criterion, optimizer_p2, epochs=phase2_epochs)  # use loop variable

            final_phase2_acc = evaluate_model(model_p2, test_loader_p2)
            print(f"Final Phase 2 Accuracy on full Phase 2 test set: {final_phase2_acc:.2f}%")

            # Evaluate Phase 2 model on Phase 1 dataset
            phase2_on_phase1_acc = evaluate_model(model_p2, test_loader_p1)
            difference_hd = phase2_on_phase1_acc - phase1_acc
            print(f"Phase 2 model Accuracy on Phase 1 test set: {phase2_on_phase1_acc:.2f}%")
            print(f"Difference (H-D, where H is Phase 2 on Phase 1 dataset accuracy, D is Phase 1 accuracy) = {difference_hd:.2f}%")

            # --- Write Results to CSV ---
            ratio_str = f"{classes_per_superclass}:5"
            with open(RESULTS_CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([
                    ratio_str,
                    f"{phase1_acc:.4f}",
                    phase1_epochs,
                    f"{final_phase2_acc:.4f}",
                    phase2_epochs,  # write current Phase 2 epochs value
                    f"{phase2_on_phase1_acc:.4f}",
                    f"{difference_hd:.4f}"
                ])
            print(f"Appended results to {RESULTS_CSV_FILE}")

print("\nAll experiments completed!")