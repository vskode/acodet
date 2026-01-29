
import torch
from torch import nn
import torch.optim as optim

from torchaudio import datasets
from torch.utils.data import DataLoader

from itertools import cycle

from tqdm import tqdm

def train(model, data_loaders, nr_epochs, device='cuda'):
    
    train_loader = data_loaders.train_loader()
    noise_loader = data_loaders.noise_loader()
    val_loader = data_loaders.val_loader()
    
    noise_iter = cycle(noise_loader)

    # Modify input size (note: torchvision EfficientNet supports arbitrary sizes)
    # No special change required; resizing inputs in transforms is enough
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(nr_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels, path, start in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            noise_batch = next(noise_iter)
            noise = noise_batch[0].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, labels, noise, path, start, training=True)          # (N, 1), raw logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)


        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{nr_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels, _, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # _, preds = torch.max(outputs, 1)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        print(f"Validation Acc: {val_acc:.4f}")

        # scheduler.step()
    return model

def test(model, test_loader, device='cuda'):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels, _, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total
    print(f"Test Acc: {test_acc:.4f}")