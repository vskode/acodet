
import torch
from torch import nn
import torch.optim as optim

from torchaudio import datasets
from torch.utils.data import DataLoader

from itertools import cycle

from tqdm import tqdm

from acodet import global_config as conf


def train(model, data_loaders, device='cuda'):
    
    train_loader = data_loaders.train_loader()
    val_loader = data_loaders.val_loader()
    
    # Setup explicit noise iterator
    noise_loader = data_loaders.noise_loader()
    noise_iter = cycle(noise_loader)

    model = model.to(device)

    # Setup Optimizer with Initial LR
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.INIT_LR)
    
    # Setup Scheduler (Cosine Decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=conf.EPOCHS, 
        eta_min=conf.FINAL_LR
    )

    for epoch in range(conf.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Determine total steps for the progress bar
        total_steps = len(train_loader)
        if conf.STEPS_PER_EPOCH is not None:
            total_steps = min(total_steps, conf.STEPS_PER_EPOCH)

        # define progressbar
        pbar = tqdm(train_loader, 
                    total=total_steps, 
                    desc=f"Epoch {epoch+1}/{conf.EPOCHS}")
        

        for i, batch in enumerate(pbar):
            # A. Check Step Limit
            if conf.STEPS_PER_EPOCH is not None and i >= conf.STEPS_PER_EPOCH:
                break
                
            # Unpack Batch
            inputs = batch[0].to(device)
            labels = batch[1].float().to(device)
            path = batch[2]   # List of strings, keeps on CPU
            start = batch[3] # List of floats
            
            # Load Noise 
            noise_batch = next(noise_iter)
            
            noise = noise_batch[0].to(device)

            optimizer.zero_grad()

            # D. Forward Pass (Ensure model accepts these args)
            outputs = model(inputs, labels, noise=noise, path=path, start=start, training=True)
            
            # If outputs is a dictionary (some models), extract logits
            if isinstance(outputs, dict):
                outputs = outputs['logits']
                
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            # E. Metrics & Progress Bar Update
            running_loss += loss.item() * inputs.size(0)

            
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            # Update the progress bar text with current Loss and LR
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{correct/total:.3f}',
                'lr': f'{current_lr:.1e}'
            })

        # End of Epoch Metrics
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # 5. Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        
        # Optional: Add pbar for validation if it takes a long time
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                
                outputs = model(inputs)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        
        # Print summary (tqdm might clear the line, so we print after)
        print(f"Epoch {epoch+1} Summary - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

        # 6. Step the Scheduler
        scheduler.step()

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
    
    

