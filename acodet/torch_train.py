import torch
from torch import nn
import torch.optim as optim

from torchaudio import datasets
from torch.utils.data import DataLoader

from itertools import cycle

from tqdm import tqdm

from acodet import global_config as conf

def train(model, data_loaders, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        try:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        except:
            device = 'cpu'
            
    
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
        train_correct = 0
        train_total = 0
        train_TP = 0
        train_FP = 0
        train_FN = 0
        train_support = {'0': 0, '1': 0}
        
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
                
            if hasattr(model, 'probe_device'):
                labels = labels.to(model.probe_device)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            # E. Metrics & Progress Bar Update
            running_loss += loss.item() * inputs.size(0)

            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().flatten()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                train_TP += ((preds == 1) & (labels == 1)).sum().item()
                train_FP += ((preds == 1) & (labels == 0)).sum().item()
                train_FN += ((preds == 0) & (labels == 1)).sum().item()
                train_support['0'] += (labels == 0).sum().item()
                train_support['1'] += (labels == 1).sum().item()

            # Update the progress bar text with current Loss and LR
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{train_correct/train_total:.3f}',
                'lr': f'{current_lr:.1e}'
            })

        # End of Epoch Metrics
        train_loss = running_loss / train_total
        train_acc = train_correct / train_total
        train_prec = train_TP / (train_TP + train_FP)
        train_recall = train_TP / (train_TP + train_FN)
        train_f1 = 2.0 * train_prec * train_recall / (train_prec + train_recall)
        
        # 5. Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        val_TP = 0
        val_FP = 0
        val_FN = 0
        val_running_loss = 0.0
        val_support = {'0': 0, '1': 1}
        
        with torch.no_grad():
            for v_idx, batch in enumerate(val_loader):
                # A. Check Step Limit
                if (
                    conf.STEPS_PER_EPOCH is not None 
                    and v_idx >= (conf.STEPS_PER_EPOCH / 5)
                    ):
                    break
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                
                if hasattr(model, 'probe_device'):
                    labels = labels.to(model.probe_device)

                outputs = model(inputs)
                val_loss = criterion(outputs.squeeze(), labels)
                val_running_loss += val_loss.item() / inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().flatten()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_TP += ((preds == 1) & (labels == 1)).sum().item()
                val_FP += ((preds == 1) & (labels == 0)).sum().item()
                val_FN += ((preds == 0) & (labels == 1)).sum().item()
                val_support['0'] += (labels == 0).sum().item()
                val_support['1'] += (labels == 1).sum().item()
                
        val_acc = val_correct / val_total
        val_loss = val_running_loss / val_total
        try:
            val_prec = val_TP / (val_TP + val_FP)
        except ZeroDivisionError:
            val_prec = 0.0
        try: 
            val_recall = val_TP / (val_TP + val_FN)
        except ZeroDivisionError:
            val_recall = 0.0
        try:
            val_f1 = 2 * val_prec * val_recall / (val_prec + val_recall)
        except ZeroDivisionError:
            val_f1 = 0.0
        
        # Print summary (tqdm might clear the line, so we print after)
        print(f"Epoch {epoch+1} Train Summary - Accuracy {train_acc:.4f} | Precision {train_prec:.4f} | Recall {train_recall:.4f} | F1 {train_f1:.4f} | Loss {train_loss:.4f} | Support ({train_support['0']},{train_support['1']})")
        print(f"Epoch {epoch+1} Validation Summary - Accuracy {val_acc:.4f} | Precision {val_prec:.4f} | Recall {val_recall:.4f} | F1 {val_f1:.4f} | Loss: {val_loss:.4f} | Support ({val_support['0']},{val_support['1']})")

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
            
            if hasattr(model, 'probe_device'):
                labels = labels.to(model.probe_device)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total
    print(f"Test Acc: {test_acc:.4f}")
    
    

