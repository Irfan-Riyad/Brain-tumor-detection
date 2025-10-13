# training_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import config

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Performs one training epoch."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

def run_training(model, train_loader, val_loader, class_names):
    """Main function to run the training and validation loop."""
    model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.SCHEDULER_T_MAX)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    print(f"Starting training for {config.EPOCHS} epochs on {config.DEVICE}...")

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        va_loss, va_acc = evaluate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

        if epoch == config.UNFREEZE_EPOCH:
            print("→ Unfreezing backbones for fine-tuning...")
            for p in model.parameters(): p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_SCHEDULER_T_MAX)
            
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": class_names}, config.BEST_MODEL_PATH)
            print(f"✓ Saved new best model to {config.BEST_MODEL_PATH} (val acc {best_acc:.4f})")
            
    total_time = time.time() - start_time
    print(f"\nFinished Training. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    return history