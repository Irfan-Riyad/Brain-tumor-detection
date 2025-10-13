# evaluation_metrics.py

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import config

@torch.no_grad()
def get_predictions(model, loader, device):
    """Get predictions and labels from a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_on_test_set(model, test_loader, class_names):
    """Loads the best model and evaluates it on the test set."""
    print("\nPerforming final evaluation on the unseen test set...")
    
    # Load the best model state
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.DEVICE)

    # Get predictions
    y_pred, y_true = get_predictions(model, test_loader, config.DEVICE)

    # Print Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Compute Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm