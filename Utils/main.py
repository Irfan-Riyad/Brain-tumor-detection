# main.py

import torch
import random
import numpy as np
import os
import config
import data_utils
import model_architectures
import training_utils
import evaluation_metrics
import visualization

def main():
    # Set seed for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    os.environ['PYTHONHASHSEED'] = str(config.SEED)

    # 1. Prepare Data
    num_classes = data_utils.prepare_and_split_data()
    train_loader, val_loader, test_loader, class_names = data_utils.get_dataloaders()
    
    # 2. Build Model
    model = model_architectures.build_model(num_classes)
    
    # 3. Run Training
    history = training_utils.run_training(model, train_loader, val_loader, class_names)
    
    # 4. Plot Training History
    visualization.plot_training_history(history)
    
    # 5. Evaluate on Test Set
    _, conf_matrix = evaluation_metrics.evaluate_on_test_set(model, test_loader, class_names)
    
    # 6. Plot Confusion Matrix
    visualization.plot_confusion_matrix(conf_matrix, class_names)

if __name__ == '__main__':
    main()