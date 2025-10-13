# Experiment Methodology: Brain Tumor Classification using Hybrid CNN Architecture

## 1. Problem Statement and Objectives

### Primary Objective
Develop and evaluate a hybrid deep learning model for multi-class brain tumor classification using medical imaging data.

### Research Questions
1. Can a hybrid CNN architecture combining ResNet50 and DenseNet121 backbones outperform single-backbone approaches?
2. How effective is the two-stage fine-tuning strategy (frozen → unfrozen backbones) for medical image classification?
3. What are the key misclassification patterns and failure modes of the hybrid model?

### Success Metrics
- Classification accuracy on held-out test set
- Per-class precision, recall, and F1-scores
- Area Under ROC Curve (AUC) - both micro and macro averaged
- Qualitative analysis through Grad-CAM visualizations

## 2. Dataset and Data Preparation

### 2.1 Dataset Source
- **Source**: Kaggle Brain Tumor Dataset (`mrnotalent/braint`)
- **Type**: Medical imaging dataset with multiple brain tumor classes
- **Format**: Digital images suitable for computer vision tasks

### 2.2 Data Splitting Strategy
**Stratified Random Split** to ensure balanced representation across all classes:
- **Training Set**: 70% of total data
- **Validation Set**: 15% of total data  
- **Test Set**: 15% of total data

**Rationale**: 
- Stratified splitting maintains class distribution consistency
- 70-15-15 split provides sufficient training data while preserving adequate validation/test sets
- Random seed (42) ensures reproducibility

### 2.3 Data Preprocessing Pipeline

#### Training Data Augmentation
```python
- Resize to 224×224 pixels (ImageNet standard)
- Random horizontal flip (p=0.5)
- Random rotation (±15 degrees)
- Random affine transformations (translation ±10%, scale 0.9-1.1×)
- Random erasing (p=0.5, scale 2-20% of image)
- Normalization using ImageNet statistics
- Grayscale to RGB conversion when necessary
```

#### Validation/Test Data Processing
```python
- Resize to 224×224 pixels
- Normalization using ImageNet statistics
- Grayscale to RGB conversion when necessary
- No augmentation to ensure consistent evaluation
```

**Rationale for Conservative Augmentation**:
- Medical images require careful augmentation to preserve diagnostic features
- Horizontal flipping is anatomically valid for brain scans
- Limited rotation (15°) prevents unrealistic orientations
- Random erasing simulates partial occlusion scenarios

## 3. Model Architecture

### 3.1 Hybrid CNN Design
**Multi-Backbone Feature Fusion Architecture**

#### Component Models
1. **ResNet50**: Deep residual network with skip connections
   - Pre-trained on ImageNet
   - Output feature dimension: 2048
   
2. **DenseNet121**: Densely connected convolutional network
   - Pre-trained on ImageNet  
   - Output feature dimension: 1024

#### Fusion Strategy
- **Late Fusion**: Concatenate feature vectors from both backbones
- **Combined Feature Dimension**: 3072 (2048 + 1024)
- **Classification Head**: 
  - Linear layer: 3072 → 1024 (with ReLU + Dropout 0.5)
  - Output layer: 1024 → num_classes

### 3.2 Architecture Rationale
- **Complementary Features**: ResNet50 (residual learning) + DenseNet121 (feature reuse)
- **Transfer Learning**: Leverage ImageNet pre-training for medical domain
- **Feature Diversity**: Multiple architectures capture different visual patterns
- **Regularization**: Dropout and label smoothing prevent overfitting

## 4. Training Methodology

### 4.1 Two-Stage Training Strategy

#### Stage 1: Frozen Backbone Training (Epochs 1-5)
- **Backbone Parameters**: Frozen (requires_grad=False)
- **Trainable Parameters**: Only classification head
- **Learning Rate**: 3×10⁻⁴
- **Optimizer**: AdamW with weight decay 1×10⁻⁴
- **Rationale**: Learn domain-specific classification head while preserving pre-trained features

#### Stage 2: End-to-End Fine-tuning (Epochs 6-25)
- **Backbone Parameters**: Unfrozen (requires_grad=True)
- **Learning Rate**: 1×10⁻⁴ (reduced for stability)
- **Optimizer**: AdamW with weight decay 1×10⁻⁴
- **Rationale**: Fine-tune entire network for medical domain adaptation

### 4.2 Training Configuration
- **Total Epochs**: 25
- **Batch Size**: 32 (training), 64 (validation/test)
- **Loss Function**: CrossEntropyLoss with label smoothing (0.1)
- **Learning Rate Scheduler**: CosineAnnealingLR
- **Early Stopping**: Best validation accuracy checkpoint saving

### 4.3 Regularization Techniques
1. **Label Smoothing** (0.1): Reduces overconfidence and improves generalization
2. **Weight Decay** (1×10⁻⁴): L2 regularization prevents overfitting
3. **Dropout** (0.5): Random neuron deactivation in classification head
4. **Data Augmentation**: Increases training data diversity

## 5. Evaluation Methodology

### 5.1 Performance Metrics

#### Primary Metrics
- **Accuracy**: Overall classification correctness
- **Per-Class Metrics**: Precision, Recall, F1-score for each tumor type
- **Confusion Matrix**: Detailed misclassification analysis

#### Advanced Metrics
- **ROC-AUC Curves**: 
  - Micro-averaged (global performance)
  - Macro-averaged (per-class average)
  - Individual class curves (when feasible)
- **Multi-class ROC Analysis**: Handles class imbalance and provides probabilistic interpretation

### 5.2 Qualitative Analysis

#### Grad-CAM Visualization
- **Purpose**: Understand model attention and decision-making process
- **Implementation**: Gradient-weighted Class Activation Mapping
- **Target Layers**: Last convolutional layers of both backbones
- **Output**: Heatmaps showing regions of high model attention

#### Error Analysis
- **Correct Predictions**: Validate expected model behavior
- **Incorrect Predictions**: Identify systematic failure patterns
- **Visual Inspection**: Compare model attention with medical expertise expectations

## 6. Experimental Controls and Reproducibility

### 6.1 Reproducibility Measures
```python
# Fixed random seeds across all libraries
- Python random: seed=42
- NumPy: seed=42  
- PyTorch: manual_seed=42
- TensorFlow: seed=42
- Environment: PYTHONHASHSEED=42
```

### 6.2 Hardware and Software Environment
- **Computing Platform**: GPU-accelerated training (CUDA when available)
- **Frameworks**: PyTorch (primary), TensorFlow (random seed control)
- **Key Libraries**: torchvision, scikit-learn, matplotlib, seaborn

### 6.3 Data Integrity
- **Consistent Splits**: Same train/val/test division across experiments
- **Standardized Preprocessing**: Identical transforms for fair comparison
- **Checkpoint Management**: Best model preservation based on validation performance

## 7. Experimental Timeline and Monitoring

### 7.1 Training Monitoring
- **Per-Epoch Metrics**: Training/validation loss and accuracy
- **Best Model Tracking**: Automatic checkpoint saving for best validation accuracy
- **Learning Rate Scheduling**: Cosine annealing for optimal convergence

### 7.2 Performance Benchmarks
- **Training Time**: Complete training duration measurement
- **Inference Time**: Test set evaluation timing
- **Memory Usage**: GPU memory requirements monitoring

## 8. Validation Strategy

### 8.1 Model Selection Criteria
- **Primary**: Validation set accuracy maximization
- **Secondary**: Training stability and convergence behavior
- **Overfitting Detection**: Training vs. validation performance gap monitoring

### 8.2 Final Model Evaluation
- **Unseen Test Set**: Final performance assessment on completely held-out data
- **Statistical Significance**: Multiple metrics for robust evaluation
- **Clinical Relevance**: Focus on medically important misclassification patterns

## 9. Expected Outcomes and Limitations

### 9.1 Expected Results
- Improved classification performance compared to single-backbone models
- Clear visualization of model decision-making process through Grad-CAM
- Identification of challenging tumor types and failure modes

### 9.2 Experimental Limitations
- **Dataset Size**: Limited by available labeled medical images
- **Class Imbalance**: Potential uneven distribution of tumor types
- **Domain Gap**: ImageNet pre-training vs. medical imaging domain
- **Computational Resources**: Training time constraints on available hardware

### 9.3 Validation Considerations
- **Medical Validation**: Model outputs require clinical expert review
- **Generalization**: Performance may vary across different imaging protocols
- **Ethical Considerations**: Model intended for research, not clinical diagnosis

## 10. Documentation and Reporting

### 10.1 Results Documentation
- **Quantitative Results**: Complete classification report with all metrics
- **Visual Results**: Training curves, confusion matrices, ROC curves
- **Qualitative Analysis**: Grad-CAM visualizations and error analysis
- **Timing Analysis**: Training and inference performance measurements

### 10.2 Reproducibility Package
- **Complete Code**: All preprocessing, training, and evaluation scripts
- **Model Checkpoints**: Best performing model weights
- **Configuration Files**: All hyperparameters and experimental settings
- **Environment Specifications**: Library versions and dependencies

---

## Summary

This methodology implements a systematic approach to brain tumor classification using a novel hybrid CNN architecture. The combination of rigorous data splitting, conservative medical image augmentation, two-stage fine-tuning, and comprehensive evaluation provides a robust framework for assessing deep learning performance in medical image analysis. The inclusion of explainability through Grad-CAM ensures the model's decision-making process can be interpreted and validated by medical professionals.