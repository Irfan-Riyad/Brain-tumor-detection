# HybridCNN — Model Architecture Details


---

## 1. High-level overview

This project implements a **Hybrid CNN classifier** that fuses feature representations from two strong ImageNet-pretrained backbones and passes the fused representation through a compact classifier head. The goals are:

- Leverage complementary features from two backbones (ResNet50 + DenseNet121).
- Keep the fusion simple and effective (concatenation → MLP head).
- Support a two-stage fine-tuning strategy (freeze backbones → unfreeze and fine-tune).

This document explains the components, layer shapes, training hyperparameters used in the notebook, Grad-CAM support, and practical tips for experimentation and deployment.

---

## 2. Components and data flow (per forward pass)

1. **Input**
   - Shape: `(B, 3, 224, 224)` (the notebook expands single-channel images to 3 channels when necessary)
   - Preprocessing: Resize to 224×224, convert to tensor, normalize with ImageNet statistics: `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`.

2. **Backbone A — ResNet50**
   - Pretrained weights: `IMAGENET1K_V2` (when `pretrained=True`).
   - Implementation detail: The model uses `models.resnet50(weights=...)` and replaces `resnet.fc` with `nn.Identity()` so the model outputs the final feature vector from the backbone's pooling layer.
   - Output feature dimension: **2048** per sample (shape `(B, 2048)`).

3. **Backbone B — DenseNet121**
   - Pretrained weights: `IMAGENET1K_V1` (when `pretrained=True`).
   - Implementation detail: The notebook uses `models.densenet121(weights=...)` and replaces `densenet.classifier` with `nn.Identity()`. The features are taken from the global pooled output.
   - Output feature dimension: **1024** per sample (shape `(B, 1024)`).

4. **Feature fusion**
   - Method: **Concatenation** along channel dimension. `fused = torch.cat([f1, f2], dim=1)`
   - Fused feature dimension: `2048 + 1024 = 3072` (shape `(B, 3072)`).

5. **Classifier head**
   - `nn.Sequential(
       nn.Linear(3072, hidden=1024),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(1024, num_classes)
     )`
   - Final logits shape: `(B, num_classes)`

6. **Loss / Output**
   - Loss: `nn.CrossEntropyLoss(label_smoothing=0.1)` during training.
   - During inference, apply `softmax` to obtain class probabilities.

---

## 3. Training strategy used in the notebook

- **Two-stage fine-tuning**:
  1. Stage 1 (warmup): Freeze both backbone parameter groups (`requires_grad=False`) and train only the classifier head for the first 5 epochs with a relatively higher LR (`3e-4`).
  2. Stage 2 (fine-tune): Unfreeze full model at epoch 5 and continue training all parameters at a lower LR (`1e-4`).

- **Optimizer**: `torch.optim.AdamW` with `weight_decay=1e-4`.
- **Scheduler**: `CosineAnnealingLR` (example `T_max=10` then re-created with `T_max=15` after unfreeze).
- **Batch sizes**: training `32`, validation/test `64` (these can be adjusted for available GPU memory).
- **Epochs**: `25` by default.
- **Regularization**: `Dropout(p=0.5)` in head, label smoothing `0.1`, Random Erasing augmentation.

---

## 4. Hyperparameters & defaults (from notebook)

- `IMG_SIZE = 224`
- `batch_size_train = 32`
- `batch_size_eval = 64`
- `EPOCHS = 25`
- `optimizer = AdamW(lr=3e-4, weight_decay=1e-4)` initially
- After unfreeze: `optimizer = AdamW(lr=1e-4, weight_decay=1e-4)`
- `scheduler = CosineAnnealingLR(T_max=10)` initially
- Loss: `CrossEntropyLoss(label_smoothing=0.1)`
- `dropout_p = 0.5`
- `hidden_units = 1024` (classifier bottleneck)

---

## 5. Shapes and parameter size (how to inspect)

To print a summary and get parameter counts you can use either `torchsummary` or iterate manually. Example snippet:

```py
from torchsummary import summary
model = HybridCNN(num_classes=NUM_CLASSES, pretrained=True).to(device)
summary(model, (3, IMG_SIZE, IMG_SIZE))

# Or count params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
```

**Typical estimates (approximate)**:
- ResNet50: ~25M parameters
- DenseNet121: ~8M parameters
- Head (3072→1024→C): ~3.2M (first linear) + a small second linear
- **Total**: typically in the **~36M** range (depends on exact versions and whether biases are counted). Use the snippet above for exact numbers on your environment.

---

## 6. Inference & exporting

- **Model saving**: notebook saves the best model as `hybrid_brain_tumor_best.pth` with a dict: `{"state_dict": model.state_dict(), "classes": train_ds.classes}`. When loading: `model.load_state_dict(torch.load(path)["state_dict"])`.
- **Switch to `eval()`** and wrap inference with `torch.no_grad()`.
- For deployment consider tracing or scripting (`torch.jit.trace` / `torch.jit.script`) or converting to ONNX for cross-platform inference.

---

## 7. Grad-CAM support in the notebook

- The notebook contains a `GradCAM` class and `_find_last_conv_layer()` helper that attempts to locate the last convolution in either the ResNet or DenseNet backbone. The Grad-CAM hooks register forward and backward hooks to capture activations and gradients.
- **How it works**: capture activations `A_k` from the target conv layer and gradients `dS/dA_k`, compute channel weights via GAP over spatial dims, form weighted sum and ReLU → resize to input image size and normalize.
- **Notes / gotchas**:
  - Make sure the target layer is a convolutional layer with spatial map (not an Identity or pool).
  - If backbones are frozen or altered, check that `model.to(device)` was called and that gradients are enabled for the required tensors when running `score.backward()`.
  - The notebook uses `retain_graph=True` in backward — only use if you need multiple backward passes.

---

## 8. Practical notes, pitfalls & improvements

### a. Memory / speed
- Two backbones increase memory and compute cost. If GPU memory is tight:
  - Reduce batch size.
  - Use `torch.cuda.amp.autocast()` with `GradScaler()` for mixed-precision training (significant speed/memory win).
  - Freeze more layers for longer or use smaller backbones (e.g., ResNet34 / MobileNetV3) as a trade-off.

### b. Better fusion strategies to try
- **Attention gating**: learn weights for each backbone's features.
- **Bottleneck + concat**: project each backbone feature to a smaller dimension (e.g., 512) before concatenating.
- **Additive fusion**: if dimensions match, learn a 1×1 conv/linear to combine.
- **Cross-attention / Transformer head**: treat backbone outputs as tokens and use a small transformer to fuse.

### c. Class imbalance handling
- If classes are imbalanced use: weighted loss (`CrossEntropyLoss(weight=...)`), focal loss, oversampling minority classes, or class-balanced sampling in the DataLoader.

### d. Data augmentations
- Current augmentations are conservative for medical images: small rotations, flips (if anatomically valid), slight affine transforms, random erasing. Avoid large photometric changes if they distort pathological features.

### e. Regularization & robustness
- Consider label smoothing (already used), weight decay, dropout in head, and test-time augmentations (TTA) for more robust predictions.

### f. Reproducibility
- Seed `numpy`, `random`, `torch`, and set `torch.backends.cudnn.deterministic=True` if strict reproducibility is required (note: will reduce throughput).
- Save `classes` mapping along with the state dict (already done in notebook).

---

## 9. Diagnostics & evaluation

- The notebook computes **multiclass ROC-AUC** (micro/macro) by binarizing labels and plotting micro/macro ROC curves.
- Use `classification_report` and confusion matrix for per-class precision/recall/F1.
- For error analysis: the notebook plots correct vs incorrect predictions and overlays Grad-CAM heatmaps (if Grad-CAM setup finds the target layer).

---

## 10. Quick checklist for reproducible experiments

- [ ] Fix random seeds for `random`, `numpy`, `torch`, `tf` (if TF is used anywhere).
- [ ] Save training hyperparameters and `git` commit hash.
- [ ] Save best model and `classes` mapping.
- [ ] Log training metrics (loss/acc) each epoch (TensorBoard / Weights & Biases recommended).
- [ ] Save sample misclassified images with predictions and confidences for manual review.

---

## 11. Example simplified model summary (pseudocode)

```py
HybridCNN(
  (resnet): ResNet50 (fc -> Identity)  # outputs 2048-d
  (densenet): DenseNet121 (classifier -> Identity)  # outputs 1024-d
  (head): Sequential(
      Linear(3072, 1024), ReLU, Dropout(0.5), Linear(1024, num_classes)
  )
)
```

---

## 12. Where to find things in the notebook

- Model class: `class HybridCNN(nn.Module):` (Cell 5)
- Training & scheduler logic: Cell 6 (two-stage unfreeze at epoch 5)
- ROC/ROC-AUC code: Cell 8
- Confusion matrix and classification report: Cell 9
- Grad-CAM implementation and visualization: Cell 11
- Best model file: `hybrid_brain_tumor_best.pth` (saved by the training loop)

---

## 13. Next-step experiments (suggested)

1. Try projecting backbone outputs to lower-dim vectors via small `nn.Linear` (e.g. 2048→512 and 1024→512) then concatenate, reducing classifier size.
2. Replace concatenation with a learned attention module that assigns weights to each backbone per-example.
3. Train with `torch.cuda.amp` and larger batch sizes if hardware allows.
4. Run a hyperparameter sweep over learning rate, weight decay, dropout, and frozen/unfrozen schedule.
5. Evaluate robustness with cross-validation folds and TTA.

---


*End of document.*

