# AI Brain Tumor Classification Project

## Group Information
- **Group Number:** 3
- **Group Members:** 
  - Shafayat Hasnat Rubaiyat (ID: 2022-3-60-029)
  - MD Riyad Hossain (ID: 2022-3-60-156)
  - M. Nura Alam Naim (ID: 2022-3-60-123)
  - Mehejarin Aklima Jerin (ID: 2022-3-60-325)

## Project Overview
This project implements and compares multiple deep learning architectures for Brain Tumor classification. The goal is to accurately classify different types of brain tumors using computer vision techniques, comparing traditional CNN architectures with modern transfer learning approaches and hybrid models.

## Model Architecture
<img width="2530" height="901" alt="times new roman drawio" src="https://github.com/user-attachments/assets/77be0501-f264-4f2d-b319-458680b2f437" />


## Dataset Information
- **Dataset Name:** Brain Tumor MRI Images 44 Classes
- **Source:** mrnotalent/braint
- **Classes:** Here is the list of 44 class tumors:
1. Astrocitoma T1
2. Astrocitoma T1C+
3. Astrocitoma T2
4. Carcinoma T1
5. Carcinoma T1C+
6. Carcinoma T2
7. Ependimoma T1
8. Ependimoma T1C+
9. Ependimoma T2
10. Ganglioglioma T1
11. Ganglioglioma T1C+
12. Ganglioglioma T2
13. Germinoma T1
14. Germinoma T1C+
15. Germinoma T2
16. Glioblastoma T1
17. Glioblastoma T1C+
18. Glioblastoma T2
19. Granuloma T1
20. Granuloma T1C+
21. Granuloma T2
22. Meduloblastoma T1
23. Meduloblastoma T1C+
24. Meduloblastoma T2
25. Meningioma T1
26. Meningioma T1C+
27. Meningioma T2
28. Neurocitoma T1
29. Neurocitoma T1C+
30. Neurocitoma T2
31. Oligodendroglioma T1
32. Oligodendroglioma T1C+
33. Oligodendroglioma T2
34. Papiloma T1
35. Papiloma T1C+
36. Papiloma T2
37. Schwannoma T1
38. Schwannoma T1C+
39. Schwannoma T2
40. Tuberculoma T1
41. Tuberculoma T1C+
42. Tuberculoma T2
43. _NORMAL T1
44. _NORMAL T2
- **Data Split:** 70% Training, 15% Validation, 15% Testing
- **Total Images:** 16236
- **Image Resolution:** 224x224, variable

## Execution Environment
- **Platform:** Google Colab Pro / Kaggle
- **Hardware Specifications:**
  - **RAM:** 29 GB / 13GB
  - **GPU:** NVIDIA Tesla T4 / NVIDIA Tesla P100 / etc.
  - **CPU:** Intel Xeon CPU / AMD Ryzen (if relevant)
  - **Storage:** SSD 512GB (if relevant)

## Experiment Methodology
1. **Data Exploration & Preprocessing**
   - Data Analysis: We began by analyzing the distribution of images across different brain tumor classes to identify any potential imbalances.
   - Train/Validation/Test Split: The dataset was partitioned into standard training, validation, and testing sets to ensure robust and unbiased model evaluation.
   - Image Augmentation: To enhance the model's ability to generalize and prevent overfitting, we applied various augmentation techniques. As demonstrated in our RestNet experiments, we compared performance on both augmented and non-augmented data. The augmentation strategies included:
    Random rotations
    Horizontal and vertical flips
    Zooming and shearing
    Brightness adjustments

2. **Baseline Model**
    To establish a performance benchmark, we first developed a Convolutional Neural Network from the ground up.
   - Architecture: The model detailed in Custom_CNN.ipynb was a straightforward CNN with multiple convolutional and pooling layers, followed by dense layers for classification.
   - Performance Metrics: This initial model was evaluated on core metrics like accuracy, providing a baseline to measure the effectiveness of more complex, pre-trained architectures.

3. **Advanced Models**
    The core of our experimentation involved leveraging the power of transfer learning with several state-of-the-art architectures. Each model was fine-tuned on our brain tumor dataset.
   - CNN from Scratch: Our Custom_CNN.ipynb served as our primary from-scratch model.
   - Transfer Learning Models: We implemented and evaluated the following pre-trained models:

    ResNet: Explored in both RestNet on Augmented Data.ipynb and RestNet on Non-Augmented Data.ipynb to explicitly measure the impact of our data augmentation strategy.

    GoogleNet: Implemented in GoogleNet.ipynb.

    DenseNet: Implemented in Densenet.ipynb.

    EfficientNet: A modern, efficient architecture explored in EfficientNet.ipynb.

   - Hybrid Architecture: To push performance further, we designed and tested a novel hybrid model in Hybrid_CNN (Densenet + Restnet).ipynb, which combined features from both the DenseNet and ResNet architectures.

4. **Model Evaluation**
    The final phase involved a comprehensive evaluation of all trained models to determine the top performer.
   - Quantitative Metrics: We assessed each model using a suite of metrics including Accuracy, Precision, Recall, and F1-Score to get a holistic view of performance.
   - Final Comparison: The results from all experiments were compiled to compare the architectures directly, leading to our final model selection based on empirical evidence.

## Repository Navigation
- `notebooks/`: Densenet.ipynb, Hybrid_CNN (Densenet + Restnet).ipynb, Custom_CNN.ipynb, EfficientNet.ipynb, GoogleNet.ipynb, RestNet on Augmented Data.ipynb, RestNet on Non-Augmented Data.ipynb
- `src/`: config.py, data_utils.py, evalutation_metrices.py, main.py, modal_architectures.py, training_utils.py, visualization.py
- `models/`: hybrid_brain_tumor.pth
- `results/`: logs, plots
- `data/`: Astrocitoma T1,strocitoma T1C+,strocitoma T2,arcinoma T1,arcinoma T1C+,arcinoma T2,pendimoma T1,pendimoma T1C+,pendimoma T2,Ganglioglioma T1,Ganglioglioma T1C+,Ganglioglioma T2,Germinoma T1,Germinoma T1C+,Germinoma T2,Glioblastoma T1,Glioblastoma T1C+,Glioblastoma T2,Granuloma T1,Granuloma T1C+,Granuloma T2,Meduloblastoma T1,Meduloblastoma T1C+,Meduloblastoma T2,Meningioma T1,Meningioma T1C+,Meningioma T2,Neurocitoma T1,Neurocitoma T1C+,Neurocitoma T2,Oligodendroglioma T1,Oligodendroglioma T1C+,Oligodendroglioma T2,Papiloma T1,Papiloma T1C+,Papiloma T2,Schwannoma T1,Schwannoma T1C+,Schwannoma T2,Tuberculoma T1,Tuberculoma T1C+,Tuberculoma T2,_NORMAL T1,_NORMAL T2

## How to Run the Code
1. **Clone Repository:**
   ```bash
   git clone https://github.com/Infi9itee/AI_GRP_3_Brain_Tumor_Classification
   cd AI_GRP_3_Brain_Tumor_Classification

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Experiments:**
   - Start with `01_Custom_CNN.ipynb`
   - Follow numerical order through notebooks
   - Each notebook is self-contained with clear instructions

4. **Data Setup:**
   - Download dataset from https://www.kaggle.com/datasets/mrnotalent/braint
   - Place in `/content` directory
   - Or modify data paths in configuration files

## Key Findings
- **Best Model:** Hybrid model (restnet + densenet)
- **Best Accuracy:** 98.69%
- **Key Insights:** 
  - Hybrid ResNet50 & DenseNet121 architecture achieved ~99% accuracy by fusing diverse feature sets.
  - A two-phase fine-tuning strategy (head-first, then full model) proved crucial for effective learning.
  - Extensive data augmentation was key to the model's high generalization and prevention of overfitting.

## Final Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 98.69% |
| Precision | 98.72% |
| Recall | 98.69% |
| F1-Score | 96.68% |

## Dependencies
- Python 3.8+
- TensorFlow 2.x / PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- OpenCV
- (See requirements.txt for complete list)

## Presentation Slide:
- https://docs.google.com/presentation/d/1DxG9ekwUIq-qw59cb0vyV1sT0qPpY1f_/edit?usp=share_link&ouid=103396683720185455463&rtpof=true&sd=true

## Acknowledgements
We express our gratitude to the following individual for their guidance and support:
- Dr. Raihan Ul Islam: Provided project supervision and technical guidance on machine learning methodologies.

## Contributors
All group members contributed equally:
- Shafayat Hasnat Rubaiyat: CNN architecture, Transfer learning      experiments
- MD Riyad Hossain: Hyperparameter tuning, Data preprocessing 
- M. Nura Alam Naim: final integration, evaluation, visualization
- Mehejarin Aklima Jerin: Baseline model, Documentation


