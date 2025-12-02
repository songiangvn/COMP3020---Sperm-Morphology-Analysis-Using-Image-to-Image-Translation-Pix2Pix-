# Sperm Morphology Analysis Using Deep Learning

This project reproduces the research paper: **"Ensembled Deep Learning for the Classification of Human Sperm Head Morphology"** (Advanced Intelligent Systems, 2022).

## 📋 Project Overview

Implementation of a stacked ensemble CNN approach for classifying human sperm head morphology into 4 categories:
- Normal
- Tapered
- Pyriform
- Amorphous

## 🏗️ Architecture

### Base Models (4)
1. **VGG16** - Custom classifier (4096→1000→4)
2. **VGG19** - Custom classifier (4096→1000→4)
3. **Modified ResNet-34** - Removed layer4 (256→4)
4. **DenseNet-161** - (2208→4)

### Meta-Classifier
- Input: 16 features (4 models × 4 class probabilities)
- Architecture: FC(16→32) → BN → ReLU → Dropout(0.2) → FC(32→32) → BN → ReLU → Dropout(0.2) → FC(32→4)

## 📊 Dataset

**HuSHeM Dataset:**
- 216 RGB images (131×131 pixels)
- 4 morphology classes
- Manual rotation annotations for proper alignment

## 🔧 Implementation Details

### Preprocessing Pipeline
1. **Alignment**: Manual rotation annotations (100% accurate)
2. **Resize**: 131×131 → 70×70 (preserves all sperm content)
3. **Normalization**: ImageNet mean/std
4. **Augmentation**: Vertical flip only (p=0.5)

### Training Strategy
- **Base Models**: 3×5-fold cross-validation per model
- **Ensemble**: Train fresh base models for each fold (60 total trainings)
- **No data leakage**: Each fold's models trained only on that fold's data
- **Hyperparameters**: Per Table S1/S2 from paper

### Key Hyperparameters

**Base Models (HuSHeM):**
- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 100 (with early stopping)
- Optimizer: Adam
- Weight Decay: 1e-4

**Meta-Classifier:**
- Learning Rate: 7.801e-2
- Batch Size: 47
- Epochs: 2000 (with early stopping)
- Weight Decay: 5.526e-2
- Momentum (beta1): 0.9855

## 📁 Project Structure

```
ML_Project/
├── HuSHem/
│   ├── 01_train_individual_models.ipynb    # Train 4 base CNN models
│   ├── 02_train_ensemble_model.ipynb       # Train meta-classifier
│   ├── head_orientation_annotations.json   # Manual rotation labels
│   └── outputs/
│       ├── saved_models/                   # Best base models
│       ├── ensemble_results/               # Ensemble outputs
│       └── *.png                           # Visualizations
└── README.md
```

## 🚀 Usage

### 1. Train Individual Models
```bash
# Run notebook: 01_train_individual_models.ipynb
# Trains VGG16, VGG19, ResNet-34, DenseNet-161
# Output: 4 best models saved to outputs/saved_models/
```

### 2. Train Ensemble Model
```bash
# Run notebook: 02_train_ensemble_model.ipynb
# Trains meta-classifier with proper cross-validation
# Output: Ensemble model and performance metrics
```

## 📈 Results

### Individual Models (Mean ± Std)
- **VGG16**: ~89.64% ± 4.89%
- **VGG19**: ~91.03% ± 3.98%
- **ResNet-34**: ~91.81% ± 3.40%
- **DenseNet-161**: ~89.64% ± 3.70%

### Ensemble Model
- Expected: ~92-95% accuracy (with proper data leakage prevention)
- F1 Score: ~0.92-0.95

## ⚠️ Critical Implementation Notes

### Data Leakage Prevention
The ensemble implementation ensures **NO DATA LEAKAGE** by:
1. Training fresh base models for each fold
2. Using only fold-specific training data
3. Generating predictions on completely unseen validation data
4. Meta-classifier never sees contaminated features

### Why This Matters
- ❌ **Wrong**: Load pre-trained models → predict on all data → 100% accuracy (leakage)
- ✅ **Correct**: Train new models per fold → predict on unseen data → realistic accuracy

## 🛠️ Requirements

```bash
# Python 3.12+
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pillow tqdm
```

## 🎯 Paper Compliance

✓ Base model architectures match Table S1  
✓ Meta-classifier architecture exact per paper  
✓ Hyperparameters from Table S1/S2 (HuSHeM)  
✓ Preprocessing pipeline consistent  
✓ 3×5-fold CV with proper data separation  
✓ Stacking methodology correctly implemented  

## 📝 Citation

Original Paper:
```
Spencer, R., Jalloh, I., Champneys, A. R., et al. (2022). 
Ensembled Deep Learning for the Classification of Human Sperm Head Morphology. 
Advanced Intelligent Systems, 4(8), 2200079.
```

## 👤 Author

Implementation by: 23giang.ns  
Repository: COMP3020 - Sperm Morphology Analysis

## 📄 License

This project is for academic research purposes.
