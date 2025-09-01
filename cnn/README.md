# CNN Training for SPS Histogram Classification

This directory contains the complete pipeline for training a CNN to classify SPS (Single Photon Sensor) histogram quality using automatically labeled data.

## Overview

The pipeline consists of two main components:

1. **`data_generator.py`**: Generates large-scale dataset (10k histograms) with parameter variance and automatic SVM labeling
2. **`cnn_trainer.py`**: Trains CNN on generated data with 80-20 train/validation split

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv cnn_env
source cnn_env/bin/activate  # Linux/Mac
# or
cnn_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
# Generate 10k histograms with SVM labeling
python data_generator.py --num_samples 10000 --output_dir dataset

# Expected output structure:
# dataset/
# ├── good/           # ~7500 GOOD histograms (based on SVM 92.4% accuracy)
# ├── bad/            # ~2500 BAD histograms
# └── metadata.csv    # Complete dataset metadata
```

### 3. Train CNN

```bash
# Train CNN with default settings
python cnn_trainer.py --data_dir dataset --epochs 50 --batch_size 32

# Results saved to:
# training_results/
# ├── best_model.h5              # Best trained model
# ├── training_history.png       # Training curves
# ├── confusion_matrix.png       # Performance visualization
# ├── evaluation_results.json    # Detailed metrics
# └── training_log.csv           # Complete training log
```

## Pipeline Details

### Data Generation (`data_generator.py`)

**Parameter Sampling:**
- `Sigma_Gain`: 0.5-6.0 (mean: 3.3, std: 1.5)
- `Sigma_0`: 0.3-3.0 (mean: 1.7, std: 0.8)  
- `Mu_p`: 0.1-3.5 (mean: 1.9, std: 1.0)

**Histogram Generation:**
- Uses Generalized Poisson distribution
- 300 bins, 0-300 ADU range
- 3000 events per histogram
- Exports as PNG images (224x224) for CNN training

**SVM Labeling:**
- Uses pre-trained SVM (92.4% accuracy)
- Automatic GOOD/BAD classification
- Confidence scores included in metadata

### CNN Training (`cnn_trainer.py`)

**Architecture:**
- 4 Convolutional blocks (32→64→128→256 filters)
- Batch normalization and dropout for regularization
- Global average pooling + dense layers
- Binary classification (GOOD vs BAD)

**Training Features:**
- 80-20 train/validation split
- Data augmentation (rotation, shifts, zoom)
- Early stopping and learning rate reduction
- Model checkpointing (saves best model)
- Comprehensive performance monitoring

**Expected Performance:**
- Target accuracy: >95% (improvement over SVM's 92.4%)
- Training time: ~1-2 hours on GPU
- Model size: ~50MB

## Advanced Usage

### Custom Dataset Generation

```bash
# Generate smaller dataset for testing
python data_generator.py --num_samples 1000 --output_dir test_dataset --seed 123

# Generate without SVM labeling (manual labeling required)
python data_generator.py --num_samples 5000 --svm_model None
```

### Custom CNN Training

```bash
# Longer training with larger batch size
python cnn_trainer.py --epochs 100 --batch_size 64 --img_size 256 256

# Different train/validation split
python cnn_trainer.py --test_size 0.3 --output_dir results_70_30
```

## File Descriptions

### Generated Files

**Dataset Structure:**
```
dataset/
├── good/
│   ├── hist_000001.png  # GOOD histogram images
│   └── ...
├── bad/
│   ├── hist_000002.png  # BAD histogram images  
│   └── ...
└── metadata.csv        # Complete metadata with parameters and labels
```

**Training Results:**
```
training_results/
├── best_model.h5              # Keras model (best validation accuracy)
├── model_architecture.json    # Model structure
├── training_history.csv       # Epoch-by-epoch metrics
├── training_history.png       # Training curves visualization
├── confusion_matrix.png       # Classification performance
├── evaluation_results.json    # Final performance metrics
├── train_metadata.csv         # Training set metadata
└── val_metadata.csv           # Validation set metadata
```

## Performance Expectations

### Data Generation
- **10k histograms**: ~30-45 minutes
- **SVM labeling**: ~74% GOOD, ~26% BAD (based on original analysis)
- **Storage**: ~2-3 GB for 10k PNG images

### CNN Training
- **50 epochs**: ~1-2 hours (GPU recommended)
- **Target accuracy**: 95%+ validation accuracy
- **Memory usage**: ~4-8 GB GPU memory (depends on batch size)

## Troubleshooting

### Common Issues

1. **ROOT not found**: Data generator falls back to matplotlib simulation
2. **TensorFlow GPU not detected**: CNN will use CPU (slower but functional)
3. **Memory errors**: Reduce batch size (`--batch_size 16`)
4. **SVM model loading fails**: Check that data-analysis pipeline was run first

### Performance Optimization

- **GPU Training**: Install `tensorflow-gpu` for faster training
- **Larger Batch Size**: Use `--batch_size 64` if sufficient GPU memory
- **Image Size**: Reduce to `--img_size 128 128` for faster training
- **Parallel Data Loading**: Set `workers=4` in data generators

## Integration with Analysis Pipeline

This CNN training pipeline integrates with the analysis results:

1. **SVM Model**: Loads from `../data-analysis/analyze.py` results
2. **Parameter Distributions**: Based on real data analysis
3. **Performance Baseline**: Aims to exceed SVM's 92.4% accuracy

The complete workflow:
```
data-generation → data-analysis → cnn (this directory)
```

## Next Steps

After training, the CNN can be:
1. **Deployed**: For real-time histogram quality assessment  
2. **Fine-tuned**: On additional manual labels for better performance
3. **Extended**: For multi-class classification (excellent/good/fair/poor)
4. **Integrated**: Into existing ROOT/C++ analysis pipeline