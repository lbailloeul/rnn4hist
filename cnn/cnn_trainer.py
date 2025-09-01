#!/usr/bin/env python3
"""
CNN Training Pipeline for SPS Histogram Classification
=====================================================
Trains a CNN on generated SPS histogram data with automatic 80-20 split.

Usage:
    python cnn_trainer.py --data_dir dataset --epochs 50 --batch_size 32

Features:
    - Automated 80-20 train/validation split
    - CNN architecture optimized for histogram images
    - Data augmentation for better generalization  
    - Comprehensive training monitoring and visualization
    - Model checkpointing and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, optimizers
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError:
    print("❌ TensorFlow not available. Install with: pip install tensorflow")
    exit(1)

class SPSCNNTrainer:
    def __init__(self, data_dir='dataset', output_dir='training_results', n_bins=300):
        """Initialize CNN trainer for histogram bin data"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.n_bins = n_bins  # Number of histogram bins
        self.model = None
        self.history = None
        
        # Neural network architecture parameters
        self.nn_config = {
            'input_shape': (n_bins,),       # 1D histogram bins
            'num_classes': 2,               # GOOD vs BAD
            'dropout_rate': 0.3,
            'l2_reg': 1e-4
        }
        
        # Check for ROOT availability
        try:
            import ROOT
            self.ROOT_available = True
            print("✅ ROOT available for loading histogram files")
        except ImportError:
            self.ROOT_available = False
            print("⚠️ ROOT not available, will use numpy files")
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Histogram bins: {n_bins}")
        
    def load_histogram_data(self, metadata):
        """Load histogram bin data from ROOT or numpy files"""
        X = []
        y = []
        
        print(f"📊 Loading {len(metadata)} histogram files...")
        
        for idx, row in metadata.iterrows():
            filename = row['filename']
            label = int(row['label'])
            
            # Determine file path based on label
            subdir = 'good' if label == 1 else 'bad'
            filepath = self.data_dir / subdir / filename
            
            if not filepath.exists():
                # Try alternative path (might be in root directory)
                filepath = self.data_dir / filename
                
            if not filepath.exists():
                print(f"⚠️ File not found: {filename}")
                continue
            
            # Load histogram data
            try:
                if filepath.suffix == '.root':
                    bin_data = self.load_root_histogram(filepath)
                elif filepath.suffix == '.npz':
                    bin_data = self.load_numpy_histogram(filepath)
                else:
                    print(f"⚠️ Unsupported file format: {filename}")
                    continue
                
                if bin_data is not None and len(bin_data) == self.n_bins:
                    X.append(bin_data)
                    y.append(label)
                else:
                    print(f"⚠️ Invalid histogram data in: {filename}")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def load_root_histogram(self, filepath):
        """Load histogram bin contents from ROOT file"""
        if not self.ROOT_available:
            return None
            
        try:
            import ROOT
            
            root_file = ROOT.TFile.Open(str(filepath), "READ")
            if not root_file or root_file.IsZombie():
                return None
            
            # Get the histogram (first histogram in file)
            key = root_file.GetListOfKeys().First()
            hist = root_file.Get(key.GetName())
            
            if not hist:
                root_file.Close()
                return None
            
            # Extract bin contents
            bin_data = np.zeros(self.n_bins)
            for i in range(self.n_bins):
                bin_data[i] = hist.GetBinContent(i + 1)  # ROOT bins start at 1
            
            root_file.Close()
            return bin_data
            
        except Exception as e:
            print(f"⚠️ ROOT loading error: {e}")
            return None
    
    def load_numpy_histogram(self, filepath):
        """Load histogram bin contents from numpy file"""
        try:
            data = np.load(filepath)
            return data['bin_counts']
        except Exception as e:
            print(f"⚠️ Numpy loading error: {e}")
            return None
    
    def load_and_split_data(self, test_size=0.2, random_state=42):
        """Load histogram data and create train/validation split"""
        print("\n📊 Loading dataset metadata...")
        
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        metadata = pd.read_csv(metadata_path)
        print(f"   Total samples: {len(metadata)}")
        
        # Check if we have labels
        if 'label' not in metadata.columns or metadata['label'].isna().all():
            raise ValueError("Dataset must have labels for supervised training")
            
        # Filter out unlabeled samples
        labeled_metadata = metadata[metadata['label'].notna()]
        print(f"   Labeled samples: {len(labeled_metadata)}")
        
        # Check class distribution
        label_counts = labeled_metadata['label'].value_counts()
        print(f"   Class distribution:")
        for label, count in label_counts.items():
            label_name = 'GOOD' if label == 1 else 'BAD'
            print(f"      {label_name}: {count} ({count/len(labeled_metadata)*100:.1f}%)")
        
        # Create train/validation split
        train_metadata, val_metadata = train_test_split(
            labeled_metadata, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labeled_metadata['label']
        )
        
        print(f"   Training samples: {len(train_metadata)}")
        print(f"   Validation samples: {len(val_metadata)}")
        
        # Load histogram data
        print("📂 Loading training data...")
        X_train, y_train = self.load_histogram_data(train_metadata)
        
        print("📂 Loading validation data...")
        X_val, y_val = self.load_histogram_data(val_metadata)
        
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Validation data shape: {X_val.shape}")
        
        # Save split metadata
        train_metadata.to_csv(self.output_dir / 'train_metadata.csv', index=False)
        val_metadata.to_csv(self.output_dir / 'val_metadata.csv', index=False)
        
        return (X_train, y_train), (X_val, y_val), train_metadata, val_metadata
        
    def create_data_generators(self, train_metadata, val_metadata, batch_size=32):
        """Create data generators with augmentation"""
        print(f"\n🔄 Creating data generators (batch size: {batch_size})...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,                    # Normalize pixel values
            rotation_range=10,                 # Slight rotations
            width_shift_range=0.1,             # Horizontal shifts
            height_shift_range=0.1,            # Vertical shifts
            zoom_range=0.1,                    # Slight zoom
            horizontal_flip=False,             # No horizontal flip (histograms are asymmetric)
            vertical_flip=False,               # No vertical flip
            fill_mode='nearest'
        )
        
        # Only rescaling for validation (no augmentation)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create temporary directories for train/val split
        temp_train_dir = self.output_dir / 'temp_train'
        temp_val_dir = self.output_dir / 'temp_val'
        
        # Create directory structure for flow_from_directory
        self._create_temp_directories(train_metadata, val_metadata, temp_train_dir, temp_val_dir)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            temp_train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            temp_val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"   Training generator: {train_generator.samples} samples")
        print(f"   Validation generator: {val_generator.samples} samples")
        print(f"   Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator, temp_train_dir, temp_val_dir
        
    def _create_temp_directories(self, train_metadata, val_metadata, temp_train_dir, temp_val_dir):
        """Create temporary directory structure for data generators"""
        import shutil
        
        # Clean up existing temp directories
        for temp_dir in [temp_train_dir, temp_val_dir]:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        # Create class directories
        for temp_dir in [temp_train_dir, temp_val_dir]:
            (temp_dir / 'good').mkdir(parents=True, exist_ok=True)
            (temp_dir / 'bad').mkdir(parents=True, exist_ok=True)
        
        # Copy/link files to appropriate directories
        def copy_files(metadata, target_dir):
            for _, row in metadata.iterrows():
                label = 'good' if row['label'] == 1 else 'bad'
                subdir = row['subdir']
                
                src_path = self.data_dir / subdir / row['filename']
                dst_path = target_dir / label / row['filename']
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
        
        copy_files(train_metadata, temp_train_dir)
        copy_files(val_metadata, temp_val_dir)
        
    def build_histogram_model(self):
        """Build neural network architecture optimized for histogram bin data"""
        print("\n🏗️ Building histogram neural network architecture...")
        
        model = keras.Sequential([
            # Input layer for histogram bins
            layers.Input(shape=self.nn_config['input_shape']),
            
            # Normalization layer
            layers.BatchNormalization(),
            
            # First dense block
            layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.nn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.nn_config['dropout_rate']),
            
            # Second dense block
            layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.nn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.nn_config['dropout_rate']),
            
            # Third dense block
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.nn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.nn_config['dropout_rate']),
            
            # Fourth dense block  
            layers.Dense(64, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.nn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.nn_config['dropout_rate']),
            
            # Output layer
            layers.Dense(self.nn_config['num_classes'], activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Print model summary
        print("📋 Model Architecture:")
        model.summary()
        
        # Save model architecture
        model_json = model.to_json()
        with open(self.output_dir / 'model_architecture.json', 'w') as json_file:
            json_file.write(model_json)
            
        return model
        
    def train_histogram_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the histogram neural network"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        # Set up callbacks
        callbacks_list = self.setup_callbacks(patience=10)
        
        start_time = datetime.now()
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        print(f"\n⏱️ Training completed in {training_time}")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        
        return self.history
        
    def evaluate_histogram_model(self, X_val, y_val):
        """Evaluate histogram model performance"""
        print("\n📊 Evaluating model performance...")
        
        # Load best model
        best_model = keras.models.load_model(self.output_dir / 'best_model.h5')
        
        # Evaluate on validation set
        val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
        
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Validation Loss: {val_loss:.4f}")
        
        # Get predictions for detailed analysis
        predictions = best_model.predict(X_val, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Classification report
        class_names = ['bad', 'good']
        class_report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Calculate F1 score
        val_f1 = class_report['weighted avg']['f1-score']
        
        # Save evaluation results
        eval_results = {
            'validation_accuracy': val_acc,
            'validation_precision': class_report['weighted avg']['precision'],
            'validation_recall': class_report['weighted avg']['recall'], 
            'validation_f1': val_f1,
            'classification_report': class_report
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results, y_val, y_pred, predictions
        
    def setup_callbacks(self, patience=10):
        """Setup training callbacks"""
        callback_list = [
            # Model checkpointing
            callbacks.ModelCheckpoint(
                self.output_dir / 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logging
            callbacks.CSVLogger(
                self.output_dir / 'training_log.csv',
                append=False
            )
        ]
        
        return callback_list
        
    def train_model(self, train_generator, val_generator, epochs=50, callbacks_list=None):
        """Train the CNN model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        start_time = datetime.now()
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        print(f"\n⏱️ Training completed in {training_time}")
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        
        return self.history
        
    def evaluate_model(self, val_generator):
        """Evaluate model performance"""
        print("\n📊 Evaluating model performance...")
        
        # Load best model
        best_model = keras.models.load_model(self.output_dir / 'best_model.h5')
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall = best_model.evaluate(val_generator, verbose=0)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
        
        print(f"   Validation Accuracy: {val_acc:.4f}")
        print(f"   Validation Precision: {val_precision:.4f}")
        print(f"   Validation Recall: {val_recall:.4f}")
        print(f"   Validation F1-Score: {val_f1:.4f}")
        
        # Get predictions for detailed analysis
        val_generator.reset()
        predictions = best_model.predict(val_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes
        
        # Classification report
        class_names = list(val_generator.class_indices.keys())
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Save evaluation results
        eval_results = {
            'validation_accuracy': val_acc,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
            'validation_f1': val_f1,
            'classification_report': class_report
        }
        
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
            
        return eval_results, y_true, y_pred, predictions
        
    def plot_training_history(self):
        """Plot training history"""
        print("\n📈 Creating training plots...")
        
        if self.history is None:
            print("   No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy
        ax = axes[0, 0]
        ax.plot(self.history.history['accuracy'], label='Training')
        ax.plot(self.history.history['val_accuracy'], label='Validation')
        ax.set_title('Model Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[0, 1]
        ax.plot(self.history.history['loss'], label='Training')
        ax.plot(self.history.history['val_loss'], label='Validation')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        ax = axes[1, 0]
        if 'learning_rate' in self.history.history:
            ax.plot(self.history.history['learning_rate'], label='Learning Rate', color='orange')
            ax.set_title('Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Rate')
        
        # Training Progress Summary
        ax = axes[1, 1]
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        summary_text = f"Final Results:\n\nTraining Accuracy: {final_train_acc:.3f}\nValidation Accuracy: {final_val_acc:.3f}\n\nTraining Loss: {final_train_loss:.3f}\nValidation Loss: {final_val_loss:.3f}"
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title('Training Summary')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Training plots saved to: {self.output_dir / 'training_history.png'}")
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def cleanup_temp_directories(self, temp_dirs):
        """Clean up temporary directories"""
        import shutil
        print("\n🧹 Cleaning up temporary directories...")
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
    def run_complete_pipeline(self, epochs=50, batch_size=32, test_size=0.2):
        """Run the complete histogram neural network training pipeline"""
        print("🎯 Starting complete histogram neural network training pipeline...")
        
        # Load histogram data and split
        (X_train, y_train), (X_val, y_val), train_metadata, val_metadata = self.load_and_split_data(test_size=test_size)
        
        # Validate data shapes
        print(f"   Training data: {X_train.shape}, labels: {y_train.shape}")
        print(f"   Validation data: {X_val.shape}, labels: {y_val.shape}")
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("No data loaded! Check histogram files and metadata.")
        
        # Build model
        self.build_histogram_model()
        
        # Train model
        self.train_histogram_model(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
        
        # Evaluate model
        eval_results, y_true, y_pred, predictions = self.evaluate_histogram_model(X_val, y_val)
        
        # Create visualizations
        self.plot_training_history()
        class_names = ['bad', 'good']
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        print(f"\n✅ Training pipeline completed successfully!")
        print(f"   Final validation accuracy: {eval_results['validation_accuracy']:.4f}")
        print(f"   Model saved to: {self.output_dir / 'best_model.h5'}")
        print(f"   Results saved to: {self.output_dir}")
        
        return eval_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='CNN Training for SPS Histogram Classification')
    parser.add_argument('--data_dir', default='dataset', help='Dataset directory')
    parser.add_argument('--output_dir', default='training_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_bins', type=int, default=300, help='Number of histogram bins')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("🧠 Histogram Neural Network Training Pipeline for SPS")
    print("=" * 50)
    print(f"📊 Dataset: {args.data_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🏃 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📊 Histogram bins: {args.n_bins}")
    print(f"✂️ Test split: {args.test_size}")
    
    # Initialize trainer
    trainer = SPSCNNTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_bins=args.n_bins
    )
    
    # Run training pipeline
    results = trainer.run_complete_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()