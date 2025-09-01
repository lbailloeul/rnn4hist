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
    def __init__(self, data_dir='dataset', output_dir='training_results', img_size=(224, 224)):
        """Initialize CNN trainer"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.img_size = img_size
        self.model = None
        self.history = None
        
        # CNN architecture parameters
        self.cnn_config = {
            'input_shape': (*img_size, 3),  # RGB images
            'num_classes': 2,               # GOOD vs BAD
            'dropout_rate': 0.5,
            'l2_reg': 1e-4
        }
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖼️ Image size: {img_size}")
        
    def load_and_split_data(self, test_size=0.2, random_state=42):
        """Load metadata and create train/validation split"""
        print("\n📊 Loading dataset metadata...")
        
        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        metadata = pd.read_csv(metadata_path)
        print(f"   Total samples: {len(metadata)}")
        
        # Check if we have labels
        if 'label' not in metadata or metadata['label'].isna().any():
            raise ValueError("Dataset must have labels for supervised training")
            
        # Print class distribution
        class_counts = metadata['label'].value_counts()
        print(f"   Class distribution:")
        for label, count in class_counts.items():
            label_name = 'GOOD' if label == 1 else 'BAD'
            print(f"     {label_name}: {count} ({count/len(metadata)*100:.1f}%)")
        
        # Create train/validation split
        train_metadata, val_metadata = train_test_split(
            metadata, test_size=test_size, random_state=random_state, 
            stratify=metadata['label']
        )
        
        print(f"\n✂️ Dataset split (80-20):")
        print(f"   Training: {len(train_metadata)} samples")
        print(f"   Validation: {len(val_metadata)} samples")
        
        # Save split information
        train_metadata.to_csv(self.output_dir / 'train_metadata.csv', index=False)
        val_metadata.to_csv(self.output_dir / 'val_metadata.csv', index=False)
        
        return train_metadata, val_metadata
        
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
        
    def build_cnn_model(self):
        """Build CNN architecture optimized for histogram classification"""
        print("\n🏗️ Building CNN architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.cnn_config['input_shape']),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(), 
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and fully connected layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.cnn_config['dropout_rate']),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(self.cnn_config['l2_reg'])),
            layers.BatchNormalization(),
            layers.Dropout(self.cnn_config['dropout_rate']),
            
            # Output layer
            layers.Dense(self.cnn_config['num_classes'], activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
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
        
        # Precision
        ax = axes[1, 0]
        ax.plot(self.history.history['precision'], label='Training')
        ax.plot(self.history.history['val_precision'], label='Validation')
        ax.set_title('Model Precision')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Recall
        ax = axes[1, 1]
        ax.plot(self.history.history['recall'], label='Training')
        ax.plot(self.history.history['val_recall'], label='Validation')
        ax.set_title('Model Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
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
        """Run the complete training pipeline"""
        print("🎯 Starting complete CNN training pipeline...")
        
        # Load and split data
        train_metadata, val_metadata = self.load_and_split_data(test_size=test_size)
        
        # Create data generators
        train_gen, val_gen, temp_train, temp_val = self.create_data_generators(
            train_metadata, val_metadata, batch_size=batch_size
        )
        
        # Build model
        self.build_cnn_model()
        
        # Setup callbacks
        callbacks_list = self.setup_callbacks()
        
        # Train model
        self.train_model(train_gen, val_gen, epochs=epochs, callbacks_list=callbacks_list)
        
        # Evaluate model
        eval_results, y_true, y_pred, predictions = self.evaluate_model(val_gen)
        
        # Create visualizations
        self.plot_training_history()
        class_names = list(val_gen.class_indices.keys())
        self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Cleanup
        self.cleanup_temp_directories([temp_train, temp_val])
        
        print(f"\n✅ Training pipeline completed!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"🎯 Final validation accuracy: {eval_results['validation_accuracy']:.4f}")
        
        return eval_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='CNN Training for SPS Histogram Classification')
    parser.add_argument('--data_dir', default='dataset', help='Dataset directory')
    parser.add_argument('--output_dir', default='training_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', nargs=2, type=int, default=[224, 224], help='Image size (height width)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    print("🧠 CNN Training Pipeline for SPS Histograms")
    print("=" * 50)
    print(f"📊 Dataset: {args.data_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🏃 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🖼️ Image size: {args.img_size}")
    print(f"✂️ Test split: {args.test_size}")
    
    # Initialize trainer
    trainer = SPSCNNTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=tuple(args.img_size)
    )
    
    # Run training pipeline
    results = trainer.run_complete_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size
    )

if __name__ == "__main__":
    main()