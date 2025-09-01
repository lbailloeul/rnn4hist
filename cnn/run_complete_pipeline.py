#!/usr/bin/env python3
"""
Complete SPS CNN Training Pipeline
==================================
Runs the complete pipeline: data generation + SVM labeling + CNN training

Usage:
    python run_complete_pipeline.py --num_samples 10000 --epochs 50

This script orchestrates:
1. Parameter sampling and histogram generation
2. Automatic SVM labeling (92.4% accuracy)
3. 80-20 train/validation split
4. CNN training with performance monitoring
5. Results visualization and model saving
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"   Command: {' '.join(command)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        print(f"✅ {description} completed in {elapsed_time:.1f}s")
        
        # Print last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')[-3:]
            for line in lines:
                print(f"   {line}")
                
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {description} failed after {elapsed_time:.1f}s")
        print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {description} failed after {elapsed_time:.1f}s")
        print(f"   Error: {str(e)}")
        return False

def check_dependencies():
    """Check if required Python packages are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scikit-learn', 'tensorflow', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True

def estimate_resources(num_samples, epochs):
    """Estimate resource requirements"""
    print(f"\n📊 Resource Estimation:")
    
    # Data generation estimates
    gen_time_mins = num_samples * 0.003  # ~3ms per histogram
    storage_gb = num_samples * 0.0003    # ~300KB per image
    
    # Training estimates  
    train_time_mins = epochs * (num_samples / 1000) * 0.5  # Rough estimate
    
    print(f"   Data Generation:")
    print(f"     Time: ~{gen_time_mins:.1f} minutes")
    print(f"     Storage: ~{storage_gb:.2f} GB")
    
    print(f"   CNN Training:")
    print(f"     Time: ~{train_time_mins:.1f} minutes (GPU recommended)")
    print(f"     Memory: ~4-8 GB GPU memory (batch_size dependent)")
    
    total_time_hours = (gen_time_mins + train_time_mins) / 60
    print(f"   Total Pipeline: ~{total_time_hours:.1f} hours")

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Complete SPS CNN Training Pipeline')
    
    # Data generation parameters
    parser.add_argument('--num_samples', type=int, default=10000, 
                       help='Number of histograms to generate')
    parser.add_argument('--dataset_dir', default='dataset', 
                       help='Dataset directory')
    
    # CNN training parameters
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Training batch size')
    parser.add_argument('--img_size', nargs=2, type=int, default=[224, 224], 
                       help='Image size for CNN')
    parser.add_argument('--results_dir', default='training_results', 
                       help='Training results directory')
    
    # Pipeline control
    parser.add_argument('--skip_generation', action='store_true', 
                       help='Skip data generation (use existing dataset)')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip CNN training (generate data only)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🎯 Complete SPS CNN Training Pipeline")
    print("=" * 60)
    print(f"📊 Samples: {args.num_samples:,}")
    print(f"🏃 Epochs: {args.epochs}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"🖼️ Image Size: {args.img_size}")
    print(f"🎲 Random Seed: {args.seed}")
    print(f"📁 Dataset: {args.dataset_dir}")
    print(f"📁 Results: {args.results_dir}")
    
    start_time = datetime.now()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return False
    
    # Estimate resources
    estimate_resources(args.num_samples, args.epochs)
    
    # Confirm execution
    print(f"\n⏳ Pipeline will take approximately {(args.num_samples * 0.003 + args.epochs * (args.num_samples / 1000) * 0.5) / 60:.1f} hours")
    
    if not args.skip_generation and not args.skip_training:
        response = input("\nProceed with complete pipeline? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Pipeline cancelled.")
            return False
    
    success = True
    
    # Step 1: Data Generation
    if not args.skip_generation:
        print(f"\n" + "="*60)
        print("STEP 1: DATA GENERATION")
        print("="*60)
        
        generation_cmd = [
            'python', 'data_generator.py',
            '--num_samples', str(args.num_samples),
            '--output_dir', args.dataset_dir,
            '--seed', str(args.seed)
        ]
        
        if not run_command(generation_cmd, "Data generation with SVM labeling"):
            success = False
        
        # Check if dataset was created successfully
        dataset_path = Path(args.dataset_dir)
        if success and dataset_path.exists():
            metadata_path = dataset_path / 'metadata.csv'
            good_dir = dataset_path / 'good'
            bad_dir = dataset_path / 'bad'
            
            if metadata_path.exists() and good_dir.exists() and bad_dir.exists():
                good_count = len(list(good_dir.glob('*.png')))
                bad_count = len(list(bad_dir.glob('*.png')))
                total_count = good_count + bad_count
                
                print(f"   ✅ Dataset created successfully:")
                print(f"      Total: {total_count:,} histograms")
                print(f"      GOOD: {good_count:,} ({good_count/total_count*100:.1f}%)")
                print(f"      BAD: {bad_count:,} ({bad_count/total_count*100:.1f}%)")
            else:
                print(f"   ⚠️ Dataset structure incomplete")
                success = False
    else:
        print(f"\n📂 Using existing dataset: {args.dataset_dir}")
    
    # Step 2: CNN Training
    if success and not args.skip_training:
        print(f"\n" + "="*60)
        print("STEP 2: CNN TRAINING")
        print("="*60)
        
        training_cmd = [
            'python', 'cnn_trainer.py',
            '--data_dir', args.dataset_dir,
            '--output_dir', args.results_dir,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--img_size', str(args.img_size[0]), str(args.img_size[1])
        ]
        
        if not run_command(training_cmd, "CNN training"):
            success = False
        
        # Check training results
        results_path = Path(args.results_dir)
        if success and results_path.exists():
            best_model_path = results_path / 'best_model.h5'
            eval_results_path = results_path / 'evaluation_results.json'
            
            if best_model_path.exists() and eval_results_path.exists():
                print(f"   ✅ Training completed successfully:")
                print(f"      Model saved: {best_model_path}")
                print(f"      Results: {results_path}")
                
                # Try to read final accuracy
                try:
                    import json
                    with open(eval_results_path) as f:
                        eval_data = json.load(f)
                    final_acc = eval_data.get('validation_accuracy', 'unknown')
                    print(f"      Final accuracy: {final_acc:.4f}" if isinstance(final_acc, float) else f"      Final accuracy: {final_acc}")
                except:
                    print(f"      Final accuracy: Check {eval_results_path}")
            else:
                print(f"   ⚠️ Training results incomplete")
                success = False
    elif args.skip_training:
        print(f"\n📊 Data generation only - CNN training skipped")
    
    # Pipeline Summary
    total_time = datetime.now() - start_time
    print(f"\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    if success:
        print(f"✅ Pipeline completed successfully in {total_time}")
        print(f"📊 Generated {args.num_samples:,} histograms with SVM labeling")
        if not args.skip_training:
            print(f"🧠 Trained CNN for {args.epochs} epochs")
            print(f"📁 Results available in: {args.results_dir}")
        print(f"📈 Next steps:")
        print(f"   - Review training results in {args.results_dir}/")
        print(f"   - Evaluate model performance on independent test set")
        print(f"   - Deploy model for histogram quality assessment")
    else:
        print(f"❌ Pipeline failed after {total_time}")
        print(f"🔧 Check error messages above for troubleshooting")
        print(f"💡 Try running individual steps manually:")
        print(f"   python data_generator.py --num_samples 1000 --output_dir test_dataset")
        print(f"   python cnn_trainer.py --data_dir test_dataset --epochs 10")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)