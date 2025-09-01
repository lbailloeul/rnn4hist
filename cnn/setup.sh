#!/bin/bash
# Setup script for CNN training environment

echo "🚀 Setting up CNN training environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv cnn_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source cnn_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To use the environment:"
echo "  source cnn_env/bin/activate"
echo ""
echo "Then run:"
echo "  python run_complete_pipeline.py --num_samples 10000 --epochs 50"