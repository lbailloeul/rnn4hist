# SPS Data Analysis Tools

This directory contains tools for analyzing Single Photon Sensor (SPS) histogram fitting quality data, with a focus on distinguishing between GOOD and BAD quality measurements.

## 🗂️ Directory Structure

```
data-analysis/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── visualize.py                # 📊 Visualization script  
├── analyze.py                  # 🔬 Statistical analysis script
├── sps_analysis_env/           # Virtual environment
└── outputs/                    # Generated results
    ├── *.png                   # Static plots
    ├── *.html                  # Interactive plots
    └── *.md                    # Analysis reports
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate the virtual environment
source sps_analysis_env/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Generate Visualizations

```bash
# Create all visualizations (static, 3D, and interactive)
python visualize.py

# Create only interactive plots
python visualize.py --interactive

# Create only static plots  
python visualize.py --static

# Create only 3D plots
python visualize.py --3d
```

### 3. Run Statistical Analysis

```bash
# Run full analysis suite
python analyze.py

# Run specific analyses
python analyze.py --bounds --pca
python analyze.py --clustering --lda
python analyze.py --report
```

## 📊 Visualization Script (`visualize.py`)

Creates comprehensive visualizations to understand the data structure and class separation.

### Features:
- **Static Plots**: Histograms, scatter plots, box plots using matplotlib
- **3D Visualizations**: Multi-angle 3D scatter plots of feature space  
- **Interactive Plots**: Plotly-based interactive 3D plots with hover details

### Generated Files:
- `feature_distributions.png` - Histograms of each feature by class
- `scatter_matrix.png` - 2D scatter plots of feature pairs
- `3d_scatter.png` - Single 3D scatter plot
- `3d_multiple_views.png` - 3D plots from different angles
- `interactive_3d.html` - Basic interactive 3D plot
- `interactive_enhanced.html` - 3D plot with centroids and statistics
- `interactive_dashboard.html` - Multi-view dashboard

### Usage Options:
```bash
python visualize.py [options]

Options:
  --interactive    Generate interactive plotly visualizations (default: True)
  --static        Generate static matplotlib visualizations (default: True)
  --3d            Generate 3D visualizations (default: True)
  --output-dir    Output directory for plots (default: outputs)
```

## 🔬 Analysis Script (`analyze.py`)

Performs comprehensive statistical analysis and machine learning to find optimal classification strategies.

### Features:
- **Boundary Analysis**: Finds optimal thresholds for each feature
- **PCA Analysis**: Principal component analysis with class separation metrics
- **Clustering**: K-means clustering with silhouette analysis
- **LDA**: Linear discriminant analysis for supervised dimensionality reduction
- **Comprehensive Reporting**: Generates detailed markdown reports

### Generated Files:
- `pca_analysis.png` - PCA visualizations and component analysis
- `clustering_analysis.png` - K-means clustering results  
- `lda_analysis.png` - Linear discriminant analysis plots
- `comprehensive_analysis_report.md` - Detailed analysis report

### Usage Options:
```bash
python analyze.py [options]

Options:
  --bounds        Find optimal classification boundaries (default: True)
  --pca          Perform PCA analysis (default: True) 
  --clustering   Run clustering analysis (default: True)
  --lda          Linear Discriminant Analysis (default: True)
  --report       Generate comprehensive report (default: True)
  --output-dir   Output directory for results (default: outputs)
```

## 📈 Data Overview

The analysis works with SPS histogram fitting quality data containing:

- **Features**: 
  - `Sigma_Gain` - Gain parameter sigma
  - `Sigma_0` - Baseline sigma parameter  
  - `Mu_p` - Occupancy parameter
- **Target**: `Rating` (GOOD/BAD classification)
- **Metadata**: `Set` (can repeat - each entry is unique data)

**Data Statistics** (typical):
- Total samples: ~1000+
- GOOD samples: ~75%
- BAD samples: ~25%
- Feature ranges: 0.5-6.0 (varies by feature)

## 🎯 Key Analysis Outputs

### Classification Boundaries
- Optimal thresholds for each feature
- Classification accuracy for single-feature classifiers
- Effect sizes (Cohen's d) measuring feature discriminative power

### Principal Component Analysis  
- Variance explanation by each component
- Feature loadings showing which original features contribute to each PC
- Class separation metrics (Fisher discriminant ratios)

### Clustering Analysis
- Optimal number of clusters using silhouette analysis
- Cluster purity analysis (how well clusters align with true labels)
- Visualization of natural data groupings

### Recommendations
- Best single-feature classifier
- Multi-feature combination strategies
- Suggestions for quality control implementation

## 🔧 Extending the Analysis

### Adding New Features
1. Update the `features` list in both scripts
2. Ensure new features are in the CSV data
3. Run analysis to see how new features perform

### Custom Visualizations
- Modify the `SPSVisualizer` class in `visualize.py`  
- Add new plotting methods following existing patterns
- Update the main function to call new methods

### Additional Analysis Methods
- Modify the `SPSAnalyzer` class in `analyze.py`
- Add methods like `perform_svm_analysis()` or `perform_random_forest_analysis()`
- Update the reporting function to include new results

## 🐛 Troubleshooting

### Common Issues:
1. **Import errors**: Ensure virtual environment is activated
2. **Data not found**: Check that `../data-generation/sps_rating_results.csv` exists
3. **Permission errors**: Ensure write access to `outputs/` directory
4. **Display issues**: Interactive plots require a web browser

### Dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0  
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- plotly >= 5.17.0

## 📝 Output Interpretation

### Visualization Outputs:
- **Static plots**: Good for publications and reports
- **3D plots**: Help understand spatial relationships
- **Interactive plots**: Best for exploration and presentations

### Analysis Metrics:
- **Classification Accuracy**: Higher = better separation
- **Fisher Discriminant Ratio**: Higher = better class separation
- **Silhouette Score**: Higher = better clustering quality  
- **Cohen's d**: >0.8 large effect, >0.5 medium effect, >0.2 small effect

## 🔬 Research Applications

This analysis framework is designed for:
- **Quality Control**: Automated classification of measurement quality
- **Feature Selection**: Identifying most informative parameters
- **Method Validation**: Understanding measurement reliability
- **Process Optimization**: Improving experimental protocols

---

*For questions or contributions, please refer to the main project documentation.*