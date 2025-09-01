#!/usr/bin/env python3
"""
SPS Data Analysis - Simple version without pandas
Analyzes bounds, correlations, and clustering for GOOD vs BAD data classification
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

class SPSDataAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with data path"""
        self.data_path = data_path
        self.data = []
        self.features = ['Sigma_Gain', 'Sigma_0', 'Mu_p']
        
    def load_data(self):
        """Load and preprocess the data"""
        with open(self.data_path, 'r') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        
        # Convert numeric columns
        for row in self.data:
            for feature in self.features:
                row[feature] = float(row[feature])
        
        print(f"Loaded {len(self.data)} records")
        
        # Count ratings
        good_count = sum(1 for row in self.data if row['Rating'] == 'GOOD')
        bad_count = len(self.data) - good_count
        print(f"GOOD samples: {good_count}")
        print(f"BAD samples: {bad_count}")
        
        return self.data
        
    def analyze_bounds(self):
        """Analyze bounds and thresholds for good vs bad classification"""
        print("\n" + "="*60)
        print("BOUNDS ANALYSIS - Finding Decision Boundaries")
        print("="*60)
        
        good_data = [row for row in self.data if row['Rating'] == 'GOOD']
        bad_data = [row for row in self.data if row['Rating'] == 'BAD']
        
        bounds_analysis = {}
        
        for feature in self.features:
            print(f"\n--- {feature} Analysis ---")
            
            # Extract values
            good_values = [row[feature] for row in good_data]
            bad_values = [row[feature] for row in bad_data]
            all_values = [row[feature] for row in self.data]
            
            # Calculate statistics
            good_stats = {
                'mean': np.mean(good_values),
                'std': np.std(good_values),
                'min': np.min(good_values),
                'max': np.max(good_values),
                'q25': np.percentile(good_values, 25),
                'q75': np.percentile(good_values, 75)
            }
            
            bad_stats = {
                'mean': np.mean(bad_values),
                'std': np.std(bad_values),
                'min': np.min(bad_values),
                'max': np.max(bad_values),
                'q25': np.percentile(bad_values, 25),
                'q75': np.percentile(bad_values, 75)
            }
            
            print(f"GOOD {feature}: mean={good_stats['mean']:.3f}, std={good_stats['std']:.3f}")
            print(f"     Range: [{good_stats['min']:.3f}, {good_stats['max']:.3f}]")
            print(f"     Q25-Q75: [{good_stats['q25']:.3f}, {good_stats['q75']:.3f}]")
            
            print(f"BAD {feature}: mean={bad_stats['mean']:.3f}, std={bad_stats['std']:.3f}")
            print(f"     Range: [{bad_stats['min']:.3f}, {bad_stats['max']:.3f}]")
            print(f"     Q25-Q75: [{bad_stats['q25']:.3f}, {bad_stats['q75']:.3f}]")
            
            # Find optimal threshold
            threshold_candidates = np.linspace(min(all_values), max(all_values), 100)
            
            best_threshold = None
            best_accuracy = 0
            
            for thresh in threshold_candidates:
                # Test both rules
                high_vals_good = sum(1 for val in good_values if val > thresh)
                high_vals_bad = sum(1 for val in bad_values if val > thresh)
                low_vals_good = sum(1 for val in good_values if val <= thresh)
                low_vals_bad = sum(1 for val in bad_values if val <= thresh)
                
                # Scenario 1: high values = GOOD
                acc1 = (high_vals_good + low_vals_bad) / len(self.data)
                
                # Scenario 2: low values = GOOD  
                acc2 = (low_vals_good + high_vals_bad) / len(self.data)
                
                if acc1 > best_accuracy:
                    best_accuracy = acc1
                    best_threshold = (thresh, "high_good")
                    
                if acc2 > best_accuracy:
                    best_accuracy = acc2
                    best_threshold = (thresh, "low_good")
            
            bounds_analysis[feature] = {
                'threshold': best_threshold[0],
                'rule': best_threshold[1], 
                'accuracy': best_accuracy,
                'good_stats': good_stats,
                'bad_stats': bad_stats
            }
            
            print(f"Best threshold: {best_threshold[0]:.3f} ({best_threshold[1]}) - Accuracy: {best_accuracy:.3f}")
        
        return bounds_analysis
    
    def visualize_data(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("DATA VISUALIZATION")
        print("="*60)
        
        good_data = [row for row in self.data if row['Rating'] == 'GOOD']
        bad_data = [row for row in self.data if row['Rating'] == 'BAD']
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Feature histograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SPS Data Feature Analysis - GOOD vs BAD', fontsize=16)
        
        colors = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}  # Sea green and crimson
        
        for i, feature in enumerate(self.features):
            ax = axes[i//2, i%2] if i < 3 else axes[1, 1]
            
            good_values = [row[feature] for row in good_data]
            bad_values = [row[feature] for row in bad_data]
            
            ax.hist(good_values, alpha=0.6, label='GOOD', color=colors['GOOD'], bins=30)
            ax.hist(bad_values, alpha=0.6, label='BAD', color=colors['BAD'], bins=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{feature} Distribution')
        
        # Remove empty subplot
        if len(self.features) == 3:
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/feature_histograms.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Feature Scatter Plots - GOOD vs BAD', fontsize=16)
        
        feature_pairs = [
            ('Sigma_Gain', 'Sigma_0'),
            ('Sigma_Gain', 'Mu_p'),
            ('Sigma_0', 'Mu_p')
        ]
        
        for i, (feat1, feat2) in enumerate(feature_pairs):
            ax = axes[i]
            
            good_x = [row[feat1] for row in good_data]
            good_y = [row[feat2] for row in good_data]
            bad_x = [row[feat1] for row in bad_data]
            bad_y = [row[feat2] for row in bad_data]
            
            ax.scatter(good_x, good_y, alpha=0.6, label='GOOD', color=colors['GOOD'], s=20)
            ax.scatter(bad_x, bad_y, alpha=0.6, label='BAD', color=colors['BAD'], s=20)
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{feat1} vs {feat2}')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/scatter_plots.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def simple_clustering(self):
        """Perform simple 2-means clustering"""
        print("\n" + "="*60)
        print("SIMPLE CLUSTERING ANALYSIS")
        print("="*60)
        
        # Extract features
        X = np.array([[row[feat] for feat in self.features] for row in self.data])
        
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        
        # Simple k-means implementation
        np.random.seed(42)
        n_clusters = 2
        
        # Initialize centroids randomly
        centroids = X_norm[np.random.choice(len(X_norm), n_clusters, replace=False)]
        
        for iteration in range(100):
            # Assign points to closest centroid
            distances = np.sqrt(((X_norm[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            cluster_labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([X_norm[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        # Analyze clusters
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_ratings = [self.data[i]['Rating'] for i in cluster_indices]
            
            good_count = cluster_ratings.count('GOOD')
            bad_count = cluster_ratings.count('BAD')
            total_count = len(cluster_indices)
            
            purity = max(good_count, bad_count) / total_count
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Total samples: {total_count}")
            print(f"  GOOD samples: {good_count} ({good_count/total_count:.1%})")
            print(f"  BAD samples: {bad_count} ({bad_count/total_count:.1%})")
            print(f"  Purity: {purity:.3f}")
        
        return cluster_labels
        
    def generate_report(self, bounds_analysis):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        good_count = sum(1 for row in self.data if row['Rating'] == 'GOOD')
        bad_count = len(self.data) - good_count
        
        report = []
        report.append("# SPS Data Analysis Report\n")
        report.append("## Dataset Overview\n")
        report.append(f"- Total samples: {len(self.data)}\n")
        report.append(f"- GOOD samples: {good_count}\n")
        report.append(f"- BAD samples: {bad_count}\n")
        report.append(f"- Features analyzed: {', '.join(self.features)}\n\n")
        
        report.append("## Optimal Decision Boundaries\n")
        for feature, analysis in bounds_analysis.items():
            rule_desc = "Higher values indicate GOOD" if analysis['rule'] == 'high_good' else "Lower values indicate GOOD"
            report.append(f"### {feature}\n")
            report.append(f"- **Threshold**: {analysis['threshold']:.3f}\n")
            report.append(f"- **Rule**: {rule_desc}\n")
            report.append(f"- **Classification Accuracy**: {analysis['accuracy']:.1%}\n")
            report.append(f"- **GOOD range**: [{analysis['good_stats']['min']:.3f}, {analysis['good_stats']['max']:.3f}]\n")
            report.append(f"- **BAD range**: [{analysis['bad_stats']['min']:.3f}, {analysis['bad_stats']['max']:.3f}]\n\n")
        
        report.append("## Key Findings\n")
        
        # Determine best single feature classifier
        best_feature = max(bounds_analysis.keys(), key=lambda k: bounds_analysis[k]['accuracy'])
        best_acc = bounds_analysis[best_feature]['accuracy']
        
        report.append(f"- **Best single feature**: {best_feature} with {best_acc:.1%} accuracy\n")
        
        # Feature importance insights
        report.append("- **Feature Insights**:\n")
        for feature, analysis in bounds_analysis.items():
            good_mean = analysis['good_stats']['mean']
            bad_mean = analysis['bad_stats']['mean']
            if good_mean != 0:
                diff_pct = abs((good_mean - bad_mean) / good_mean * 100)
            else:
                diff_pct = 0
            report.append(f"  - {feature}: GOOD samples average {good_mean:.3f}, BAD samples average {bad_mean:.3f} (difference: {diff_pct:.1f}%)\n")
        
        report_text = ''.join(report)
        
        # Save report
        with open('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text

def main():
    """Main analysis function"""
    print("SPS Data Analysis - Finding bounds for GOOD vs BAD classification")
    print("Note: Set numbers can repeat - each entry represents unique data")
    
    # Initialize analyzer
    analyzer = SPSDataAnalyzer('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-generation/sps_rating_results.csv')
    
    # Load data
    data = analyzer.load_data()
    
    # Analyze bounds and thresholds
    bounds_analysis = analyzer.analyze_bounds()
    
    # Create visualizations
    analyzer.visualize_data(save_plots=True)
    
    # Perform clustering analysis
    cluster_labels = analyzer.simple_clustering()
    
    # Generate comprehensive report
    analyzer.generate_report(bounds_analysis)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Generated files:")
    print("- analysis_report.md")
    print("- feature_histograms.png")
    print("- scatter_plots.png")

if __name__ == "__main__":
    main()