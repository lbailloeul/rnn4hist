#!/usr/bin/env python3
"""
SPS Data Analysis - Histogram Fitting Quality Assessment
Analyzes bounds, correlations, and clustering for GOOD vs BAD data classification
Note: Set numbers can repeat - each entry represents unique measurement data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

class SPSDataAnalyzer:
    def __init__(self, data_path):
        """Initialize analyzer with data path"""
        self.data_path = data_path
        self.data = None
        self.features = ['Sigma_Gain', 'Sigma_0', 'Mu_p']
        
    def load_data(self):
        """Load and preprocess the data"""
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data)} records")
        print(f"Features: {self.features}")
        print(f"Target distribution:")
        print(self.data['Rating'].value_counts())
        return self.data
        
    def analyze_bounds(self):
        """Analyze bounds and thresholds for good vs bad classification"""
        print("\n" + "="*60)
        print("BOUNDS ANALYSIS - Finding Decision Boundaries")
        print("="*60)
        
        good_data = self.data[self.data['Rating'] == 'GOOD']
        bad_data = self.data[self.data['Rating'] == 'BAD']
        
        print(f"\nGOOD samples: {len(good_data)}")
        print(f"BAD samples: {len(bad_data)}")
        
        bounds_analysis = {}
        
        for feature in self.features:
            print(f"\n--- {feature} Analysis ---")
            
            good_stats = good_data[feature].describe()
            bad_stats = bad_data[feature].describe()
            
            print(f"GOOD {feature}: mean={good_stats['mean']:.3f}, std={good_stats['std']:.3f}")
            print(f"     Range: [{good_stats['min']:.3f}, {good_stats['max']:.3f}]")
            print(f"     Q25-Q75: [{good_stats['25%']:.3f}, {good_stats['75%']:.3f}]")
            
            print(f"BAD {feature}: mean={bad_stats['mean']:.3f}, std={bad_stats['std']:.3f}")
            print(f"     Range: [{bad_stats['min']:.3f}, {bad_stats['max']:.3f}]")
            print(f"     Q25-Q75: [{bad_stats['25%']:.3f}, {bad_stats['75%']:.3f}]")
            
            # Find optimal threshold using median split
            combined_values = self.data[feature].values
            threshold_candidates = np.linspace(combined_values.min(), combined_values.max(), 100)
            
            best_threshold = None
            best_accuracy = 0
            
            for thresh in threshold_candidates:
                # Test if high values are good or bad
                high_vals_good = len(good_data[good_data[feature] > thresh])
                high_vals_bad = len(bad_data[bad_data[feature] > thresh])
                low_vals_good = len(good_data[good_data[feature] <= thresh])
                low_vals_bad = len(bad_data[bad_data[feature] <= thresh])
                
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
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        
        # 1. Pairplot with classification
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('SPS Data Feature Analysis - GOOD vs BAD', fontsize=16, y=0.95)
        
        colors = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}  # Sea green and crimson
        
        # Diagonal: histograms
        for i, feature in enumerate(self.features):
            ax = axes[i, i]
            for rating in ['GOOD', 'BAD']:
                data_subset = self.data[self.data['Rating'] == rating][feature]
                ax.hist(data_subset, alpha=0.6, label=rating, color=colors[rating], bins=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Off-diagonal: scatter plots
        for i in range(3):
            for j in range(3):
                if i != j:
                    ax = axes[i, j]
                    for rating in ['GOOD', 'BAD']:
                        data_subset = self.data[self.data['Rating'] == rating]
                        ax.scatter(data_subset[self.features[j]], data_subset[self.features[i]], 
                                 alpha=0.6, label=rating, color=colors[rating], s=20)
                    ax.set_xlabel(self.features[j])
                    ax.set_ylabel(self.features[i])
                    ax.grid(True, alpha=0.3)
                    if i == 0 and j == 1:  # Add legend only once
                        ax.legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/feature_analysis.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Box plots for each feature
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Feature Distributions by Rating', fontsize=16)
        
        for i, feature in enumerate(self.features):
            ax = axes[i]
            self.data.boxplot(column=feature, by='Rating', ax=ax)
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel('Rating')
            ax.set_ylabel(feature)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/boxplots.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(10, 8))
        # Encode rating as numeric for correlation
        data_numeric = self.data.copy()
        data_numeric['Rating_num'] = data_numeric['Rating'].map({'GOOD': 1, 'BAD': 0})
        
        corr_matrix = data_numeric[self.features + ['Rating_num']].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Matrix\n(Rating_num: 1=GOOD, 0=BAD)')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_3d(self, save_plots=True):
        """Create 3D visualizations of the feature space"""
        print("\n" + "="*60)
        print("3D FEATURE SPACE VISUALIZATION")
        print("="*60)
        
        # Prepare data
        good_data = self.data[self.data['Rating'] == 'GOOD']
        bad_data = self.data[self.data['Rating'] == 'BAD']
        
        colors = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}
        
        # 1. Original 3D feature space
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Sigma_Gain vs Sigma_0 vs Mu_p
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(good_data['Sigma_Gain'], good_data['Sigma_0'], good_data['Mu_p'], 
                   c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax1.scatter(bad_data['Sigma_Gain'], bad_data['Sigma_0'], bad_data['Mu_p'], 
                   c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax1.set_xlabel('Sigma_Gain')
        ax1.set_ylabel('Sigma_0')
        ax1.set_zlabel('Mu_p')
        ax1.set_title('Original 3D Feature Space')
        ax1.legend()
        
        # Plot 2: Different angle view
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(good_data['Sigma_Gain'], good_data['Sigma_0'], good_data['Mu_p'], 
                   c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax2.scatter(bad_data['Sigma_Gain'], bad_data['Sigma_0'], bad_data['Mu_p'], 
                   c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax2.set_xlabel('Sigma_Gain')
        ax2.set_ylabel('Sigma_0')
        ax2.set_zlabel('Mu_p')
        ax2.set_title('3D Feature Space (Different View)')
        ax2.view_init(elev=20, azim=45)
        ax2.legend()
        
        # Standardize features for PCA
        X = self.data[self.features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA with 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe for easier handling
        pca_df = self.data.copy()
        pca_df['PC1'] = X_pca[:, 0]
        pca_df['PC2'] = X_pca[:, 1]
        pca_df['PC3'] = X_pca[:, 2]
        
        good_pca = pca_df[pca_df['Rating'] == 'GOOD']
        bad_pca = pca_df[pca_df['Rating'] == 'BAD']
        
        # Plot 3: PCA 3D space
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.scatter(good_pca['PC1'], good_pca['PC2'], good_pca['PC3'], 
                   c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax3.scatter(bad_pca['PC1'], bad_pca['PC2'], bad_pca['PC3'], 
                   c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
        ax3.set_title('PCA 3D Space')
        ax3.legend()
        
        # Plot 4: PCA different angle
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.scatter(good_pca['PC1'], good_pca['PC2'], good_pca['PC3'], 
                   c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax4.scatter(bad_pca['PC1'], bad_pca['PC2'], bad_pca['PC3'], 
                   c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax4.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
        ax4.set_title('PCA 3D Space (Side View)')
        ax4.view_init(elev=0, azim=0)
        ax4.legend()
        
        # Plot 5: Density visualization using alpha blending
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        # Use smaller alpha for density effect
        ax5.scatter(good_pca['PC1'], good_pca['PC2'], good_pca['PC3'], 
                   c=colors['GOOD'], label='GOOD', alpha=0.3, s=20)
        ax5.scatter(bad_pca['PC1'], bad_pca['PC2'], bad_pca['PC3'], 
                   c=colors['BAD'], label='BAD', alpha=0.3, s=20)
        ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax5.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
        ax5.set_title('PCA 3D Density View')
        ax5.legend()
        
        # Plot 6: Feature importance visualization
        ax6 = fig.add_subplot(2, 3, 6)
        # Show PCA component loadings
        components = pca.components_
        feature_names = self.features
        
        x = np.arange(len(feature_names))
        width = 0.25
        
        ax6.bar(x - width, components[0, :], width, label=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', alpha=0.8)
        ax6.bar(x, components[1, :], width, label=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', alpha=0.8)
        ax6.bar(x + width, components[2, :], width, label=f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', alpha=0.8)
        
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Component Loading')
        ax6.set_title('PCA Component Loadings')
        ax6.set_xticks(x)
        ax6.set_xticklabels(feature_names)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/3d_visualization.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        return X_pca, pca
    
    def advanced_pca_analysis(self, save_plots=True):
        """Comprehensive PCA analysis to understand data separation"""
        print("\n" + "="*60)
        print("ADVANCED PCA ANALYSIS")
        print("="*60)
        
        # Prepare data
        X = self.data[self.features].values
        y = (self.data['Rating'] == 'GOOD').astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Original feature statistics (after standardization):")
        for i, feature in enumerate(self.features):
            print(f"  {feature}: mean=0.000, std=1.000")
        
        # PCA Analysis
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"\nPCA Results:")
        print(f"Total Components: {pca.n_components_}")
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        for i in range(len(pca.explained_variance_ratio_)):
            print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} variance ({pca.explained_variance_ratio_[i]:.1%})")
            print(f"     Cumulative: {cumulative_variance[i]:.3f} ({cumulative_variance[i]:.1%})")
        
        # Feature contributions to PCs
        print(f"\nFeature Contributions to Principal Components:")
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.features
        )
        print(components_df)
        
        # Create comprehensive PCA visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced PCA Analysis', fontsize=16)
        
        colors = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}
        
        # Plot 1: Explained Variance
        ax = axes[0, 0]
        pc_numbers = range(1, len(pca.explained_variance_ratio_) + 1)
        ax.bar(pc_numbers, pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
        ax.plot(pc_numbers, cumulative_variance, 'ro-', color='red', linewidth=2)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.legend(['Cumulative', 'Individual'])
        ax.grid(True, alpha=0.3)
        
        # Plot 2: PC1 vs PC2
        ax = axes[0, 1]
        good_mask = y == 1
        bad_mask = y == 0
        
        ax.scatter(X_pca[good_mask, 0], X_pca[good_mask, 1], 
                  c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax.scatter(X_pca[bad_mask, 0], X_pca[bad_mask, 1], 
                  c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PC1 vs PC2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: PC1 vs PC3
        ax = axes[0, 2]
        ax.scatter(X_pca[good_mask, 0], X_pca[good_mask, 2], 
                  c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax.scatter(X_pca[bad_mask, 0], X_pca[bad_mask, 2], 
                  c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax.set_title('PC1 vs PC3')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: PC2 vs PC3
        ax = axes[1, 0]
        ax.scatter(X_pca[good_mask, 1], X_pca[good_mask, 2], 
                  c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax.scatter(X_pca[bad_mask, 1], X_pca[bad_mask, 2], 
                  c=colors['BAD'], label='BAD', alpha=0.7, s=30)
        ax.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax.set_title('PC2 vs PC3')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Component loadings heatmap
        ax = axes[1, 1]
        im = ax.imshow(pca.components_, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(self.features)))
        ax.set_xticklabels(self.features)
        ax.set_yticks(range(pca.n_components_))
        ax.set_yticklabels([f'PC{i+1}' for i in range(pca.n_components_)])
        ax.set_title('PCA Component Loadings')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loading Strength')
        
        # Add text annotations
        for i in range(pca.n_components_):
            for j in range(len(self.features)):
                text = ax.text(j, i, f'{pca.components_[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        # Plot 6: Separation analysis
        ax = axes[1, 2]
        
        # Calculate separation metrics for each PC
        separations = []
        for i in range(pca.n_components_):
            good_pc = X_pca[good_mask, i]
            bad_pc = X_pca[bad_mask, i]
            
            # Fisher's discriminant ratio: (mean_diff)^2 / (var1 + var2)
            mean_diff = np.abs(np.mean(good_pc) - np.mean(bad_pc))
            var_sum = np.var(good_pc) + np.var(bad_pc)
            separation = (mean_diff ** 2) / var_sum if var_sum > 0 else 0
            separations.append(separation)
        
        pc_labels = [f'PC{i+1}' for i in range(pca.n_components_)]
        bars = ax.bar(pc_labels, separations, alpha=0.7, color='lightcoral')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Fisher Discriminant Ratio')
        ax.set_title('Class Separation by PC')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, sep in zip(bars, separations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{sep:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/advanced_pca_analysis.png', 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # Linear Discriminant Analysis for comparison
        print(f"\n" + "="*60)
        print("LINEAR DISCRIMINANT ANALYSIS (LDA)")
        print("="*60)
        
        try:
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X_scaled, y)
            
            print(f"LDA Components: {lda.n_components_}")
            print(f"LDA Explained Variance Ratio: {lda.explained_variance_ratio_}")
            
            # Plot LDA results
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.scatter(X_lda[good_mask, 0], np.zeros(np.sum(good_mask)), 
                       c=colors['GOOD'], label='GOOD', alpha=0.7, s=30)
            plt.scatter(X_lda[bad_mask, 0], np.zeros(np.sum(bad_mask)), 
                       c=colors['BAD'], label='BAD', alpha=0.7, s=30)
            plt.xlabel('LDA Component 1')
            plt.ylabel('')
            plt.title('LDA Projection (1D)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(X_lda[good_mask, 0], alpha=0.7, label='GOOD', color=colors['GOOD'], bins=30)
            plt.hist(X_lda[bad_mask, 0], alpha=0.7, label='BAD', color=colors['BAD'], bins=30)
            plt.xlabel('LDA Component 1')
            plt.ylabel('Frequency')
            plt.title('LDA Component Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/lda_analysis.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"LDA analysis failed: {e}")
            X_lda = None
        
        return X_pca, pca, X_lda, separations
    
    def kmeans_clustering(self, n_clusters=2):
        """Perform K-means clustering analysis"""
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING ANALYSIS")
        print("="*60)
        
        # Prepare features for clustering
        X = self.data[self.features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        self.data['Cluster'] = cluster_labels
        
        # Analyze cluster performance
        print(f"K-means with {n_clusters} clusters:")
        
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_data = self.data[self.data['Cluster'] == cluster_id]
            good_count = len(cluster_data[cluster_data['Rating'] == 'GOOD'])
            bad_count = len(cluster_data[cluster_data['Rating'] == 'BAD'])
            total_count = len(cluster_data)
            
            purity = max(good_count, bad_count) / total_count
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Total samples: {total_count}")
            print(f"  GOOD samples: {good_count} ({good_count/total_count:.1%})")
            print(f"  BAD samples: {bad_count} ({bad_count/total_count:.1%})")
            print(f"  Purity: {purity:.3f}")
            
            cluster_analysis[cluster_id] = {
                'total': total_count,
                'good': good_count,
                'bad': bad_count,
                'purity': purity
            }
        
        # Visualize clusters in 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: True labels
        colors_true = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}
        for rating in ['GOOD', 'BAD']:
            mask = self.data['Rating'] == rating
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors_true[rating], label=rating, alpha=0.7, s=30)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('True Labels (GOOD vs BAD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster labels
        colors_cluster = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            mask = cluster_labels == i
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors_cluster[i]], label=f'Cluster {i}', alpha=0.7, s=30)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax2.set_title('K-means Clusters')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/clustering_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cluster_analysis, X_pca, pca
        
    def generate_report(self, bounds_analysis):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        report = []
        report.append("# SPS Data Analysis Report\n")
        report.append("## Dataset Overview\n")
        report.append(f"- Total samples: {len(self.data)}\n")
        report.append(f"- GOOD samples: {len(self.data[self.data['Rating'] == 'GOOD'])}\n")
        report.append(f"- BAD samples: {len(self.data[self.data['Rating'] == 'BAD'])}\n")
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
            diff_pct = abs((good_mean - bad_mean) / good_mean * 100)
            report.append(f"  - {feature}: GOOD samples average {good_mean:.3f}, BAD samples average {bad_mean:.3f} (difference: {diff_pct:.1f}%)\n")
        
        report_text = ''.join(report)
        
        # Save report
        with open('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text
    
    def generate_enhanced_report(self, bounds_analysis, pca, separations):
        """Generate enhanced report with PCA insights"""
        print("\n" + "="*60)
        print("GENERATING ENHANCED ANALYSIS REPORT")
        print("="*60)
        
        good_count = sum(1 for row in self.data if row['Rating'] == 'GOOD')
        bad_count = len(self.data) - good_count
        
        report_lines = [
            "# Enhanced SPS Data Analysis Report",
            "",
            "## Dataset Overview",
            f"- **Total samples**: {len(self.data)}",
            f"- **GOOD samples**: {good_count} ({good_count/len(self.data)*100:.1f}%)",
            f"- **BAD samples**: {bad_count} ({bad_count/len(self.data)*100:.1f}%)",
            f"- **Features analyzed**: {', '.join(self.features)}",
            "",
            "Note: Set numbers can repeat - each entry represents unique measurement data.",
            "",
            "## 🔍 Data Separation Analysis",
            "",
            f"### PCA Results",
            f"The Principal Component Analysis reveals the following structure:",
            ""
        ]
        
        # PCA variance explanation
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        for i in range(len(pca.explained_variance_ratio_)):
            report_lines.append(f"- **PC{i+1}**: {pca.explained_variance_ratio_[i]:.1%} of variance (cumulative: {cumulative_variance[i]:.1%})")
        
        report_lines.extend([
            "",
            f"### Feature Contributions to Principal Components",
            ""
        ])
        
        # Feature loadings
        components_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.features
        )
        
        for feature in self.features:
            report_lines.append(f"**{feature}**:")
            for i in range(pca.n_components_):
                loading = components_df.loc[feature, f'PC{i+1}']
                report_lines.append(f"- PC{i+1}: {loading:+.3f}")
            report_lines.append("")
        
        # Separation analysis
        report_lines.extend([
            "### 🎯 Class Separation Analysis",
            "",
            "Fisher Discriminant Ratios (higher = better separation):",
            ""
        ])
        
        for i, sep in enumerate(separations):
            interpretation = "Excellent" if sep > 0.5 else "Good" if sep > 0.2 else "Moderate" if sep > 0.1 else "Poor"
            report_lines.append(f"- **PC{i+1}**: {sep:.3f} ({interpretation} separation)")
        
        best_pc = np.argmax(separations) + 1
        report_lines.extend([
            "",
            f"**Best separating component**: PC{best_pc} (Fisher ratio: {max(separations):.3f})",
            ""
        ])
        
        # Original feature bounds analysis
        report_lines.extend([
            "## 📊 Feature-Based Classification Bounds",
            ""
        ])
        
        # Sort features by accuracy
        sorted_features = sorted(bounds_analysis.items(), 
                               key=lambda x: x[1]['accuracy'], 
                               reverse=True)
        
        best_feature, best_analysis = sorted_features[0]
        
        report_lines.extend([
            f"### 🏆 Best Single Feature: {best_feature}",
            f"- **Classification Accuracy**: {best_analysis['accuracy']:.1%}",
            f"- **Threshold**: {best_analysis['threshold']:.3f}",
            f"- **Rule**: {'Higher values indicate GOOD' if best_analysis['rule'] == 'high_good' else 'Lower values indicate GOOD'}",
            ""
        ])
        
        # Recommendations based on analysis
        total_variance_2pc = cumulative_variance[1] if len(cumulative_variance) > 1 else cumulative_variance[0]
        
        report_lines.extend([
            "## 💡 Analysis-Based Recommendations",
            "",
            "### Data Quality Assessment Strategy:",
            ""
        ])
        
        if max(separations) > 0.3:
            report_lines.extend([
                f"1. **Good Linear Separability**: The data shows good separation in PC{best_pc}",
                f"2. **Recommended Approach**: Use PCA transformation + linear classifier",
                f"3. **Alternative**: Single feature classifier using {best_feature}"
            ])
        elif best_analysis['accuracy'] > 0.75:
            report_lines.extend([
                f"1. **Moderate Separability**: Best single feature ({best_feature}) achieves {best_analysis['accuracy']:.1%} accuracy",
                f"2. **Recommended Approach**: Use {best_feature} with threshold {best_analysis['threshold']:.3f}",
                f"3. **Enhancement**: Consider ensemble methods or feature combinations"
            ])
        else:
            report_lines.extend([
                "1. **Challenging Separation**: No single feature or PC provides clear separation",
                "2. **Recommended Approach**: Consider non-linear methods (SVM, Random Forest, Neural Networks)",
                "3. **Feature Engineering**: May need additional derived features or domain knowledge"
            ])
        
        if total_variance_2pc < 0.8:
            report_lines.extend([
                "",
                f"4. **Dimensionality Note**: First 2 PCs only explain {total_variance_2pc:.1%} of variance",
                "   - Consider keeping more components for classification",
                "   - 3D visualization may be necessary for full understanding"
            ])
        
        report_lines.extend([
            "",
            "### Quality Control Implementation:",
            "",
            f"- **Primary Check**: {best_feature} {'>' if best_analysis['rule'] == 'high_good' else '≤'} {best_analysis['threshold']:.3f}",
            f"- **Expected Accuracy**: ~{best_analysis['accuracy']:.1%}",
            "- **Secondary Check**: Consider multi-feature approach for higher accuracy",
            "- **Validation**: Test thresholds on new data before deployment",
            "",
            "### Visualization Insights:",
            "",
            "- **3D Feature Space**: Check 3d_visualization.png for spatial distribution",
            "- **PCA Space**: Review advanced_pca_analysis.png for transformed space",
            "- **LDA Projection**: Examine lda_analysis.png for supervised dimension reduction"
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save enhanced report
        with open('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-analysis/enhanced_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print("✅ Enhanced report saved to: enhanced_analysis_report.md")
        
        return report_text

def main():
    """Main analysis function"""
    print("🔬 SPS Data Quality Analysis - Comprehensive 3D and PCA Analysis")
    print("="*70)
    print("Note: Set numbers can repeat - each entry represents unique measurement data")
    print("="*70)
    
    # Initialize analyzer
    analyzer = SPSDataAnalyzer('/Users/leobailloeul/Documents/coding/desy/rnn4hist/data-generation/sps_rating_results.csv')
    
    # Load data
    data = analyzer.load_data()
    
    # Analyze bounds and thresholds
    bounds_analysis = analyzer.analyze_bounds()
    
    # Create standard visualizations
    analyzer.visualize_data(save_plots=True)
    
    # NEW: 3D visualizations
    X_pca_3d, pca_3d = analyzer.visualize_3d(save_plots=True)
    
    # NEW: Advanced PCA analysis
    X_pca_full, pca_full, X_lda, separations = analyzer.advanced_pca_analysis(save_plots=True)
    
    # Perform clustering analysis
    cluster_analysis, X_pca_cluster, pca_cluster = analyzer.kmeans_clustering(n_clusters=2)
    
    # Enhanced report generation with PCA insights
    analyzer.generate_enhanced_report(bounds_analysis, pca_full, separations)
    
    print(f"\n{'='*70}")
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("Generated files:")
    print("📊 Standard Analysis:")
    print("  - analysis_report.md")
    print("  - feature_analysis.png")
    print("  - boxplots.png") 
    print("  - correlation_matrix.png")
    print("  - clustering_analysis.png")
    print("\n🔍 Advanced Analysis:")
    print("  - 3d_visualization.png")
    print("  - advanced_pca_analysis.png")
    print("  - lda_analysis.png")
    print("  - enhanced_analysis_report.md")
    
    # Print key findings
    print(f"\n📋 KEY FINDINGS:")
    best_feature = max(bounds_analysis.keys(), key=lambda k: bounds_analysis[k]['accuracy'])
    best_acc = bounds_analysis[best_feature]['accuracy']
    print(f"  🏆 Best classifier: {best_feature} ({best_acc:.1%} accuracy)")
    
    if separations:
        best_pc = np.argmax(separations)
        print(f"  🔄 Best PC separation: PC{best_pc+1} (Fisher ratio: {separations[best_pc]:.3f})")
    
    total_variance_2pc = np.sum(pca_full.explained_variance_ratio_[:2])
    print(f"  📈 First 2 PCs explain {total_variance_2pc:.1%} of variance")
    
    if total_variance_2pc < 0.8:
        print("  ⚠️  Data separation may be challenging - consider non-linear methods")
    else:
        print("  ✅ Good dimensional structure for classification")

if __name__ == "__main__":
    main()