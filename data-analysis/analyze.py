#!/usr/bin/env python3
"""
SPS Data Analysis Script
========================
Comprehensive statistical analysis and machine learning for SPS histogram fitting quality data.

Usage:
    python analyze.py [--bounds] [--pca] [--clustering] [--lda] [--report]
    
Options:
    --bounds        Find optimal classification boundaries (default: True)
    --pca          Perform PCA analysis (default: True)
    --clustering   Run clustering analysis (default: True) 
    --lda          Linear Discriminant Analysis (default: True)
    --report       Generate comprehensive report (default: True)
    --help         Show this help message

Output:
    Analysis results saved to outputs/ directory
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Scientific computing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class SPSAnalyzer:
    def __init__(self, data_path='../data-generation/parameter-rating-app/sps_rating_results.csv', output_dir='outputs'):
        """Initialize the SPS data analyzer"""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.features = ['Sigma_Gain', 'Sigma_0', 'Mu_p']
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare SPS data"""
        print(f"📊 Loading SPS data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        print(f"   Total samples: {len(self.data)}")
        good_count = len(self.data[self.data['Rating'] == 'GOOD'])
        bad_count = len(self.data[self.data['Rating'] == 'BAD'])
        print(f"   GOOD: {good_count} ({good_count/len(self.data)*100:.1f}%)")
        print(f"   BAD: {bad_count} ({bad_count/len(self.data)*100:.1f}%)")
        
        return self.data
    
    def find_optimal_boundaries(self):
        """Find optimal classification boundaries for each feature"""
        print("\n🎯 Finding optimal classification boundaries...")
        
        good_data = self.data[self.data['Rating'] == 'GOOD']
        bad_data = self.data[self.data['Rating'] == 'BAD']
        
        boundaries = {}
        
        for feature in self.features:
            print(f"\n--- {feature} Analysis ---")
            
            good_values = good_data[feature].values
            bad_values = bad_data[feature].values
            all_values = self.data[feature].values
            
            # Statistics
            good_stats = {
                'mean': np.mean(good_values),
                'std': np.std(good_values),
                'min': np.min(good_values),
                'max': np.max(good_values),
                'median': np.median(good_values),
                'q25': np.percentile(good_values, 25),
                'q75': np.percentile(good_values, 75)
            }
            
            bad_stats = {
                'mean': np.mean(bad_values),
                'std': np.std(bad_values),
                'min': np.min(bad_values),
                'max': np.max(bad_values),
                'median': np.median(bad_values),
                'q25': np.percentile(bad_values, 25),
                'q75': np.percentile(bad_values, 75)
            }
            
            print(f"GOOD {feature}: μ={good_stats['mean']:.3f}±{good_stats['std']:.3f}, range=[{good_stats['min']:.3f}, {good_stats['max']:.3f}]")
            print(f"BAD {feature}:  μ={bad_stats['mean']:.3f}±{bad_stats['std']:.3f}, range=[{bad_stats['min']:.3f}, {bad_stats['max']:.3f}]")
            
            # Find optimal threshold
            threshold_candidates = np.linspace(all_values.min(), all_values.max(), 200)
            best_threshold = None
            best_accuracy = 0
            best_rule = None
            
            for thresh in threshold_candidates:
                # Rule 1: high values = GOOD
                correct_1 = np.sum(good_values > thresh) + np.sum(bad_values <= thresh)
                accuracy_1 = correct_1 / len(self.data)
                
                # Rule 2: low values = GOOD
                correct_2 = np.sum(good_values <= thresh) + np.sum(bad_values > thresh)
                accuracy_2 = correct_2 / len(self.data)
                
                if accuracy_1 > best_accuracy:
                    best_accuracy = accuracy_1
                    best_threshold = thresh
                    best_rule = "high_good"
                    
                if accuracy_2 > best_accuracy:
                    best_accuracy = accuracy_2
                    best_threshold = thresh
                    best_rule = "low_good"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(good_values)-1)*good_stats['std']**2 + (len(bad_values)-1)*bad_stats['std']**2) / (len(good_values)+len(bad_values)-2))
            cohens_d = abs(good_stats['mean'] - bad_stats['mean']) / pooled_std
            
            boundaries[feature] = {
                'threshold': best_threshold,
                'rule': best_rule,
                'accuracy': best_accuracy,
                'good_stats': good_stats,
                'bad_stats': bad_stats,
                'cohens_d': cohens_d
            }
            
            rule_desc = "Values > threshold = GOOD" if best_rule == "high_good" else "Values ≤ threshold = GOOD"
            print(f"Optimal threshold: {best_threshold:.3f} ({rule_desc})")
            print(f"Classification accuracy: {best_accuracy:.1%}")
            print(f"Effect size (Cohen's d): {cohens_d:.3f}")
        
        self.results['boundaries'] = boundaries
        return boundaries
    
    def perform_pca_analysis(self):
        """Comprehensive PCA analysis"""
        print("\n🔄 Performing PCA analysis...")
        
        # Prepare data
        X = self.data[self.features].values
        y = (self.data['Rating'] == 'GOOD').astype(int)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA Results:")
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        for i, (var_ratio, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance)):
            print(f"  PC{i+1}: {var_ratio:.3f} ({var_ratio:.1%}) - Cumulative: {cum_var:.1%}")
        
        # Feature loadings
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.features
        )
        
        print(f"\nFeature Loadings:")
        print(loadings_df)
        
        # Class separation analysis
        separations = []
        for i in range(pca.n_components_):
            good_pc = X_pca[y == 1, i]
            bad_pc = X_pca[y == 0, i]
            
            # Fisher discriminant ratio
            mean_diff = abs(np.mean(good_pc) - np.mean(bad_pc))
            var_sum = np.var(good_pc) + np.var(bad_pc)
            separation = (mean_diff ** 2) / var_sum if var_sum > 0 else 0
            separations.append(separation)
        
        print(f"\nClass Separation (Fisher ratios):")
        for i, sep in enumerate(separations):
            quality = "Excellent" if sep > 0.5 else "Good" if sep > 0.2 else "Moderate" if sep > 0.1 else "Poor"
            print(f"  PC{i+1}: {sep:.3f} ({quality})")
        
        # Create PCA visualization
        self._plot_pca_analysis(X_pca, y, pca, separations)
        
        pca_results = {
            'pca': pca,
            'X_pca': X_pca,
            'scaler': scaler,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'loadings': loadings_df,
            'separations': separations
        }
        
        self.results['pca'] = pca_results
        return pca_results
    
    def _plot_pca_analysis(self, X_pca, y, pca, separations):
        """Create PCA analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Principal Component Analysis', fontsize=16)
        
        colors = {1: '#2E8B57', 0: '#DC143C'}  # GOOD=1, BAD=0
        
        # Plot 1: Explained variance
        ax = axes[0, 0]
        pc_numbers = range(1, len(pca.explained_variance_ratio_) + 1)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        bars = ax.bar(pc_numbers, pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
        ax.plot(pc_numbers, cumulative_variance, 'ro-', color='red', linewidth=2)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance')
        ax.legend(['Cumulative', 'Individual'])
        ax.grid(True, alpha=0.3)
        
        # Plot 2-4: PC scatter plots
        pc_pairs = [(0, 1), (0, 2), (1, 2)]
        positions = [(0, 1), (0, 2), (1, 0)]
        
        for (pc1, pc2), (row, col) in zip(pc_pairs, positions):
            ax = axes[row, col]
            for label in [0, 1]:
                mask = y == label
                label_name = 'GOOD' if label == 1 else 'BAD'
                ax.scatter(X_pca[mask, pc1], X_pca[mask, pc2], 
                          c=colors[label], label=label_name, alpha=0.7, s=20)
            
            ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]:.1%})')
            ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]:.1%})')
            ax.set_title(f'PC{pc1+1} vs PC{pc2+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Component loadings heatmap
        ax = axes[1, 1]
        im = ax.imshow(pca.components_, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(self.features)))
        ax.set_xticklabels(self.features)
        ax.set_yticks(range(pca.n_components_))
        ax.set_yticklabels([f'PC{i+1}' for i in range(pca.n_components_)])
        ax.set_title('Component Loadings')
        
        # Add text annotations
        for i in range(pca.n_components_):
            for j in range(len(self.features)):
                ax.text(j, i, f'{pca.components_[i, j]:.2f}',
                       ha="center", va="center", fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Loading')
        
        # Plot 6: Separation quality
        ax = axes[1, 2]
        pc_labels = [f'PC{i+1}' for i in range(len(separations))]
        bars = ax.bar(pc_labels, separations, alpha=0.7, color='lightcoral')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Fisher Discriminant Ratio')
        ax.set_title('Class Separation Quality')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, sep in zip(bars, separations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{sep:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_clustering_analysis(self):
        """K-means clustering analysis"""
        print("\n🎲 Performing clustering analysis...")
        
        X = self.data[self.features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different numbers of clusters (extended to 14)
        cluster_range = range(2, 15)
        silhouette_scores = []
        inertias = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Find optimal number of clusters
        best_k = cluster_range[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {best_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        # Analyze purity for ALL cluster numbers (2-14)
        all_cluster_analysis = {}
        print(f"\n🔍 Detailed Purity Analysis for All Cluster Numbers:")
        print("=" * 60)
        
        for k in cluster_range:
            kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_k = kmeans_k.fit_predict(X_scaled)
            
            cluster_purities = []
            cluster_details = {}
            
            for cluster_id in range(k):
                mask = labels_k == cluster_id
                cluster_data = self.data[mask]
                good_count = len(cluster_data[cluster_data['Rating'] == 'GOOD'])
                bad_count = len(cluster_data[cluster_data['Rating'] == 'BAD'])
                total = len(cluster_data)
                
                purity = max(good_count, bad_count) / total if total > 0 else 0
                cluster_purities.append(purity)
                majority_class = 'GOOD' if good_count > bad_count else 'BAD'
                
                cluster_details[cluster_id] = {
                    'total': total,
                    'good': good_count,
                    'bad': bad_count,
                    'purity': purity,
                    'majority_class': majority_class
                }
            
            avg_purity = np.mean(cluster_purities)
            min_purity = np.min(cluster_purities)
            max_purity = np.max(cluster_purities)
            
            all_cluster_analysis[k] = {
                'avg_purity': avg_purity,
                'min_purity': min_purity, 
                'max_purity': max_purity,
                'cluster_details': cluster_details,
                'silhouette_score': silhouette_scores[k-2]  # Adjust index
            }
            
            print(f"\n📊 k={k} clusters:")
            print(f"   Silhouette Score: {silhouette_scores[k-2]:.3f}")
            print(f"   Purity - Avg: {avg_purity:.3f}, Min: {min_purity:.3f}, Max: {max_purity:.3f}")
            
            # Show individual cluster details for k=14 or if specifically requested
            if k == 14 or k == best_k:
                print(f"   Individual cluster breakdown:")
                for cluster_id in range(k):
                    details = cluster_details[cluster_id]
                    print(f"     Cluster {cluster_id}: {details['total']:3d} samples, "
                          f"{details['good']:3d} GOOD, {details['bad']:3d} BAD, "
                          f"purity: {details['purity']:.3f} ({details['majority_class']})")
        
        # Use optimal k for final results
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(X_scaled)
        self.data['Cluster'] = cluster_labels
        cluster_analysis = all_cluster_analysis[best_k]['cluster_details']
        
        # Create clustering visualization
        self._plot_clustering_analysis(X_scaled, cluster_labels, silhouette_scores, inertias, cluster_range)
        
        clustering_results = {
            'best_k': best_k,
            'kmeans': kmeans_final,
            'cluster_labels': cluster_labels,
            'silhouette_scores': silhouette_scores,
            'cluster_analysis': cluster_analysis,
            'all_cluster_analysis': all_cluster_analysis  # Complete analysis for k=2 to k=14
        }
        
        self.results['clustering'] = clustering_results
        return clustering_results
    
    def _plot_clustering_analysis(self, X_scaled, cluster_labels, silhouette_scores, inertias, cluster_range):
        """Create clustering analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Means Clustering Analysis', fontsize=16)
        
        # Plot 1: Silhouette scores
        ax = axes[0, 0]
        ax.plot(cluster_range, silhouette_scores, 'bo-')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score vs Number of Clusters')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal k
        best_k_idx = np.argmax(silhouette_scores)
        ax.scatter(cluster_range[best_k_idx], silhouette_scores[best_k_idx], 
                  color='red', s=100, zorder=5)
        ax.annotate(f'Optimal k={cluster_range[best_k_idx]}', 
                   xy=(cluster_range[best_k_idx], silhouette_scores[best_k_idx]),
                   xytext=(10, 10), textcoords='offset points')
        
        # Plot 2: Elbow curve
        ax = axes[0, 1]
        ax.plot(cluster_range, inertias, 'ro-')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Curve')
        ax.grid(True, alpha=0.3)
        
        # Plot 3-4: Cluster visualization (using PCA for visualization)
        if hasattr(self, 'results') and 'pca' in self.results:
            X_pca = self.results['pca']['X_pca']
            
            # Plot 3: Clusters in PCA space
            ax = axes[1, 0]
            colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(cluster_labels))))
            for i, cluster_id in enumerate(np.unique(cluster_labels)):
                mask = cluster_labels == cluster_id
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=20)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Clusters in PCA Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: True labels in PCA space
            ax = axes[1, 1]
            y = (self.data['Rating'] == 'GOOD').astype(int)
            colors_true = {1: '#2E8B57', 0: '#DC143C'}
            for label in [0, 1]:
                mask = y == label
                label_name = 'GOOD' if label == 1 else 'BAD'
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          c=colors_true[label], label=label_name, alpha=0.7, s=20)
            
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('True Labels in PCA Space')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_cluster_purity_and_classifiers(self):
        """Evaluate cluster purity and train supervised classifiers if purity is insufficient"""
        print("\n🔍 Evaluating cluster purity and training supervised classifiers...")
        
        if 'clustering' not in self.results:
            print("   ⚠️ Clustering analysis must be performed first")
            return None
            
        clustering = self.results['clustering']
        
        # Calculate overall cluster purity
        cluster_purities = []
        for cluster_id, analysis in clustering['cluster_analysis'].items():
            cluster_purities.append(analysis['purity'])
        
        avg_purity = np.mean(cluster_purities)
        min_purity = np.min(cluster_purities)
        purity_threshold = 0.85  # Threshold for "high enough" purity
        
        print(f"   Average cluster purity: {avg_purity:.3f}")
        print(f"   Minimum cluster purity: {min_purity:.3f}")
        print(f"   Purity threshold: {purity_threshold}")
        
        # Prepare data for supervised learning
        X = self.data[self.features].values
        y = (self.data['Rating'] == 'GOOD').astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        classifier_results = {}
        
        # Always train classifiers for comparison
        print("\n   Training supervised classifiers...")
        
        # SVM Classifier
        print("   🔹 Training SVM...")
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_proba = svm_model.predict_proba(X_test)[:, 1]
        
        svm_metrics = {
            'accuracy': accuracy_score(y_test, svm_pred),
            'precision': precision_score(y_test, svm_pred),
            'recall': recall_score(y_test, svm_pred),
            'f1': f1_score(y_test, svm_pred)
        }
        
        classifier_results['svm'] = {
            'model': svm_model,
            'predictions': svm_pred,
            'probabilities': svm_proba,
            'metrics': svm_metrics
        }
        
        print(f"      SVM Accuracy: {svm_metrics['accuracy']:.3f}")
        print(f"      SVM F1-Score: {svm_metrics['f1']:.3f}")
        
        # Random Forest Classifier
        print("   🌳 Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        
        classifier_results['random_forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_proba,
            'metrics': rf_metrics,
            'feature_importance': rf_model.feature_importances_
        }
        
        print(f"      RF Accuracy: {rf_metrics['accuracy']:.3f}")
        print(f"      RF F1-Score: {rf_metrics['f1']:.3f}")
        
        # Feature importance from Random Forest
        print(f"\n   Random Forest Feature Importance:")
        feature_importance = list(zip(self.features, rf_model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance:
            print(f"      {feature}: {importance:.3f}")
        
        # Assess if clustering purity is sufficient
        if min_purity < purity_threshold:
            print(f"\n   ⚠️ Cluster purity insufficient (min: {min_purity:.3f} < {purity_threshold})")
            print("   📊 Supervised classifiers recommended for reliable classification")
            recommendation = "supervised"
        else:
            print(f"\n   ✅ Cluster purity sufficient (min: {min_purity:.3f} ≥ {purity_threshold})")
            print("   🎯 Clustering approach can be used for classification")
            recommendation = "clustering"
        
        # Create classifier comparison visualization
        self._plot_classifier_comparison(X_test, y_test, classifier_results, clustering)
        
        classifier_eval_results = {
            'avg_cluster_purity': avg_purity,
            'min_cluster_purity': min_purity,
            'purity_threshold': purity_threshold,
            'recommendation': recommendation,
            'classifiers': classifier_results,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler
        }
        
        self.results['classifier_evaluation'] = classifier_eval_results
        return classifier_eval_results
    
    def _plot_classifier_comparison(self, X_test, y_test, classifier_results, clustering):
        """Create comprehensive classifier comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Supervised Classifiers vs Clustering Analysis', fontsize=16)
        
        colors = {1: '#2E8B57', 0: '#DC143C'}
        
        # Use PCA for visualization if available
        if 'pca' in self.results:
            pca = self.results['pca']['pca']
            X_test_pca = pca.transform(X_test)
        else:
            # Simple 2D projection using first two features
            X_test_pca = X_test[:, :2]
        
        # Plot 1: True labels
        ax = axes[0, 0]
        for label in [0, 1]:
            mask = y_test == label
            label_name = 'GOOD' if label == 1 else 'BAD'
            ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                      c=colors[label], label=label_name, alpha=0.7, s=30)
        ax.set_title('True Labels')
        ax.set_xlabel('PC1' if 'pca' in self.results else 'Feature 1')
        ax.set_ylabel('PC2' if 'pca' in self.results else 'Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: SVM predictions
        ax = axes[0, 1]
        svm_pred = classifier_results['svm']['predictions']
        for label in [0, 1]:
            mask = svm_pred == label
            label_name = 'GOOD' if label == 1 else 'BAD'
            ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                      c=colors[label], label=f'SVM: {label_name}', alpha=0.7, s=30)
        ax.set_title(f'SVM Predictions (Acc: {classifier_results["svm"]["metrics"]["accuracy"]:.3f})')
        ax.set_xlabel('PC1' if 'pca' in self.results else 'Feature 1')
        ax.set_ylabel('PC2' if 'pca' in self.results else 'Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Random Forest predictions
        ax = axes[0, 2]
        rf_pred = classifier_results['random_forest']['predictions']
        for label in [0, 1]:
            mask = rf_pred == label
            label_name = 'GOOD' if label == 1 else 'BAD'
            ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                      c=colors[label], label=f'RF: {label_name}', alpha=0.7, s=30)
        ax.set_title(f'Random Forest Predictions (Acc: {classifier_results["random_forest"]["metrics"]["accuracy"]:.3f})')
        ax.set_xlabel('PC1' if 'pca' in self.results else 'Feature 1')
        ax.set_ylabel('PC2' if 'pca' in self.results else 'Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison
        ax = axes[1, 0]
        methods = ['SVM', 'Random Forest']
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(methods))
        width = 0.2
        
        for i, metric in enumerate(metrics_names):
            values = [
                classifier_results['svm']['metrics'][metric],
                classifier_results['random_forest']['metrics'][metric]
            ]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Classifier')
        ax.set_ylabel('Score')
        ax.set_title('Classifier Performance Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 5: Feature importance (Random Forest)
        ax = axes[1, 1]
        importance = classifier_results['random_forest']['feature_importance']
        bars = ax.bar(self.features, importance, alpha=0.7, color='lightblue')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Random Forest Feature Importance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{imp:.3f}', ha='center', va='bottom')
        
        # Plot 6: Cluster purity analysis
        ax = axes[1, 2]
        cluster_ids = list(clustering['cluster_analysis'].keys())
        purities = [clustering['cluster_analysis'][cid]['purity'] for cid in cluster_ids]
        
        bars = ax.bar(cluster_ids, purities, alpha=0.7, color='lightcoral')
        ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.7, label='Purity Threshold')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Purity')
        ax.set_title('Cluster Purity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, purity in zip(bars, purities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{purity:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classifier_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_lda_analysis(self):
        """Linear Discriminant Analysis"""
        print("\n📈 Performing Linear Discriminant Analysis...")
        
        X = self.data[self.features].values
        y = (self.data['Rating'] == 'GOOD').astype(int)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            # LDA
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X_scaled, y)
            
            print(f"LDA Results:")
            print(f"  Number of components: {lda.n_components_}")
            if hasattr(lda, 'explained_variance_ratio_'):
                print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
            
            # LDA coefficients (discriminant function)
            print(f"\nLinear Discriminant Function:")
            for i, feature in enumerate(self.features):
                coef = lda.scalings_[i, 0] if lda.scalings_.ndim > 1 else lda.scalings_[i]
                print(f"  {feature}: {coef:.3f}")
            
            # Classification performance
            lda_pred = lda.predict(X_scaled)
            accuracy = np.mean(lda_pred == y)
            print(f"\nLDA Classification Accuracy: {accuracy:.3f}")
            
            # Create LDA visualization
            self._plot_lda_analysis(X_lda, y, lda)
            
            lda_results = {
                'lda': lda,
                'X_lda': X_lda,
                'accuracy': accuracy,
                'predictions': lda_pred
            }
            
            self.results['lda'] = lda_results
            return lda_results
            
        except Exception as e:
            print(f"LDA analysis failed: {e}")
            return None
    
    def _plot_lda_analysis(self, X_lda, y, lda):
        """Create LDA analysis plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Linear Discriminant Analysis', fontsize=16)
        
        colors = {1: '#2E8B57', 0: '#DC143C'}
        
        # Plot 1: LDA projection (1D)
        ax = axes[0]
        for label in [0, 1]:
            mask = y == label
            label_name = 'GOOD' if label == 1 else 'BAD'
            y_offset = 0.1 * (label - 0.5)  # Slight vertical separation
            ax.scatter(X_lda[mask, 0], np.full(np.sum(mask), y_offset),
                      c=colors[label], label=label_name, alpha=0.7, s=30)
        
        ax.set_xlabel('LDA Component 1')
        ax.set_ylabel('')
        ax.set_title('LDA Projection (1D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 0.2)
        
        # Plot 2: LDA histograms
        ax = axes[1]
        for label in [0, 1]:
            mask = y == label
            label_name = 'GOOD' if label == 1 else 'BAD'
            ax.hist(X_lda[mask, 0], alpha=0.7, label=label_name, 
                   color=colors[label], bins=30)
        
        ax.set_xlabel('LDA Component 1')
        ax.set_ylabel('Frequency')
        ax.set_title('LDA Component Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'lda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n📝 Generating comprehensive analysis report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "# SPS Data Analysis Report",
            f"*Generated: {timestamp}*",
            "",
            "## Executive Summary",
            "",
            f"This report presents a comprehensive analysis of {len(self.data)} SPS histogram fitting quality samples,",
            f"with {len(self.data[self.data['Rating'] == 'GOOD'])} GOOD samples ({len(self.data[self.data['Rating'] == 'GOOD'])/len(self.data)*100:.1f}%) and {len(self.data[self.data['Rating'] == 'BAD'])} BAD samples ({len(self.data[self.data['Rating'] == 'BAD'])/len(self.data)*100:.1f}%).",
            "",
            "## 🎯 Classification Boundaries",
            ""
        ]
        
        # Boundary analysis
        if 'boundaries' in self.results:
            boundaries = self.results['boundaries']
            
            # Find best feature
            best_feature = max(boundaries.keys(), key=lambda k: boundaries[k]['accuracy'])
            best_acc = boundaries[best_feature]['accuracy']
            
            report_lines.extend([
                f"### Best Single Feature Classifier: **{best_feature}**",
                f"- **Accuracy**: {best_acc:.1%}",
                f"- **Threshold**: {boundaries[best_feature]['threshold']:.3f}",
                f"- **Rule**: {'Higher values indicate GOOD' if boundaries[best_feature]['rule'] == 'high_good' else 'Lower values indicate GOOD'}",
                f"- **Effect Size (Cohen's d)**: {boundaries[best_feature]['cohens_d']:.3f}",
                "",
                "### All Features Performance:",
                ""
            ])
            
            # Sort by accuracy
            sorted_features = sorted(boundaries.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for i, (feature, analysis) in enumerate(sorted_features, 1):
                effect_size_desc = "Large" if analysis['cohens_d'] > 0.8 else "Medium" if analysis['cohens_d'] > 0.5 else "Small"
                rule_desc = "Higher = GOOD" if analysis['rule'] == 'high_good' else "Lower = GOOD"
                
                report_lines.extend([
                    f"**{i}. {feature}**",
                    f"- Accuracy: {analysis['accuracy']:.1%}",
                    f"- Threshold: {analysis['threshold']:.3f} ({rule_desc})",
                    f"- Effect size: {analysis['cohens_d']:.3f} ({effect_size_desc})",
                    ""
                ])
        
        # PCA analysis
        if 'pca' in self.results:
            pca_results = self.results['pca']
            
            report_lines.extend([
                "## 🔄 Principal Component Analysis",
                "",
                "### Variance Explanation:",
                ""
            ])
            
            cumulative_var = np.cumsum(pca_results['explained_variance_ratio'])
            for i, (var_ratio, cum_var) in enumerate(zip(pca_results['explained_variance_ratio'], cumulative_var)):
                report_lines.append(f"- **PC{i+1}**: {var_ratio:.1%} (cumulative: {cum_var:.1%})")
            
            report_lines.extend([
                "",
                "### Class Separation Quality:",
                ""
            ])
            
            for i, sep in enumerate(pca_results['separations']):
                quality = "Excellent" if sep > 0.5 else "Good" if sep > 0.2 else "Moderate" if sep > 0.1 else "Poor"
                report_lines.append(f"- **PC{i+1}**: Fisher ratio = {sep:.3f} ({quality} separation)")
            
            best_pc = np.argmax(pca_results['separations']) + 1
            best_sep = max(pca_results['separations'])
            
            report_lines.extend([
                "",
                f"**Best separating component**: PC{best_pc} (Fisher ratio: {best_sep:.3f})",
                ""
            ])
        
        # Clustering analysis
        if 'clustering' in self.results:
            clustering = self.results['clustering']
            
            report_lines.extend([
                "## 🎲 Clustering Analysis",
                "",
                f"### Optimal Clustering: {clustering['best_k']} clusters",
                f"- **Silhouette Score**: {max(clustering['silhouette_scores']):.3f}",
                "",
                "### Cluster Purity Analysis:",
                ""
            ])
            
            for cluster_id, analysis in clustering['cluster_analysis'].items():
                report_lines.extend([
                    f"**Cluster {cluster_id}**:",
                    f"- Size: {analysis['total']} samples",
                    f"- Composition: {analysis['good']} GOOD, {analysis['bad']} BAD",
                    f"- Purity: {analysis['purity']:.1%} (majority: {analysis['majority_class']})",
                    ""
                ])
        
        # LDA analysis  
        if 'lda' in self.results:
            lda_results = self.results['lda']
            
            report_lines.extend([
                "## 📈 Linear Discriminant Analysis",
                "",
                f"- **Classification Accuracy**: {lda_results['accuracy']:.1%}",
                "- **Interpretation**: LDA finds the optimal linear combination of features for class separation",
                ""
            ])
        
        # Classifier evaluation analysis
        if 'classifier_evaluation' in self.results:
            classifier_eval = self.results['classifier_evaluation']
            
            report_lines.extend([
                "## 🔍 Supervised Classification Analysis",
                "",
                f"### Cluster Purity Assessment:",
                f"- **Average cluster purity**: {classifier_eval['avg_cluster_purity']:.3f}",
                f"- **Minimum cluster purity**: {classifier_eval['min_cluster_purity']:.3f}",
                f"- **Purity threshold**: {classifier_eval['purity_threshold']}",
                f"- **Recommendation**: {classifier_eval['recommendation'].title()} learning approach",
                "",
                "### Supervised Classifier Performance:",
                ""
            ])
            
            # SVM results
            svm_metrics = classifier_eval['classifiers']['svm']['metrics']
            report_lines.extend([
                "**Support Vector Machine (SVM)**:",
                f"- Accuracy: {svm_metrics['accuracy']:.3f}",
                f"- Precision: {svm_metrics['precision']:.3f}",
                f"- Recall: {svm_metrics['recall']:.3f}",
                f"- F1-Score: {svm_metrics['f1']:.3f}",
                ""
            ])
            
            # Random Forest results
            rf_metrics = classifier_eval['classifiers']['random_forest']['metrics']
            rf_importance = classifier_eval['classifiers']['random_forest']['feature_importance']
            
            report_lines.extend([
                "**Random Forest**:",
                f"- Accuracy: {rf_metrics['accuracy']:.3f}",
                f"- Precision: {rf_metrics['precision']:.3f}",
                f"- Recall: {rf_metrics['recall']:.3f}",
                f"- F1-Score: {rf_metrics['f1']:.3f}",
                "",
                "**Feature Importance (Random Forest)**:"
            ])
            
            # Add feature importance
            feature_importance_pairs = list(zip(self.features, rf_importance))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            for feature, importance in feature_importance_pairs:
                report_lines.append(f"- {feature}: {importance:.3f}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## 💡 Recommendations",
            "",
            "### For Quality Control Implementation:",
            ""
        ])
        
        if 'boundaries' in self.results:
            best_feature = max(self.results['boundaries'].keys(), 
                             key=lambda k: self.results['boundaries'][k]['accuracy'])
            best_analysis = self.results['boundaries'][best_feature]
            
            if best_analysis['accuracy'] > 0.8:
                strategy = "Single-feature approach recommended"
                report_lines.extend([
                    f"1. **Primary Strategy**: Use {best_feature} as main quality indicator",
                    f"2. **Decision Rule**: Data is GOOD if {best_feature} {'>' if best_analysis['rule'] == 'high_good' else '≤'} {best_analysis['threshold']:.3f}",
                    f"3. **Expected Performance**: ~{best_analysis['accuracy']:.1%} classification accuracy"
                ])
            elif 'pca' in self.results and max(self.results['pca']['separations']) > 0.3:
                report_lines.extend([
                    "1. **Primary Strategy**: Use PCA transformation for classification",
                    f"2. **Best Component**: Focus on PC{np.argmax(self.results['pca']['separations'])+1}",
                    f"3. **Multi-feature Approach**: Combine features using PCA weights"
                ])
            else:
                report_lines.extend([
                    "1. **Complex Classification**: No single feature provides clear separation",
                    "2. **Recommended Approach**: Use ensemble methods (Random Forest, SVM)",
                    "3. **Feature Engineering**: Consider additional derived features"
                ])
        
        report_lines.extend([
            "",
            "### Next Steps:",
            "",
            "- Validate findings on independent test set",
            "- Consider ensemble methods for improved performance", 
            "- Implement real-time quality monitoring based on findings",
            "- Regular model retraining as new data becomes available",
            "",
            "---",
            f"*Analysis completed with {len(self.features)} features: {', '.join(self.features)}*"
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.output_dir / 'comprehensive_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print(f"   ✅ Report saved to {self.output_dir / 'comprehensive_analysis_report.md'}")
        
        return report_text

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='SPS Data Analysis Tool')
    parser.add_argument('--bounds', action='store_true', default=True, help='Find classification boundaries')
    parser.add_argument('--pca', action='store_true', default=True, help='Perform PCA analysis')
    parser.add_argument('--clustering', action='store_true', default=True, help='Clustering analysis (up to 14 clusters)')
    parser.add_argument('--classifiers', action='store_true', default=True, help='Evaluate SVM and Random Forest classifiers')
    parser.add_argument('--lda', action='store_true', default=True, help='Linear discriminant analysis')
    parser.add_argument('--report', action='store_true', default=True, help='Generate comprehensive report')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    print("🔬 SPS Data Analysis Tool")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = SPSAnalyzer(output_dir=args.output_dir)
    
    # Load data
    analyzer.load_data()
    
    # Run analyses
    if args.bounds:
        analyzer.find_optimal_boundaries()
    
    if args.pca:
        analyzer.perform_pca_analysis()
        
    if args.clustering:
        analyzer.perform_clustering_analysis()
        
        # Evaluate cluster purity and train supervised classifiers
        if args.classifiers:
            analyzer.evaluate_cluster_purity_and_classifiers()
        
    if args.lda:
        analyzer.perform_lda_analysis()
        
    if args.report:
        analyzer.generate_comprehensive_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Check the '{args.output_dir}/' directory for all results")
    
    # Summary of key findings
    if 'boundaries' in analyzer.results:
        best_feature = max(analyzer.results['boundaries'].keys(),
                          key=lambda k: analyzer.results['boundaries'][k]['accuracy'])
        best_acc = analyzer.results['boundaries'][best_feature]['accuracy']
        print(f"\n🎯 Key Finding: {best_feature} is the best classifier ({best_acc:.1%} accuracy)")
    
    if 'pca' in analyzer.results:
        total_var_2pc = sum(analyzer.results['pca']['explained_variance_ratio'][:2])
        print(f"🔄 PCA Insight: First 2 PCs explain {total_var_2pc:.1%} of variance")
    
    if 'clustering' in analyzer.results:
        best_k = analyzer.results['clustering']['best_k']
        best_sil = max(analyzer.results['clustering']['silhouette_scores'])
        print(f"🎲 Clustering: Optimal k={best_k} (silhouette={best_sil:.3f})")

if __name__ == "__main__":
    main()