#!/usr/bin/env python3
"""
SPS Data Visualization Script
============================
Creates comprehensive visualizations for SPS histogram fitting quality data.

Usage:
    python visualize.py [--interactive] [--static] [--3d]
    
Options:
    --interactive    Generate interactive plotly visualizations (default: True)
    --static        Generate static matplotlib visualizations (default: True)  
    --3d            Generate 3D visualizations (default: True)
    --help          Show this help message

Output:
    All plots saved to outputs/ directory
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Plotly for interactive plots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class SPSVisualizer:
    def __init__(self, data_path='../data-generation/sps_rating_results.csv', output_dir='outputs'):
        """Initialize the SPS data visualizer"""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme
        self.colors = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}  # Sea green, Crimson
        self.data = None
        
    def load_data(self):
        """Load and validate SPS data"""
        print(f"📊 Loading SPS data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        print(f"   Total samples: {len(self.data)}")
        good_count = len(self.data[self.data['Rating'] == 'GOOD'])
        bad_count = len(self.data[self.data['Rating'] == 'BAD'])
        print(f"   GOOD: {good_count} ({good_count/len(self.data)*100:.1f}%)")
        print(f"   BAD: {bad_count} ({bad_count/len(self.data)*100:.1f}%)")
        
        # Data ranges
        features = ['Sigma_Gain', 'Sigma_0', 'Mu_p']
        print(f"   Feature ranges:")
        for feature in features:
            print(f"     {feature}: {self.data[feature].min():.2f} to {self.data[feature].max():.2f}")
        
        return self.data
    
    def create_static_visualizations(self):
        """Generate static matplotlib visualizations"""
        print("\n🎨 Creating static visualizations...")
        
        # 1. Feature distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        features = ['Sigma_Gain', 'Sigma_0', 'Mu_p']
        
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            
            good_data = self.data[self.data['Rating'] == 'GOOD'][feature]
            bad_data = self.data[self.data['Rating'] == 'BAD'][feature]
            
            ax.hist(good_data, alpha=0.7, label='GOOD', color=self.colors['GOOD'], bins=30)
            ax.hist(bad_data, alpha=0.7, label='BAD', color=self.colors['BAD'], bins=30)
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feature} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(features) == 3:
            fig.delaxes(axes[1, 1])
            
        plt.suptitle('SPS Feature Distributions - GOOD vs BAD', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Scatter plot matrix
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        feature_pairs = [
            ('Sigma_Gain', 'Sigma_0'),
            ('Sigma_Gain', 'Mu_p'),
            ('Sigma_0', 'Mu_p')
        ]
        
        for i, (feat1, feat2) in enumerate(feature_pairs):
            ax = axes[i]
            
            good_data = self.data[self.data['Rating'] == 'GOOD']
            bad_data = self.data[self.data['Rating'] == 'BAD']
            
            ax.scatter(good_data[feat1], good_data[feat2], 
                      alpha=0.6, label='GOOD', color=self.colors['GOOD'], s=20)
            ax.scatter(bad_data[feat1], bad_data[feat2],
                      alpha=0.6, label='BAD', color=self.colors['BAD'], s=20)
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{feat1} vs {feat2}')
        
        plt.suptitle('Feature Scatter Plots - GOOD vs BAD', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scatter_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, feature in enumerate(features):
            self.data.boxplot(column=feature, by='Rating', ax=axes[i])
            axes[i].set_title(f'{feature} by Rating')
            axes[i].set_xlabel('Rating')
            axes[i].set_ylabel(feature)
        
        plt.suptitle('Feature Box Plots by Rating', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Static plots saved to {self.output_dir}/")
    
    def create_3d_visualizations(self):
        """Generate 3D visualizations"""
        print("\n🔮 Creating 3D visualizations...")
        
        good_data = self.data[self.data['Rating'] == 'GOOD']
        bad_data = self.data[self.data['Rating'] == 'BAD']
        
        # Single 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(good_data['Sigma_Gain'], good_data['Sigma_0'], good_data['Mu_p'],
                   c=self.colors['GOOD'], label='GOOD', alpha=0.7, s=30)
        ax.scatter(bad_data['Sigma_Gain'], bad_data['Sigma_0'], bad_data['Mu_p'],
                   c=self.colors['BAD'], label='BAD', alpha=0.7, s=30)
        
        ax.set_xlabel('Sigma_Gain')
        ax.set_ylabel('Sigma_0')
        ax.set_zlabel('Mu_p')
        ax.set_title('SPS Data: 3D Feature Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Multiple angle views
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
        angles = [(20, 45), (20, 135), (60, 45), (0, 0)]
        angle_names = ['Default', 'Side', 'Top-down', 'Front']
        
        for ax, (elev, azim), name in zip(axes.flat, angles, angle_names):
            ax.scatter(good_data['Sigma_Gain'], good_data['Sigma_0'], good_data['Mu_p'],
                       c=self.colors['GOOD'], label='GOOD', alpha=0.7, s=20)
            ax.scatter(bad_data['Sigma_Gain'], bad_data['Sigma_0'], bad_data['Mu_p'],
                       c=self.colors['BAD'], label='BAD', alpha=0.7, s=20)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('Sigma_Gain')
            ax.set_ylabel('Sigma_0')
            ax.set_zlabel('Mu_p')
            ax.set_title(f'{name} View')
            
            if name == 'Default':
                ax.legend()
        
        plt.suptitle('SPS Data: Multiple 3D Views', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / '3d_multiple_views.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 3D plots saved to {self.output_dir}/")
    
    def create_interactive_visualizations(self):
        """Generate interactive plotly visualizations"""
        print("\n⚡ Creating interactive visualizations...")
        
        # 1. Basic interactive 3D scatter
        fig = go.Figure()
        
        for rating in ['GOOD', 'BAD']:
            subset = self.data[self.data['Rating'] == rating]
            
            hover_text = [
                f"Set: {row['Set']}<br>" +
                f"Sigma_Gain: {row['Sigma_Gain']:.3f}<br>" +
                f"Sigma_0: {row['Sigma_0']:.3f}<br>" +
                f"Mu_p: {row['Mu_p']:.3f}<br>" +
                f"Rating: {row['Rating']}"
                for _, row in subset.iterrows()
            ]
            
            fig.add_trace(go.Scatter3d(
                x=subset['Sigma_Gain'],
                y=subset['Sigma_0'],
                z=subset['Mu_p'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.colors[rating],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=f'{rating} ({len(subset)} samples)',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='🔬 Interactive SPS Data: 3D Feature Space<br><sub>Rotate • Zoom • Hover for details</sub>',
            scene=dict(
                xaxis_title='Sigma_Gain',
                yaxis_title='Sigma_0',
                zaxis_title='Mu_p',
                bgcolor="white"
            ),
            width=1000,
            height=700
        )
        
        fig.write_html(self.output_dir / 'interactive_3d.html')
        
        # 2. Enhanced plot with centroids
        good_data = self.data[self.data['Rating'] == 'GOOD']
        bad_data = self.data[self.data['Rating'] == 'BAD']
        
        good_centroid = [good_data['Sigma_Gain'].mean(), good_data['Sigma_0'].mean(), good_data['Mu_p'].mean()]
        bad_centroid = [bad_data['Sigma_Gain'].mean(), bad_data['Sigma_0'].mean(), bad_data['Mu_p'].mean()]
        
        fig_enhanced = go.Figure()
        
        # Add data points
        for rating in ['GOOD', 'BAD']:
            subset = self.data[self.data['Rating'] == rating]
            
            hover_text = [
                f"Set: {row['Set']}<br>" +
                f"Sigma_Gain: {row['Sigma_Gain']:.3f}<br>" +
                f"Sigma_0: {row['Sigma_0']:.3f}<br>" +
                f"Mu_p: {row['Mu_p']:.3f}<br>" +
                f"Rating: {row['Rating']}"
                for _, row in subset.iterrows()
            ]
            
            fig_enhanced.add_trace(go.Scatter3d(
                x=subset['Sigma_Gain'],
                y=subset['Sigma_0'],
                z=subset['Mu_p'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.colors[rating],
                    opacity=0.6
                ),
                name=f'{rating} Data',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add centroids
        fig_enhanced.add_trace(go.Scatter3d(
            x=[good_centroid[0]],
            y=[good_centroid[1]], 
            z=[good_centroid[2]],
            mode='markers',
            marker=dict(size=15, color=self.colors['GOOD'], symbol='diamond', line=dict(width=3, color='black')),
            name='GOOD Centroid'
        ))
        
        fig_enhanced.add_trace(go.Scatter3d(
            x=[bad_centroid[0]],
            y=[bad_centroid[1]],
            z=[bad_centroid[2]],
            mode='markers',
            marker=dict(size=15, color=self.colors['BAD'], symbol='diamond', line=dict(width=3, color='black')),
            name='BAD Centroid'
        ))
        
        # Connection line
        fig_enhanced.add_trace(go.Scatter3d(
            x=[good_centroid[0], bad_centroid[0]],
            y=[good_centroid[1], bad_centroid[1]],
            z=[good_centroid[2], bad_centroid[2]],
            mode='lines',
            line=dict(color='purple', width=4, dash='dash'),
            name='Centroid Distance'
        ))
        
        fig_enhanced.update_layout(
            title='🎯 Enhanced Interactive SPS Analysis<br><sub>With centroids and separation metrics</sub>',
            scene=dict(
                xaxis_title='Sigma_Gain',
                yaxis_title='Sigma_0', 
                zaxis_title='Mu_p',
                bgcolor="white"
            ),
            width=1100,
            height=800
        )
        
        fig_enhanced.write_html(self.output_dir / 'interactive_enhanced.html')
        
        # 3. Dashboard with multiple views
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D View', 'Sigma_Gain vs Sigma_0', 'Sigma_Gain vs Mu_p', 'Sigma_0 vs Mu_p'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 3D plot
        for rating in ['GOOD', 'BAD']:
            subset = self.data[self.data['Rating'] == rating]
            fig_dashboard.add_trace(
                go.Scatter3d(
                    x=subset['Sigma_Gain'], y=subset['Sigma_0'], z=subset['Mu_p'],
                    mode='markers',
                    marker=dict(size=3, color=self.colors[rating], opacity=0.7),
                    name=rating,
                    showlegend=(rating == 'GOOD')
                ),
                row=1, col=1
            )
        
        # 2D projections
        projections = [
            ('Sigma_Gain', 'Sigma_0', 1, 2),
            ('Sigma_Gain', 'Mu_p', 2, 1), 
            ('Sigma_0', 'Mu_p', 2, 2)
        ]
        
        for x_var, y_var, row, col in projections:
            for rating in ['GOOD', 'BAD']:
                subset = self.data[self.data['Rating'] == rating]
                fig_dashboard.add_trace(
                    go.Scatter(
                        x=subset[x_var], y=subset[y_var],
                        mode='markers',
                        marker=dict(size=4, color=self.colors[rating], opacity=0.7),
                        name=rating,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig_dashboard.update_layout(
            title='📊 Interactive Dashboard - Multiple Views',
            height=800, width=1200
        )
        
        fig_dashboard.write_html(self.output_dir / 'interactive_dashboard.html')
        
        print(f"   ✅ Interactive plots saved to {self.output_dir}/")
        
        # Print centroid info
        centroid_distance = np.linalg.norm(np.array(good_centroid) - np.array(bad_centroid))
        print(f"   📍 GOOD Centroid: ({good_centroid[0]:.3f}, {good_centroid[1]:.3f}, {good_centroid[2]:.3f})")
        print(f"   📍 BAD Centroid:  ({bad_centroid[0]:.3f}, {bad_centroid[1]:.3f}, {bad_centroid[2]:.3f})")
        print(f"   📏 Centroid Distance: {centroid_distance:.3f}")

def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(description='SPS Data Visualization Tool')
    parser.add_argument('--interactive', action='store_true', default=True, help='Generate interactive plots')
    parser.add_argument('--static', action='store_true', default=True, help='Generate static plots')
    parser.add_argument('--3d', action='store_true', default=True, help='Generate 3D plots')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("🔬 SPS Data Visualization Tool")
    print("=" * 40)
    
    # Initialize visualizer
    viz = SPSVisualizer(output_dir=args.output_dir)
    
    # Load data
    viz.load_data()
    
    # Generate visualizations
    if args.static:
        viz.create_static_visualizations()
    
    if getattr(args, '3d'):
        viz.create_3d_visualizations() 
        
    if args.interactive:
        viz.create_interactive_visualizations()
    
    print(f"\n✅ All visualizations complete!")
    print(f"📁 Check the '{args.output_dir}/' directory for all generated plots")
    
    # List generated files
    output_files = list(Path(args.output_dir).glob('*'))
    print(f"\n📄 Generated files:")
    for file in sorted(output_files):
        if file.is_file():
            print(f"   • {file.name}")

if __name__ == "__main__":
    main()