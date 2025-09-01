#!/usr/bin/env python3
"""
Simple 3D Visualization for SPS Data
Just shows the data in 3D space - Sigma_Gain vs Sigma_0 vs Mu_p
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_simple_3d_plot():
    """Create a simple 3D scatter plot of the SPS data"""
    
    print("Loading SPS data...")
    # Load the data
    data = pd.read_csv('../data-generation/sps_rating_results.csv')
    
    print(f"Loaded {len(data)} samples")
    print(f"GOOD: {len(data[data['Rating'] == 'GOOD'])}")
    print(f"BAD: {len(data[data['Rating'] == 'BAD'])}")
    
    # Separate GOOD and BAD data
    good_data = data[data['Rating'] == 'GOOD']
    bad_data = data[data['Rating'] == 'BAD']
    
    print("Creating 3D visualization...")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors
    good_color = '#2E8B57'  # Sea green
    bad_color = '#DC143C'   # Crimson
    
    # Plot GOOD data points
    ax.scatter(good_data['Sigma_Gain'], 
               good_data['Sigma_0'], 
               good_data['Mu_p'],
               c=good_color, 
               label='GOOD', 
               alpha=0.7, 
               s=30)
    
    # Plot BAD data points  
    ax.scatter(bad_data['Sigma_Gain'],
               bad_data['Sigma_0'],
               bad_data['Mu_p'],
               c=bad_color,
               label='BAD',
               alpha=0.7,
               s=30)
    
    # Set labels and title
    ax.set_xlabel('Sigma_Gain')
    ax.set_ylabel('Sigma_0') 
    ax.set_zlabel('Mu_p')
    ax.set_title('SPS Data: 3D Feature Space (GOOD vs BAD)', pad=20)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('simple_3d_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ 3D plot saved as: simple_3d_visualization.png")
    
    # Also create views from different angles
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': '3d'})
    
    angles = [
        (20, 45),   # Default view
        (20, 135),  # Side view
        (60, 45),   # Top-down view  
        (0, 0)      # Front view
    ]
    
    angle_names = ['Default', 'Side', 'Top-down', 'Front']
    axes = [ax1, ax2, ax3, ax4]
    
    for i, (ax, (elev, azim), name) in enumerate(zip(axes, angles, angle_names)):
        # Plot data
        ax.scatter(good_data['Sigma_Gain'], good_data['Sigma_0'], good_data['Mu_p'],
                   c=good_color, label='GOOD', alpha=0.7, s=20)
        ax.scatter(bad_data['Sigma_Gain'], bad_data['Sigma_0'], bad_data['Mu_p'],
                   c=bad_color, label='BAD', alpha=0.7, s=20)
        
        # Set view angle
        ax.view_init(elev=elev, azim=azim)
        
        # Labels and title
        ax.set_xlabel('Sigma_Gain')
        ax.set_ylabel('Sigma_0')
        ax.set_zlabel('Mu_p')
        ax.set_title(f'{name} View')
        
        if i == 0:  # Add legend only to first plot
            ax.legend()
    
    plt.suptitle('SPS Data: Multiple 3D Views', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig('multiple_3d_views.png', dpi=300, bbox_inches='tight')
    print("✅ Multiple views saved as: multiple_3d_views.png")
    
    # Print some basic statistics
    print(f"\n📊 Data Statistics:")
    print(f"Sigma_Gain range: {data['Sigma_Gain'].min():.2f} to {data['Sigma_Gain'].max():.2f}")
    print(f"Sigma_0 range: {data['Sigma_0'].min():.2f} to {data['Sigma_0'].max():.2f}")
    print(f"Mu_p range: {data['Mu_p'].min():.2f} to {data['Mu_p'].max():.2f}")
    
    print(f"\n✅ Done! Check the generated PNG files.")

if __name__ == "__main__":
    create_simple_3d_plot()