#!/usr/bin/env python3
"""
Interactive 3D Visualization for SPS Data
Creates an interactive 3D plot using plotly - you can rotate, zoom, and hover over points
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

def create_interactive_3d_plot():
    """Create an interactive 3D scatter plot using plotly"""
    
    print("🔬 Loading SPS data for interactive visualization...")
    # Load the data
    data = pd.read_csv('../data-generation/sps_rating_results.csv')
    
    print(f"📊 Loaded {len(data)} samples")
    print(f"   GOOD: {len(data[data['Rating'] == 'GOOD'])} ({len(data[data['Rating'] == 'GOOD'])/len(data)*100:.1f}%)")
    print(f"   BAD: {len(data[data['Rating'] == 'BAD'])} ({len(data[data['Rating'] == 'BAD'])/len(data)*100:.1f}%)")
    
    # Color mapping
    color_map = {'GOOD': '#2E8B57', 'BAD': '#DC143C'}  # Sea green, Crimson
    
    print("🎨 Creating interactive 3D plot...")
    
    # Create the main interactive 3D scatter plot
    fig = go.Figure()
    
    for rating in ['GOOD', 'BAD']:
        subset = data[data['Rating'] == rating]
        
        # Create hover text with all information
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
                color=color_map[rating],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            name=f'{rating} ({len(subset)} samples)',
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Update layout for better interactivity
    fig.update_layout(
        title={
            'text': '🔬 Interactive SPS Data: 3D Feature Space Analysis<br><sub>Click and drag to rotate • Scroll to zoom • Hover for details</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(title='Sigma_Gain', backgroundcolor="lightgray"),
            yaxis=dict(title='Sigma_0', backgroundcolor="lightgray"),
            zaxis=dict(title='Mu_p', backgroundcolor="lightgray"),
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8)  # Nice initial viewing angle
            )
        ),
        width=1000,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    # Save as HTML file
    fig.write_html('interactive_3d_plot.html')
    print("✅ Interactive plot saved as: interactive_3d_plot.html")
    
    # Create a second plot with statistical overlays
    print("📈 Creating enhanced interactive plot with statistics...")
    
    # Calculate centroids for each class
    good_data = data[data['Rating'] == 'GOOD']
    bad_data = data[data['Rating'] == 'BAD']
    
    good_centroid = [good_data['Sigma_Gain'].mean(), good_data['Sigma_0'].mean(), good_data['Mu_p'].mean()]
    bad_centroid = [bad_data['Sigma_Gain'].mean(), bad_data['Sigma_0'].mean(), bad_data['Mu_p'].mean()]
    
    # Create enhanced plot with centroids
    fig_enhanced = go.Figure()
    
    # Add data points
    for rating in ['GOOD', 'BAD']:
        subset = data[data['Rating'] == rating]
        
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
                color=color_map[rating],
                opacity=0.6,
                line=dict(width=0.3, color='white')
            ),
            name=f'{rating} Data',
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
    
    # Add centroids as larger markers
    fig_enhanced.add_trace(go.Scatter3d(
        x=[good_centroid[0]],
        y=[good_centroid[1]], 
        z=[good_centroid[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='#2E8B57',
            symbol='diamond',
            line=dict(width=3, color='black')
        ),
        name='GOOD Centroid',
        text=[f"GOOD Centroid<br>Sigma_Gain: {good_centroid[0]:.3f}<br>Sigma_0: {good_centroid[1]:.3f}<br>Mu_p: {good_centroid[2]:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig_enhanced.add_trace(go.Scatter3d(
        x=[bad_centroid[0]],
        y=[bad_centroid[1]],
        z=[bad_centroid[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='#DC143C',
            symbol='diamond',
            line=dict(width=3, color='black')
        ),
        name='BAD Centroid',
        text=[f"BAD Centroid<br>Sigma_Gain: {bad_centroid[0]:.3f}<br>Sigma_0: {bad_centroid[1]:.3f}<br>Mu_p: {bad_centroid[2]:.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add line connecting centroids
    fig_enhanced.add_trace(go.Scatter3d(
        x=[good_centroid[0], bad_centroid[0]],
        y=[good_centroid[1], bad_centroid[1]],
        z=[good_centroid[2], bad_centroid[2]],
        mode='lines',
        line=dict(
            color='purple',
            width=4,
            dash='dash'
        ),
        name='Centroid Distance',
        text=[f"Distance between centroids: {np.linalg.norm(np.array(good_centroid) - np.array(bad_centroid)):.3f}"],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Update layout
    fig_enhanced.update_layout(
        title={
            'text': '🎯 Enhanced Interactive SPS Analysis<br><sub>Includes class centroids and separation metrics</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis=dict(title='Sigma_Gain', backgroundcolor="lightgray"),
            yaxis=dict(title='Sigma_0', backgroundcolor="lightgray"), 
            zaxis=dict(title='Mu_p', backgroundcolor="lightgray"),
            bgcolor="white",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8)
            )
        ),
        width=1100,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Save enhanced plot
    fig_enhanced.write_html('enhanced_interactive_3d_plot.html')
    print("✅ Enhanced plot saved as: enhanced_interactive_3d_plot.html")
    
    # Create a subplot version with multiple views
    print("🔄 Creating multi-view interactive dashboard...")
    
    # Create subplot with 2x2 layout (but we'll use 3D plots)
    fig_multi = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Full 3D View', 'XY Projection', 'XZ Projection', 'YZ Projection'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Main 3D plot (top-left)
    for rating in ['GOOD', 'BAD']:
        subset = data[data['Rating'] == rating]
        fig_multi.add_trace(
            go.Scatter3d(
                x=subset['Sigma_Gain'],
                y=subset['Sigma_0'],
                z=subset['Mu_p'],
                mode='markers',
                marker=dict(size=3, color=color_map[rating], opacity=0.7),
                name=f'{rating}',
                showlegend=(rating == 'GOOD')  # Only show legend once
            ),
            row=1, col=1
        )
    
    # 2D projections
    projections = [
        ('Sigma_Gain', 'Sigma_0', 'XY', 1, 2),
        ('Sigma_Gain', 'Mu_p', 'XZ', 2, 1), 
        ('Sigma_0', 'Mu_p', 'YZ', 2, 2)
    ]
    
    for x_var, y_var, title, row, col in projections:
        for rating in ['GOOD', 'BAD']:
            subset = data[data['Rating'] == rating]
            fig_multi.add_trace(
                go.Scatter(
                    x=subset[x_var],
                    y=subset[y_var],
                    mode='markers',
                    marker=dict(size=4, color=color_map[rating], opacity=0.7),
                    name=f'{rating} ({title})',
                    showlegend=False
                ),
                row=row, col=col
            )
    
    # Update layout for multi-view
    fig_multi.update_layout(
        title={
            'text': '📊 Multi-View Interactive Dashboard',
            'x': 0.5,
            'font': {'size': 16}
        },
        height=800,
        width=1200
    )
    
    # Save multi-view plot
    fig_multi.write_html('multi_view_interactive_dashboard.html') 
    print("✅ Multi-view dashboard saved as: multi_view_interactive_dashboard.html")
    
    # Print statistics and instructions
    centroid_distance = np.linalg.norm(np.array(good_centroid) - np.array(bad_centroid))
    
    print(f"\n📊 **3D Analysis Summary:**")
    print(f"   GOOD Centroid: Sigma_Gain={good_centroid[0]:.3f}, Sigma_0={good_centroid[1]:.3f}, Mu_p={good_centroid[2]:.3f}")
    print(f"   BAD Centroid:  Sigma_Gain={bad_centroid[0]:.3f}, Sigma_0={bad_centroid[1]:.3f}, Mu_p={bad_centroid[2]:.3f}")
    print(f"   Centroid Distance: {centroid_distance:.3f}")
    
    print(f"\n🎮 **How to Use the Interactive Plots:**")
    print(f"   • **Rotate**: Click and drag")
    print(f"   • **Zoom**: Scroll wheel or pinch")
    print(f"   • **Pan**: Shift + click and drag")
    print(f"   • **Hover**: Mouse over points for details")
    print(f"   • **Legend**: Click to show/hide categories")
    print(f"   • **Reset**: Double-click to reset view")
    
    print(f"\n✅ **Generated Files:**")
    print(f"   1. interactive_3d_plot.html - Basic interactive 3D scatter")
    print(f"   2. enhanced_interactive_3d_plot.html - With centroids and statistics")
    print(f"   3. multi_view_interactive_dashboard.html - Multiple viewing perspectives")
    print(f"\n🌐 Open any HTML file in your web browser to explore!")

if __name__ == "__main__":
    create_interactive_3d_plot()