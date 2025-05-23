"""
Visualization module for house price prediction.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_3d_scatter(df, highlight_point=None, dark_mode=False):
    """
    Create a 3D scatter plot of house data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing house data
    highlight_point : dict, optional
        Dictionary with keys 'Size', 'Bedrooms', 'Bathrooms' to highlight
    dark_mode : bool, default=False
        Whether to use dark mode theme
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D scatter plot figure
    """
    # Set color scheme based on theme
    colorscale = "Viridis" if dark_mode else "Turbo"
    grid_color = "rgba(255, 255, 255, 0.1)" if dark_mode else "rgba(0, 0, 0, 0.1)"
    bg_color = "rgba(0, 0, 0, 0)" if dark_mode else "rgba(255, 255, 255, 1)"
    text_color = "white" if dark_mode else "black"
    
    fig = px.scatter_3d(
        df, 
        x='Size', 
        y='Bedrooms', 
        z='Bathrooms', 
        color='Price',
        opacity=0.7,
        color_continuous_scale=colorscale,
        title="House Features vs. Price",
        labels={
            "Size": "Size (sq ft)",
            "Bedrooms": "Number of Bedrooms",
            "Bathrooms": "Number of Bathrooms",
            "Price": "Price ($)"
        }
    )
    
    # Add custom styling
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Price ($)",
            tickprefix="$",
            tickformat=",.0f"
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        scene=dict(
            xaxis=dict(gridcolor=grid_color, showbackground=False),
            yaxis=dict(gridcolor=grid_color, showbackground=False),
            zaxis=dict(gridcolor=grid_color, showbackground=False)
        )
    )
    
    # Add highlight point if provided
    if highlight_point:
        marker_color = "red" if not dark_mode else "#FF5E5E"
        fig.add_trace(
            go.Scatter3d(
                x=[highlight_point['Size']], 
                y=[highlight_point['Bedrooms']], 
                z=[highlight_point['Bathrooms']],
                mode='markers',
                marker=dict(size=10, color=marker_color),
                name="Selected House"
            )
        )
    
    return fig

def create_feature_importance_plot(importance_dict, dark_mode=False):
    """
    Create a bar chart of feature importance.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping feature names to importance values
    dark_mode : bool, default=False
        Whether to use dark mode theme
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    # Set color scheme based on theme
    bg_color = "rgba(0, 0, 0, 0)" if dark_mode else "rgba(255, 255, 255, 1)"
    text_color = "white" if dark_mode else "black"
    bar_color = "#4CAF50" if dark_mode else "#1E88E5"
    
    fig = go.Figure(
        go.Bar(
            x=features,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
            marker_color=bar_color
        )
    )
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Coefficient Value",
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color)
    )
    
    return fig 