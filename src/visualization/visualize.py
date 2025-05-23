"""
Visualization module for house price prediction.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_3d_scatter(df, highlight_point=None):
    """
    Create a 3D scatter plot of house data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing house data
    highlight_point : dict, optional
        Dictionary with keys 'Size', 'Bedrooms', 'Bathrooms' to highlight
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D scatter plot figure
    """
    fig = px.scatter_3d(
        df, 
        x='Size', 
        y='Bedrooms', 
        z='Bathrooms', 
        color='Price',
        opacity=0.7,
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
        )
    )
    
    # Add highlight point if provided
    if highlight_point:
        fig.add_trace(
            go.Scatter3d(
                x=[highlight_point['Size']], 
                y=[highlight_point['Bedrooms']], 
                z=[highlight_point['Bathrooms']],
                mode='markers',
                marker=dict(size=10, color='red'),
                name="Selected House"
            )
        )
    
    return fig

def create_feature_importance_plot(importance_dict):
    """
    Create a bar chart of feature importance.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary mapping feature names to importance values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart figure
    """
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    fig = go.Figure(
        go.Bar(
            x=features,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto',
        )
    )
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Coefficient Value",
        template="plotly_white"
    )
    
    return fig 