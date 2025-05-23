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
            "Bedrooms": "Bedrooms",
            "Bathrooms": "Bathrooms",
            "Price": "Price ($)"
        }
    )
    
    # Add custom styling - Improved for mobile readability
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Price ($)",
            title_font=dict(size=16),
            tickfont=dict(size=14),
            tickprefix="$",
            tickformat=",.0f",
            thickness=25  # Make colorbar thicker
        ),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(
            color=text_color,
            size=16  # Increase base font size
        ),
        title=dict(
            font=dict(size=20)  # Larger title
        ),
        scene=dict(
            xaxis=dict(
                title=dict(font=dict(size=16)),  # Larger axis title
                tickfont=dict(size=14),  # Larger tick labels
                gridcolor=grid_color, 
                showbackground=False
            ),
            yaxis=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor=grid_color, 
                showbackground=False
            ),
            zaxis=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor=grid_color, 
                showbackground=False
            ),
            # Adjust camera angle for better mobile viewing
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        # Make the legend more visible
        legend=dict(
            font=dict(size=14),
            bgcolor="rgba(0,0,0,0.1)" if dark_mode else "rgba(255,255,255,0.5)",
            bordercolor="rgba(255,255,255,0.2)" if dark_mode else "rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        # Add margin to ensure labels don't get cut off
        margin=dict(l=10, r=10, t=60, b=10)
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
                marker=dict(size=12, color=marker_color),  # Larger marker
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
            marker_color=bar_color,
            textfont=dict(size=14)  # Larger text on bars
        )
    )
    
    fig.update_layout(
        title=dict(
            text="Feature Importance",
            font=dict(size=20)  # Larger title
        ),
        xaxis=dict(
            title=dict(
                text="Feature",
                font=dict(size=16)  # Larger axis title
            ),
            tickfont=dict(size=14)  # Larger tick labels
        ),
        yaxis=dict(
            title=dict(
                text="Coefficient Value",
                font=dict(size=16)
            ),
            tickfont=dict(size=14)
        ),
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(
            color=text_color,
            size=14  # Base font size
        ),
        # Add margin to ensure labels don't get cut off
        margin=dict(l=10, r=10, t=60, b=50)
    )
    
    return fig 