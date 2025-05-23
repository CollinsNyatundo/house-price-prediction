"""
Streamlit deployment entry point for House Price Prediction App
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

# Define exchange rate constant
KES_EXCHANGE_RATE = 129.25  # 1 USD = 129.25 KES as of today

# Define functions from src modules directly to avoid import issues
def generate_house_data(n_samples=10000, random_seed=50):
    """Generate synthetic house data for model training and testing."""
    np.random.seed(random_seed)
    
    # Generate features
    size = np.random.randint(1000, 3500, n_samples)
    bedrooms = np.random.randint(1, 5, n_samples)
    bathrooms = np.random.randint(1, 3, n_samples)
    
    # Create DataFrame
    house_data = pd.DataFrame({
        'Size': size, 
        'Bedrooms': bedrooms, 
        'Bathrooms': bathrooms
    })
    
    # Add target variable with some noise
    house_data['Price'] = (
        100 * house_data['Size'] + 
        50000 * house_data['Bedrooms'] + 
        30000 * house_data['Bathrooms'] + 
        np.random.normal(0, 50000, n_samples)
    )
    
    return house_data

def create_3d_scatter(df, highlight_point=None, dark_mode=True):
    """Create a 3D scatter plot of house data."""
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

def create_feature_importance_plot(importance_dict, dark_mode=True):
    """Create a bar chart of feature importance."""
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

def train_model(X, y, random_state=42):
    """Train a linear regression model."""
    # Convert to DataFrame if not already to ensure feature names
    feature_names = ['Size', 'Bedrooms', 'Bathrooms']
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Feature importance
    importance = dict(zip(feature_names, model.coef_))
    
    return model, importance

def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title="House Price Prediction",
        page_icon="🏠",
        layout="wide"
    )
    
    # App title and description
    st.title("House Price Prediction")
    st.write("Predict house prices based on features")
    
    # Detect if on mobile (approximation based on screen width)
    is_mobile = False
    
    # Add CSS to make UI more responsive
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    
    /* Improve slider labels on mobile */
    .st-emotion-cache-1l269u1 p {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Make 3D plot container taller on mobile */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        
        .js-plotly-plot, .plot-container {
            min-height: 400px !important;
        }
        
        /* Increase button size on mobile */
        .stButton>button {
            font-size: 1rem;
            padding: 0.5rem 1rem;
        }
        
        /* Larger text for metric values */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sliders at the top of the page
    st.subheader("House Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        size = st.slider(
            "Size (sq ft)", 
            min_value=1000,
            max_value=3500,
            value=2000
        )
    
    with col2:
        bedrooms = st.slider(
            "Bedrooms",
            min_value=1,
            max_value=5,
            value=2
        )
    
    with col3:
        bathrooms = st.slider(
            "Bathrooms",
            min_value=1,
            max_value=3,
            value=2
        )
    
    # Generate data and train model
    house_data = generate_house_data()
    
    X = house_data[['Size', 'Bedrooms', 'Bathrooms']]
    y = house_data['Price']
    
    model, importance = train_model(X, y)
    
    # Make prediction
    prediction = model.predict([[size, bedrooms, bathrooms]])
    predicted_price = prediction[0]
    
    # Prediction results
    st.subheader("Prediction Results")
    st.metric("Predicted House Price", f"${predicted_price:,.2f}")
    
    # Currency converter (more reliable implementation)
    st.write("---")
    st.write("**Currency Conversion**")
    
    if 'show_kes' not in st.session_state:
        st.session_state.show_kes = False
        
    if st.button("Convert to Kenyan Shillings (KES)", key="kes_converter"):
        st.session_state.show_kes = not st.session_state.show_kes
        
    if st.session_state.show_kes:
        kes_price = predicted_price * KES_EXCHANGE_RATE
        st.metric("Predicted Price in KES", f"KSh {kes_price:,.2f}")
        st.caption("Exchange rate: $1 = KSh 129.25 (as of today)")
    
    # Visualization
    st.subheader("Visualization")
    fig = create_3d_scatter(
        house_data,
        highlight_point={'Size': size, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms},
        dark_mode=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_fig = create_feature_importance_plot(importance, dark_mode=True)
    st.plotly_chart(importance_fig, use_container_width=True)
    
    # Add footer
    st.markdown("---")
    st.markdown(
        "House Price Prediction App | Built with Streamlit | "
        "Data is simulated for demonstration purposes | "
        "Made by Collins N. Nyagaka"
    )

if __name__ == "__main__":
    main() 