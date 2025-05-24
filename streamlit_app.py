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
from datetime import datetime
import random
import string
from streamlit.components.v1 import html

# Helper function to generate a random ID for forcing Streamlit refreshes
def get_random_id(length=8):
    """Generate a random string to use as a unique ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

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
    grid_color = "rgba(255, 255, 255, 0.3)" if dark_mode else "rgba(0, 0, 0, 0.3)"
    bg_color = "rgba(0, 0, 0, 0)" if dark_mode else "rgba(255, 255, 255, 0)"
    text_color = "white" if dark_mode else "black"
    
    fig = px.scatter_3d(
        df.sample(1000), # Sample for better performance
        x='Size', 
        y='Bedrooms', 
        z='Bathrooms', 
        color='Price',
        opacity=0.7,
        color_continuous_scale=colorscale,
        title="House Features vs. Price Distribution",
        labels={
            "Size": "Size (sq ft)",
            "Bedrooms": "Bedrooms",
            "Bathrooms": "Bathrooms",
            "Price": "Price ($)"
        },
        hover_data={
            "Size": True,
            "Bedrooms": True,
            "Bathrooms": True,
            "Price": ':.2f'
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
                showbackground=True,
                backgroundcolor="rgba(0, 0, 0, 0.02)" if not dark_mode else "rgba(0, 0, 0, 0.2)",
                showline=True,
                linecolor=grid_color,
                linewidth=2
            ),
            yaxis=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor=grid_color, 
                showbackground=True,
                backgroundcolor="rgba(0, 0, 0, 0.02)" if not dark_mode else "rgba(0, 0, 0, 0.2)",
                showline=True,
                linecolor=grid_color,
                linewidth=2
            ),
            zaxis=dict(
                title=dict(font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor=grid_color, 
                showbackground=True,
                backgroundcolor="rgba(0, 0, 0, 0.02)" if not dark_mode else "rgba(0, 0, 0, 0.2)",
                showline=True,
                linecolor=grid_color,
                linewidth=2
            ),
            # Adjust camera angle for better mobile viewing
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        # Make the legend more visible
        legend=dict(
            font=dict(size=14),
            bgcolor="rgba(0,0,0,0.1)" if dark_mode else "rgba(255,255,255,0.7)",
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
                name="Your House"
            )
        )
    
    return fig

def create_feature_importance_plot(importance_dict, dark_mode=True):
    """Create a bar chart of feature importance."""
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    # Set color scheme based on theme
    bg_color = "rgba(0, 0, 0, 0)" if dark_mode else "rgba(255, 255, 255, 0)"
    text_color = "white" if dark_mode else "black"
    
    # Use a gradient color scheme for bars
    bar_colors = ["#4CAF50", "#2196F3", "#FFC107"] if not dark_mode else ["#00E676", "#00B0FF", "#FFEA00"]
    
    fig = go.Figure()
    
    for i, (feature, value) in enumerate(zip(features, values)):
        fig.add_trace(
            go.Bar(
                x=[feature],
                y=[value],
                text=[f"{value:.2f}"],
                textposition='auto',
                marker_color=bar_colors[i],
                name=feature,
                textfont=dict(size=14)  # Larger text on bars
            )
        )
    
    # Add grid lines for better readability in light mode
    grid_color = "rgba(0, 0, 0, 0.1)" if not dark_mode else "rgba(255, 255, 255, 0.1)"
    
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
            tickfont=dict(size=14),  # Larger tick labels
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color if not dark_mode else "rgba(255, 255, 255, 0.3)",
            linewidth=2
        ),
        yaxis=dict(
            title=dict(
                text="Coefficient Value ($)",
                font=dict(size=16)
            ),
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color if not dark_mode else "rgba(255, 255, 255, 0.3)",
            linewidth=2
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

def create_price_distribution_plot(df, predicted_price=None, dark_mode=True):
    """Create a histogram of price distribution with prediction marker."""
    bg_color = "rgba(0, 0, 0, 0)" if dark_mode else "rgba(255, 255, 255, 0)"
    text_color = "white" if dark_mode else "black"
    marker_color = "#FF5E5E" if dark_mode else "red"
    hist_color = "#4CAF50" if not dark_mode else "#00E676"
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=df['Price'],
            nbinsx=50,
            marker_color=hist_color,
            opacity=0.7,
            name="Price Distribution"
        )
    )
    
    # Add vertical line for prediction
    if predicted_price is not None:
        fig.add_vline(
            x=predicted_price, 
            line_width=3, 
            line_dash="dash", 
            line_color=marker_color,
            annotation_text="Your Prediction",
            annotation_position="top right",
            annotation_font_size=14,
            annotation_font_color=marker_color
        )
    
    # Add grid lines for better readability in light mode
    grid_color = "rgba(0, 0, 0, 0.1)" if not dark_mode else "rgba(255, 255, 255, 0.1)"
    
    fig.update_layout(
        title=dict(
            text="House Price Distribution",
            font=dict(size=20)
        ),
        xaxis=dict(
            title=dict(
                text="Price ($)",
                font=dict(size=16)
            ),
            tickfont=dict(size=14),
            tickprefix="$",
            tickformat=",.0f",
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color if not dark_mode else "rgba(255, 255, 255, 0.3)",
            linewidth=2
        ),
        yaxis=dict(
            title=dict(
                text="Frequency",
                font=dict(size=16)
            ),
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color if not dark_mode else "rgba(255, 255, 255, 0.3)",
            linewidth=2
        ),
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(
            color=text_color,
            size=14
        ),
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
    train_rmse = np.sqrt(np.mean((train_preds - y_train) ** 2))
    
    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((test_preds - y_test) ** 2))
    
    # R2 score for test data
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, test_preds)
    
    # Feature importance
    importance = dict(zip(feature_names, model.coef_))
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_r2': r2
    }
    
    return model, importance, metrics

def display_feature_explanation():
    """Display explanations of features for non-technical users."""
    with st.expander("Understanding House Features"):
        st.markdown("""
        ### What do these features mean?
        
        - **Size (sq ft)**: The total interior living space measured in square feet. Larger homes generally cost more.
        
        - **Bedrooms**: The number of bedrooms in the house. More bedrooms typically increase the value.
        
        - **Bathrooms**: The number of bathrooms. Additional bathrooms add significant value to a home.
        
        ### How does the model work?
        
        This model uses linear regression to find relationships between these features and house prices.
        The model learns from thousands of sample houses to predict prices for new houses with similar features.
        
        ### What are coefficients?
        
        The coefficients (shown in the Feature Importance chart) represent how much each feature contributes to the price:
        - If Size has a coefficient of 100, it means each additional square foot adds about $100 to the price
        - If Bedrooms has a coefficient of 50,000, it means each additional bedroom adds about $50,000 to the price
        """)

def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title="House Price Prediction",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize or update prediction timestamp in session state
    # This ensures the prediction date is updated whenever the app recalculates
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = datetime.now()
    
    # Force a refresh by adding a hidden element with a random ID
    # This helps ensure state is fresh with each interaction
    random_id = get_random_id()
    st.markdown(f'<div id="{random_id}" style="display: none;"></div>', unsafe_allow_html=True)
    
    # Sidebar with app info and inputs
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/cottage.png", width=80)
        st.title("House Price Prediction")
        
        st.markdown("### About This App")
        st.info(
            "This application helps predict house prices based on key features. "
            "Adjust the sliders to see how different features affect the predicted price."
        )
        
        st.markdown("### Input Parameters")
        # Add unique keys to force re-execution when values change
        # Update the timestamp whenever any slider value changes
        def update_prediction_time():
            st.session_state.last_prediction_time = datetime.now()
        
        size = st.slider(
            "Size (sq ft)", 
            min_value=1000,
            max_value=3500,
            value=2000,
            help="The total interior living space of the house",
            key="size_slider",
            on_change=update_prediction_time
        )
        
        bedrooms = st.slider(
            "Bedrooms",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of bedrooms in the house",
            key="bedrooms_slider",
            on_change=update_prediction_time
        )
        
        bathrooms = st.slider(
            "Bathrooms",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of bathrooms in the house",
            key="bathrooms_slider",
            on_change=update_prediction_time
        )
        
        # Theme selection
        st.markdown("### App Settings")
        dark_mode = st.checkbox("Dark Mode", value=True, key="dark_mode_checkbox")
        
        # Information
        st.markdown("### Data Source")
        st.caption(
            "This app uses simulated data for demonstration purposes. "
            "The model is trained on 10,000 synthetic house samples."
        )
    
    # Add custom CSS for styling - based on chosen theme
    if dark_mode:
        # Dark mode styles
        card_bg = "rgba(255, 255, 255, 0.1)"
        card_border = "rgba(255, 255, 255, 0.2)"
        separator_gradient = "linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.5), rgba(255,255,255,0))"
        metric_bg = "rgba(33, 150, 243, 0.1)"
        metric_title_color = "#90CAF9"
        metric_value_color = "white"
        text_color = "white"
        footer_bg = "rgba(255, 255, 255, 0.05)"
        footer_text = "rgba(255, 255, 255, 0.7)"
        header_gradient = "linear-gradient(to right, #4CAF50, #2196F3)"
    else:
        # Light mode styles
        card_bg = "rgba(0, 0, 0, 0.03)"
        card_border = "rgba(0, 0, 0, 0.1)"
        separator_gradient = "linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0))"
        metric_bg = "rgba(33, 150, 243, 0.05)"
        metric_title_color = "#1565C0"
        metric_value_color = "#212121"
        text_color = "#212121"
        footer_bg = "rgba(0, 0, 0, 0.03)"
        footer_text = "rgba(0, 0, 0, 0.7)"
        header_gradient = "linear-gradient(to right, #4CAF50, #2196F3)"
    
    # Apply theme using CSS
    st.markdown(f"""
    <style>
    .stApp {{
        max-width: 100%;
        color: {text_color};
    }}
    
    /* Custom card styling */
    .custom-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid {card_border};
    }}
    
    /* Colored headings */
    .green-heading {{
        color: {'#4CAF50' if dark_mode else '#2E7D32'};
        font-size: 1.5rem;
        margin-bottom: 10px;
    }}
    
    .blue-heading {{
        color: {'#2196F3' if dark_mode else '#1565C0'};
        font-size: 1.5rem;
        margin-bottom: 10px;
    }}
    
    .orange-heading {{
        color: {'#FF9800' if dark_mode else '#E65100'};
        font-size: 1.5rem;
        margin-bottom: 10px;
    }}
    
    /* Styled metrics */
    .metric-container {{
        background-color: {metric_bg};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid {'#2196F3' if dark_mode else '#1565C0'};
    }}
    
    .metric-title {{
        font-size: 1rem;
        color: {metric_title_color};
        margin-bottom: 5px;
    }}
    
    .metric-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {metric_value_color};
    }}
    
    /* Separator line */
    .separator {{
        height: 1px;
        background: {separator_gradient};
        margin: 30px 0;
    }}
    
    /* Improve slider labels on mobile */
    .st-emotion-cache-1l269u1 p {{
        font-weight: 600 !important;
        font-size: 1rem !important;
    }}
    
    /* Make 3D plot container taller on mobile */
    @media (max-width: 768px) {{
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column;
        }}
        
        .js-plotly-plot, .plot-container {{
            min-height: 400px !important;
        }}
        
        /* Increase button size on mobile */
        .stButton>button {{
            font-size: 1rem;
            padding: 0.5rem 1rem;
        }}
        
        /* Larger text for metric values */
        [data-testid="stMetricValue"] {{
            font-size: 2rem !important;
        }}
    }}
    
    /* Adjust expander styling based on theme */
    .streamlit-expanderHeader {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
    }}
    
    /* Fix tabs to match theme */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {card_bg};
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main content area
    # App header with animated gradient (using custom HTML)
    st.markdown(f"""
    <div style="background: {header_gradient}; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <h1 style="color: white; margin:0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🏠 House Price Predictor</h1>
        <p style="color: white; margin-top: 0.5rem; font-size: 1.1rem;">Interactive tool for real estate price estimation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Force app to recalculate on every run by adding a random element
    # This ensures the prediction date stays current
    current_runtime = datetime.now()
    st.write(f"<div style='display: none;'>{current_runtime.timestamp()}</div>", unsafe_allow_html=True)
    
    # Generate data and train model
    house_data = generate_house_data()
    
    X = house_data[['Size', 'Bedrooms', 'Bathrooms']]
    y = house_data['Price']
    
    model, importance, metrics = train_model(X, y)
    
    # Make prediction - Fix feature names issue by creating a DataFrame with proper column names
    input_data = pd.DataFrame([[size, bedrooms, bathrooms]], columns=['Size', 'Bedrooms', 'Bathrooms'])
    prediction = model.predict(input_data)
    predicted_price = prediction[0]
    
    # Update prediction timestamp whenever a new prediction is made
    st.session_state.last_prediction_time = datetime.now()
    
    # Display feature explanations for non-technical users
    display_feature_explanation()
    
    # Top section - Predictions and Key Metrics
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    st.markdown('<p class="green-heading">Prediction Results</p>', unsafe_allow_html=True)
    
    # Display prediction in a more visually appealing way
    cols = st.columns([2, 2, 1])
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Predicted House Price (USD)</div>
            <div class="metric-value">$""" + f"{predicted_price:,.2f}" + """</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Per square foot calculation
        price_per_sqft = predicted_price / size
        st.markdown(f"Price per sq ft: **${price_per_sqft:.2f}**")
    
    with cols[1]:
        # Currency converter (more reliable implementation with improved styling)
        if 'show_kes' not in st.session_state:
            st.session_state.show_kes = False
            
        kes_price = predicted_price * KES_EXCHANGE_RATE
        
        if st.session_state.show_kes:
            st.markdown(f"""
            <div class="metric-container" style="border-left: 5px solid {'#FF9800' if dark_mode else '#E65100'};">
                <div class="metric-title">Predicted Price (KES)</div>
                <div class="metric-value">KSh """ + f"{kes_price:,.2f}" + """</div>
            </div>
            """, unsafe_allow_html=True)
            button_label = "Show in USD"
        else:
            button_label = "Convert to Kenyan Shillings (KES)"
        
        def toggle_currency_and_update_time():
            st.session_state.show_kes = not st.session_state.show_kes
            # Also update the prediction time when toggling currency
            st.session_state.last_prediction_time = datetime.now()
        
        if st.button(button_label, key="kes_converter", on_click=toggle_currency_and_update_time):
            # Instead, perform the rerun here after the callback completes
            st.rerun()
            
        if st.session_state.show_kes:
            st.caption(f"Exchange rate: $1 = KSh {KES_EXCHANGE_RATE} (as of today)")
    
    with cols[2]:
        # Use the timestamp from session state for displaying the prediction date
        # This ensures the date updates whenever the user changes input values
        now = st.session_state.last_prediction_time
        current_date = now.strftime("%d %b %Y")
        current_time = now.strftime("%H:%M:%S")
        tz_display = "System local time"
        
        # Force refresh with a unique key to ensure the component updates
        refresh_key = get_random_id()
        
        # Display the prediction date with current time to show it's updating
        # Using a more prominent and colorful style
        date_color = "#4CAF50" if dark_mode else "#1E88E5"
        time_color = "#757575" if dark_mode else "#9E9E9E"
        
        # Get the current date and time for server-side display (fallback)
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%d %b %Y")
        
        # Create a very simple HTML component with JavaScript
        local_time_html = f"""
        <div style="background-color: rgba(0,0,0,0.05); padding: 15px; border-radius: 5px; 
                  border-left: 4px solid #4CAF50; margin-bottom: 10px;">
            <div style="font-weight: bold; font-size: 0.9rem; margin-bottom: 8px; color: white;">Prediction Date:</div>
            <div id="date" style="font-size: 1.2rem; color: #4CAF50; font-weight: bold;">{current_date}</div>
            <div id="time" style="font-size: 0.9rem; color: #888888; margin-top: 5px;">Generated at {current_time}</div>
            <div id="tz" style="font-size: 0.75rem; color: #666666; margin-top: 5px;">Server time</div>
        </div>

        <script>
        try {{
            // Get local time
            var now = new Date();
            
            // Format date: DD MMM YYYY
            var day = now.getDate().toString().padStart(2, '0');
            var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            var month = months[now.getMonth()];
            var year = now.getFullYear();
            
            // Format time: HH:MM:SS
            var hours = now.getHours().toString().padStart(2, '0');
            var minutes = now.getMinutes().toString().padStart(2, '0');
            var seconds = now.getSeconds().toString().padStart(2, '0');
            
            // Update HTML with exact colors from the screenshot
            document.getElementById('date').textContent = day + ' ' + month + ' ' + year;
            document.getElementById('date').style.color = '#4CAF50';  // Green date color
            
            document.getElementById('time').textContent = 'Generated at ' + hours + ':' + minutes + ':' + seconds;
            document.getElementById('time').style.color = '#888888';  // Gray time text
            
            document.getElementById('tz').textContent = 'Your local time';
            document.getElementById('tz').style.color = '#666666';  // Light gray timezone text
        }} catch(e) {{}}
        </script>
        """
        
        # Use components.html instead of st.markdown
        html(local_time_html, height=130)
        
        # Add a visible fallback message just in case
        st.caption("If date is not updating: Local JavaScript may be disabled in your browser")
        
        # Add a refresh button with a properly displayed calendar icon
        refresh_label = "📅 Update Date & Time"  # Using a different calendar emoji that works across platforms
        
        if st.button(refresh_label, 
                   key=f"refresh_date_{refresh_key}",
                   help="Click to update the prediction date and time to current moment"):
            st.rerun()
            
        # Display accuracy underneath
        model_accuracy = metrics['test_r2'] * 100
        st.markdown(f"**Model Accuracy:**  \n{model_accuracy:.1f}%")
    
    # Metric cards for model performance
    st.markdown('<p class="blue-heading">Model Performance Metrics</p>', unsafe_allow_html=True)
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Training Error (RMSE)", f"${metrics['train_rmse']:,.2f}", 
                 help="Root Mean Square Error on training data - lower is better")
    with metric_cols[1]:
        st.metric("Testing Error (RMSE)", f"${metrics['test_rmse']:,.2f}", 
                 help="Root Mean Square Error on testing data - lower is better")
    with metric_cols[2]:
        st.metric("R² Score", f"{metrics['test_r2']:.4f}", 
                 help="Coefficient of determination - closer to 1 is better")
    
    # Vizualization section
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<p class="orange-heading">Data Visualization</p>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["3D Feature Plot", "Price Distribution", "Feature Importance"])
    
    with tab1:
        st.markdown("""
        This 3D visualization shows how house Size, Bedrooms, and Bathrooms relate to Price.
        Your selected house is highlighted in red.
        """)
        fig_3d = create_3d_scatter(
            house_data,
            highlight_point={'Size': size, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms},
            dark_mode=dark_mode
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab2:
        st.markdown("""
        This histogram shows the distribution of house prices in the dataset.
        Your predicted price is marked with a dashed red line.
        """)
        price_dist_fig = create_price_distribution_plot(
            house_data, 
            predicted_price=predicted_price,
            dark_mode=dark_mode
        )
        st.plotly_chart(price_dist_fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        This chart shows how much each feature contributes to the house price.
        Higher values mean the feature has a stronger impact on the price.
        """)
        importance_fig = create_feature_importance_plot(importance, dark_mode=dark_mode)
        st.plotly_chart(importance_fig, use_container_width=True)
    
    # Detailed explanation for technical stakeholders
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    with st.expander("Technical Details (For Data Scientists)"):
        st.markdown("""
        ### Model Information
        
        - **Algorithm**: Linear Regression
        - **Features**: Size (sq ft), Bedrooms, Bathrooms
        - **Target**: House Price
        - **Training Data**: 10,000 synthetic samples
        - **Train/Test Split**: 80% training, 20% testing
        - **Random Seed**: 42
        
        ### Model Equation
        
        The linear regression model uses the following equation to predict house prices:
        
        ```
        Price = (Intercept) + Size * Coef_Size + Bedrooms * Coef_Bedrooms + Bathrooms * Coef_Bathrooms
        ```
        
        Where:
        - Intercept = {:.2f}
        - Coef_Size = {:.2f}
        - Coef_Bedrooms = {:.2f}
        - Coef_Bathrooms = {:.2f}
        
        ### Data Generation
        
        The synthetic data is generated with the following parameters:
        - Size: Random integers between 1000 and 3500 sq ft
        - Bedrooms: Random integers between 1 and 5
        - Bathrooms: Random integers between 1 and 3
        - Price: Weighted sum of features with normal noise (μ=0, σ=50000)
        """.format(model.intercept_, importance['Size'], importance['Bedrooms'], importance['Bathrooms']))
        
        # Show a sample of the data
        st.subheader("Sample Data")
        st.dataframe(house_data.sample(5), use_container_width=True)
    
    # Add footer
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Enhanced footer with more information
    st.markdown(f"""
    <div style="background-color: {footer_bg}; padding: 15px; border-radius: 10px; text-align: center;">
        <div style="margin-bottom: 10px;">
            <span style="font-weight: bold;">House Price Prediction App</span> | 
            <span>Built with Streamlit & Python</span> | 
            <span>Data is simulated for demonstration</span>
        </div>
        <div style="font-size: 0.8rem; color: {footer_text};">
            Made by Collins N. Nyagaka | Last Updated: May 2025 | Uses system local time for prediction timestamps
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 