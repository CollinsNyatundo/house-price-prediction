"""
House Price Prediction App - Main Application

This Streamlit application predicts house prices based on features
such as size, number of bedrooms, and bathrooms using a linear regression model.
"""
import streamlit as st
import logging
import hashlib
import socket

from src.data.data_generator import generate_house_data
from src.models.house_price_model import HousePriceModel
from src.visualization.visualize import create_3d_scatter, create_feature_importance_plot
from src.config.config import APP_CONFIG
from src.config.security_config import get_all_security_headers
from src.utils.logger import setup_logger
from src.utils.security_middleware import secure_endpoint, sanitize_input

# Set up logger
logger = setup_logger()

# Define exchange rate constant
KES_EXCHANGE_RATE = 129.25  # 1 USD = 129.25 KES as of today

# Add security headers
def add_security_headers():
    """Add security headers to the Streamlit app."""
    # Get security headers from configuration
    headers = get_all_security_headers()
    
    # Add headers to the HTML
    headers_html = ""
    for header, value in headers.items():
        headers_html += f'<meta http-equiv="{header}" content="{value}">\n'
    
    # Inject headers into the page
    st.markdown(headers_html, unsafe_allow_html=True)

@secure_endpoint
def main():
    """Main application function."""
    try:
        # Set page config
        st.set_page_config(
            page_title=APP_CONFIG["title"],
            page_icon="🏠",
            layout="wide"
        )
        
        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = hashlib.md5(str(id(st.session_state)).encode()).hexdigest()
        
        # Add security headers
        add_security_headers()
        
        # Add CSS using safer approach with st.markdown and <style> tags
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
        """)
        
        # App title and description
        st.title(APP_CONFIG["title"])
        st.write(sanitize_input(APP_CONFIG["description"]))
        
        # Sidebar for inputs
        st.sidebar.header("House Features")
        
        size = st.sidebar.slider(
            "Size (sq ft)", 
            min_value=APP_CONFIG["size_range"][0],
            max_value=APP_CONFIG["size_range"][1],
            value=APP_CONFIG["default_size"]
        )
        
        bedrooms = st.sidebar.slider(
            "Bedrooms",
            min_value=APP_CONFIG["bedrooms_range"][0],
            max_value=APP_CONFIG["bedrooms_range"][1],
            value=APP_CONFIG["default_bedrooms"]
        )
        
        bathrooms = st.sidebar.slider(
            "Bathrooms",
            min_value=APP_CONFIG["bathrooms_range"][0],
            max_value=APP_CONFIG["bathrooms_range"][1],
            value=APP_CONFIG["default_bathrooms"]
        )
        
        # Validate inputs
        if size < 100 or size > 10000:
            st.error("Size must be between 100 and 10000 sq ft")
            return
            
        if bedrooms < 1 or bedrooms > 10:
            st.error("Bedrooms must be between 1 and 10")
            return
            
        if bathrooms < 1 or bathrooms > 10:
            st.error("Bathrooms must be between 1 and 10")
            return
        
        # Generate data and train model
        logger.info("Generating house data and training model")
        house_data = generate_house_data()
        
        X = house_data[['Size', 'Bedrooms', 'Bathrooms']]
        y = house_data['Price']
        
        model = HousePriceModel()
        metrics = model.train(X, y)
        
        # Make prediction
        prediction = model.predict([[size, bedrooms, bathrooms]])
        predicted_price = prediction[0]
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Visualization")
            # Create and display 3D scatter plot
            fig = create_3d_scatter(
                house_data,
                highlight_point={'Size': size, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms},
                dark_mode=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
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
            
            st.subheader("Model Performance")
            st.write(f"Training RMSE: ${metrics['train_rmse']:,.2f}")
            st.write(f"Testing RMSE: ${metrics['test_rmse']:,.2f}")
            st.write(f"R² Score: {metrics['test_r2']:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            importance = model.get_feature_importance()
            importance_fig = create_feature_importance_plot(importance, dark_mode=True)
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Add footer
        st.markdown("---")
        st.markdown(
            sanitize_input(
                "House Price Prediction App | Built with Streamlit | "
                "Data is simulated for demonstration purposes | "
                "Made by Collins N. Nyagaka"
            )
        )
        
        logger.info(f"Prediction made: ${predicted_price:,.2f} for house with "
                   f"{size} sq ft, {bedrooms} bedrooms, {bathrooms} bathrooms")
        
    except Exception as e:
        logger.error(f"Error in application: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 