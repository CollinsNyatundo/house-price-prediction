"""
House Price Prediction App - Main Application

This Streamlit application predicts house prices based on features
such as size, number of bedrooms, and bathrooms using a linear regression model.
"""
import streamlit as st
import logging

from src.data.data_generator import generate_house_data
from src.models.house_price_model import HousePriceModel
from src.visualization.visualize import create_3d_scatter, create_feature_importance_plot
from src.config.config import APP_CONFIG
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger()

# Define exchange rate constant
KES_EXCHANGE_RATE = 129.25  # 1 USD = 129.25 KES as of today

def main():
    """Main application function."""
    try:
        # Set page config
        st.set_page_config(
            page_title=APP_CONFIG["title"],
            page_icon="🏠",
            layout="wide"
        )
        
        # App title and description
        st.title(APP_CONFIG["title"])
        st.write(APP_CONFIG["description"])
        
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
            
            # Currency converter
            show_kes = st.button("Convert to Kenyan Shillings (KES)")
            if show_kes:
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
            "House Price Prediction App | Built with Streamlit | "
            "Data is simulated for demonstration purposes | "
            "Made with ❤️ by Collins N. Nyagaka"
        )
        
        logger.info(f"Prediction made: ${predicted_price:,.2f} for house with "
                   f"{size} sq ft, {bedrooms} bedrooms, {bathrooms} bathrooms")
        
    except Exception as e:
        logger.error(f"Error in application: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 