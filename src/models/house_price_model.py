"""
House price prediction model module.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class HousePriceModel:
    """
    Linear regression model for house price prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.model = LinearRegression()
        self.random_state = random_state
        self.is_trained = False
        self.feature_names = ['Size', 'Bedrooms', 'Bathrooms']
        
    def train(self, X, y, test_size=0.2):
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features for training
        y : array-like
            Target values
        test_size : float, default=0.2
            Test size for train-test split
            
        Returns:
        --------
        dict
            Dictionary containing model performance metrics
        """
        # Input validation
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
            
        # Convert to DataFrame if not already to ensure feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Additional validation for feature values
        for feature in self.feature_names:
            if X[feature].isnull().any():
                raise ValueError(f"Feature '{feature}' contains null values")
            if not np.issubdtype(X[feature].dtype, np.number):
                raise ValueError(f"Feature '{feature}' must contain numeric values")
                
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds)),
            'train_r2': r2_score(y_train, train_preds),
            'test_r2': r2_score(y_test, test_preds)
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            Features for prediction
            
        Returns:
        --------
        array-like
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Convert to DataFrame if not already to ensure feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Validate input data
        for feature in self.feature_names:
            if feature not in X.columns:
                raise ValueError(f"Missing required feature: {feature}")
            if X[feature].isnull().any():
                raise ValueError(f"Feature '{feature}' contains null values")
            if not np.issubdtype(X[feature].dtype, np.number):
                raise ValueError(f"Feature '{feature}' must contain numeric values")
                
        # Apply reasonable bounds checking
        if (X['Size'] < 100).any() or (X['Size'] > 10000).any():
            raise ValueError("Size must be between 100 and 10000 sq ft")
        if (X['Bedrooms'] < 1).any() or (X['Bedrooms'] > 10).any():
            raise ValueError("Bedrooms must be between 1 and 10")
        if (X['Bathrooms'] < 1).any() or (X['Bathrooms'] > 10).any():
            raise ValueError("Bathrooms must be between 1 and 10")
            
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return dict(zip(self.feature_names, self.model.coef_)) 