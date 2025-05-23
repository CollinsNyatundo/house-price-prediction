"""
Tests for the house price prediction model.
"""
import unittest
import numpy as np
import pandas as pd

from src.models.house_price_model import HousePriceModel
from src.data.data_generator import generate_house_data

class TestHousePriceModel(unittest.TestCase):
    """Test cases for HousePriceModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = HousePriceModel(random_state=42)
        self.data = generate_house_data(n_samples=100, random_seed=42)
        self.X = self.data[['Size', 'Bedrooms', 'Bathrooms']]
        self.y = self.data['Price']
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.is_trained)
        self.assertEqual(self.model.random_state, 42)
    
    def test_model_training(self):
        """Test model training."""
        metrics = self.model.train(self.X, self.y)
        
        # Check if model is trained
        self.assertTrue(self.model.is_trained)
        
        # Check if metrics are returned
        self.assertIn('train_rmse', metrics)
        self.assertIn('test_rmse', metrics)
        self.assertIn('train_r2', metrics)
        self.assertIn('test_r2', metrics)
        
        # Check if metrics are valid
        self.assertGreater(metrics['train_r2'], 0)
        self.assertGreater(metrics['test_r2'], 0)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model
        self.model.train(self.X, self.y)
        
        # Test prediction on a single sample
        sample = np.array([[2000, 3, 2]])
        prediction = self.model.predict(sample)
        
        # Check if prediction is a number
        self.assertEqual(len(prediction), 1)
        self.assertIsInstance(prediction[0], float)
        
        # Test prediction on multiple samples
        samples = np.array([[2000, 3, 2], [1500, 2, 1]])
        predictions = self.model.predict(samples)
        
        # Check if predictions are returned for all samples
        self.assertEqual(len(predictions), 2)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        # Train model
        self.model.train(self.X, self.y)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check if all features are included
        self.assertEqual(set(importance.keys()), {'Size', 'Bedrooms', 'Bathrooms'})
        
        # Check if importance values are numbers
        for value in importance.values():
            self.assertIsInstance(value, float)
    
    def test_error_before_training(self):
        """Test error handling when predicting before training."""
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[2000, 3, 2]]))
        
        with self.assertRaises(ValueError):
            self.model.get_feature_importance()

if __name__ == '__main__':
    unittest.main() 