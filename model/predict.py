"""
Model Prediction Module
Loads saved model and makes predictions
"""

import pickle
import os
import pandas as pd

class SustainabilityPredictor:
    """Predictor class for sustainability classification"""
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.load_model()
    
    def load_model(self):
        """Load saved model and encoders"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
    
    def predict(self, item_type, waste_category, energy_level, pollution_risk, 
                recyclable, disposal_method):
        """Make prediction for new item"""
        # Encode features
        features = []
        feature_values = {
            'item_type': str(item_type),
            'waste_category': str(waste_category),
            'energy_level': str(energy_level),
            'pollution_risk': str(pollution_risk),
            'recyclable': str(recyclable),
            'disposal_method': str(disposal_method)
        }
        
        for feature in ['item_type', 'waste_category', 'energy_level', 
                        'pollution_risk', 'recyclable', 'disposal_method']:
            le = self.label_encoders[feature]
            try:
                encoded = le.transform([feature_values[feature]])[0]
            except ValueError:
                # Handle unseen labels - use most common value
                encoded = 0
            features.append(encoded)
        
        # Scale and predict
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        score_category = self.label_encoders['target'].inverse_transform([prediction])[0]
        
        # Map category to score range
        score_map = {'Low': (1, 3), 'Medium': (4, 6), 'High': (7, 9)}
        score_range = score_map.get(score_category, (1, 9))
        
        return {
            'prediction': score_category,
            'score_range': score_range,
            'probabilities': {
                self.label_encoders['target'].classes_[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            },
            'confidence': float(max(probabilities))
        }
    
    def predict_from_text(self, item_text):
        """Predict from natural language text (simple keyword matching)"""
        item_text_lower = item_text.lower()
        
        # Simple keyword matching (can be enhanced with NLP)
        item_type = "Other"
        waste_category = "Non-Recyclable"
        energy_level = "Low"
        pollution_risk = "Medium"
        recyclable = "No"
        disposal_method = "Landfill"
        
        # Detect item type
        if any(word in item_text_lower for word in ['battery', 'electronic', 'phone', 'computer', 'tv']):
            item_type = "Electronics"
            waste_category = "E-Waste"
            pollution_risk = "High"
            recyclable = "Yes"
            disposal_method = "Special Collection"
        elif any(word in item_text_lower for word in ['plastic', 'bottle', 'bag', 'container']):
            item_type = "Plastic"
            waste_category = "Recyclable"
            recyclable = "Yes"
            disposal_method = "Recycling Bin"
        elif any(word in item_text_lower for word in ['paper', 'cardboard', 'newspaper']):
            item_type = "Paper"
            waste_category = "Recyclable"
            recyclable = "Yes"
            disposal_method = "Recycling Bin"
        elif any(word in item_text_lower for word in ['food', 'organic', 'compost', 'vegetable', 'fruit']):
            item_type = "Organic"
            waste_category = "Biodegradable"
            recyclable = "No"
            disposal_method = "Compost Bin"
        elif any(word in item_text_lower for word in ['glass', 'jar', 'bottle']):
            item_type = "Glass"
            waste_category = "Recyclable"
            recyclable = "Yes"
            disposal_method = "Recycling Bin"
        elif any(word in item_text_lower for word in ['chemical', 'paint', 'oil', 'medicine', 'pharmaceutical']):
            item_type = "Chemical"
            waste_category = "Hazardous"
            pollution_risk = "High"
            recyclable = "No"
            disposal_method = "Special Collection"
        
        return self.predict(item_type, waste_category, energy_level, pollution_risk, 
                          recyclable, disposal_method)

