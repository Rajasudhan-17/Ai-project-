"""
Machine Learning Model Training Script
Trains a classification model for sustainability item classification
Supports: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class SustainabilityModelTrainer:
    """Trainer class for sustainability classification model"""
    
    def __init__(self, dataset_path='../dataset/sustainability_data.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Load and display dataset information"""
        self.df = pd.read_csv(self.dataset_path)
        return self.df
    
    def preprocess_data(self):
        """Preprocess data: encode categorical features"""
        df = self.df.copy()
        
        # Features to use for prediction
        categorical_features = ['item_type', 'waste_category', 'energy_level', 
                               'pollution_risk', 'recyclable', 'disposal_method']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                self.feature_columns.append(feature + '_encoded')
        
        # Target variable: sustainability_score (convert to categories)
        # Convert score to categories: Low (1-3), Medium (4-6), High (7-9)
        df['score_category'] = pd.cut(df['sustainability_score'], 
                                      bins=[0, 3, 6, 10], 
                                      labels=['Low', 'Medium', 'High'])
        
        le_target = LabelEncoder()
        df['target'] = le_target.fit_transform(df['score_category'])
        self.label_encoders['target'] = le_target
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, df
    
    def train_model(self, model_type='random_forest', test_size=0.2, random_state=42):
        """Train the selected model"""
        X, y, df = self.preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Select model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=random_state, 
                                          multi_class='multinomial', solver='lbfgs')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                               max_depth=10, min_samples_split=5)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(random_state=random_state, max_depth=6,
                                          n_estimators=100, learning_rate=0.1)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def save_model(self, model_path='model.pkl'):
        """Save trained model and encoders"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def predict(self, item_type, waste_category, energy_level, pollution_risk, 
                recyclable, disposal_method):
        """Make prediction for new item"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Encode features
        features = []
        feature_values = {
            'item_type': item_type,
            'waste_category': waste_category,
            'energy_level': energy_level,
            'pollution_risk': pollution_risk,
            'recyclable': recyclable,
            'disposal_method': disposal_method
        }
        
        for feature in ['item_type', 'waste_category', 'energy_level', 
                        'pollution_risk', 'recyclable', 'disposal_method']:
            le = self.label_encoders[feature]
            try:
                encoded = le.transform([feature_values[feature]])[0]
            except ValueError:
                # Handle unseen labels
                encoded = 0
            features.append(encoded)
        
        # Scale and predict
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        score_category = self.label_encoders['target'].inverse_transform([prediction])[0]
        
        return {
            'prediction': score_category,
            'probabilities': dict(zip(self.label_encoders['target'].classes_, probabilities))
        }


def main():
    """Main training function"""
    trainer = SustainabilityModelTrainer()
    trainer.load_data()
    accuracy = trainer.train_model(model_type='random_forest')
    model_path = 'model.pkl'
    trainer.save_model(model_path)


if __name__ == '__main__':
    main()

