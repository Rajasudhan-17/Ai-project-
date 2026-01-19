"""
IBM Watson Visual Recognition / Watsonx.ai Visual Models
For image-based waste classification
"""

import os
import base64
from typing import Dict, Optional
from io import BytesIO

try:
    from ibm_watson import VisualRecognitionV3
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    import requests
    VISUAL_RECOGNITION_AVAILABLE = True
except ImportError:
    VISUAL_RECOGNITION_AVAILABLE = False


class IBMVisualRecognition:
    """IBM Visual Recognition for image-based waste classification"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 service_url: Optional[str] = None):
        """
        Initialize IBM Visual Recognition
        
        Args:
            api_key: Visual Recognition API key
            service_url: Service URL
        """
        self.visual_recognition = None
        
        if VISUAL_RECOGNITION_AVAILABLE and api_key and service_url:
            try:
                authenticator = IAMAuthenticator(api_key)
                self.visual_recognition = VisualRecognitionV3(
                    version='2018-03-19',
                    authenticator=authenticator
                )
                self.visual_recognition.set_service_url(service_url)
            except Exception as e:
                self.visual_recognition = None
    
    def classify_waste_from_image(self, image_data: bytes, image_name: str = "image.jpg") -> Dict:
        """
        Classify waste item from image
        
        Args:
            image_data: Image file bytes
            image_name: Image filename
            
        Returns:
            Dictionary with classification results
        """
        if not self.visual_recognition:
            return {'error': 'Visual Recognition not available'}
        
        try:
            # Classify image
            response = self.visual_recognition.classify(
                images_file=image_data,
                images_filename=image_name,
                threshold=0.5
            ).get_result()
            
            images = response.get('images', [])
            if not images:
                return {'error': 'No classification results'}
            
            classifiers = images[0].get('classifiers', [])
            
            # Extract relevant classifications
            waste_categories = []
            for classifier in classifiers:
                classes = classifier.get('classes', [])
                for cls in classes:
                    class_name = cls.get('class', '')
                    score = cls.get('score', 0)
                    
                    # Filter for waste-related classifications
                    waste_keywords = ['plastic', 'paper', 'glass', 'metal', 'electronic', 
                                    'battery', 'bottle', 'container', 'waste', 'trash']
                    if any(keyword in class_name.lower() for keyword in waste_keywords):
                        waste_categories.append({
                            'category': class_name,
                            'confidence': score
                        })
            
            return {
                'waste_categories': waste_categories[:5],
                'all_classifications': classes[:10],
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def classify_from_url(self, image_url: str) -> Dict:
        """Classify waste from image URL"""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                return self.classify_waste_from_image(response.content, "remote_image.jpg")
            else:
                return {'error': f'Failed to download image: {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def is_available(self) -> bool:
        """Check if Visual Recognition is available"""
        return self.visual_recognition is not None

