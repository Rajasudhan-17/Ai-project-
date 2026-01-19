"""
IBM Watson Chatbot Integration
Uses IBM Watson Assistant and Natural Language Understanding for intelligent responses
"""

import os
from typing import Dict, List, Optional
import json

# Try importing IBM Watson SDK
try:
    from ibm_watson import AssistantV2, NaturalLanguageUnderstandingV1
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False

class IBMWatsonChatbot:
    """IBM Watson-powered chatbot for sustainability queries"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 assistant_url: Optional[str] = None,
                 assistant_id: Optional[str] = None,
                 nlu_api_key: Optional[str] = None,
                 nlu_url: Optional[str] = None):
        """
        Initialize IBM Watson services
        
        Args:
            api_key: IBM Watson Assistant API key
            assistant_url: IBM Watson Assistant service URL
            assistant_id: IBM Watson Assistant Assistant ID
            nlu_api_key: IBM Watson NLU API key
            nlu_url: IBM Watson NLU service URL
        """
        self.watson_available = WATSON_AVAILABLE
        self.assistant = None
        self.nlu = None
        self.assistant_id = assistant_id
        self.session_id = None
        
        if WATSON_AVAILABLE and api_key and assistant_url:
            try:
                # Initialize Watson Assistant
                authenticator = IAMAuthenticator(api_key)
                self.assistant = AssistantV2(
                    version='2021-06-14',
                    authenticator=authenticator
                )
                self.assistant.set_service_url(assistant_url)
                self.assistant_id = assistant_id or 'default'
                
                # Create session
                if self.assistant_id:
                    response = self.assistant.create_session(
                        assistant_id=self.assistant_id
                    ).get_result()
                    self.session_id = response.get('session_id')
            except Exception as e:
                self.assistant = None
        
        if WATSON_AVAILABLE and nlu_api_key and nlu_url:
            try:
                # Initialize Watson NLU
                authenticator = IAMAuthenticator(nlu_api_key)
                self.nlu = NaturalLanguageUnderstandingV1(
                    version='2021-08-01',
                    authenticator=authenticator
                )
                self.nlu.set_service_url(nlu_url)
            except Exception as e:
                self.nlu = None
    
    def analyze_text(self, text: str) -> Dict:
        """Real IBM Watson NLU analysis - extracts entities, keywords, and concepts"""
        if not self.nlu:
            return {}
        
        try:
            # Comprehensive NLU analysis with multiple features
            response = self.nlu.analyze(
                text=text,
                features=Features(
                    entities=EntitiesOptions(emotion=False, sentiment=False, limit=20),
                    keywords=KeywordsOptions(emotion=False, sentiment=False, limit=20),
                    concepts=dict(limit=10)
                )
            ).get_result()
            
            entities = response.get('entities', [])
            keywords = response.get('keywords', [])
            concepts = response.get('concepts', [])
            
            # Extract entity information
            entity_list = []
            entity_types = {}
            for e in entities:
                entity_text = e.get('text', '')
                entity_type = e.get('type', 'Unknown')
                entity_list.append(entity_text)
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity_text)
            
            # Extract keywords with relevance
            keyword_list = []
            keyword_scores = {}
            for k in keywords:
                keyword_text = k.get('text', '')
                relevance = k.get('relevance', 0)
                keyword_list.append(keyword_text)
                keyword_scores[keyword_text] = relevance
            
            # Sort keywords by relevance
            keyword_list_sorted = sorted(keyword_list, key=lambda x: keyword_scores.get(x, 0), reverse=True)
            
            # Extract concepts
            concept_list = [c.get('text', '') for c in concepts]
            
            return {
                'entities': entity_list,
                'keywords': keyword_list_sorted,
                'concepts': concept_list,
                'top_entities': entity_list[:5],
                'top_keywords': keyword_list_sorted[:10],
                'entity_types': entity_types,
                'keyword_scores': keyword_scores
            }
        except Exception as e:
            return {}
    
    def chat(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Send message to IBM Watson Assistant
        
        Args:
            message: User message
            context: Additional context for the conversation
            
        Returns:
            Dictionary with response text and metadata
        """
        if self.assistant and self.session_id and self.assistant_id:
            try:
                response = self.assistant.message(
                    assistant_id=self.assistant_id,
                    session_id=self.session_id,
                    input={
                        'message_type': 'text',
                        'text': message
                    },
                    context=context or {}
                ).get_result()
                
                # Extract response
                output = response.get('output', {})
                generic = output.get('generic', [])
                
                if generic:
                    response_text = '\n'.join([g.get('text', '') for g in generic])
                else:
                    response_text = "I'm here to help with sustainability questions!"
                
                return {
                    'response': response_text,
                    'intents': output.get('intents', []),
                    'entities': output.get('entities', []),
                    'confidence': output.get('intents', [{}])[0].get('confidence', 0.0) if output.get('intents') else 0.0
                }
            except Exception as e:
                return {'response': None, 'error': str(e)}
        
        return {'response': None}
    
    def is_available(self) -> bool:
        """Check if IBM Watson services are available"""
        return self.assistant is not None or self.nlu is not None


class EnhancedWastePredictor:
    """Enhanced waste prediction using NLP and comprehensive classification"""
    
    # Comprehensive waste classification rules
    WASTE_CATEGORIES = {
        'electronics': {
            'keywords': ['battery', 'batteries', 'phone', 'computer', 'laptop', 'tablet', 'tv', 'television', 
                        'electronic', 'e-waste', 'ewaste', 'circuit', 'charger', 'cable', 'wire', 
                        'monitor', 'keyboard', 'mouse', 'printer', 'scanner', 'speaker', 
                        'headphone', 'remote', 'camera', 'gadget', 'device', 'lithium', 'lead'],
            'category': 'E-Waste',
            'type': 'Electronics',
            'pollution_risk': 'High',
            'recyclable': 'Yes',
            'disposal': 'Special Collection',
            'score_range': (1, 4)
        },
        'plastic': {
            'keywords': ['plastic', 'bottle', 'container', 'bag', 'wrap', 'packaging', 
                        'straw', 'cup', 'plate', 'cutlery', 'tupperware', 'polyethylene',
                        'polystyrene', 'styrofoam', 'pvc', 'pet', 'hdpe', 'ldpe'],
            'category': 'Recyclable',
            'type': 'Plastic',
            'pollution_risk': 'Medium',
            'recyclable': 'Yes',
            'disposal': 'Recycling Bin',
            'score_range': (5, 8)
        },
        'organic': {
            'keywords': ['food', 'organic', 'compost', 'vegetable', 'fruit', 'vegetable',
                        'scrap', 'waste', 'leftover', 'peel', 'core', 'seed', 'coffee',
                        'ground', 'tea', 'leaf', 'yard', 'garden', 'grass', 'branch',
                        'flower', 'plant', 'biodegradable', 'decompose'],
            'category': 'Biodegradable',
            'type': 'Organic',
            'pollution_risk': 'Low',
            'recyclable': 'No',
            'disposal': 'Compost Bin',
            'score_range': (7, 9)
        },
        'paper': {
            'keywords': ['paper', 'cardboard', 'newspaper', 'magazine', 'book', 'notebook',
                        'envelope', 'box', 'carton', 'tissue', 'napkin', 'toilet paper',
                        'wrapper', 'receipt', 'flyer', 'brochure', 'catalog'],
            'category': 'Recyclable',
            'type': 'Paper',
            'pollution_risk': 'Low',
            'recyclable': 'Yes',
            'disposal': 'Recycling Bin',
            'score_range': (8, 9)
        },
        'glass': {
            'keywords': ['glass', 'bottle', 'jar', 'container', 'window', 'mirror',
                        'lightbulb', 'ceramic', 'crystal'],
            'category': 'Recyclable',
            'type': 'Glass',
            'pollution_risk': 'Low',
            'recyclable': 'Yes',
            'disposal': 'Recycling Bin',
            'score_range': (7, 9)
        },
        'metal': {
            'keywords': ['metal', 'aluminum', 'steel', 'iron', 'can', 'tin', 'copper',
                        'brass', 'bronze', 'foil', 'aerosol', 'scrap metal'],
            'category': 'Recyclable',
            'type': 'Metal',
            'pollution_risk': 'Low',
            'recyclable': 'Yes',
            'disposal': 'Recycling Bin',
            'score_range': (8, 9)
        },
        'textile': {
            'keywords': ['cloth', 'clothing', 'fabric', 'textile', 'garment', 'shirt',
                        'pants', 'dress', 'shoe', 'towel', 'blanket', 'curtain',
                        'carpet', 'rug', 'yarn', 'thread'],
            'category': 'Reusable',
            'type': 'Textile',
            'pollution_risk': 'Low',
            'recyclable': 'Yes',
            'disposal': 'Donation/Recycling',
            'score_range': (6, 8)
        },
        'hazardous': {
            'keywords': ['chemical', 'paint', 'oil', 'cooking oil', 'solvent', 'medicine',
                        'pharmaceutical', 'pesticide', 'herbicide', 'battery acid',
                        'mercury', 'lead', 'asbestos', 'hazardous', 'toxic', 'poison',
                        'flammable', 'corrosive', 'radioactive', 'used oil', 'motor oil'],
            'category': 'Hazardous',
            'type': 'Chemical',
            'pollution_risk': 'High',
            'recyclable': 'No',
            'disposal': 'Special Collection',
            'score_range': (1, 3)
        },
        'medical': {
            'keywords': ['medicine', 'medication', 'pill', 'syringe', 'needle', 'bandage',
                        'medical', 'pharmaceutical', 'prescription', 'drug', 'expired medicine',
                        'expired medication', 'old medicine'],
            'category': 'Hazardous',
            'type': 'Pharmaceutical',
            'pollution_risk': 'High',
            'recyclable': 'No',
            'disposal': 'Pharmacy Return',
            'score_range': (2, 4)
        },
        'construction': {
            'keywords': ['concrete', 'brick', 'wood', 'lumber', 'drywall', 'insulation',
                        'roofing', 'tile', 'pipe', 'construction', 'demolition'],
            'category': 'Non-Recyclable',
            'type': 'Construction',
            'pollution_risk': 'Medium',
            'recyclable': 'No',
            'disposal': 'Special Collection',
            'score_range': (3, 5)
        },
        'tires': {
            'keywords': ['tire', 'tyre', 'rubber', 'wheel'],
            'category': 'Non-Recyclable',
            'type': 'Rubber',
            'pollution_risk': 'High',
            'recyclable': 'Yes',
            'disposal': 'Special Collection',
            'score_range': (2, 4)
        }
    }
    
    def predict_waste(self, text: str, watson_nlu_result: Optional[Dict] = None) -> Dict:
        """
        Predict waste category and properties from text input
        
        Args:
            text: User input text describing the waste item
            watson_nlu_result: Optional NLU analysis from IBM Watson
            
        Returns:
            Dictionary with waste classification results
        """
        text_lower = text.lower()
        
        # Use Watson NLU entities/keywords if available
        entities = []
        keywords = []
        if watson_nlu_result:
            entities = watson_nlu_result.get('entities', [])
            keywords = watson_nlu_result.get('keywords', [])
        
        # Combine user text with Watson analysis
        search_text = text_lower + ' ' + ' '.join(entities) + ' ' + ' '.join(keywords)
        
        # Score each category
        category_scores = {}
        for cat_name, cat_info in self.WASTE_CATEGORIES.items():
            score = 0
            matched_keywords = []
            
            for keyword in cat_info['keywords']:
                if keyword in search_text:
                    score += 2
                    matched_keywords.append(keyword)
            
            # Bonus for exact matches in entities
            if entities:
                for entity in entities:
                    if any(kw in entity.lower() for kw in cat_info['keywords']):
                        score += 3
            
            if score > 0:
                category_scores[cat_name] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'info': cat_info
                }
        
        # Get top category
        if category_scores:
            top_category = max(category_scores.items(), key=lambda x: x[1]['score'])
            cat_name, cat_data = top_category
            cat_info = cat_data['info']
            
            # Calculate sustainability score
            score_range = cat_info['score_range']
            avg_score = (score_range[0] + score_range[1]) / 2
            
            return {
                'item': text,
                'category': cat_info['category'],
                'type': cat_info['type'],
                'pollution_risk': cat_info['pollution_risk'],
                'recyclable': cat_info['recyclable'],
                'disposal_method': cat_info['disposal'],
                'sustainability_score': avg_score,
                'score_range': score_range,
                'confidence': min(cat_data['score'] / 10.0, 1.0),
                'matched_keywords': cat_data['matched_keywords'],
                'prediction_category': 'High' if avg_score >= 7 else ('Medium' if avg_score >= 4 else 'Low')
            }
        else:
            # Default/unknown waste
            return {
                'item': text,
                'category': 'Other',
                'type': 'Unknown',
                'pollution_risk': 'Medium',
                'recyclable': 'Unknown',
                'disposal_method': 'General Waste',
                'sustainability_score': 5.0,
                'score_range': (4, 6),
                'confidence': 0.5,
                'matched_keywords': [],
                'prediction_category': 'Medium'
            }

