"""
IBM Granite Open-Source LLM Integration
Uses IBM Granite foundation models for conversational AI
Alternative to Watson Assistant - fully open-source
"""

import os
import json
import requests
from typing import Dict, List, Optional

class IBMGraniteChatbot:
    """
    IBM Granite LLM Chatbot
    Uses IBM's open-source Granite models for conversational AI
    Can use IBM watsonx.ai API or Hugging Face models
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 api_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize IBM Granite LLM
        
        Args:
            api_key: IBM watsonx.ai API key (optional, can use Hugging Face)
            api_url: API endpoint URL
            model_name: Granite model name (default: granite-8b-instruct)
        """
        self.api_key = api_key
        self.api_url = api_url or "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        self.model_name = model_name or "ibm/granite-8b-instruct"
        self.conversation_history = []
        self.use_huggingface = False
        
        # Try Hugging Face as fallback (completely open-source, no API key needed)
        if not api_key:
            self.use_huggingface = True
    
    def _create_prompt(self, user_message: str, context: Optional[str] = None) -> str:
        """Create prompt for Granite model"""
        system_prompt = """You are an AI Sustainability Assistant helping users make sustainable decisions. 
You provide information about waste disposal, recycling, environmental impact, and sustainability practices.
You are knowledgeable about UN SDG goals, especially SDG 12 (Responsible Consumption and Production).
Be helpful, accurate, and encourage sustainable practices."""
        
        if context:
            system_prompt += f"\n\nContext: {context}"
        
        prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
"""
        return prompt
    
    def chat(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Chat with IBM Granite LLM
        
        Args:
            message: User message
            context: Additional context (like waste analysis results)
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Add context if available (e.g., waste analysis results)
            context_text = None
            if context:
                if isinstance(context, dict):
                    context_text = json.dumps(context, indent=2)
                else:
                    context_text = str(context)
            
            prompt = self._create_prompt(message, context_text)
            
            # Try IBM watsonx.ai API first if API key is available
            if self.api_key and not self.use_huggingface:
                return self._chat_watsonx(prompt, message)
            else:
                # Use Hugging Face Inference API (completely free, open-source)
                return self._chat_huggingface(prompt, message)
                
        except Exception as e:
            return {
                'response': None,
                'error': str(e),
                'model': 'granite'
            }
    
    def _chat_watsonx(self, prompt: str, original_message: str) -> Dict:
        """Use IBM watsonx.ai API for Granite models"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        payload = {
            "model_id": self.model_name,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('results', [{}])[0].get('generated_text', '')
                
                # Clean up response
                generated_text = generated_text.strip()
                
                return {
                    'response': generated_text,
                    'model': 'granite-watsonx',
                    'confidence': 0.9
                }
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            return self._chat_huggingface(prompt, original_message)
    
    def _chat_huggingface(self, prompt: str, original_message: str) -> Dict:
        """Use Hugging Face Inference API for Granite models (open-source, free)"""
        # Hugging Face API endpoint for Granite models
        hf_api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                hf_api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                else:
                    generated_text = str(result)
                
                # Clean up response
                generated_text = generated_text.strip()
                
                # Remove prompt prefix if present
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, '').strip()
                
                return {
                    'response': generated_text,
                    'model': 'granite-huggingface',
                    'confidence': 0.85
                }
            elif response.status_code == 503:
                # Model is loading, provide fallback response
                return {
                    'response': self._get_fallback_response(original_message),
                    'model': 'granite-fallback',
                    'confidence': 0.7
                }
            else:
                raise Exception(f"API error: {response.status_code}")
                
        except Exception as e:
            return {
                'response': self._get_fallback_response(original_message),
                'model': 'granite-fallback',
                'confidence': 0.6,
                'error': str(e)
            }
    
    def _get_fallback_response(self, message: str) -> str:
        """Fallback response when API is unavailable"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['battery', 'electronic', 'e-waste']):
            return "For electronic waste like batteries, you should take them to a certified e-waste collection center. Batteries contain toxic materials and should never be thrown in regular trash."
        
        elif any(word in message_lower for word in ['plastic', 'bottle', 'container']):
            return "Plastic items should be cleaned and placed in recycling bins if they're recyclable. Check local recycling guidelines for specific plastic types."
        
        elif any(word in message_lower for word in ['recycle', 'recycling']):
            return "Recycling helps reduce waste and conserve resources. Make sure to clean items before recycling and follow local recycling guidelines."
        
        elif any(word in message_lower for word in ['organic', 'food', 'compost']):
            return "Organic waste like food scraps can be composted at home or through municipal composting programs. This reduces methane emissions from landfills."
        
        else:
            return "I'm here to help with sustainability questions! Please ask about waste disposal, recycling, or environmental impact. For specific items, I can analyze their sustainability score and provide disposal recommendations."

