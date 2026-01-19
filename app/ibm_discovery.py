"""
IBM Watson Discovery Integration
Advanced knowledge retrieval and search for sustainability information
"""

import os
from typing import Dict, List, Optional

try:
    from ibm_watson import DiscoveryV2
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    DISCOVERY_AVAILABLE = True
except ImportError:
    DISCOVERY_AVAILABLE = False


class IBMDiscoveryService:
    """IBM Watson Discovery for enhanced knowledge retrieval"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 service_url: Optional[str] = None,
                 project_id: Optional[str] = None):
        """
        Initialize IBM Watson Discovery
        
        Args:
            api_key: Discovery API key
            service_url: Discovery service URL
            project_id: Discovery project ID
        """
        self.discovery = None
        self.project_id = project_id
        
        if DISCOVERY_AVAILABLE and api_key and service_url:
            try:
                authenticator = IAMAuthenticator(api_key)
                self.discovery = DiscoveryV2(
                    version='2020-08-30',
                    authenticator=authentator
                )
                self.discovery.set_service_url(service_url)
            except Exception as e:
                self.discovery = None
    
    def search(self, query: str, limit: int = 5) -> Dict:
        """
        Search sustainability knowledge base using Discovery
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        if not self.discovery or not self.project_id:
            return {'results': [], 'count': 0}
        
        try:
            response = self.discovery.query(
                project_id=self.project_id,
                natural_language_query=query,
                count=limit,
                highlight=True
            ).get_result()
            
            results = response.get('results', [])
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'text': result.get('text', ''),
                    'relevance': result.get('relevance_score', 0),
                    'url': result.get('url', '')
                })
            
            return {
                'results': formatted_results,
                'count': len(formatted_results),
                'query': query
            }
        except Exception as e:
            return {'results': [], 'count': 0, 'error': str(e)}
    
    def is_available(self) -> bool:
        """Check if Discovery is available"""
        return self.discovery is not None

