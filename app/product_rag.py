"""
Enhanced RAG System for Product Analysis
Analyzes any product using Retrieval-Augmented Generation
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class ProductRAGAnalyzer:
    """RAG-based product analyzer - can analyze ANY product"""
    
    def __init__(self, knowledge_base_path, product_data_path):
        """
        Initialize Product RAG Analyzer
        
        Args:
            knowledge_base_path: Path to sustainability knowledge base
            product_data_path: Path to product dataset
        """
        # Load knowledge base
        self.kb_df = pd.read_csv(knowledge_base_path)
        
        # Load product dataset
        try:
            self.product_df = pd.read_csv(product_data_path)
        except:
            self.product_df = pd.DataFrame()
        
        # Build comprehensive search index
        self.vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1, 2))
        self.vectors = None
        self.all_data = None
        self._build_index()
    
    def _build_index(self):
        """Build comprehensive search index from all data"""
        # Combine knowledge base and product data
        all_items = []
        
        # Add knowledge base topics
        for idx, row in self.kb_df.iterrows():
            item = {
                'type': 'knowledge',
                'text': f"{row.get('topic', '')} {row.get('category', '')} {row.get('information', '')} {row.get('action_items', '')}",
                'data': row.to_dict()
            }
            all_items.append(item)
        
        # Add product data
        if not self.product_df.empty:
            for idx, row in self.product_df.iterrows():
                item = {
                    'type': 'product',
                    'text': f"{row.get('item', '')} {row.get('item_type', '')} {row.get('waste_category', '')} {row.get('disposal_method', '')} {row.get('recommendation', '')} {row.get('environmental_impact', '')}",
                    'data': row.to_dict()
                }
                all_items.append(item)
        
        self.all_data = pd.DataFrame(all_items)
        
        if len(self.all_data) > 0:
            # Vectorize all text
            self.vectors = self.vectorizer.fit_transform(self.all_data['text'])
    
    def analyze_product(self, product_description: str, top_k: int = 5) -> dict:
        """
        Analyze any product using RAG
        
        Args:
            product_description: Product/item description
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with comprehensive analysis
        """
        if self.vectors is None or len(self.all_data) == 0:
            return {
                'error': 'RAG system not properly initialized',
                'product': product_description
            }
        
        # Vectorize query
        query_vector = self.vectorizer.transform([product_description.lower()])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
        
        # Retrieve relevant contexts
        relevant_contexts = []
        product_matches = []
        knowledge_contexts = []
        
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            if similarity_score > 0.01:  # Lower threshold to get more results
                item = self.all_data.iloc[idx]
                context = {
                    'similarity': similarity_score,
                    'type': item['type'],
                    'data': item['data']
                }
                relevant_contexts.append(context)
                
                if item['type'] == 'product':
                    product_matches.append(context)
                else:
                    knowledge_contexts.append(context)
        
        # Sort by similarity
        relevant_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        product_matches.sort(key=lambda x: x['similarity'], reverse=True)
        knowledge_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate confidence based on matches
        max_similarity = float(max(similarities)) if len(similarities) > 0 else 0.0
        # If we extracted info via keyword matching, we still have confidence
        # RAG system can analyze products even without exact matches
        if len(relevant_contexts) > 0:
            confidence = max(max_similarity, 0.3)  # Minimum confidence if matches found
        else:
            # Even without matches, keyword matching provides analysis
            confidence = 0.2  # Lower confidence but still usable
        
        # Build comprehensive analysis
        analysis = {
            'product': product_description,
            'rag_confidence': confidence,
            'matches_found': len(relevant_contexts),
            'product_matches': product_matches[:3],
            'knowledge_contexts': knowledge_contexts[:3]
        }
        
        # Extract sustainability information (always extract, even with low confidence)
        sustainability_info = self._extract_sustainability_info(relevant_contexts, product_description, product_description)
        analysis.update(sustainability_info)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis, relevant_contexts)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _extract_sustainability_info(self, contexts, product_description, original_query):
        """Extract sustainability information from contexts and product description"""
        info = {
            'category': 'Unknown',
            'waste_type': 'Unknown',
            'sustainability_score': 5.0,
            'recyclable': 'Unknown',
            'disposal_method': 'General Waste',
            'environmental_impact': 'Medium',
            'pollution_risk': 'Medium'
        }
        
        product_lower = product_description.lower()
        
        # Look for product matches first
        product_matches = [c for c in contexts if c['type'] == 'product']
        if product_matches:
            best_match = product_matches[0]['data']
            info['category'] = best_match.get('waste_category', 'Unknown')
            info['waste_type'] = best_match.get('item_type', 'Unknown')
            info['sustainability_score'] = best_match.get('sustainability_score', 5.0)
            info['recyclable'] = best_match.get('recyclable', 'Unknown')
            info['disposal_method'] = best_match.get('disposal_method', 'General Waste')
            info['environmental_impact'] = best_match.get('environmental_impact', 'Medium')
            # Infer pollution risk from environmental impact
            impact_map = {'High': 'High', 'Medium': 'Medium', 'Low': 'Low', 'Very High': 'Very High'}
            info['pollution_risk'] = impact_map.get(info['environmental_impact'], 'Medium')
        
        # Enhance with knowledge base information
        knowledge_matches = [c for c in contexts if c['type'] == 'knowledge']
        if knowledge_matches:
            kb_info = knowledge_matches[0]['data']
            topic = kb_info.get('topic', '').lower()
            
            # Map knowledge base topics to product categories
            if 'e-waste' in topic or 'electronic' in topic:
                info['category'] = 'E-Waste'
                info['waste_type'] = 'Electronics'
                info['pollution_risk'] = 'High'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Special Collection'
                info['sustainability_score'] = 3.0
            elif 'plastic' in topic:
                info['category'] = 'Recyclable'
                info['waste_type'] = 'Plastic'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Recycling Bin'
                info['sustainability_score'] = 7.0
            elif 'organic' in topic or 'compost' in topic:
                info['category'] = 'Biodegradable'
                info['waste_type'] = 'Organic'
                info['pollution_risk'] = 'Low'
                info['recyclable'] = 'No'
                info['disposal_method'] = 'Compost Bin'
                info['sustainability_score'] = 8.0
        
        # Fallback keyword matching for ANY product (this is why it can analyze ALL products)
        if info['category'] == 'Unknown':
            # Electronics/E-Waste
            if any(word in product_lower for word in ['battery', 'batteries', 'electronic', 'computer', 'laptop', 'phone', 'tv', 'television', 'circuit', 'charger']):
                info['category'] = 'E-Waste'
                info['waste_type'] = 'Electronics'
                info['pollution_risk'] = 'High'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Special Collection'
                info['sustainability_score'] = 3.0
            # Plastic
            elif any(word in product_lower for word in ['plastic', 'bottle', 'container', 'bag', 'wrap']):
                info['category'] = 'Recyclable'
                info['waste_type'] = 'Plastic'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Recycling Bin'
                info['sustainability_score'] = 7.0
            # Organic
            elif any(word in product_lower for word in ['food', 'organic', 'compost', 'vegetable', 'fruit', 'scrap']):
                info['category'] = 'Biodegradable'
                info['waste_type'] = 'Organic'
                info['pollution_risk'] = 'Low'
                info['recyclable'] = 'No'
                info['disposal_method'] = 'Compost Bin'
                info['sustainability_score'] = 8.0
            # Hazardous/Chemical
            elif any(word in product_lower for word in ['oil', 'paint', 'chemical', 'medicine', 'pharmaceutical', 'hazardous', 'toxic']):
                info['category'] = 'Hazardous'
                info['waste_type'] = 'Chemical'
                info['pollution_risk'] = 'High'
                info['recyclable'] = 'No'
                info['disposal_method'] = 'Special Collection'
                info['sustainability_score'] = 2.0
            # Paper
            elif any(word in product_lower for word in ['paper', 'cardboard', 'newspaper', 'magazine']):
                info['category'] = 'Recyclable'
                info['waste_type'] = 'Paper'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Recycling Bin'
                info['sustainability_score'] = 9.0
            # Glass
            elif any(word in product_lower for word in ['glass', 'jar', 'bottle']):
                info['category'] = 'Recyclable'
                info['waste_type'] = 'Glass'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Recycling Bin'
                info['sustainability_score'] = 8.0
            # Metal
            elif any(word in product_lower for word in ['metal', 'aluminum', 'steel', 'can', 'tin']):
                info['category'] = 'Recyclable'
                info['waste_type'] = 'Metal'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Recycling Bin'
                info['sustainability_score'] = 9.0
            # Textile
            elif any(word in product_lower for word in ['cloth', 'clothing', 'fabric', 'textile', 'garment']):
                info['category'] = 'Reusable'
                info['waste_type'] = 'Textile'
                info['recyclable'] = 'Yes'
                info['disposal_method'] = 'Donation/Recycling'
                info['sustainability_score'] = 7.0
        
        # Categorize sustainability score
        score = info['sustainability_score']
        if score >= 7:
            info['score_category'] = 'High'
        elif score >= 4:
            info['score_category'] = 'Medium'
        else:
            info['score_category'] = 'Low'
        
        return info
    
    def _generate_recommendations(self, analysis, contexts):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Get disposal method
        disposal = analysis.get('disposal_method', 'General Waste')
        recyclable = analysis.get('recyclable', 'Unknown')
        pollution_risk = analysis.get('pollution_risk', 'Medium')
        
        # Primary recommendation
        if recyclable == 'Yes':
            recommendations.append({
                'priority': 'high',
                'action': f'Recycle via {disposal}',
                'reason': 'This item is recyclable and should be recycled to reduce waste'
            })
        elif pollution_risk in ['High', 'Very High']:
            recommendations.append({
                'priority': 'high',
                'action': f'Use {disposal} - Hazardous Waste',
                'reason': 'This item contains hazardous materials and requires special disposal'
            })
        elif analysis.get('category') == 'Biodegradable':
            recommendations.append({
                'priority': 'medium',
                'action': 'Compost or use municipal composting',
                'reason': 'Organic waste can be composted to reduce methane emissions'
            })
        else:
            recommendations.append({
                'priority': 'medium',
                'action': f'Dispose via {disposal}',
                'reason': 'Follow proper disposal guidelines'
            })
        
        # Add recommendations from knowledge base
        for context in contexts[:3]:
            if context['type'] == 'knowledge':
                action_items = context['data'].get('action_items', '')
                if action_items:
                    recommendations.append({
                        'priority': 'low',
                        'action': action_items,
                        'reason': f"Based on {context['data'].get('topic', 'sustainability best practices')}"
                    })
        
        return recommendations
    
    def get_product_insights(self, product_description: str) -> dict:
        """Get comprehensive insights for a product"""
        analysis = self.analyze_product(product_description)
        
        insights = {
            'product': product_description,
            'classification': {
                'category': analysis.get('category'),
                'waste_type': analysis.get('waste_type'),
                'recyclable': analysis.get('recyclable')
            },
            'sustainability': {
                'score': analysis.get('sustainability_score'),
                'category': analysis.get('score_category'),
                'environmental_impact': analysis.get('environmental_impact'),
                'pollution_risk': analysis.get('pollution_risk')
            },
            'disposal': {
                'method': analysis.get('disposal_method'),
                'recommendations': analysis.get('recommendations', [])
            },
            'rag_confidence': analysis.get('rag_confidence', 0.0),
            'sources': {
                'product_matches': len(analysis.get('product_matches', [])),
                'knowledge_contexts': len(analysis.get('knowledge_contexts', []))
            }
        }
        
        return insights

