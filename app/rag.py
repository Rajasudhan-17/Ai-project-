"""
Retrieval-Augmented Generation (RAG) System
Provides context-aware responses using sustainability knowledge base
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class RAGSystem:
    """RAG system for sustainability knowledge retrieval"""
    
    def __init__(self, knowledge_base_path):
        """Initialize RAG system with knowledge base"""
        self.knowledge_base_path = knowledge_base_path
        self.df = pd.read_csv(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.vectors = None
        self._build_index()
    
    def _build_index(self):
        """Build search index from knowledge base"""
        # Combine relevant columns for search
        self.df['search_text'] = (
            self.df['topic'].fillna('') + ' ' +
            self.df['category'].fillna('') + ' ' +
            self.df['information'].fillna('') + ' ' +
            self.df['action_items'].fillna('')
        )
        
        # Vectorize knowledge base
        self.vectors = self.vectorizer.fit_transform(self.df['search_text'])
    
    def _retrieve_relevant_context(self, query, top_k=3):
        """Retrieve top-k most relevant contexts"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant contexts
        contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                contexts.append({
                    'topic': self.df.iloc[idx]['topic'],
                    'information': self.df.iloc[idx]['information'],
                    'action_items': self.df.iloc[idx]['action_items'],
                    'sdg_target': self.df.iloc[idx]['sdg_target'],
                    'similarity': float(similarities[idx])
                })
        
        return contexts
    
    def _generate_response(self, query, contexts):
        """Generate response using retrieved contexts"""
        if not contexts:
            return "I don't have specific information about that topic. Could you rephrase your question?", []
        
        # Build response from contexts
        response_parts = []
        sources = []
        
        # Primary context (most relevant)
        primary = contexts[0]
        response_parts.append(primary['information'])
        
        if primary['action_items']:
            response_parts.append(f"\n\nRecommended actions: {primary['action_items']}")
        
        sources.append({
            'topic': primary['topic'],
            'sdg_target': primary['sdg_target']
        })
        
        # Add additional relevant contexts if available
        if len(contexts) > 1:
            response_parts.append("\n\nAdditional information:")
            for ctx in contexts[1:]:
                if ctx['similarity'] > 0.2:
                    response_parts.append(f"\n• {ctx['topic']}: {ctx['information'][:100]}...")
                    sources.append({
                        'topic': ctx['topic'],
                        'sdg_target': ctx['sdg_target']
                    })
        
        response = ' '.join(response_parts)
        
        # Add SDG alignment note
        if primary['sdg_target']:
            response += f"\n\nThis aligns with {primary['sdg_target']} of the UN Sustainable Development Goals."
        
        return response, sources
    
    def query(self, user_query):
        """Main query function - retrieve and generate"""
        # Retrieve relevant contexts
        contexts = self._retrieve_relevant_context(user_query.lower(), top_k=3)
        
        # Generate response
        response, sources = self._generate_response(user_query, contexts)
        
        return response, sources
    
    def get_knowledge_base_info(self):
        """Get information about the knowledge base"""
        return {
            'total_topics': len(self.df),
            'categories': self.df['category'].unique().tolist(),
            'sdg_targets': self.df['sdg_target'].unique().tolist()
        }

