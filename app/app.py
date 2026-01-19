"""
Flask Application - AI-Powered Sustainability Web App
Main application file with routes, ML inference, and chatbot
"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'model'))
# Add current directory (app) to path for local imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from model.predict import SustainabilityPredictor
from rag import RAGSystem
from product_rag import ProductRAGAnalyzer  # RAG-based product analyzer
from watson_chatbot import EnhancedWastePredictor  # Keep enhanced predictor
from ibm_granite import IBMGraniteChatbot  # Open-source IBM Granite LLM
from ibm_discovery import IBMDiscoveryService  # IBM Watson Discovery
from ibm_visual_recognition import IBMVisualRecognition  # Image classification
from dotenv import load_dotenv

# Get the base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = 'sustainability-app-secret-key-2024'  # Change in production

# Load environment variables for IBM Watson
load_dotenv()

# Initialize IBM Granite LLM (Open-Source IBM Technology)
granite_chatbot = None
try:
    granite_api_key = os.getenv('IBM_GRANITE_API_KEY')  # Optional - can use Hugging Face
    granite_api_url = os.getenv('IBM_GRANITE_API_URL')
    granite_model = os.getenv('IBM_GRANITE_MODEL', 'ibm/granite-8b-instruct')
    
    granite_chatbot = IBMGraniteChatbot(
        api_key=granite_api_key,
        api_url=granite_api_url,
        model_name=granite_model
    )
except Exception as e:
    pass
    granite_chatbot = None

# Initialize IBM Watson NLU (for text analysis)
watson_nlu = None
try:
    nlu_api_key = os.getenv('WATSON_NLU_API_KEY')
    nlu_url = os.getenv('WATSON_NLU_URL')
    
    if nlu_api_key and nlu_url:
        from watson_chatbot import IBMWatsonChatbot
        watson_nlu = IBMWatsonChatbot(nlu_api_key=nlu_api_key, nlu_url=nlu_url)
        if watson_nlu.nlu:
            pass
        else:
            watson_nlu = None
except Exception as e:
    pass
    watson_nlu = None

# Initialize IBM Watson Discovery (for enhanced knowledge retrieval)
discovery_service = None
try:
    discovery_api_key = os.getenv('WATSON_DISCOVERY_API_KEY')
    discovery_url = os.getenv('WATSON_DISCOVERY_URL')
    discovery_project_id = os.getenv('WATSON_DISCOVERY_PROJECT_ID')
    
    discovery_service = IBMDiscoveryService(
        api_key=discovery_api_key,
        service_url=discovery_url,
        project_id=discovery_project_id
    )
    if discovery_service.is_available():
        pass
except Exception as e:
    pass
    discovery_service = None

# Initialize IBM Visual Recognition (for image classification)
visual_recognition = None
try:
    visual_api_key = os.getenv('WATSON_VISUAL_API_KEY')
    visual_url = os.getenv('WATSON_VISUAL_URL')
    
    visual_recognition = IBMVisualRecognition(
        api_key=visual_api_key,
        service_url=visual_url
    )
    if visual_recognition.is_available():
        pass
except Exception as e:
    pass
    visual_recognition = None

# Initialize Enhanced Waste Predictor (works with any waste type)
enhanced_predictor = EnhancedWastePredictor()

# Initialize ML predictor
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'model', 'model.pkl')
    predictor = SustainabilityPredictor(model_path)
except Exception as e:
    pass
    predictor = None

# Initialize RAG system (for chatbot)
try:
    knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'dataset', 'sustainability_knowledge_base.csv')
    rag_system = RAGSystem(knowledge_base_path)
except Exception as e:
    pass
    rag_system = None

# Initialize Product RAG Analyzer (for product analysis - can analyze ANY product)
product_rag_analyzer = None
try:
    knowledge_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'dataset', 'sustainability_knowledge_base.csv')
    product_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'dataset', 'sustainability_data.csv')
    product_rag_analyzer = ProductRAGAnalyzer(knowledge_base_path, product_data_path)
except Exception as e:
    pass
    product_rag_analyzer = None

# Initialize prediction history in before_request
@app.before_request
def initialize_session():
    """Initialize session variables if they don't exist"""
    if 'prediction_history' not in session:
        session['prediction_history'] = []


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    """Chatbot page"""
    return render_template('chatbot.html')


@app.route('/dashboard')
def dashboard():
    """Admin/Analytics dashboard"""
    return render_template('dashboard.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint - predicts ANY waste type using NLP"""
    try:
        data = request.json
        item_text = data.get('item', '').strip()
        
        if not item_text:
            return jsonify({'error': 'Item description is required'}), 400
        
        # Step 1: Use RAG Model to analyze the product (Primary method - can analyze ANY product)
        rag_analysis = None
        using_rag = False
        if product_rag_analyzer:
            try:
                rag_analysis = product_rag_analyzer.analyze_product(item_text, top_k=5)
                using_rag = True
            except Exception as e:
                pass
        
        # Step 2: Use REAL IBM Watson NLU for text analysis (Optional enhancement)
        watson_nlu_result = None
        using_watson_nlu = False
        if watson_nlu and watson_nlu.nlu:
            try:
                watson_nlu_result = watson_nlu.analyze_text(item_text)
                using_watson_nlu = True
            except Exception as e:
                pass
        
        # Step 3: Get prediction - ALWAYS use RAG analysis (it can analyze ANY product)
        # RAG system uses knowledge base + keyword matching to analyze ALL products
        use_rag_primary = True  # Always use RAG for product analysis
        
        if rag_analysis:
            # Use RAG analysis results (RAG can analyze ANY product using knowledge base + keywords)
            rag_score = rag_analysis.get('sustainability_score', 5.0)
            # Use RAG confidence if available, otherwise use minimum confidence (RAG always provides analysis)
            rag_conf = rag_analysis.get('rag_confidence', 0.0)
            if rag_conf < 0.1 and rag_analysis.get('category') != 'Unknown':
                # If RAG categorized via keyword matching, boost confidence
                rag_conf = 0.5
            
            result = {
                'item': item_text,
                'category': rag_analysis.get('category', 'Unknown'),
                'type': rag_analysis.get('waste_type', 'Unknown'),
                'pollution_risk': rag_analysis.get('pollution_risk', 'Medium'),
                'recyclable': rag_analysis.get('recyclable', 'Unknown'),
                'disposal_method': rag_analysis.get('disposal_method', 'General Waste'),
                'sustainability_score': rag_score,
                'score_range': (1, 3) if rag_score < 4 else ((4, 6) if rag_score < 7 else (7, 9)),
                'confidence': max(rag_conf, 0.5),  # RAG always provides analysis
                'matched_keywords': [],
                'prediction_category': rag_analysis.get('score_category', 'Medium'),
                'rag_based': True
            }
        else:
            # Fallback to enhanced predictor if RAG unavailable
            result = enhanced_predictor.predict_waste(item_text, watson_nlu_result)
            result['rag_based'] = False
        
        # Generate recommendation (use RAG recommendations if available)
        if rag_analysis and rag_analysis.get('recommendations'):
            rag_recs = rag_analysis.get('recommendations', [])
            if rag_recs:
                recommendation = rag_recs[0].get('action', '')
                recommendation += f" {rag_recs[0].get('reason', '')}"
            else:
                recommendation = f"Based on RAG analysis, this is classified as {result['type']} ({result['category']}). "
                recommendation += f"Disposal method: {result['disposal_method']}."
        else:
            # Standard recommendation
            recommendation = f"Based on analysis, this is classified as {result['type']} ({result['category']}). "
            if result['recyclable'] == 'Yes':
                recommendation += f"Please {result['disposal_method'].lower()}. This item is recyclable!"
            elif result['pollution_risk'] == 'High' or result['pollution_risk'] == 'Very High':
                recommendation += f"⚠️ CAUTION: This is hazardous waste. {result['disposal_method']} required. "
                recommendation += "Never dispose in regular trash or down drains."
            else:
                recommendation += f"Disposal method: {result['disposal_method']}."
        
        # Store in history
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'item': item_text,
            'prediction': result['prediction_category'],
            'score_range': result['score_range'],
            'confidence': result['confidence'],
            'category': result['category'],
            'type': result['type']
        }
        session['prediction_history'].append(prediction_entry)
        session.modified = True
        
        # Build comprehensive response with RAG insights
        response = {
            'success': True,
            'item': item_text,
            'prediction': result['prediction_category'],
            'score_range': result['score_range'],
            'score': result['sustainability_score'],
            'confidence': result['confidence'],
            'category': result['category'],
            'type': result['type'],
            'pollution_risk': result['pollution_risk'],
            'recyclable': result['recyclable'],
            'disposal_method': result['disposal_method'],
            'recommendation': recommendation,
            'environmental_impact': result['pollution_risk'],
            'explanation': f"This item is classified as {result['type']} ({result['category']}) with a {result['prediction_category']} sustainability score ({result['score_range'][0]}-{result['score_range'][1]}). "
                          f"Environmental impact: {result['pollution_risk']}. {recommendation}",
            'matched_keywords': result.get('matched_keywords', []),
            'using_rag_model': using_rag and result.get('rag_based', False),
            'rag_confidence': rag_analysis.get('rag_confidence', 0.0) if rag_analysis else 0.0,
            'rag_sources': {
                'product_matches': len(rag_analysis.get('product_matches', [])) if rag_analysis else 0,
                'knowledge_contexts': len(rag_analysis.get('knowledge_contexts', [])) if rag_analysis else 0
            } if rag_analysis else {},
            'rag_recommendations': rag_analysis.get('recommendations', []) if rag_analysis else [],
            'probabilities': result.get('probabilities', None),  # Only include if available
            'using_watson_nlu': using_watson_nlu,
            'watson_entities': watson_nlu_result.get('entities', []) if watson_nlu_result else [],
            'watson_keywords': watson_nlu_result.get('keywords', []) if watson_nlu_result else []
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Real IBM Watson Chatbot endpoint with NLU analysis"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        response_text = None
        sources = []
        using_ibm_services = False
        watson_nlu_result = None
        discovery_results = None
        
        # Step 1: Use IBM Watson NLU for text analysis (if available)
        if watson_nlu and watson_nlu.nlu:
            try:
                watson_nlu_result = watson_nlu.analyze_text(user_message)
                using_ibm_services = True
            except Exception as e:
                pass
        
        # Step 2: Use IBM Watson Discovery for enhanced knowledge retrieval (if available)
        if discovery_service and discovery_service.is_available():
            try:
                discovery_results = discovery_service.search(user_message, limit=3)
                if discovery_results.get('count', 0) > 0:
                    using_ibm_services = True
                    sources = discovery_results.get('results', [])
            except Exception as e:
                pass
        
        # Step 3: Use IBM Granite LLM for conversational response (Open-Source)
        using_granite = False
        granite_response = None
        if granite_chatbot:
            try:
                # Prepare context from Discovery and NLU
                context_data = {}
                if discovery_results:
                    context_data['discovery_results'] = discovery_results
                if watson_nlu_result:
                    context_data['nlu_analysis'] = watson_nlu_result
                
                granite_response = granite_chatbot.chat(user_message, context_data)
                if granite_response.get('response'):
                    response_text = granite_response['response']
                    using_granite = True
                    using_ibm_services = True
            except Exception as e:
                pass
        
        # Step 4: Fallback to enhanced RAG if Granite didn't provide response
        if not response_text:
            if rag_system:
                response_text, sources = rag_system.query(user_message)
            else:
                response_text = "I'm here to help with sustainability questions. Please ask about waste disposal, recycling, or environmental impact."
                sources = []
        
        # Step 5: Detect waste-related questions and provide enhanced analysis
        waste_indicators = ['dispose', 'throw', 'recycle', 'waste', 'trash', 'item', 'where', 'how', 'what', 'which']
        detected_waste_items = []
        
        # Extract waste items from Watson NLU entities if available
        if watson_nlu_result:
            entities = watson_nlu_result.get('entities', [])
            keywords = watson_nlu_result.get('keywords', [])
            # Look for waste-related entities
            for entity in entities:
                if any(w in entity.lower() for w in ['waste', 'trash', 'garbage', 'item', 'product', 'material']):
                    detected_waste_items.append(entity)
            
            # Also check keywords for waste items
            for keyword in keywords[:10]:  # Check top 10 keywords
                keyword_lower = keyword.lower()
                # Check if keyword might be a waste item
                if any(waste_word in keyword_lower for waste_word in ['battery', 'plastic', 'bottle', 'electronic', 'glass', 'paper', 'metal']):
                    if keyword not in detected_waste_items:
                        detected_waste_items.append(keyword)
        
        is_waste_question = any(indicator in user_message.lower() for indicator in waste_indicators)
        
        if is_waste_question or detected_waste_items:
            # Use REAL Watson NLU result for enhanced prediction
            try:
                # Combine detected items with original message
                analysis_text = user_message
                if detected_waste_items:
                    analysis_text += ' ' + ' '.join(detected_waste_items)
                
                # Get REAL enhanced prediction using Watson NLU analysis
                pred_result = enhanced_predictor.predict_waste(analysis_text, watson_nlu_result)
                
                # Create comprehensive analysis response
                pred_info = "\n\n🔍 **Real-Time Sustainability Analysis:**\n"
                pred_info += f"• **Item Type:** {pred_result['type']}\n"
                pred_info += f"• **Category:** {pred_result['category']}\n"
                pred_info += f"• **Sustainability Score:** {pred_result['prediction_category']} ({pred_result['score_range'][0]}-{pred_result['score_range'][1]}/9)\n"
                pred_info += f"• **Recyclable:** {pred_result['recyclable']}\n"
                pred_info += f"• **Disposal Method:** {pred_result['disposal_method']}\n"
                pred_info += f"• **Environmental Impact:** {pred_result['pollution_risk']}\n"
                pred_info += f"• **Confidence:** {pred_result['confidence']:.1%}\n"
                
                # Add Watson NLU insights if available
                if watson_nlu_result:
                    if watson_nlu_result.get('entities'):
                        pred_info += f"\n📊 **Detected Entities:** {', '.join(watson_nlu_result['entities'][:5])}\n"
                    if watson_nlu_result.get('keywords'):
                        pred_info += f"🔑 **Key Terms:** {', '.join(watson_nlu_result['keywords'][:5])}\n"
                
                response_text += pred_info
                
                # Add recommendation
                if pred_result['recyclable'] == 'Yes' and pred_result['pollution_risk'] not in ['High', 'Very High']:
                    response_text += "\n✅ **Recommendation:** This item should be recycled. Please clean it and place in the appropriate recycling bin."
                elif pred_result['pollution_risk'] in ['High', 'Very High']:
                    response_text += f"\n⚠️ **Important:** This is hazardous waste. Use {pred_result['disposal_method']}. Never dispose in regular trash."
                
            except Exception as e:
                pass
        
        response = {
            'success': True,
            'response': response_text,
            'sources': sources if isinstance(sources, list) else [],
            'timestamp': datetime.now().isoformat(),
            'using_ibm_services': using_ibm_services,
            'using_granite_llm': using_granite,
            'granite_model': granite_response.get('model', '') if granite_response else '',
            'watson_nlu_used': watson_nlu_result is not None,
            'discovery_used': discovery_results is not None,
            'entities_detected': detected_waste_items,
            'granite_confidence': granite_response.get('confidence', 0.0) if granite_response else 0.0
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        history = session.get('prediction_history', [])
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics for dashboard"""
    try:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   'dataset', 'sustainability_data.csv')
        df = pd.read_csv(dataset_path)
        
        # Calculate statistics
        stats = {
            'total_items': len(df),
            'categories': df['waste_category'].value_counts().to_dict(),
            'recyclable_count': int(df['recyclable'].value_counts().get('Yes', 0)),
            'avg_sustainability_score': float(df['sustainability_score'].mean()),
            'environmental_impact_dist': df['environmental_impact'].value_counts().to_dict(),
            'prediction_count': len(session.get('prediction_history', []))
        }
        
        return jsonify({'success': True, 'stats': stats})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    try:
        session['prediction_history'] = []
        session.modified = True
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

