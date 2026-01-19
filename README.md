# 🌱 AI-Powered Sustainability Web Application

A production-level AI system that helps users make sustainable decisions using Machine Learning, Conversational AI (RAG), and Responsible AI principles. Aligned with UN Sustainable Development Goals (SDG 12, 11, 13).

## 🎯 Project Overview

This application combines:
- **Machine Learning Models** (Random Forest, Logistic Regression, XGBoost) for sustainability classification
- **RAG-Powered Chatbot** for answering sustainability questions
- **Analytics Dashboard** for insights and statistics
- **Mobile-First UI** with modern, responsive design
- **Responsible AI** principles throughout

## 🚀 Features

### 1. ML-Powered Sustainability Classification (Enhanced)
- **Predicts ANY waste type** - not just predefined items
- Uses advanced NLP with IBM Watson NLU (optional)
- 11 waste categories: Electronics, Plastic, Organic, Paper, Glass, Metal, Textile, Hazardous, Medical, Construction, Tires
- 100+ keywords for automatic recognition
- Provides sustainability scores (1-9 scale)
- Explains predictions with confidence scores
- Recommends proper disposal methods

### 2. AI Chatbot with IBM Watson + RAG
- **IBM Watson Assistant integration** for intelligent conversations
- **Watson NLU** for entity extraction and keyword analysis
- Retrieval-Augmented Generation from sustainability knowledge base
- SDG-aligned responses
- Context-aware recommendations
- Falls back to enhanced RAG if Watson not configured

### 3. Analytics Dashboard
- Dataset statistics
- Prediction history
- Category distribution charts
- Environmental impact analysis

### 4. Responsible AI
- No personal data collection
- Transparent predictions with explanations
- Bias-aware dataset design
- Ethical recommendations

## 📁 Project Structure

```
IBM2/
├── dataset/
│   ├── sustainability_data.csv          # ML training dataset
│   └── sustainability_knowledge_base.csv # RAG knowledge base
├── model/
│   ├── train_model.py                    # ML model training script
│   ├── predict.py                        # Prediction module
│   └── model.pkl                         # Trained model (generated)
├── app/
│   ├── app.py                            # Flask application
│   ├── rag.py                            # RAG system implementation
│   └── __init__.py
├── templates/
│   ├── index.html                        # Home page
│   ├── chatbot.html                      # Chatbot interface
│   └── dashboard.html                    # Analytics dashboard
├── static/
│   ├── css/
│   │   ├── style.css                     # Main styles
│   │   ├── chatbot.css                   # Chatbot styles
│   │   └── dashboard.css                 # Dashboard styles
│   └── js/
│       ├── main.js                       # Home page logic
│       ├── chatbot.js                    # Chatbot logic
│       └── dashboard.js                  # Dashboard logic
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Project
```bash
cd IBM2
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3.5: (Optional) Configure IBM Watson
For enhanced chatbot with IBM Watson services:
1. See `IBM_WATSON_SETUP.md` for detailed instructions
2. Copy `.env.example` to `.env` and add your credentials
3. **Note**: The app works without Watson (uses enhanced RAG fallback)

### Step 4: Train the ML Model
```bash
cd model
python train_model.py
cd ..
```

This will:
- Load the sustainability dataset
- Preprocess and encode features
- Train a Random Forest classifier
- Save the model as `model.pkl`
- Display accuracy metrics

### Step 5: Run the Flask Application
```bash
python app/app.py
```

The application will start on `http://localhost:5000`

## 📖 Usage Guide

### Home Page (`/`)
1. Enter an item description (e.g., "Used Battery", "Plastic Bottle")
2. Click "Analyze" to get:
   - Sustainability score (1-9)
   - Environmental impact assessment
   - Disposal recommendations
   - Confidence scores

### Chatbot (`/chatbot`)
1. Ask questions about:
   - Waste disposal methods
   - Recycling guidelines
   - Environmental impact
   - Sustainability best practices
2. Get RAG-powered responses with SDG alignment

### Dashboard (`/dashboard`)
- View dataset statistics
- See prediction history
- Analyze category distributions
- Monitor environmental impact trends

## 🧠 Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: Item type, waste category, energy level, pollution risk, recyclability, disposal method
- **Target**: Sustainability score category (Low: 1-3, Medium: 4-6, High: 7-9)
- **Accuracy**: ~85-95% (depends on dataset)

### Training Process
1. Data loading and exploration
2. Categorical feature encoding (Label Encoding)
3. Feature scaling (StandardScaler)
4. Train-test split (80-20)
5. Model training and evaluation
6. Model serialization (.pkl)

### Supported Models
- ✅ Random Forest (Default)
- ✅ Logistic Regression
- ✅ XGBoost (Optional, requires xgboost package)

## 🔍 RAG System

### Knowledge Base
- Sustainability topics from UN SDG guidelines
- Municipal waste disposal rules
- Environmental impact information
- Best practices and action items

### Retrieval Process
1. User query vectorization (TF-IDF)
2. Cosine similarity search
3. Top-k context retrieval
4. Response generation with sources

## 🌍 SDG Alignment

### Primary SDG: SDG 12 - Responsible Consumption and Production
- Target 12.4: Reduce chemical and waste generation
- Target 12.5: Substantially reduce waste generation

### Secondary SDGs
- **SDG 11**: Sustainable Cities and Communities
- **SDG 13**: Climate Action

## 🤖 Responsible AI Principles

### 1. Privacy First
- No personal data collection
- Anonymous predictions
- No tracking or profiling

### 2. Bias-Aware Design
- Diverse dataset representation
- Transparent feature selection
- Fair classification across categories

### 3. Explainability
- Every prediction includes:
  - Confidence scores
  - Probability distributions
  - Clear explanations
  - Source citations (for chatbot)

### 4. Ethical Recommendations
- Prioritizes environmental protection
- Aligned with UN SDG guidelines
- Promotes sustainable practices

## 📊 API Endpoints

### POST `/api/predict`
Predict sustainability score for an item.

**Request:**
```json
{
  "item": "Used Battery"
}
```

**Response:**
```json
{
  "success": true,
  "item": "Used Battery",
  "prediction": "Low",
  "score_range": [1, 3],
  "score": 2.0,
  "confidence": 0.95,
  "recommendation": "Take to certified e-waste collection center...",
  "environmental_impact": "High"
}
```

### POST `/api/chat`
Chat with AI assistant.

**Request:**
```json
{
  "message": "Where should I throw a used battery?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Electronic waste contains toxic materials...",
  "sources": [{"topic": "E-Waste Disposal", "sdg_target": "SDG 12.4"}]
}
```

### GET `/api/stats`
Get dashboard statistics.

### GET `/api/history`
Get prediction history.

## 🧪 Testing

### Test ML Model
```bash
cd model
python train_model.py
```

### Test Flask App
```bash
python app/app.py
# Visit http://localhost:5000
```

## 🚀 Deployment

### Production Considerations
1. Set `debug=False` in `app.py`
2. Use a production WSGI server (e.g., Gunicorn)
3. Set secure `SECRET_KEY` for Flask sessions
4. Use environment variables for configuration
5. Set up proper logging
6. Use HTTPS
7. Implement rate limiting

### Example with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

## 📝 Dataset Information

### sustainability_data.csv
- 20+ items with sustainability attributes
- Features: item_type, waste_category, energy_level, pollution_risk, recyclable, disposal_method
- Target: sustainability_score (1-9)

### sustainability_knowledge_base.csv
- 10+ sustainability topics
- SDG-aligned information
- Action items and recommendations

## 🐛 Troubleshooting

### Model Not Found
- Ensure `model.pkl` exists in `model/` directory
- Run `python model/train_model.py` first

### Import Errors
- Activate virtual environment
- Install all dependencies: `pip install -r requirements.txt`

### Port Already in Use
- Change port in `app.py`: `app.run(port=5001)`

## 📄 License

This project is created for educational purposes as part of IBM SkillsBuild / 1M1B Virtual Internship.

## 👥 Credits

- Built with Flask, scikit-learn, pandas
- Aligned with UN Sustainable Development Goals
- Designed for IBM SkillsBuild / 1M1B Virtual Internship

## 🔮 Future Enhancements

- [ ] Integration with IBM Watson NLP
- [ ] Image recognition for waste classification
- [ ] Multi-language support
- [ ] User accounts and personalized recommendations
- [ ] Mobile app version
- [ ] Real-time data from municipal APIs
- [ ] Carbon footprint calculator

---

**Built with ❤️ for a Sustainable Future**

