#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
           ULTIMATE AI CHATBOT – Machine Learning, Deep Learning & LLMs
================================================================================
A single‑file Streamlit application that integrates:

- Multiple LLM backends: Ollama (local), GROQ, OpenAI (streaming)
- Llama 3.2:3b model support
- Conversation memory (last 10 messages)
- Sentiment analysis (Hugging Face transformers)
- Intent classification fallback (scikit‑learn)
- Like/dislike buttons on bot messages
- Extensive logging and error handling

All in one file – no external HTML/CSS needed.
Run with: streamlit run chatbot.py
================================================================================
"""

import os
import streamlit as st
import time
import json
import hashlib
import logging

from datetime import datetime
from io import BytesIO
from typing import List, Dict, Generator, Tuple, Optional

# -------------------- 1. SESSION STATE INITIALIZATION --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can we help?"}]
if "feedback" not in st.session_state:
    st.session_state.feedback = {}          # message_id -> rating (1 or -1)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "stats" not in st.session_state:
    st.session_state.stats = {"messages": 0, "users": 1, "feedback_count": 0}
if "last_message_id" not in st.session_state:
    st.session_state.last_message_id = 0
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = os.getenv("AI_PROVIDER", "ollama").lower()
if "ai_model" not in st.session_state:
    st.session_state.ai_model = os.getenv("AI_MODEL", "llama3.2:3b").lower()

# -------------------- 2. AI PROVIDER SETUP --------------------
AI_PROVIDER = st.session_state.ai_provider
AI_MODEL = st.session_state.ai_model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Available models by provider
AVAILABLE_MODELS = {
    "ollama": ["llama3.2:3b", "llama3.1:8b", "phi3", "mistral", "llama2"],
    "groq": ["llama-3.2-3b-preview", "mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
}

# Initialize clients based on provider
if AI_PROVIDER == "ollama":
    try:
        import ollama
        # Check if llama3.2:3b is available, pull if not
        try:
            ollama_client = ollama
            # Optional: Check if model exists, you might want to pull it
            # ollama_client.pull(AI_MODEL)
        except Exception as e:
            st.warning(f"Model {AI_MODEL} may not be available locally. Please pull it first: ollama pull {AI_MODEL}")
    except ImportError:
        st.error("Ollama not installed. Run: pip install ollama")
        st.stop()
elif AI_PROVIDER == "openai":
    try:
        import openai
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY environment variable not set.")
            st.stop()
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
    except ImportError:
        st.error("OpenAI not installed. Run: pip install openai")
        st.stop()
elif AI_PROVIDER == "groq":
    try:
        import groq
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY environment variable not set.")
            st.stop()
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
    except ImportError:
        st.error("Groq not installed. Run: pip install groq")
        st.stop()
else:
    st.error(f"Unknown AI_PROVIDER: {AI_PROVIDER}. Use 'ollama', 'openai', or 'groq'.")
    st.stop()

# -------------------- 3. MACHINE LEARNING IMPORTS (FALLBACKS) --------------------
try:
    from transformers import pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    ML_AVAILABLE = True
except ImportError:
    st.warning("ML dependencies not available. Install with: pip install transformers scikit-learn")
    ML_AVAILABLE = False

# -------------------- 4. PDF EXPORT --------------------
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("FPDF not available. Install with: pip install fpdf")

# -------------------- 5. LOGGING SETUP --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- 6. CACHED MODELS (for performance) --------------------
@st.cache_resource
def load_sentiment_pipeline():
    """Load sentiment analysis model from Hugging Face."""
    if not ML_AVAILABLE:
        return None
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        logger.error(f"Sentiment pipeline error: {e}")
        return None

@st.cache_resource
def train_intent_classifier():
    """Train a simple intent classifier on the fly."""
    if not ML_AVAILABLE:
        return None, None
    
    try:
        intent_data = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'goodbye': ['bye', 'goodbye', 'see you', 'talk to you later'],
            'hours': ['what are your hours', 'when are you open', 'business hours'],
            'pricing': ['how much', 'pricing', 'cost', 'price', 'subscription'],
            'support': ['help', 'support', 'issue', 'problem', 'contact']
        }
        patterns, labels = [], []
        for intent, pats in intent_data.items():
            for p in pats:
                patterns.append(p.lower())
                labels.append(intent)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform(patterns)
        clf = MultinomialNB()
        clf.fit(X, labels)
        return vectorizer, clf
    except Exception as e:
        logger.error(f"Intent classifier error: {e}")
        return None, None

sentiment_pipeline = load_sentiment_pipeline()
vectorizer, intent_clf = train_intent_classifier()

# -------------------- 7. HELPER FUNCTIONS --------------------
def predict_intent(text: str) -> str:
    """Fallback intent prediction using local ML model."""
    if not intent_clf or not vectorizer:
        return "unknown"
    try:
        X_test = vectorizer.transform([text.lower()])
        return intent_clf.predict(X_test)[0]
    except:
        return "unknown"

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Return sentiment label and confidence."""
    if not sentiment_pipeline:
        return "NEUTRAL", 0.5
    try:
        result = sentiment_pipeline(text)[0]
        return result['label'], result['score']
    except:
        return "NEUTRAL", 0.5

def fallback_response(user_message: str) -> str:
    """Generate a canned response based on intent (used when LLM fails)."""
    intent = predict_intent(user_message)
    responses = {
        'greeting': "Hello! How can I assist you today?",
        'goodbye': "Goodbye! Have a great day!",
        'hours': "We are open Monday–Friday, 9am to 5pm (your timezone).",
        'pricing': "Please visit our pricing page at example.com/pricing for details.",
        'support': "I'm sorry you're having trouble. Could you provide more details?"
    }
    return responses.get(intent, "I'm not sure I understand. Could you rephrase?")

def check_admin_password(password: str) -> bool:
    """Simple password check (in production, use hashed env var)."""
    return hashlib.sha256(password.encode()).hexdigest() == \
           hashlib.sha256("admin123".encode()).hexdigest()

def generate_pdf(messages: List[Dict]) -> bytes:
    """Generate PDF from conversation history."""
    if not PDF_AVAILABLE:
        return b"PDF export not available"
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for msg in messages:
            role = "You" if msg["role"] == "user" else "ST"
            pdf.multi_cell(0, 10, f"{role}: {msg['content']}")
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return b"Error generating PDF"

# -------------------- 8. STREAMING GENERATORS (per provider) --------------------
def generate_ollama(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from Ollama with Llama 3.2:3b."""
    try:
        # Using Llama 3.2:3b model
        stream = ollama.chat(
            model=AI_MODEL,  # Using the selected model
            messages=messages,
            stream=True,
            options={
                "num_predict": 200,  # Slightly increased for Llama
                "temperature": 0.3,
                "top_p": 0.9,
                "stop": ["</s>", "Human:", "Assistant:"]  # Llama-specific stop tokens
            }
        )
        for chunk in stream:
            if chunk and "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama error with {AI_MODEL}: {e}")
        yield f"[Error with {AI_MODEL}: {e}]"

def generate_openai(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from OpenAI."""
    try:
        stream = openai_client.ChatCompletion.create(
            model=AI_MODEL,  # Using the selected model
            messages=messages,
            temperature=0.3,
            max_tokens=200,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.get("content"):
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        yield f"[Error: {e}]"

def generate_groq(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from GROQ with Llama 3.2 support."""
    try:
        # For GROQ, map llama3.2:3b to the correct GROQ model name
        groq_model = AI_MODEL
        if AI_MODEL == "llama3.2:3b":
            groq_model = "llama-3.2-3b-preview"
        
        stream = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"GROQ error: {e}")
        yield f"[Error: {e}]"

def get_ai_response(messages: List[Dict]) -> Generator[str, None, None]:
    """Route to the appropriate generator based on AI_PROVIDER."""
    if AI_PROVIDER == "ollama":
        return generate_ollama(messages)
    elif AI_PROVIDER == "openai":
        return generate_openai(messages)
    elif AI_PROVIDER == "groq":
        return generate_groq(messages)
    else:
        def fake_stream():
            yield fallback_response(messages[-1]["content"] if messages else "")
        return fake_stream()

# -------------------- 9. PAGE CONFIG & CUSTOM CSS --------------------
st.set_page_config(
    page_title="AI Chatbot with Llama 3.2",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Removed Chatway references
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f5;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        overflow: hidden;
    }
    .chat-header {
        background: #007bff;
        color: white;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    .provider-badge {
        font-size: 10px;
        color: #e0e0e0;
        margin-top: 5px;
    }
    .chat-messages {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f9f9f9;
    }
    .message {
        display: flex;
        margin: 10px 0;
    }
    .message.bot {
        justify-content: flex-start;
    }
    .message.user {
        justify-content: flex-end;
    }
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #007bff;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
        font-size: 14px;
    }
    .bubble {
        padding: 8px 12px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .bot .bubble {
        background: white;
    }
    .user .bubble {
        background: #007bff;
        color: white;
    }
    .chat-footer {
        padding: 10px;
        text-align: center;
        font-size: 12px;
        border-top: 1px solid #eee;
    }
    .model-selector {
        padding: 10px;
        background: #f0f2f5;
        border-radius: 5px;
        margin: 10px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 1px solid #ccc;
        padding: 8px 15px;
    }
    .stButton>button {
        background: #007bff;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 15px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background: #0056b3;
    }
    div[data-testid="column"] {
        padding: 0 5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- 10. SIDEBAR FOR MODEL SELECTION --------------------
with st.sidebar:
    st.title("🦙 Model Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "Select AI Provider",
        options=["ollama", "groq", "openai"],
        index=["ollama", "groq", "openai"].index(st.session_state.ai_provider)
    )
    
    if provider != st.session_state.ai_provider:
        st.session_state.ai_provider = provider
        st.rerun()
    
    # Model selection based on provider
    available_models = AVAILABLE_MODELS.get(provider, [])
    
    # Set default model to llama3.2:3b if available
    default_index = 0
    if "llama3.2:3b" in available_models:
        default_index = available_models.index("llama3.2:3b")
    elif "llama-3.2-3b-preview" in available_models:
        default_index = available_models.index("llama-3.2-3b-preview")
    
    model = st.selectbox(
        "Select Model",
        options=available_models,
        index=default_index
    )
    
    if model != st.session_state.ai_model:
        st.session_state.ai_model = model
        st.rerun()
    
    st.divider()
    
    # Model info
    if provider == "ollama" and model == "llama3.2:3b":
        st.info("📌 Make sure you have pulled the model: `ollama pull llama3.2:3b`")
    elif provider == "groq" and model == "llama-3.2-3b-preview":
        st.info("📌 Using Llama 3.2 3B via GROQ API")
    
    # Stats
    st.divider()
    st.subheader("📊 Stats")
    st.write(f"Messages: {st.session_state.stats['messages']}")
    st.write(f"Feedback: {st.session_state.stats['feedback_count']}")

# -------------------- 11. MAIN CHAT UI --------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header with Llama 3.2 info
model_display = "🦙 Llama 3.2 3B" if "llama" in AI_MODEL.lower() else AI_MODEL
st.markdown(f"""
<div class="chat-header">
    AI Chat Assistant
    <div class="provider-badge">Powered by {AI_PROVIDER.upper()} | {model_display}</div>
</div>
""", unsafe_allow_html=True)

# Messages area
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

# Display all messages with feedback buttons for bot messages
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "assistant":
        msg_id = idx
        fb_display = ""
        if msg_id in st.session_state.feedback:
            fb = st.session_state.feedback[msg_id]
            fb_display = "👍" if fb == 1 else "👎"
        st.markdown(f"""
        <div class="message bot">
            <div class="avatar">AI</div>
            <div class="bubble">{msg['content']}</div>
            <div class="feedback-buttons" id="fb-{msg_id}">
                <button onclick="sendFeedback({msg_id}, 1)">👍</button>
                <button onclick="sendFeedback({msg_id}, -1)">👎</button>
                <span>{fb_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message user">
            <div class="bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close chat-messages

# JavaScript to handle feedback (via query parameters)
st.markdown("""
<script>
function sendFeedback(msgId, rating) {
    const url = new URL(window.location.href);
    url.searchParams.set('feedback', '1');
    url.searchParams.set('msg_id', msgId);
    url.searchParams.set('rating', rating);
    window.location.href = url.toString();
}
</script>
""", unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your message...", label_visibility="collapsed")
    with col2:
        send = st.form_submit_button("Send", use_container_width=True)

# Process user input
if send and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stats["messages"] += 1
    logger.info(f"User: {user_input}")
    st.rerun()

# Streaming response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_message = st.session_state.messages[-1]["content"]

    message_placeholder = st.empty()
    message_placeholder.markdown(
        '<div class="message bot"><div class="avatar">AI</div><div class="bubble" id="streaming-bubble"></div></div>',
        unsafe_allow_html=True
    )
    full_response = ""

    # Build context for AI
    system_prompt = "You are a helpful AI assistant. Be concise, friendly, and keep responses under 150 tokens."
    sentiment_label, sentiment_score = analyze_sentiment(last_user_message)
    if sentiment_label == "NEGATIVE" and sentiment_score > 0.8:
        system_prompt += " The user seems upset. Respond with extra empathy and kindness."

    messages_for_ai = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-10:]:
        messages_for_ai.append({"role": msg["role"], "content": msg["content"]})

    try:
        for chunk in get_ai_response(messages_for_ai):
            full_response += chunk
            message_placeholder.markdown(
                f'<div class="message bot"><div class="avatar">AI</div><div class="bubble">{full_response}▌</div></div>',
                unsafe_allow_html=True
            )
        message_placeholder.markdown(
            f'<div class="message bot"><div class="avatar">AI</div><div class="bubble">{full_response}</div></div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        full_response = fallback_response(last_user_message)
        words = full_response.split()
        for i, word in enumerate(words):
            time.sleep(0.05)
            current_text = " ".join(words[:i+1])
            message_placeholder.markdown(
                f'<div class="message bot"><div class="avatar">AI</div><div class="bubble">{current_text}▌</div></div>',
                unsafe_allow_html=True
            )
        message_placeholder.markdown(
            f'<div class="message bot"><div class="avatar">AI</div><div class="bubble">{full_response}</div></div>',
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.stats["messages"] += 1
    logger.info(f"Bot: {full_response}")
    st.rerun()

# Footer - Simplified with only New Chat button
st.markdown('<div class="chat-footer">', unsafe_allow_html=True)
if st.button("🆕 New Chat", key="new_chat", use_container_width=True):
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
    st.session_state.feedback = {}
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)  # close chat-footer

st.markdown('</div>', unsafe_allow_html=True)  # close chat-container

# -------------------- 12. HANDLE FEEDBACK FROM QUERY PARAMETERS --------------------
query_params = st.query_params
if "feedback" in query_params:
    try:
        msg_id = int(query_params.get("msg_id", [0])[0])
        rating = int(query_params.get("rating", [0])[0])
        if msg_id not in st.session_state.feedback:
            st.session_state.feedback[msg_id] = rating
            st.session_state.stats["feedback_count"] += 1
            logger.info(f"Feedback: message {msg_id} rated {rating}")
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        st.query_params.clear()

# -------------------- 13. MAIN GUARD --------------------
if __name__ == "__main__":
    pass