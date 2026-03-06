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

All in one file – no external HTML/CSS needed.
Run with: streamlit run chatbot.py
================================================================================
"""

import os
import streamlit as st
import time
import hashlib
from datetime import datetime
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
    st.session_state.ai_provider = os.getenv("AI_PROVIDER", "openai").lower()
if "ai_model" not in st.session_state:
    st.session_state.ai_model = os.getenv("AI_MODEL", "gpt-3.5-turbo").lower()

# -------------------- 2. CHECK DEPENDENCIES --------------------
def check_dependencies():
    """Check which dependencies are available"""
    deps = {
        'openai': False,
        'groq': False,
        'ollama': False,
        'transformers': False,
        'sklearn': False,
        'fpdf': False,
        'torch': False
    }
    
    # Check OpenAI
    try:
        import openai
        deps['openai'] = True
    except ImportError:
        pass
    
    # Check Groq
    try:
        import groq
        deps['groq'] = True
    except ImportError:
        pass
    
    # Check Ollama
    try:
        import ollama
        deps['ollama'] = True
    except ImportError:
        pass
    
    # Check transformers
    try:
        from transformers import pipeline
        deps['transformers'] = True
    except ImportError:
        pass
    
    # Check sklearn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        deps['sklearn'] = True
    except ImportError:
        pass
    
    # Check fpdf
    try:
        from fpdf import FPDF
        deps['fpdf'] = True
    except ImportError:
        pass
    
    # Check torch
    try:
        import torch
        deps['torch'] = True
    except ImportError:
        pass
    
    return deps

deps_available = check_dependencies()

# -------------------- 3. AI PROVIDER SETUP --------------------
AI_PROVIDER = st.session_state.ai_provider
AI_MODEL = st.session_state.ai_model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Available models by provider
AVAILABLE_MODELS = {
    "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "groq": ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
    "ollama": ["llama3.2:3b", "llama3.1:8b", "phi3", "mistral", "llama2"]
}

# Filter providers based on available dependencies
AVAILABLE_PROVIDERS = []
if deps_available['openai']:
    AVAILABLE_PROVIDERS.append("openai")
if deps_available['groq']:
    AVAILABLE_PROVIDERS.append("groq")
if deps_available['ollama']:
    AVAILABLE_PROVIDERS.append("ollama")

if not AVAILABLE_PROVIDERS:
    st.error("No AI providers available. Please install at least one: openai, groq, or ollama")
    st.stop()

# Initialize clients based on provider
if AI_PROVIDER == "openai" and deps_available['openai']:
    import openai
    if not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set. Using fallback responses.")
        openai_client = None
    else:
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
elif AI_PROVIDER == "groq" and deps_available['groq']:
    import groq
    if not GROQ_API_KEY:
        st.warning("GROQ_API_KEY not set. Using fallback responses.")
        groq_client = None
    else:
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
elif AI_PROVIDER == "ollama" and deps_available['ollama']:
    import ollama
    ollama_client = ollama
else:
    st.warning(f"Provider {AI_PROVIDER} not available. Switching to {AVAILABLE_PROVIDERS[0]}")
    st.session_state.ai_provider = AVAILABLE_PROVIDERS[0]
    st.rerun()

# -------------------- 4. ML MODELS (with fallbacks) --------------------
@st.cache_resource
def load_sentiment_pipeline():
    """Load sentiment analysis model if available."""
    if not deps_available['transformers'] or not deps_available['torch']:
        return None
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception:
        return None

@st.cache_resource
def train_intent_classifier():
    """Train intent classifier if sklearn is available."""
    if not deps_available['sklearn']:
        return None, None
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        
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
    except Exception:
        return None, None

sentiment_pipeline = load_sentiment_pipeline() if deps_available['transformers'] else None
vectorizer, intent_clf = train_intent_classifier() if deps_available['sklearn'] else (None, None)

# -------------------- 5. PDF EXPORT --------------------
if deps_available['fpdf']:
    from fpdf import FPDF
    PDF_AVAILABLE = True
else:
    PDF_AVAILABLE = False

# -------------------- 6. HELPER FUNCTIONS --------------------
def predict_intent(text: str) -> str:
    """Fallback intent prediction."""
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
    """Generate a canned response."""
    intent = predict_intent(user_message)
    responses = {
        'greeting': "Hello! How can I assist you today?",
        'goodbye': "Goodbye! Have a great day!",
        'hours': "We are open Monday–Friday, 9am to 5pm (your timezone).",
        'pricing': "Please visit our pricing page at example.com/pricing for details.",
        'support': "I'm sorry you're having trouble. Could you provide more details?"
    }
    return responses.get(intent, "I'm not sure I understand. Could you rephrase?")

def generate_pdf(messages: List[Dict]) -> bytes:
    """Generate PDF from conversation history."""
    if not PDF_AVAILABLE:
        return b"PDF export not available"
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for msg in messages:
            role = "You" if msg["role"] == "user" else "AI"
            pdf.multi_cell(0, 10, f"{role}: {msg['content']}")
        return pdf.output(dest='S').encode('latin1')
    except Exception:
        return b"Error generating PDF"

# -------------------- 7. STREAMING GENERATORS --------------------
def generate_openai(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from OpenAI."""
    if not deps_available['openai'] or not OPENAI_API_KEY:
        yield fallback_response(messages[-1]["content"] if messages else "")
        return
    
    try:
        import openai
        stream = openai.ChatCompletion.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.get("content"):
                yield chunk.choices[0].delta.content
    except Exception:
        yield f"[Error: Unable to get response from OpenAI]"

def generate_groq(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from GROQ."""
    if not deps_available['groq'] or not GROQ_API_KEY:
        yield fallback_response(messages[-1]["content"] if messages else "")
        return
    
    try:
        import groq
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        stream = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        yield f"[Error: Unable to get response from GROQ]"

def generate_ollama(messages: List[Dict]) -> Generator[str, None, None]:
    """Yields tokens from Ollama."""
    if not deps_available['ollama']:
        yield fallback_response(messages[-1]["content"] if messages else "")
        return
    
    try:
        import ollama
        stream = ollama.chat(
            model=AI_MODEL,
            messages=messages,
            stream=True,
            options={
                "num_predict": 200,
                "temperature": 0.3,
                "top_p": 0.9
            }
        )
        for chunk in stream:
            if chunk and "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
    except Exception:
        yield f"[Error: Unable to get response from Ollama]"

def get_ai_response(messages: List[Dict]) -> Generator[str, None, None]:
    """Route to the appropriate generator."""
    if AI_PROVIDER == "openai":
        return generate_openai(messages)
    elif AI_PROVIDER == "groq":
        return generate_groq(messages)
    elif AI_PROVIDER == "ollama":
        return generate_ollama(messages)
    else:
        def fake_stream():
            yield fallback_response(messages[-1]["content"] if messages else "")
        return fake_stream()

# -------------------- 8. PAGE CONFIG & UI --------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
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

# -------------------- 9. SIDEBAR --------------------
with st.sidebar:
    st.title("🤖 Configuration")
    
    # Provider selection (only show available providers)
    if AVAILABLE_PROVIDERS:
        provider = st.selectbox(
            "Select AI Provider",
            options=AVAILABLE_PROVIDERS,
            index=AVAILABLE_PROVIDERS.index(st.session_state.ai_provider) if st.session_state.ai_provider in AVAILABLE_PROVIDERS else 0
        )
        
        if provider != st.session_state.ai_provider:
            st.session_state.ai_provider = provider
            st.rerun()
        
        # Model selection
        available_models = AVAILABLE_MODELS.get(provider, [])
        if available_models:
            model = st.selectbox(
                "Select Model",
                options=available_models,
                index=0
            )
            
            if model != st.session_state.ai_model:
                st.session_state.ai_model = model
                st.rerun()
    
    st.divider()
    
    # API Key status
    st.subheader("🔑 API Status")
    if deps_available['openai']:
        st.write(f"OpenAI: {'✅' if OPENAI_API_KEY else '❌ No API Key'}")
    if deps_available['groq']:
        st.write(f"GROQ: {'✅' if GROQ_API_KEY else '❌ No API Key'}")
    if deps_available['ollama']:
        st.write("Ollama: ✅ Available (local)")
    
    st.divider()
    
    # Stats
    st.subheader("📊 Stats")
    st.write(f"Messages: {st.session_state.stats['messages']}")
    st.write(f"Feedback: {st.session_state.stats['feedback_count']}")

# -------------------- 10. MAIN CHAT UI --------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Header
model_display = AI_MODEL
st.markdown(f"""
<div class="chat-header">
    AI Chat Assistant
    <div class="provider-badge">Powered by {AI_PROVIDER.upper()} | {model_display}</div>
</div>
""", unsafe_allow_html=True)

# Messages area
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

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
            <div class="feedback-buttons" style="display: inline-block; margin-left: 10px;">
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

st.markdown('</div>', unsafe_allow_html=True)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", placeholder="Type your message...", label_visibility="collapsed")
    with col2:
        send = st.form_submit_button("Send", use_container_width=True)

if send and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stats["messages"] += 1
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

    system_prompt = "You are a helpful AI assistant. Be concise and friendly."
    sentiment_label, sentiment_score = analyze_sentiment(last_user_message)
    if sentiment_label == "NEGATIVE" and sentiment_score > 0.8:
        system_prompt += " The user seems upset. Respond with extra empathy."

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
    except Exception:
        full_response = fallback_response(last_user_message)
        message_placeholder.markdown(
            f'<div class="message bot"><div class="avatar">AI</div><div class="bubble">{full_response}</div></div>',
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.stats["messages"] += 1
    st.rerun()

# Footer
st.markdown('<div class="chat-footer">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("🆕 New Chat", key="new_chat", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I help you today?"}]
        st.session_state.feedback = {}
        st.rerun()
with col2:
    if PDF_AVAILABLE and st.button("📥 Export PDF", key="export_pdf", use_container_width=True):
        pdf_bytes = generate_pdf(st.session_state.messages)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- 11. HANDLE FEEDBACK FROM QUERY PARAMETERS --------------------
query_params = st.query_params
if "feedback" in query_params:
    try:
        msg_id = int(query_params.get("msg_id", [0])[0])
        rating = int(query_params.get("rating", [0])[0])
        if msg_id not in st.session_state.feedback:
            st.session_state.feedback[msg_id] = rating
            st.session_state.stats["feedback_count"] += 1
        st.query_params.clear()
        st.rerun()
    except Exception:
        st.query_params.clear()

# -------------------- 12. MAIN GUARD --------------------
if __name__ == "__main__":
    pass