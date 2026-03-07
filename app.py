import streamlit as st
from groq import Groq
from openai import OpenAI

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Llama Chat", page_icon="🦙", layout="centered")

# ------------------------------
# Custom CSS for a modern look
# ------------------------------
st.markdown("""
<style>
    /* Import a nice font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }

    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        border-radius: 30px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 40px 20px;
        background: #f9f9f9;
        border-radius: 20px;
        margin: 20px 0;
    }
    .welcome-icon {
        font-size: 60px;
        margin-bottom: 20px;
    }
    .welcome-title {
        font-size: 2em;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .welcome-subtitle {
        font-size: 1.2em;
        color: #666;
        margin-bottom: 30px;
    }
    .start-chat-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 40px;
        font-size: 1.1em;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    .start-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }

    /* Message bubbles */
    .user-message, .assistant-message {
        padding: 15px 20px;
        border-radius: 25px;
        max-width: 70%;
        word-wrap: break-word;
        margin: 10px 0;
        position: relative;
        animation: fadeIn 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .assistant-message {
        background: #f0f0f0;
        color: #333;
        align-self: flex-start;
        border-bottom-left-radius: 5px;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 50px;
        border: 2px solid #e0e0e0;
        padding: 15px 20px;
        font-size: 1em;
        transition: border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: none;
    }
    .stButton > button {
        border-radius: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        font-weight: 600;
        transition: transform 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load API keys
# ------------------------------
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"Missing API keys: {e}. Please add them to `.streamlit/secrets.toml`.")
    st.stop()

# ------------------------------
# Fixed model and parameters
# ------------------------------
MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 1024
SYSTEM_PROMPT = "You are a helpful assistant."

# ------------------------------
# Session state
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ------------------------------
# UI Layout
# ------------------------------
# Main container with gradient background
st.markdown('<div class="main">', unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display welcome or chat history
if len(st.session_state.messages) == 1:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">🦙✨</div>
        <div class="welcome-title">Our team is here for you</div>
        <div class="welcome-subtitle">Hi, how can we help?</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start a new chat", key="welcome_start", help="Begin a conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        elif msg["role"] == "assistant":
            st.markdown(
                f'<div class="assistant-message">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)  # close chat-container

st.markdown('</div>', unsafe_allow_html=True)  # close main

# ------------------------------
# Chat input
# ------------------------------
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Error: {e}")