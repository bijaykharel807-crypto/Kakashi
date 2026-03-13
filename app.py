# app.py
import streamlit as st
import requests
import os
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="HubBot", page_icon="🤖", layout="centered")

# ---------- GROQ API KEY (from environment or Streamlit secrets) ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    st.error(
        "Groq API key not found. Please set it as an environment variable "
        "`GROQ_API_KEY` or add it to `.streamlit/secrets.toml`."
    )
    st.stop()

# ---------- CUSTOM CSS (exact look from the image) ----------
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background-color: #f5f8fa;
    }
    /* Chat message container */
    .stChatMessage {
        padding: 0 !important;
    }
    /* User message bubble */
    .stChatMessage[data-testid="chat-message-user"] div[data-testid="chat-message-content"] {
        background-color: #0b2b4a !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 12px 16px !important;
        max-width: 80%;
        margin-left: auto;
    }
    /* Assistant message bubble */
    .stChatMessage[data-testid="chat-message-assistant"] div[data-testid="chat-message-content"] {
        background-color: #f1f3f5 !important;
        color: #1e2a3a !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 12px 16px !important;
        max-width: 80%;
    }
    /* Timestamp style */
    .timestamp {
        font-size: 11px;
        color: #8a9aa8;
        text-align: right;
        margin-top: 6px;
    }
    .user-timestamp {
        color: #b0c4de;
    }
    /* Options container – 4 buttons in a row */
    .option-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 16px 0;
        justify-content: center;
    }
    /* Individual option button (Streamlit button override) */
    div.stButton > button {
        background: white !important;
        border: 1px solid #ccd7e4 !important;
        border-radius: 30px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        color: #1e2a3a !important;
        font-weight: normal !important;
        transition: all 0.2s !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        width: 100% !important;
        justify-content: center !important;
    }
    div.stButton > button:hover {
        background: #e9ecf0 !important;
        border-color: #0b2b4a !important;
    }
    /* Disclaimer and AI warning */
    .disclaimer {
        font-size: 12px;
        color: #5a6b7c;
        margin: 16px 0 8px;
        line-height: 1.4;
        text-align: center;
    }
    .disclaimer a {
        color: #0b2b4a;
        text-decoration: none;
    }
    .ai-warning {
        font-size: 12px;
        color: #8a9aa8;
        font-style: italic;
        margin-bottom: 6px;
        text-align: center;
    }
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE INIT ----------
if "messages" not in st.session_state:
    # Initial bot message with timestamp (exactly as in screenshot)
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Want to add a chatbot (like this one) to your website? I’m an AI bot that’s here to help! 😊\n\n"
                "What would you like to do next?"
            ),
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
    ]
if "first_message_sent" not in st.session_state:
    st.session_state.first_message_sent = False

# ---------- HELPER FUNCTION TO CALL GROQ ----------
def call_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------- HEADER WITH LOGO (safe loading) ----------
# Check if the logo file exists; if not, show a text header
logo_path = "hubspot_logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=150)
else:
    st.markdown("<h2 style='text-align: center; color:#0b2b4a;'>HubBot</h2>", unsafe_allow_html=True)

# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    # Determine avatar
    if msg["role"] == "assistant":
        # Try to use custom bot avatar if it exists
        bot_avatar_path = "bot_avatar.png"
        if os.path.exists(bot_avatar_path):
            avatar = bot_avatar_path
        else:
            avatar = "🤖"
    else:
        avatar = "👤"
    
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Display timestamp if available
        if "timestamp" in msg:
            ts_class = "user-timestamp" if msg["role"] == "user" else ""
            st.markdown(
                f'<div class="timestamp {ts_class}">{msg["timestamp"]}</div>',
                unsafe_allow_html=True
            )

# ---------- OPTIONS BUTTONS (only before first user message) ----------
if not st.session_state.first_message_sent:
    # Use columns to place four buttons in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("☐ Chat with sales", key="opt1", use_container_width=True):
            st.session_state.first_message_sent = True
            user_msg = "Chat with sales"
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": now})
            # Get bot reply
            reply = call_groq(user_msg)
            st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%I:%M %p")})
            st.rerun()
    
    with col2:
        if st.button("🗹 Book a demo", key="opt2", use_container_width=True):
            st.session_state.first_message_sent = True
            user_msg = "Book a demo"
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": now})
            reply = call_groq(user_msg)
            st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%I:%M %p")})
            st.rerun()
    
    with col3:
        if st.button("❌ Get started for free", key="opt3", use_container_width=True):
            st.session_state.first_message_sent = True
            user_msg = "Get started for free"
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": now})
            reply = call_groq(user_msg)
            st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%I:%M %p")})
            st.rerun()
    
    with col4:
        if st.button("☐ Get help with my account", key="opt4", use_container_width=True):
            st.session_state.first_message_sent = True
            user_msg = "Get help with my account"
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_msg, "timestamp": now})
            reply = call_groq(user_msg)
            st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%I:%M %p")})
            st.rerun()
    
    # Disclaimer and AI warning (exactly as in image)
    st.markdown("""
    <div class="disclaimer">
        HubSpot uses the information you provide to us to contact you about our relevant content, products, and services. Check out our <a href="#">privacy policy</a> here.
    </div>
    <div class="ai-warning">AI-generated content may be inaccurate.</div>
    """, unsafe_allow_html=True)

# ---------- CHAT INPUT (free text) ----------
if prompt := st.chat_input("Ask me anything..."):
    # Mark that a message has been sent (hides options)
    st.session_state.first_message_sent = True
    now = datetime.now().strftime("%I:%M %p")
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})
    # Show immediately (rerun will display it)
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.markdown(f'<div class="timestamp user-timestamp">{now}</div>', unsafe_allow_html=True)
    
    # Get bot response
    bot_avatar = "bot_avatar.png" if os.path.exists("bot_avatar.png") else "🤖"
    with st.chat_message("assistant", avatar=bot_avatar):
        with st.spinner("Thinking..."):
            reply = call_groq(prompt)
        st.markdown(reply)
        st.markdown(f'<div class="timestamp">{datetime.now().strftime("%I:%M %p")}</div>', unsafe_allow_html=True)
    
    # Store bot response
    st.session_state.messages.append({"role": "assistant", "content": reply, "timestamp": datetime.now().strftime("%I:%M %p")})
    
    # Force a rerun to update the full chat history
    st.rerun()