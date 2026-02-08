"""
EVA Pharma AI Customer Support Agent
Built for EVA AI Hackathon - Challenge 03
Author: Muhammed Ahmed Esmail
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="EVA Pharma - AI Support Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE ====================
@st.cache_resource
def load_rag_model():
    """Load the RAG embedding model (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_knowledge_base():
    """Load and process the knowledge base"""
    # Try to load from uploaded file or default
    try:
        kb_articles = pd.read_csv('kb_articles.csv')
    except:
        # Create a sample KB if file not found
        kb_articles = pd.DataFrame({
            'article_id': ['A001', 'A002', 'A003'],
            'title': [
                'Cough Relief Products',
                'Return Policy',
                'Shipping Information'
            ],
            'content': [
                'EVA Soothing Cough Syrup (P123) helps relieve persistent coughs with natural ingredients.',
                'Products can be returned within 30 days with original receipt.',
                'Standard shipping takes 2-3 business days. Express shipping available.'
            ]
        })
    
    # Create searchable text
    kb_articles['searchable'] = (
        'article_id: ' + kb_articles['article_id'] + 
        ' | title: ' + kb_articles['title'] + 
        ' | content: ' + kb_articles['content']
    )
    
    return kb_articles

@st.cache_resource
def create_faiss_index(_rag_model, kb_articles):
    """Create FAISS index for semantic search"""
    kb_list = kb_articles['searchable'].tolist()
    embeddings = _rag_model.encode(kb_list, show_progress_bar=False)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, kb_list

# ==================== CORE FUNCTIONS ====================
def get_relevant_context(query, rag_model, index, kb_list, k=3):
    """Retrieve relevant context from knowledge base"""
    query_vector = rag_model.encode([query])
    distances, indices = index.search(np.array(query_vector).astype('float32'), k=k)
    
    relevant_docs = [kb_list[idx] for idx in indices[0]]
    return relevant_docs

def prompt_convo(max_new_tokens):
    """System prompt for Sarah the support agent"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are Sarah, a customer support representative at EVA Pharma. "
                
                "YOU CAN HELP WITH: "
                "- Product recommendations for symptoms (cough, pain, allergies, etc.) "
                "- Product information, ingredients, usage instructions "
                "- Order tracking, delivery, and shipping questions "
                "- Returns, exchanges, and refunds "
                "- Billing and payment questions "
                "- Side effects, product complaints, or technical issues "
                "- General questions about EVA Pharma services "
                "- Health-related questions that EVA Pharma products can address "
                
                "YOU CANNOT HELP WITH: "
                "- Non-medical topics (sports, weather, general trivia, cooking, etc.) "
                "- Competitor products or other pharmaceutical companies "
                "- Topics completely unrelated to healthcare or EVA Pharma "
                
                "For out-of-scope questions: "
                "Politely say: 'That's outside my area! I focus on EVA Pharma's medical products. "
                "Is there anything health-related I can help you with?' "
                
                "Be friendly, empathetic, and helpful. "
                "When customers describe symptoms, recommend relevant EVA products from the knowledge base. "
                "Ask clarifying questions if needed. "
                
                "Keep responses concise and complete. Finish your thoughts before ending."
            )
        }
    ]
    return messages

def ask_llama_api(client, conversation_history, max_tokens=250):
    """Call Groq API with conversation history"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversation_history,
            max_tokens=max_tokens,
            temperature=0.5
        )
        answer = response.choices[0].message.content
        
        # Debug: Check if response is empty
        if not answer or answer.strip() == "":
            return "I apologize, but I didn't generate a response. Could you please rephrase your question?"
        
        return answer
    except Exception as e:
        error_msg = f"Error communicating with AI: {str(e)}"
        st.error(error_msg)
        return f"I'm having trouble processing your request right now. Error: {str(e)}"

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-header">EVA Pharma AI Support Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Built by Muhammed Ahmed Esmail • EVA AI Hackathon 2026</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input (hidden)
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your free API key at https://console.groq.com"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            max_tokens = st.slider("Max Response Length", 100, 500, 250)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
            num_context_docs = st.slider("Context Documents", 1, 5, 3)
        
        st.divider()
        
        # Info
        st.markdown("### Project Info")
        st.info("""
        **Challenge:** CH-03 AI Customer Support Agent
        
        **Author:** Muhammed A. Esmail
        
        **Features:**
        - RAG-powered responses
        - Conversational memory
        - Semantic search over knowledge base
        
        **Tech Stack:**
        - Llama-3.1-8B (via Groq)
        - Sentence Transformers
        - FAISS vector search
        """)
        
        # Reset button
        if st.button("Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Check API key
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to start chatting.")
        st.info("Get a free API key at: https://console.groq.com")
        st.stop()
    
    # Initialize
    try:
        client = Groq(api_key=api_key)
        rag_model = load_rag_model()
        kb_articles = load_knowledge_base()
        index, kb_list = create_faiss_index(rag_model, kb_articles)
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask Sarah about EVA Pharma products..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get relevant context
        with st.spinner("Searching knowledge base..."):
            context = get_relevant_context(prompt, rag_model, index, kb_list, k=num_context_docs)
        
        # Prepare full conversation history
        conversation_history = prompt_convo(max_tokens).copy()
        
        # Add previous messages (last 4 exchanges for context)
        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages[-8:]:  # Last 4 exchanges (8 messages)
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message with context
        current_user_message = {
            "role": "user",
            "content": f"Context from knowledge base: {context}\n\nCustomer Question: {prompt}"
        }
        conversation_history.append(current_user_message)
        
        # Get response
        with st.spinner("💭 Sarah is thinking..."):
            response = ask_llama_api(client, conversation_history, max_tokens=max_tokens)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Built by <strong>Muhammed Ahmed Esmail</strong> for EVA AI Hackathon 2026</p>
        <p style='font-size: 0.9rem;'>
            <a href='https://github.com/Muhammed-Esmail' target='_blank'>GitHub</a> • 
            <a href='https://linkedin.com/in/muhammed-esmail0' target='_blank'>LinkedIn</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()