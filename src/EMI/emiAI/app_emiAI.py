import streamlit as st
import sys
import os

# Ensure the script can access parent directory for 'docling' package and project-level dirs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(_SCRIPT_DIR)) # Adds 'knowledge' directory to sys.path

# This MUST be the first Streamlit command
st.set_page_config(page_title="emiAI", layout="wide")

import lancedb
import ollama
from sentence_transformers import SentenceTransformer

# --- Project Root Definition ---
# Assuming this script is in PROJECT_ROOT/knowledge/docling/
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# --- Configuration ---
DB_PATH = os.path.join(_PROJECT_ROOT, "data", "lancedb")
TABLE_NAME = "emiAI_docs"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b" # Make sure this matches your pulled Ollama model
CONTEXT_WINDOW_SIZE = 4096 # Example, adjust based on DeepSeek model
MAX_TOKENS_RESPONSE = 512 # Example, adjust as needed
NUM_RESULTS_TO_RETRIEVE = 5 # Number of chunks to retrieve from LanceDB
SIMILARITY_THRESHOLD = 0.3 # Optional: minimum similarity score for a chunk to be considered

# --- Global Variables / Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_table" not in st.session_state:
    st.session_state.db_table = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# --- Helper Functions ---
@st.cache_resource # Cache the DB connection and model loading
def initialize_database_and_model():
    """Initializes LanceDB connection and loads the embedding model."""
    try:
        db = lancedb.connect(DB_PATH)
        table = db.open_table(TABLE_NAME)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return table, embedding_model
    except Exception as e:
        st.error(f"Error initializing database or embedding model: {e}")
        st.error(f"Please ensure the database exists at '{DB_PATH}' and was created by '0-preprocess_pdfs.py'.")
        st.error(f"Also check if the embedding model '{EMBEDDING_MODEL_NAME}' is accessible.")
        return None, None

def retrieve_relevant_context(query_text, table, embedding_model, top_k=NUM_RESULTS_TO_RETRIEVE):
    """Retrieves relevant text chunks from LanceDB based on the query."""
    if table is None or embedding_model is None:
        return "Error: Database or embedding model not initialized."

    query_vector = embedding_model.encode(query_text).tolist()
    
    try:
        results = table.search(query_vector).limit(top_k).to_df()
        
        # DEBUG: Print DataFrame columns and first row to the terminal
        print(f"DEBUG: DataFrame columns: {results.columns.tolist()}")
        if not results.empty:
            print(f"DEBUG: First row of results (metadata content):\n{results.head(1)['metadata'].values if 'metadata' in results.columns else 'metadata column not found'}")
            print(f"DEBUG: Full first row of results:\n{results.head(1)}")


        if results.empty:
            return "" # No results found

        # Optional: Filter by similarity score if a threshold is set
        if SIMILARITY_THRESHOLD > 0 and '_score' in results.columns:
             results = results[results['_score'] >= SIMILARITY_THRESHOLD]

        if results.empty:
            return "" # No results after filtering

        context_parts = []
        for _, row in results.iterrows():
            # Access metadata as a dictionary from the 'metadata' column
            metadata_dict = row['metadata'] 
            filename = metadata_dict.get('filename', 'Unknown Filename')
            page_numbers_list = metadata_dict.get('page_numbers')

            if page_numbers_list and isinstance(page_numbers_list, list):
                page_numbers_str = ", ".join(map(str, page_numbers_list))
            else:
                page_numbers_str = "N/A"

            context_parts.append(f"Source: {filename} (Page(s): {page_numbers_str})\nContent: {row['text']}\n---")
        
        return "\n\n".join(context_parts)

    except Exception as e:
        # This will print the full traceback to the terminal as well
        print(f"ERROR in retrieve_relevant_context: {e}") 
        import traceback
        traceback.print_exc() 
        st.error(f"Error during context retrieval: {e}")
        return "Error: Could not retrieve context from the database."

def get_chat_response_ollama(messages_history, context):
    """Gets a response from Ollama DeepSeek model with context."""
    try:
        # Prepare the prompt for Ollama
        # The system prompt instructs the model on how to behave.
        # The user prompt includes the retrieved context and the latest user question.
        
        # Find the last user message
        last_user_message = ""
        for msg in reversed(messages_history):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return "Error: Could not find the user's question."

        prompt_with_context = f"""You are emiAI, a helpful AI assistant for students.
Use the following retrieved context from school course documents to answer the student's question.
If the context doesn't provide the answer, state that the information is not found in the provided documents.
Do not make up answers. Be concise and focus on the information from the context.

Retrieved Context:
---
{context}
---

Student's Question: {last_user_message}

Answer:
"""
        # For Ollama, we typically send the full conversation history if the model supports it,
        # or just the latest prompt with context. For simplicity here, we'll send the constructed prompt.
        # If you want to send history, you'd format `messages_history` appropriately.
        
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are emiAI, a helpful AI assistant. Use the provided context to answer questions. If the context is insufficient, say so."},
                {"role": "user", "content": prompt_with_context} # Send the combined prompt
            ],
            options={
                "num_ctx": CONTEXT_WINDOW_SIZE,
                "temperature": 0.3, # Lower for more factual, higher for more creative
                # "num_predict": MAX_TOKENS_RESPONSE # Max tokens for the response
            }
        )
        return response['message']['content']
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        return "Sorry, I encountered an error trying to generate a response."

# --- Main App Logic ---
st.title("📚 emiAI - Your School Document Assistant")
st.caption(f"Powered by DeepSeek ({OLLAMA_MODEL_NAME}) and local course documents.")

# Initialize DB and model if not already done
if st.session_state.db_table is None or st.session_state.embedding_model is None:
    with st.spinner("Initializing knowledge base... Please wait."):
        st.session_state.db_table, st.session_state.embedding_model = initialize_database_and_model()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask emiAI about your course materials..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = "Thinking..." # Placeholder
        message_placeholder = st.empty() # For streaming-like updates
        message_placeholder.markdown(response + "...")
        
        table = st.session_state.db_table
        embedding_model = st.session_state.embedding_model
        
        if table and embedding_model:
            with st.status("Searching documents...", expanded=False) as status_search:
                retrieved_context = retrieve_relevant_context(prompt, table, embedding_model)
                
                if not retrieved_context.strip() or "Error:" in retrieved_context :
                    response = "I couldn't find any relevant information in the documents to answer your question."
                    if "Error:" in retrieved_context:
                        response = retrieved_context # Show the specific error
                    status_search.update(label="No relevant sections found.", state="complete")
                else:
                    status_search.update(label="Found relevant sections. Generating answer...", state="running")
                    
                    # Display retrieved context directly to avoid expander nesting issues
                    st.markdown("---") # Visual separator
                    st.caption("Retrieved Context:")
                    st.markdown(f"<small>{retrieved_context}</small>", unsafe_allow_html=True)
                    st.markdown("---") # Visual separator

                    response = get_chat_response_ollama(st.session_state.messages, retrieved_context)
                    status_search.update(label="Answer generated.", state="complete")
        else:
            response = "Error: Database not initialized. Please check the setup."
            st.markdown(response) # Show error directly if table is None
            
        message_placeholder.markdown(response) # Update with the final response
        st.session_state.messages.append({"role": "assistant", "content": response})

# Add a sidebar note
st.sidebar.header("About emiAI")
st.sidebar.info(
    "emiAI helps you find answers from your school's course documents. "
    "All information is sourced locally from PDFs provided by professors. "
    "The AI uses these documents to answer your questions."
)
st.sidebar.markdown("---")
st.sidebar.caption("Ensure Ollama is running with the DeepSeek model.")
