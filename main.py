import streamlit as st
import os
import shutil
import uuid
import re # For calculation parsing
import requests # For dictionary API
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Assuming pdf_utils and helpers are in app.utils
# Adjust path if necessary based on how Streamlit runs the script.
# If run from Backend/, then app.utils should work.
try:
    from app.utils.pdf_utils import extract_text_from_pdf_path
    from app.utils.helpers import sanitize_index_name
except ImportError:
    # Fallback for local development if 'app' is not in python path directly
    from utils.pdf_utils import extract_text_from_pdf_path
    from utils.helpers import sanitize_index_name


# --- Constants ---
UPLOAD_DIR = "temp_uploads"  # Relative to where run.py executes (project root)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
PINECONE_DIMENSION = 768  # Dimension for models/embedding-001
SIMILARITY_TOP_K = 3

# --- Initialization Function ---
def initialize_clients():
    """Initializes and caches clients and models in Streamlit session state."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        print("--- Running Initialization ---")
        load_dotenv()
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        if not PINECONE_API_KEY:
            st.error("PINECONE_API_KEY environment variable not set.")
            st.stop()
        if not GOOGLE_API_KEY:
            st.error("GOOGLE_API_KEY environment variable not set.")
            st.stop()

        try:
            print("Attempting to configure Google Generative AI...")
            genai.configure(api_key=GOOGLE_API_KEY)
            st.session_state.genai_configured = True
            print("Google Generative AI configured successfully.")
        except Exception as e:
            st.error(f"ERROR configuring Google Generative AI: {e}")
            st.session_state.genai_configured = False
            st.stop()

        try:
            print("Attempting to initialize Pinecone client...")
            st.session_state.pc = Pinecone(api_key=PINECONE_API_KEY)
            print("Pinecone client initialized successfully.")
        except Exception as e:
            st.error(f"ERROR initializing Pinecone client: {e}")
            st.session_state.pc = None
            st.stop()

        try:
            print("Attempting to initialize Google Generative AI Embeddings...")
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Google Generative AI Embeddings initialized successfully.")
        except Exception as e:
            st.error(f"ERROR initializing Google Generative AI Embeddings: {e}")
            st.session_state.embeddings = None
            st.stop()

        st.session_state.chatbot_metadata_store = {}
        st.session_state.current_chatbot_index_name = None
        
        try:
            print("Fetching existing Pinecone indexes...")
            indexes_result = st.session_state.pc.list_indexes()
            # Ensure 'names' attribute exists and is callable, or handle appropriately
            if hasattr(indexes_result, 'names') and callable(indexes_result.names):
                st.session_state.available_pinecone_indexes = indexes_result.names()
            elif isinstance(indexes_result, list) and all(isinstance(item, str) for item in indexes_result): # Fallback for list of strings
                st.session_state.available_pinecone_indexes = indexes_result
            elif hasattr(indexes_result, 'indexes') and isinstance(indexes_result.indexes, list): # For list_indexes().indexes
                 st.session_state.available_pinecone_indexes = [idx.get('name') for idx in indexes_result.indexes if idx.get('name')]
            else: # Default to empty list if structure is unexpected
                st.session_state.available_pinecone_indexes = []
            print(f"Found indexes: {st.session_state.available_pinecone_indexes}")
        except Exception as e:
            st.error(f"ERROR fetching Pinecone indexes: {e}")
            st.session_state.available_pinecone_indexes = []

        st.session_state.initialized = True
        print("Initialized shared state (metadata store, current index name, available indexes).")
        print("--- Initialization Complete ---")

# --- Helper Functions (Adapted from chatbot.py) ---

def create_chatbot_logic(company_name: str, company_industry: str, knowledge_base_file):
    """Logic to create a chatbot knowledge base."""
    if not st.session_state.get("initialized") or \
       st.session_state.pc is None or \
       st.session_state.embeddings is None:
        st.error("System not initialized. Please check logs.")
        return None

    pc_client = st.session_state.pc
    embeddings_model = st.session_state.embeddings

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    index_name = sanitize_index_name(company_name)
    temp_pdf_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{knowledge_base_file.name}")

    try:
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(knowledge_base_file.getbuffer())

        with st.spinner(f"Extracting text from {knowledge_base_file.name}..."):
            document_text = extract_text_from_pdf_path(temp_pdf_path)
            if not document_text.strip():
                st.error("PDF content is empty or could not be extracted.")
                return None
        st.success(f"Extracted {len(document_text)} characters.")

        with st.spinner("Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_text(document_text)
            if not texts:
                st.error("No text chunks generated. Check PDF content and chunking.")
                return None
        st.success(f"Split text into {len(texts)} chunks.")

        with st.spinner(f"Checking/Creating Pinecone index '{index_name}'..."):
            indexes_result = pc_client.list_indexes()
            existing_indexes = indexes_result.names() if hasattr(indexes_result, 'names') and callable(indexes_result.names) else []
            if index_name not in existing_indexes:
                pc_client.create_index(
                    name=index_name,
                    dimension=PINECONE_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                st.info(f"Pinecone index '{index_name}' created.")
            else:
                st.info(f"Pinecone index '{index_name}' already exists.")

        with st.spinner(f"Generating embeddings and upserting to '{index_name}'..."):
            text_embeddings = embeddings_model.embed_documents(texts)
            if len(texts) != len(text_embeddings):
                st.error("Mismatch between text chunks and embeddings count.")
                return None

            vectors_to_upsert = []
            for i, (text_chunk, embedding) in enumerate(zip(texts, text_embeddings)):
                vector_id = f"{index_name}-chunk-{i}"
                vectors_to_upsert.append((vector_id, embedding, {"text": text_chunk}))
            
            index = pc_client.Index(index_name)
            upsert_response = index.upsert(vectors=vectors_to_upsert)
        st.success(f"Successfully upserted {upsert_response.upserted_count} vectors to index '{index_name}'.")

        st.session_state.chatbot_metadata_store[index_name] = {
            "company_name": company_name,
            "company_industry": company_industry
        }
        # Add to available indexes if newly created and not listed yet
        if index_name not in st.session_state.get("available_pinecone_indexes", []):
            st.session_state.available_pinecone_indexes.append(index_name)
            
        st.session_state.current_chatbot_index_name = index_name
        st.success(f"Chatbot for '{company_name}' created and set as active!")
        return index_name

    except Exception as e:
        st.error(f"Error creating chatbot: {e}")
        return None
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if os.path.exists(UPLOAD_DIR) and not os.listdir(UPLOAD_DIR):
            try:
                os.rmdir(UPLOAD_DIR)
            except OSError as e:
                print(f"Warning: Could not remove temp directory {UPLOAD_DIR}: {e}")

# --- Keyword-based Routers ---

def handle_calculation_python(query_text: str):
    """
    Handles simple arithmetic calculations from a query string.
    Example: "calculate 2 + 2" or "what is 5 * 10"
    """
    query_text = query_text.lower()
    # Remove "calculate", "what is", etc.
    query_text = query_text.replace("calculate", "").replace("what is", "").replace("compute", "").strip()

    # Regex to find simple expressions like "number operator number"
    match = re.fullmatch(r"\s*([-+]?\d*\.?\d+)\s*([+\-*/])\s*([-+]?\d*\.?\d+)\s*", query_text)
    
    if not match:
        # Try to evaluate if it's a more complex expression, but be very careful.
        # This is a simplified approach and not a full math expression parser.
        # Avoid direct eval() for security.
        # For now, only handle "num op num"
        return "I can only handle simple calculations like 'number operator number' (e.g., '5 * 3' or '10 / 2'). Please try that format."

    num1_str, operator, num2_str = match.groups()

    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        return "Invalid numbers in calculation."

    result = None
    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 == 0:
            return "Error: Division by zero."
        result = num1 / num2
    else:
        return "Unsupported operator. I can only handle +, -, *, /."

    # Format result to int if it's a whole number
    answer_string = ""
    if result is not None and result == int(result):
        answer_string = f"The result is {int(result)}."
    else:
        answer_string = f"The result is {result}."
    
    return {
        "tool_used": "Python Calculator",
        "final_answer": answer_string,
        "details": f"Input expression: {query_text}" 
    }

def dictionary_tool_api(word: str) -> dict: # Changed return type
    """Fetches definition from api.dictionaryapi.dev, returns structured info"""
    clean_word = word.strip().lower()
    if not clean_word:
        return {"error": "Please provide a word to define."}
    try:
        api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{clean_word}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data and isinstance(data, list):
            for entry in data:
                if 'meanings' in entry and isinstance(entry['meanings'], list):
                    for meaning_item in entry['meanings']: # Renamed to avoid conflict
                        if 'definitions' in meaning_item and isinstance(meaning_item['definitions'], list):
                            for definition_obj in meaning_item['definitions']:
                                if 'definition' in definition_obj:
                                    return { # Return structured data
                                        "definition": definition_obj['definition'],
                                        "word": word,
                                        "raw_response_snippet": data[0] # For debugging or more info
                                    }
            return {"error": f"No definition found for '{word}' in the expected API response structure.", "word": word}
        
        if isinstance(data, dict) and data.get('title') == "No Definitions Found":
            return {"error": f"Sorry, no definitions were found for '{word}' from the API.", "word": word}
            
        return {"error": f"Unexpected response format from dictionary API for '{word}'.", "word": word, "raw_response_snippet": data}

    except requests.exceptions.HTTPError as http_err:
        error_message = f"An HTTP error occurred: {http_err}."
        if http_err.response.status_code == 404:
            error_message = f"Sorry, I couldn't find a definition for '{word}' (API returned 404)."
        return {"error": error_message, "word": word}
    except requests.exceptions.Timeout:
        return {"error": f"The request to the dictionary API timed out for '{word}'.", "word": word}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network request error occurred for '{word}': {req_err}.", "word": word}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        return {"error": f"Error parsing API response for '{word}': {e}.", "word": word}
    except Exception as e:
        return {"error": f"An unexpected error occurred for '{word}': {e}.", "word": word}


def handle_definition_api(query_text: str): # Returns structured dict
    """
    Extracts term from query and uses dictionary_tool_api.
    """
    term_to_define = query_text.lower()
    triggers = ["define ", "definition of ", "what is the meaning of "]
    processed_term = False
    for trigger in triggers:
        if trigger in term_to_define:
            term_to_define = term_to_define.split(trigger, 1)[-1]
            processed_term = True
            break
    
    if not processed_term and term_to_define.startswith("define"): # e.g. user types "define apple"
         term_to_define = term_to_define.replace("define", "", 1).strip()

    term_to_define = term_to_define.replace("?", "").strip()

    if not term_to_define:
        return {
            "tool_used": "Dictionary API",
            "final_answer": "Please specify a term you would like me to define.",
            "details": "No term extracted from query."
        }
    
    dict_response = dictionary_tool_api(term_to_define)
    
    final_answer = dict_response.get("definition", dict_response.get("error", "Could not process definition request."))
    
    return {
        "tool_used": "Dictionary API",
        "final_answer": final_answer,
        "details": f"Term: {term_to_define}. Raw API info (if any): {dict_response.get('raw_response_snippet', 'N/A') if isinstance(dict_response, dict) else 'N/A'}"
    }


def query_chatbot_logic(query: str): # Returns structured dict
    """Logic to query the active chatbot. Routes to specialized handlers or RAG."""
    
    query_lower = query.lower()

    # Keyword-based routing
    if "define" in query_lower or "definition of" in query_lower or \
       (query_lower.startswith("what is") and " meaning of" in query_lower): # "what is the meaning of X"
        return handle_definition_api(query)

    if "calculate" in query_lower or \
       (query_lower.startswith("what is") and re.search(r"[-+]?\d*\.?\d+\s*[+\-*/]\s*[-+]?\d*\.?\d+", query_lower)): # "what is 5+5"
        return handle_calculation_python(query)


    # If no keywords matched, proceed to RAG pipeline
    # RAG Pipeline Error/Guard Checks
    if not st.session_state.get("initialized") or \
       st.session_state.pc is None or \
       st.session_state.embeddings is None or \
       not st.session_state.get("genai_configured"):
        return {
            "tool_used": "System Error",
            "final_answer": "Error: System not initialized or GenAI not configured. Please check logs.",
            "details": "Pre-RAG initialization checks failed."
        }

    active_index_name = st.session_state.get("current_chatbot_index_name")
    if not active_index_name:
        return {
            "tool_used": "User Setup",
            "final_answer": "No active chatbot. Please create or select a chatbot first.",
            "details": "No Pinecone index selected to query."
        }

    pc_client = st.session_state.pc
    embeddings_model = st.session_state.embeddings
    metadata_store = st.session_state.chatbot_metadata_store

    try:
        # Check if active index exists in Pinecone
        # This check is important as list_indexes can be slow if called every time.
        # However, for robustness, ensuring the index exists before querying is good.
        # For now, assume if it's in available_pinecone_indexes and selected, it's likely there.
        # A more robust check could be pc_client.describe_index(active_index_name)
        # but that's another API call.
        
        # The existing check from before:
        all_pinecone_indexes = pc_client.list_indexes().names() if hasattr(pc_client.list_indexes(), 'names') else [] # Simplified
        if active_index_name not in all_pinecone_indexes:
             st.error(f"Active Pinecone index '{active_index_name}' not found. It might have been deleted. Please select another.")
             st.session_state.current_chatbot_index_name = None
             st.session_state.available_pinecone_indexes = all_pinecone_indexes # Refresh list
             return {
                 "tool_used": "System Error",
                 "final_answer": "Error: Active chatbot index not found. Please select from the updated list.",
                 "details": f"Index '{active_index_name}' not in Pinecone."
             }

        metadata = metadata_store.get(active_index_name)
        company_name = metadata.get("company_name", active_index_name) if metadata else active_index_name
        company_industry = metadata.get("company_industry", "general") if metadata else "general"

        with st.spinner(f"Searching knowledge base '{active_index_name}'..."):
            vector_store = PineconeVectorStore(index_name=active_index_name, embedding=embeddings_model)
            retrieved_docs = vector_store.similarity_search(query, k=SIMILARITY_TOP_K)
            context_str = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "None"

        prompt_template = f"""You are a specialized AI assistant for {company_name}, a company operating in the {company_industry} industry. Your primary goal is to answer questions accurately based on the provided knowledge base.

        Follow these instructions carefully:
        1.  **Check Context Relevance:** Examine the "Knowledge Base Context" provided below. Does it directly answer the specific "User Query"?
        2.  **Answer from Relevant Context:** If the context directly answers the query, formulate your answer based ONLY on that context.
        3.  **Answer from General Knowledge (If Context Irrelevant/Missing):** If the "Knowledge Base Context" is "None" OR if it does *not* directly answer the specific User Query, disregard the context. Instead, use your general knowledge to answer, BUT ONLY IF the User Query is clearly related to the {company_industry} industry. If using general knowledge, state that the specific answer wasn't found in the provided documents.
        4.  **Refuse Unrelated Questions:** If the User Query is unrelated to the {company_industry} industry, politely decline to answer, stating you can only assist with {company_industry}-related topics. Do not use irrelevant context or general knowledge for unrelated questions.
        5.  **Be Direct and Concise:** Provide clear answers without unnecessary preamble about evaluating the context.

        Knowledge Base Context:
        ---
        {context_str}
        ---

        User Query: {query}

        Answer:"""

        with st.spinner("Generating response with Gemini..."):
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt_template)
            final_answer_text = response.text
            # Handle cases where response.text might be empty due to safety filters etc.
            if not final_answer_text and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                final_answer_text = f"I apologize, but I encountered an issue generating the response due to: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}."
            elif not final_answer_text:
                 final_answer_text = "I apologize, but I encountered an issue generating the response. The response was empty."


        return {
            "tool_used": "RAG Pipeline",
            "final_answer": final_answer_text,
            "retrieved_context": context_str,
            "details": f"Queried Pinecone index '{active_index_name}' for company '{company_name}'."
        }

    except Exception as e:
        st.error(f"Error querying RAG pipeline: {e}")
        # Ensure context_str is available to be returned even if Gemini call failed
        # It should have been defined in the 'try' block before the Gemini call.
        current_context_str = "Context not available (error might have occurred before retrieval or context was empty)."
        if 'context_str' in locals() and context_str is not None: 
            current_context_str = context_str
        elif 'context_str' in globals() and context_str is not None: # Fallback if in global, less likely
             current_context_str = context_str


        return {
            "tool_used": "RAG Pipeline Error",
            "final_answer": f"An error occurred during RAG processing: {e}",
            "retrieved_context": current_context_str, # Ensure context is passed back
            "details": f"Exception during query of index '{active_index_name}'. Error: {str(e)[:500]}" # Show more of the error
        }

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot Admin", layout="wide")

# Initialize clients and models on first run or if not already initialized
initialize_clients()

st.title("AI Chatbot Management System")

# Section to select active chatbot (if multiple exist)
# Uses available_pinecone_indexes fetched during initialization
if st.session_state.get("available_pinecone_indexes"):
    st.sidebar.subheader("Available Chatbots (Pinecone Indexes)")
    
    # All indexes fetched from Pinecone
    available_indexes_from_pinecone = st.session_state.available_pinecone_indexes
    
    current_selection = st.session_state.get("current_chatbot_index_name")

    # Ensure current_selection is valid among available Pinecone indexes
    if current_selection not in available_indexes_from_pinecone:
        current_selection = None 
        # If current_selection was set but index deleted from Pinecone, clear it
        if st.session_state.get("current_chatbot_index_name"):
             st.session_state.current_chatbot_index_name = None


    # Determine the index for the radio button pre-selection
    radio_index = 0 # Default to first option
    if current_selection and current_selection in available_indexes_from_pinecone:
        try:
            radio_index = available_indexes_from_pinecone.index(current_selection)
        except ValueError: # Should not happen if current_selection is in list
            current_selection = None # Reset if error, select first
            st.session_state.current_chatbot_index_name = None


    selected_bot_index_name = st.sidebar.radio(
        "Select Active Chatbot:",
        options=available_indexes_from_pinecone,
        format_func=lambda x: st.session_state.chatbot_metadata_store.get(x, {}).get("company_name", x), # Show company name if available in session, else index name
        index=radio_index,
        key="active_chatbot_selector"
    )

    if selected_bot_index_name and selected_bot_index_name != st.session_state.get("current_chatbot_index_name"):
        st.session_state.current_chatbot_index_name = selected_bot_index_name
        st.rerun() 

    if st.session_state.get("current_chatbot_index_name"):
        active_display_name = st.session_state.chatbot_metadata_store.get(
            st.session_state.current_chatbot_index_name, {}
        ).get("company_name", st.session_state.current_chatbot_index_name)
        st.sidebar.success(f"Active: {active_display_name}")
    else:
        st.sidebar.warning("No chatbot selected as active. Please select one from the list if available.")
elif st.session_state.get("initialized"): # Initialized but no indexes found
    st.sidebar.info("No Pinecone indexes found or accessible. Create a new chatbot to begin.")


# --- Create Chatbot Section ---
# Expand if no chatbot is currently selected AND (there are no available OR user wants to create new)
expand_create_section = not bool(st.session_state.get("current_chatbot_index_name"))
if not st.session_state.get("available_pinecone_indexes") and st.session_state.get("initialized"):
    expand_create_section = True # Always expand if no indexes exist after init

with st.expander("Create New Chatbot", expanded=expand_create_section):
    st.subheader("1. Create Chatbot Knowledge Base")
    with st.form("create_chatbot_form"):
        company_name = st.text_input("Company Name", placeholder="e.g., Acme Corp")
        company_industry = st.text_input("Company Industry", placeholder="e.g., Software Development")
        knowledge_base_file = st.file_uploader("Upload Knowledge Base (PDF)", type="pdf")
        
        submit_create = st.form_submit_button("Create Chatbot")

        if submit_create:
            if not company_name or not company_industry or not knowledge_base_file:
                st.warning("Please fill in all fields and upload a PDF.")
            else:
                with st.spinner("Processing... Please wait."):
                    created_index_name = create_chatbot_logic(company_name, company_industry, knowledge_base_file)
                    if created_index_name:
                        # Automatically select the newly created bot and rerun
                        st.session_state.current_chatbot_index_name = created_index_name
                        st.rerun()


# --- Query Chatbot Section ---
if st.session_state.get("current_chatbot_index_name"):
    # Gracefully get active_bot_name, defaulting to index name if not in session metadata
    active_bot_name = st.session_state.chatbot_metadata_store.get(
        st.session_state.current_chatbot_index_name, {}
    ).get("company_name", st.session_state.current_chatbot_index_name)
    st.subheader(f"2. Query: {active_bot_name}")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input(f"Ask {active_bot_name}...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            detailed_response_placeholder = st.empty() # For tool info

            with st.spinner("Thinking..."):
                response_data = query_chatbot_logic(user_query) # This now returns a dict

                final_answer = response_data.get("final_answer", "Sorry, I could not process that.")
                tool_used = response_data.get("tool_used", "Unknown")
                details = response_data.get("details", "")
                retrieved_context = response_data.get("retrieved_context")

                message_placeholder.markdown(final_answer)
                
                # Display tool usage and context in an expander
                with detailed_response_placeholder.expander("Processing Details", expanded=False):
                    st.write(f"**Tool Used:** {tool_used}")
                    if details:
                        st.write(f"**Details:** {details}")
                    if retrieved_context:
                        st.text_area("Retrieved Context:", value=retrieved_context, height=200, disabled=True)
                
        # Store the structured response or just the answer for history?
        # For simplicity, let's store the final answer for chat history display.
        # If we want to redisplay details for old messages, session_state.messages needs to store the dict.
        # For now, only new responses show details.
        st.session_state.messages.append({"role": "assistant", "content": final_answer}) 
else:
    st.info("Create or select a chatbot to start querying.")

# --- Health Check (Optional - for completeness) ---
# Streamlit apps don't typically have a / path like FastAPI. The main page serves this.
# st.sidebar.markdown("---")
# st.sidebar.markdown("App Status: Running")
