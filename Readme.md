# AI Chatbot Management System

## Description

This project is an AI Chatbot Management System built with Streamlit. It allows users to create and manage multiple chatbots, each with its own knowledge base derived from uploaded PDF documents. The system uses Pinecone for vector storage and retrieval, and Google Generative AI (Gemini) for generating embeddings and conversational responses. It also includes simple tools for calculations and dictionary lookups.

## Features

-   **Chatbot Creation**: Upload PDF files to create a knowledge base for a new chatbot.
-   **Chatbot Interaction**: Select an active chatbot and query it.
-   **RAG Pipeline**: Implements a Retrieval Augmented Generation (RAG) pipeline using Pinecone for semantic search and Google Generative AI for response generation.
-   **Vector Storage**: Utilizes Pinecone for efficient storage and similarity search of text embeddings.
-   **AI Models**: Leverages Google Generative AI for text embeddings (`models/embedding-001`) and chat completions (`gemini-2.0-flash`).
-   **Built-in Tools**:
    -   **Calculator**: Performs simple arithmetic calculations (e.g., "calculate 5 * 3").
    -   **Dictionary**: Provides definitions for words (e.g., "define serendipity").
-   **Dynamic UI**: User-friendly interface built with Streamlit.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py             # Core Streamlit application logic
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py      # Helper functions (e.g., index name sanitization)
│       └── pdf_utils.py    # PDF text extraction utilities
├── .env.example            # Example environment file (users should create .env)
├── .gitignore              # Specifies intentionally untracked files by Git
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── run.py                  # Script to run the Streamlit application
```

## Setup

1.  **Clone the Repository** (if applicable)
    ```bash
    git clone https://github.com/Piyush2510verma/inflera.git
    cd inflera
    ```

2.  **Create and Activate a Virtual Environment** (recommended)
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the project root directory by copying `.env.example` (if provided) or creating it manually. Add the following variables:
    ```env
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # Optional: For customizing the server
    # SERVER_ADDRESS="0.0.0.0"
    # SERVER_PORT="8502"
    ```
    Replace `"YOUR_PINECONE_API_KEY"` and `"YOUR_GOOGLE_API_KEY"` with your actual API keys.

## Running the Application

1.  Execute the `run.py` script:
    ```bash
    streamlit run app/main.py
    ```
2.  The application will typically be accessible at `http://localhost:8502` or the address specified by `SERVER_ADDRESS` and `SERVER_PORT` in your `.env` file or terminal.

## Key Technologies Used

-   **Python**
-   **Streamlit**: For the web application interface.
-   **Pinecone**: Vector database for storing and searching embeddings.
-   **Google Generative AI (Gemini)**: For text embeddings and language model capabilities.
-   **Langchain**: Framework for developing applications powered by language models.
-   **PyMuPDF (fitz)**: For extracting text from PDF files.
-   **Requests**: For making HTTP requests (used by the dictionary tool).
-   **python-dotenv**: For managing environment variables.
