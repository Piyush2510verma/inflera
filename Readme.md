# AI Chatbot Management System

 Try it out yourself - https://inflera-mtz3djddpzrczzhn36tjey.streamlit.app/


This project is an AI Chatbot Management System that allows users to create and query chatbots based on PDF knowledge bases. It uses Streamlit for the UI, Pinecone for vector storage, and Google Generative AI (Gemini) for embeddings and responses.

## Architecture Overview

-   **Frontend**: Streamlit provides the user interface for chatbot creation, selection, and interaction.
-   **Backend Logic**: Python scripts handle the application's core functionalities:
    -   `app/main.py`: Contains the main Streamlit application logic, UI elements, and workflow orchestration.
    -   `app/utils/`: Includes helper modules for PDF text extraction (`pdf_utils.py`) and other utilities (`helpers.py`).
    -   `run.py`: A simple script to launch the Streamlit application.
-   **Vector Database**: Pinecone stores and manages the embeddings of text chunks from PDF documents, enabling efficient similarity searches.
-   **Language Models**: Google Generative AI (Gemini models) are used for:
    -   Generating text embeddings (`models/embedding-001`).
    -   Generating conversational responses (`gemini-2.0-flash`) as part of a Retrieval Augmented Generation (RAG) pipeline.

## Key Design Choices

-   **Streamlit for UI**: Chosen for its simplicity and speed in building interactive web applications for Python-based data science and AI projects.
-   **Pinecone for Vector Search**: Selected for its scalable and managed vector database capabilities, crucial for efficient RAG.
-   **Google Generative AI (Gemini)**: Utilized for its powerful embedding models and generative capabilities.
-   **Modular PDF Processing**: PDF text extraction is handled by a separate utility, promoting cleaner code.
-   **RAG Pattern**: The core Q&A functionality relies on the RAG pattern, retrieving relevant context from the knowledge base before generating an answer.
-   **Environment Variables**: API keys and server configurations are managed via a `.env` file for security and flexibility.
-   **Built-in Tools**: Simple calculator and dictionary tools are included as examples of extending chatbot functionality beyond RAG.

## How to Run

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Piyush2510verma/inflera.git
    cd inflera
    ```

2.  **Set Up Environment**:
    -   Create and activate a Python virtual environment (e.g., using `venv`).
    -   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    -   Create a `.env` file in the project root (you can copy/rename `.env.example` if it exists or create one manually) and add your API keys:
        ```env
        PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        # Optional: SERVER_ADDRESS="0.0.0.0"
        # Optional: SERVER_PORT="8502"
        ```

3.  **Run the Application**:
    ```bash
    streamlit run app/main.py
    ```
    Access the application in your browser, typically at `http://localhost:8502`.

## Limitations

-   **Active Chatbot Focus**: The system is designed for querying only one active chatbot at a time. You must select a chatbot from the available list to interact with it.
-   **Embedding Generation Costs (Gemini API)**: Creating new chatbots involves generating text embeddings using the Google Generative AI API. This process can consume API credits. It is recommended to use a paid Gemini API key for extensive chatbot creation, especially with large documents, to avoid hitting free tier limits or incurring unexpected costs. Monitor your API usage accordingly.


![image](https://github.com/user-attachments/assets/cc278725-aec2-4bf3-956f-ec0569a4915b)
![image](https://github.com/user-attachments/assets/f40d4455-17a5-4265-b924-7d717c40b97f)
![image](https://github.com/user-attachments/assets/5076375e-6bbf-4473-8cfd-fd749c03a472)


