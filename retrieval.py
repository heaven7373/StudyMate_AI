import streamlit as st
from indexing import indexing
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()

try:
    if 'QDRANT_URL' in st.secrets:
        QDRANT_URL = st.secrets["QDRANT_URL"]
        QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    else:
        QDRANT_URL = os.getenv("QDRANT_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
except Exception:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_url =QDRANT_URL
qdrant_key= QDRANT_API_KEY


# # --- Page Configuration ---
st.set_page_config(page_title="PDF RAG Agent", page_icon="📄")
st.header("📄 PDF RAG Agent")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexing_done" not in st.session_state:
    st.session_state.indexing_done = False

# --- File Uploader and Indexing ---
uploaded_file = st.file_uploader("Choose a PDF file to begin", type="pdf")

if uploaded_file:
    if not st.session_state.indexing_done:
        with st.spinner("Analyzing and indexing the document... This may take a moment."):
            indexing(uploaded_file)
            st.session_state.indexing_done = True
        st.success("Document indexed successfully! You can now ask questions.")
    
    # --- Chat Logic ---
    if query := st.chat_input("Ask a question about your document"):
        # Add user message to history immediately
        st.session_state.messages.append({"role": "user", "content": query})

        # --- Retrieval and AI Response Generation ---
        with st.spinner("Thinking..."):
            client = Groq()
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = QdrantVectorStore.from_existing_collection(
                url=qdrant_url,
                api_key=qdrant_key,
                collection_name="PDF_Rag_Agent",
                embedding=embedding_model
            )

            # Retrieve relevant context
            search_results = vector_db.similarity_search(query=query)
            context = "\n\n".join(
                f"Page Content: {result.page_content}\nPage Number: {result.metadata['page']}"
                for result in search_results
            )

            # Create the system prompt
            SYSTEM_PROMPT = f"""
            You are a helpful AI assistant. Your task is to answer the user's query based ONLY on the provided context from a PDF file.
            Provide a concise answer and cite the page number(s) from where the information was retrieved.
            If the information is not present, state that you cannot find the answer in the document.

            Context:
            {context}
            """

            # Get the response from the AI
            
            response_stream = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ],
                stream=True,
            )
            
            # Combine the streamed chunks into a single response string
            full_response = ""
            for chunk in response_stream:
                full_response += chunk.choices[0].delta.content or ""

            # Add the complete AI response to the history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # --- Display all messages from history ---
    # This loop is now the ONLY place where messages are displayed.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

else:
    # Reset the session state if no file is uploaded
    st.session_state.messages = []
    st.session_state.indexing_done = False



        


    


    

