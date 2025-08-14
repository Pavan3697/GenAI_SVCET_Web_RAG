import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.documents import Document
import time

# Load environment variables for local development (will be ignored by Streamlit Cloud)
load_dotenv()

# Set up API keys using Streamlit's secrets management
# For local development, these will be read from .streamlit/secrets.toml
# On Streamlit Cloud, they will be read from the repository's secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
os.environ["SERPAPI_API_KEY"] = st.secrets.get("SERPAPI_API_KEY")


st.set_page_config(page_title="Web RAG with Groq & SerpApi", layout="wide")
st.image("PragyanAI_Transperent_github.png")
st.title("Web RAG: Q&A with Website Content and Google Search")

# Initialize session state and embeddings model
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Sidebar for user input
with st.sidebar:
    st.header("Input Source")
    website_url = st.text_input("Enter Website URL to scrape")
    
    if st.button("Process Website"):
        if not website_url:
            st.warning("Please enter a website URL.")
        else:
            with st.spinner("Processing website..."):
                try:
                    loader = WebBaseLoader(website_url)
                    web_docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    final_documents = text_splitter.split_documents(web_docs)

                    st.session_state.vector = FAISS.from_documents(final_documents, st.session_state.embeddings)
                    st.success("Website processed successfully!")
                except Exception as e:
                    st.error(f"Failed to scrape or process website: {e}")


# Main chat interface
st.header("Chat with the Web")

# Initialize the language model
# Check if the API keys are available before initializing the model
if groq_api_key and os.environ.get("SERPAPI_API_KEY"):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Create the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate and comprehensive response based on the question.
        <context>
        {context}
        <context>
        Questions:{input}
        """
    )

    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt_input := st.chat_input("Ask a question..."):
        if st.session_state.vector is None:
            st.warning("Please process a website URL first.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt_input)
            st.session_state.chat_history.append({"role": "user", "content": prompt_input})

            # --- Part 1: Get answer from the website context ---
            with st.spinner("Analyzing website content..."):
                website_retriever = st.session_state.vector.as_retriever()
                website_chain = create_retrieval_chain(website_retriever, create_stuff_documents_chain(llm, prompt_template))
                response_website = website_chain.invoke({"input": prompt_input})
                website_answer = response_website['answer']

            # --- Part 2: Get answer from Google Search ---
            with st.spinner("Searching Google and analyzing results..."):
                search = SerpAPIWrapper()
                search_results = search.results(prompt_input)
                
                google_docs = []
                reference_links = []
                if "organic_results" in search_results:
                    for result in search_results["organic_results"]:
                        google_docs.append(Document(page_content=result.get("snippet", ""), metadata={"source": result.get("link", "")}))
                        if result.get("link"):
                            reference_links.append(f"- [{result.get('title', 'Source')}]({result.get('link')})")

                google_answer = "Could not find relevant information from Google search."
                if google_docs:
                    google_vector_store = FAISS.from_documents(google_docs, st.session_state.embeddings)
                    google_retriever = google_vector_store.as_retriever()
                    google_chain = create_retrieval_chain(google_retriever, create_stuff_documents_chain(llm, prompt_template))
                    response_google = google_chain.invoke({"input": prompt_input})
                    google_answer = response_google['answer']

            # --- Part 3: Display combined results ---
            with st.chat_message("assistant"):
                st.markdown("### From Website Content")
                st.markdown(website_answer)
                st.markdown("---")
                st.markdown("### From Google Search")
                st.markdown(google_answer)
                if reference_links:
                    st.markdown("#### References:")
                    st.markdown("\n".join(reference_links))
            
            # Save combined answer to history
            combined_answer = (
                f"**From Website Content:**\n{website_answer}\n\n---\n\n"
                f"**From Google Search:**\n{google_answer}"
            )
            if reference_links:
                combined_answer += "\n\n**References:**\n" + "\n".join(reference_links)
            st.session_state.chat_history.append({"role": "assistant", "content": combined_answer})

else:
    st.error("API keys not found. Please set them in your Streamlit secrets.")
