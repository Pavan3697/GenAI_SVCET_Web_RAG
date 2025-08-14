import os
import time
import streamlit as st
from dotenv import load_dotenv

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

# --- Load env vars for local dev ---
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Web RAG with Groq & SerpApi", layout="wide")

# --- API Keys ---
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
serp_api_key = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY"))
if not groq_api_key or not serp_api_key:
    st.error("Missing API keys. Set GROQ_API_KEY and SERPAPI_API_KEY in Streamlit secrets or .env")
    st.stop()

os.environ["SERPAPI_API_KEY"] = serp_api_key

# --- Static UI ---
if os.path.exists("PragyanAI_Transperent_github.png"):
    st.image("PragyanAI_Transperent_github.png")
st.title("Web RAG: Q&A with Website Content and Google Search")

# --- Session state ---
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Sidebar input ---
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

# --- Main chat ---
st.header("Chat with the Web")

try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
except Exception as e:
    st.error(f"Error initializing Groq LLM: {e}")
    st.stop()

prompt_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate and comprehensive response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
if prompt_input := st.chat_input("Ask a question..."):
    if st.session_state.vector is None:
        st.warning("Please process a website URL first.")
    else:
        # User message
        with st.chat_message("user"):
            st.markdown(prompt_input)
        st.session_state.chat_history.append({"role": "user", "content": prompt_input})

        # --- Website context answer ---
        with st.spinner("Analyzing website content..."):
            website_retriever = st.session_state.vector.as_retriever()
            website_chain = create_retrieval_chain(website_retriever, create_stuff_documents_chain(llm, prompt_template))
            response_website = website_chain.invoke({"input": prompt_input})
            website_answer = response_website.get("answer", "No answer from website context.")

        # --- Google search answer ---
        with st.spinner("Searching Google and analyzing results..."):
            search = SerpAPIWrapper()
            search_results = search.results(prompt_input)

            google_docs = []
            reference_links = []
            if search_results and "organic_results" in search_results:
                for result in search_results["organic_results"]:
                    snippet = result.get("snippet", "")
                    link = result.get("link", "")
                    title = result.get("title", "Source")
                    google_docs.append(Document(page_content=snippet, metadata={"source": link}))
                    if link:
                        reference_links.append(f"- [{title}]({link})")

            google_answer = "Could not find relevant information from Google search."
            if google_docs:
                google_vector_store = FAISS.from_documents(google_docs, st.session_state.embeddings)
                google_retriever = google_vector_store.as_retriever()
                google_chain = create_retrieval_chain(google_retriever, create_stuff_documents_chain(llm, prompt_template))
                response_google = google_chain.invoke({"input": prompt_input})
                google_answer = response_google.get("answer", google_answer)

        # --- Display results ---
        with st.chat_message("assistant"):
            st.markdown("### From Website Content")
            st.markdown(website_answer)
            st.markdown("---")
            st.markdown("### From Google Search")
            st.markdown(google_answer)
            if reference_links:
                st.markdown("#### References:")
                st.markdown("\n".join(reference_links))

        # Save answer in history
        combined_answer = (
            f"**From Website Content:**\n{website_answer}\n\n---\n\n"
            f"**From Google Search:**\n{google_answer}"
        )
        if reference_links:
            combined_answer += "\n\n**References:**\n" + "\n".join(reference_links)
        st.session_state.chat_history.append({"role": "assistant", "content": combined_answer})
