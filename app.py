import streamlit as st
import requests
import PyPDF2
from itertools import chain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from googletrans import Translator
from langdetect import detect
import os
import io

# Load environment variables from Streamlit secrets
together_api_key = st.secrets["TOGETHER_API_KEY"]

# Function to fetch content from a website
def fetch_website_content(url):
    response = requests.get(url)
    return response.text

# Function to extract text from a PDF file
def extract_pdf_text(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Split the combined content into smaller chunks
def split_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Initialize embeddings and vector store
def initialize_vector_store(contents):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    web_chunks = list(chain.from_iterable(split_text(content) for content in contents))
    db = Chroma.from_texts(web_chunks, embedding_function)
    return db

llm = Together(
    model="meta-llama/Llama-2-70b-chat-hf",
    max_tokens=256,
    temperature=0.1,
    top_k=1,
    together_api_key=together_api_key
)

# Set up the retrieval QA chain
def setup_retrieval_qa(db):
    retriever = db.as_retriever(similarity_score_threshold=0.6)

    # Define the prompt template
    prompt_template = """Please answer questions related to Agriculture. Try explaining in simple words. Answer in less than 100 words. If you don't know the answer, simply respond with 'Don't know.'
     CONTEXT: {context}
     QUESTION: {question}"""

    PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

    # Initialize the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=True
    )
    return chain

def translate_text(text):
    """Translate text to English if it is not already in English."""
    try:
        if len(text) <= 5:
            return text, "en"
        
        detected_lang = detect(text)
        if detected_lang != "en":
            translator = Translator()
            translation = translator.translate(text, src=detected_lang, dest='en')
            return translation.text, detected_lang
        return text, "en"
    except Exception as e:
        return str(e), None

def translate_from_english(result, target_lang):
    """Translate text from English to the specified target language."""
    try:
        if target_lang != "en":
            translator = Translator()
            translation = translator.translate(result, src='en', dest=target_lang)
            return translation.text
        return result
    except Exception as e:
        return str(e)

# Function to initialize the content and chain based on user input
def initialize_content_and_chain(urls=None, pdf_files=None):
    website_contents = [fetch_website_content(url) for url in urls] if urls else []
    
    if pdf_files:
        pdf_texts = [extract_pdf_text(io.BytesIO(file.read())) for file in pdf_files]
    else:
        pdf_texts = [extract_pdf_text(open("Data/Agri.pdf", "rb"))]

    all_contents = website_contents + pdf_texts

    db = initialize_vector_store(all_contents)
    chain = setup_retrieval_qa(db)
    return chain

def process_query(chain, query):
    translated_query, detected_lang = translate_text(query)
    response = chain.invoke({"query": translated_query})
    
    if isinstance(response, dict) and 'result' in response:
        result = response['result']
    elif isinstance(response, str):
        result = response
    else:
        result = str(response)
    
    translated_response = translate_from_english(result, detected_lang)
    return translated_response

# Streamlit UI
def main():
    st.set_page_config(page_title="AgriGenius", layout="wide")  # Set the browser tab title

    st.title("AgriGenius")  # Title for the app
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.header("Upload Files")
    st.sidebar.markdown("""
    You can enter URLs or upload your own document to proceed. 
    Can also proceed without data just click to proceed.
                        
    Note: Bigger the data will take more processing time.

    Able to interact in Multiple languages.
    """)

    urls_input = st.sidebar.text_area("Enter URLs (separated by commas):")
    pdf_files_input = st.sidebar.file_uploader("Upload PDF files (multiple allowed)", type="pdf", accept_multiple_files=True)

    urls = [url.strip() for url in urls_input.split(",")] if urls_input else []
    pdf_files = pdf_files_input

    if st.sidebar.button("Proceed"):
        with st.spinner("It will take some time(depend on data)..."):
            try:
                chain = initialize_content_and_chain(urls, pdf_files)
                st.session_state.chain = chain
                st.write("Content and chain initialized successfully.")
            except Exception as e:
                st.write("Error initializing content and chain:", e)

    if 'chain' in st.session_state:
        query = st.text_input("Enter your query:")
        if st.button("Submit"):
            with st.spinner("Processing your query..."):
                response = process_query(st.session_state.chain, query)
                # Save query and response to chat history
                st.session_state.chat_history.append({"query": query, "response": response})
            
            # Display chat history
            st.write("### Chat History")
            for entry in st.session_state.chat_history:
                st.write(f"**You:** {entry['query']}")
                st.write(f"**Bot:** {entry['response']}")
    else:
        st.write("Ask Any query related to Agriculture")

if __name__ == "__main__":
    main()
