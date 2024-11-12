import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import docx2txt
import streamlit_lottie as st_lottie
import json


load_dotenv()

# Import the templates from htmlTemplates.py
from htmlTemplates import css, bot_template, user_template

# Other helper functions remain the same
def get_docs_text(docs):
    ''' Extract text from document files (txt, pdf, docx) '''
    text = ""
    for doc in docs:
        if doc is not None:
            if doc.type == "text/plain":  # txt doc
                text += str(doc.read(), encoding="utf-8")
            elif doc.type == "application/pdf":  # pdf
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                text += docx2txt.process(doc)
    return text

def get_text_chunks(text):
    ''' Split text into smaller chunks '''
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    print('Chunks: ',chunks)
    return chunks

def get_vectorstore(text_chunks):
    ''' Convert text chunks into a vector store '''
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    ''' Create a conversation chain '''
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_retries=2,
)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    ''' Handle user input and show chat messages using templates '''
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents before asking questions.")
        return
    
    # Call the conversation chain to get a response
    response = st.session_state.conversation({"question": user_question})
    
    # Display user and bot chat messages using templates
    st.markdown(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    st.markdown(bot_template.replace("{{MSG}}", response['answer']), unsafe_allow_html=True)


def load_lottiefile(filepath: str):
    ''' Load a Lottie animation file '''
    with open(filepath, 'rb') as f:
        return json.load(f)

def main():
    ''' Main function to run the Streamlit app '''
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple Documents", page_icon=":books:")
    cover_pic = load_lottiefile('img/books.json')
    st.lottie(cover_pic, speed=0.5, reverse=False, loop=True, quality='low', height=200, key='first_animate')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple Documents 🔍")
    
    # Display chat interface CSS
    st.markdown(css, unsafe_allow_html=True)
    
    # User input field
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv']
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_docs_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                
                # Create or update vector store
                if st.session_state.vectorstore is None:
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                else:
                    new_vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore.merge(new_vectorstore)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)

if __name__ == "__main__":
    main()
