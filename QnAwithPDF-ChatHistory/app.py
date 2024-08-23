import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os

# API to secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

## LLM model
llm = ChatGroq(model="gemma2-9b-it")

## Embeddings : 2304 size of vector for each sent/ word.
embeddings = OllamaEmbeddings(model="gemma2:2b")

## set up our streamlit app 
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload pdf and chat with the content")

## Take session from user to procced
session_id = st.text_input("Session id",value="Default session")

if "store" not in st.session_state:
    st.session_state.store = {} # dictionary where all the chat history will be stored

uploadedPDFs = st.file_uploader("Choose the PDF file",type="pdf",accept_multiple_files=True)

# chat bot type 
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerrun

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




## Process Uploaded PDF
if uploadedPDFs:
    # uploaded pdfs we will store temparorily in our system for processing.
    documents = []

    for pdf_file in uploadedPDFs:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(pdf_file.getvalue())
            file_name = pdf_file.name
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)


    # Split the documents and create embeddings
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
    splittedDocs=textSplitter.split_documents(documents)

    vectorStore = FAISS.from_documents(documents=splittedDocs, embedding= embeddings)
    retrieverDB  = vectorStore.as_retriever()

    # Contextualize 
    contextualize_q_system_prompt = (
        """ 
            Given a chat history and the latest user question. 
            which might reference context in the chat history 
            Formulate a standalone question which can be understood. 
            without the chat history. Do not answer the question. Just formulate it if needed
            otherwise return it  as it is. 
        """
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_awareRetriever=create_history_aware_retriever(llm, retrieverDB, contextualize_q_prompt)

    ## answer question prompt

    system_prompt = (
        """ 
            You are the assistant for question answering tasks. 
            Use the following pieces of retriever context to answer the question.
            If you don't know the answer, Say thank you don't know. Use the three sentences maximum
            and keep the answer concise.
            \n\n
            {context}.
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    que_ans_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_awareRetriever, que_ans_chain)

    
    # history for session
    def get_session_history(session_id:str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    

    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" # optional
    )   



    #user_input = st.text_input("Your question")
    
    user_input = st.chat_input("What you want to ask about PDF?")
    if user_input:
        #session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input":user_input},
            config = {
                "configurable":{"session_id":session_id}
            }
        )
        
        # display user message in chat history container

        with st.chat_message("user"):
            st.markdown(user_input)

        # add user message to chat history 
        st.session_state.messages.append({"role":"user","content":user_input})

        res = response["answer"]
        # display assistant response

        with st.chat_message("assistant"):
            st.markdown(res)

        # add assistant response to chat history 
        st.session_state.messages.append({"role":"assistant","content":res})
    

        #st.write("Assistant",response['answer'])
        #st.write("Chat history",session_history.messages)
        #st.write("Store",st.session_state.store)

