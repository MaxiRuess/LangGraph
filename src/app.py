import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from htmlTemplate import css, bot_template, user_template



from dotenv import load_dotenv

load_dotenv()



def get_response(user_input): 
    
    
    return "I don't know that yet..."

def get_vectorstor_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(documents, OpenAIEmbeddings() )
    
    return vector_store


def get_context_retriever(vector_store):
    
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag(retriever_chain):
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documment_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documment_chain)

def get_response(user_input): 
    
    retriever_chain = get_context_retriever(st.session_state.vector_store)
    converstaion_rag = get_conversational_rag(retriever_chain)
    
    response = converstaion_rag.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']
    


st.set_page_config(page_title="Streamlit Chat App", 
                   page_icon=":shark:", 
                   layout="wide")
st.title("Chat App", ":shark:")
st.write(css, unsafe_allow_html=True)



with st.sidebar:
    st.header("Settings")
    url = st.text_input("Enter a Website...")
    
if url is None or url == "":
    st.info("Please enter a website to get started")

else:
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content= "Hello! How can I help you?"),
            ]
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstor_from_url(url)

    
    user_input = st.chat_input("Type a message...")
    if user_input is not None and user_input != "":
        with st.spinner("Thinking..."):
            response = get_response(user_input)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))
        
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)