import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings =  HuggingFaceEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    print(result['result'])
    return result



def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF APP")
    st.header("ChatPDF")
    #upload the file
    pdf = st.file_uploader("Upload your PDF",type="pdf")
    

    #extract the text
    if pdf is not None:
        #write the file to data directory
        with open("data/" + pdf.name,"wb") as f:
            f.write(pdf.getbuffer())
        st.write("File uploaded successfully")
        
        #loader = PyPDFLoader("data/"+pdf.name)
        #documents = loader.load()
        # split the documents into chunks   
        #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        #texts = text_splitter.split_documents(documents)
        ###pdf_reader=PdfReader(pdf)
        ###text=""
        ###for page in pdf_reader.pages:
            ###text +=page.extract_text()
       
        #split the documents into chunks
        ###text_splitter = CharacterTextSplitter(
           ### separator="\n",
            ###chunk_size=1000, 
            ###chunk_overlap=0,
            ###length_function=len)
        ###chunks = text_splitter.split_text(text)    
        #select which embeddings we want to use
        #embeddings = HuggingFaceEmbeddings()
        #db = FAISS.from_texts(chunks, embeddings)
        #db = Chroma.from_documents(texts, embeddings)
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        file = "data/" + pdf.name
        chaintype = st.radio("Please choose chain type",options=("stuff","map_reduce","refine","map_rerank"))
        k=st.slider("k value",min_value=0,max_value=10,value=5)
        if user_question:
            #docs = db.similarity_search(user_question)      
            #chain = load_qa_chain(llm=OpenAI(temperature=0), chain_type="map_reduce")
            #response = chain.run(input_documents=docs, question=user_question)
            response = qa(file, user_question, chaintype,k)
            st.write(response)



if __name__=='__main__':
    main()
