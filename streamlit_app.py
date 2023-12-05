!pip install google-cloud-aiplatform 

from google.colab import auth
auth.authenticate_user()

import streamlit as st
import ast

st.header("BigQuery Chatbot for Social Listening")

import vertexai
import pandas as pd
import numpy as np

PROJECT_ID = "sysomosapi2"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
persist_directory = '/content/drive/My Drive/Database/'
embedding = VertexAIEmbeddings()




def main():
    #st.sidebar.title("User Input")
    #user_query = st.sidebar.text_input("Enter your query here:")

    question= st.text_input("Ask a question to discover insights from tweets on BigQuery")


    if st.button("Ask"):
        if question:
            with st.spinner('Generating response...'):

               vectordb=Chroma(persist_directory=persist_directory,embedding_function=embedding)
               vectordb.get()

               #from langchain.chat_models import ChatVertexAI
               #llm = ChatVertexAI(temperature=0,max_output_tokens=500)
               

               from langchain.llms import VertexAI
               llm=VertexAI(model_name="text-bison@001", temperature=0.9,max_output_tokens=500)

               # Build prompt


               template = """You are a chatbot designed for product marketing managers to derive social listening insights on BigQuery.
               You must answer the question below in detail, strictly based on the context provided. 
               Note that you must strictly use the context provided below to answer user's question. Don't use your own knowledge
               If the context below doesn't provide any relevant information, answer with
               "I couldn't find a good match in the document database for your query".  

               {context}
               Question: {question}
               Helpful Answer:"""

               QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

               # Run chain
               from langchain.chains import RetrievalQA


               qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(search_kwargs={"k":8}),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       verbose=True)
              
               response = qa_chain({"query": question})

               #response = result["result"]

               st.success("Response:")

               st.write(qa_chain({"query": question}))
              # generated_texts=[]
              # for candidate in response.result:
              #    generated_texts.append(candidate.text)

               #if generated_texts:
               #   st.write(generated_texts[0])  # Display the first generated text
               #else:
                #  st.write("No response generated.")
        else:
          st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
