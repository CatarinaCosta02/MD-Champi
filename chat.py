import os
import sys
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone_handler import get_index


def chat_with_ollama(query, context):
    
    llm = ChatOllama(model="llama3")

    template = """
    Answer the question based on the context below. If you can't 
            answer the question, reply "I don't know". If the 
            question has nothing to do with the context, 
            answer the question normally.

            Context: {context}

            Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    prompt.format(context=context, question=query)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({
        "context": context,
        "question": query
    })
    return response.strip()



def chatbot(vectorstore):
    query = input("\nEnter your message: ")
    if query.lower() == "exit":
        print("Exiting...")
        return False
    elif query is None:
        query = "Give me a workout plan for beginners"
    else:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        response = chat_with_ollama(query, context)
        print(f'\nChampi: \n{response}')
    return response


def main():
    index_name = 'champi'
    index = get_index(index_name)
    
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

    print(f'\n-------------------CHATBOT - Champi------------------')
    start_time = time.time()
    response = chatbot(vectorstore)
    end_time = time.time()
    elapsed_time_in_minutes = (end_time - start_time) / 60
    print(f"Response time: {elapsed_time_in_minutes:.2f} minutes")
    print(response)


if __name__ == "__main__":
    main()
