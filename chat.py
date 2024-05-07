from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
"""

def chat_with_ollama():
    print("Entrei")
    messages = []
    # Inicializando o modelo Ollama com o modelo "llama3"
    llm = ChatOllama(model="llama3")
    #prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

    prompt = ChatPromptTemplate.from_template(template)
    prompt.format(context="I´m bad at physics", question="I'd like to understand string theory")
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = chain.invoke({
        "context": "I´m bad at physics",
        "question": "I'd like to understand string theory"
    })
    
    content = response.strip().lower()
    # add latest response to the chat
    messages.append(content)
    print(f'Champi: {content}\n')

    if "true" in content:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {content}" + "\033[0m")
        return True

# Chama a função para iniciar a interação
chat_with_ollama()
