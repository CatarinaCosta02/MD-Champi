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
    """
    # Loop infinito para continuar pedindo perguntas ao usuário
    while True:
        # Solicita uma pergunta ao usuário
        # user_input = input("Digite sua pergunta ou 'exit' para sair: ")
        user_input = input(f'>> ')

        # Verifica se o usuário digitou 'exit' para sair do loop
        if user_input.lower() == 'exit':
            break
        
        # Invoca o modelo de chat com a pergunta do usuário
        # Aqui, estamos passando a mensagem diretamente para o modelo,
        # sem a necessidade de um dicionário com um tópico.
        # Como estamos em um loop, podemos simplesmente passar a mensagem do usuário.
        
        # Invoca o modelo de chat com a entrada do usuário
        response = llm.invoke(user_input)
        
        # Extrai o conteúdo da resposta
        content = response.content.strip()
        
        # Imprime a resposta do modelo de chat
        print(f'Champi: {content}\n')
        
        # Imprime a resposta do modelo de chat
        #print(chat_model_response)
    """

# Chama a função para iniciar a interação
chat_with_ollama()
