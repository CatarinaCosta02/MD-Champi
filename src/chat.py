from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone_handler import get_index
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage


def chatbot_first_message(vectorstore):
    question = input("\nEnter your message: ")
    if question.lower() == "exit":
        print("Exiting...")
        return None, None, False
    elif not question:
        question = "Give me a workout plan for beginners"
    
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

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
    prompt.format(context=context, question=question)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    response = str(chain.invoke({
        "context": context,
        "question": question
    }).strip())

    return question, response


def chatbot(agent_executor, human_msg, aimsg):
    # print(f'\nhuman_msg: {human_msg.content}\naimsg: {aimsg.content}')
    question = input("\nEnter your message: ")
    
    if question.lower() == "exit":
        print("Exiting...")
        return None, None, False
    elif not question:
        question = "Give me a workout plan for beginners"
        print(f'It seems that you didn\'t ask anything to Champi... The chat will give you a workout routine.')
    
    response = str(agent_executor.invoke({
        "input": question,
        "chat_history": [human_msg, aimsg],
    }).get('output'))

    human_msg = HumanMessage(content=question)
    aimsg = AIMessage(content=response)

    return human_msg, aimsg, True


def main():
    index_name = 'champi'
    index = get_index(index_name)
    
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")

    # First message (the history is saved for the structured chat agent to analyze and continue the conversation)
    question, response = chatbot_first_message(vectorstore)
    print(f'\nChampi: {response}')

    
    human_msg = HumanMessage(content=question)
    aimsg = AIMessage(content=response)

    # structured chat agent
    # https://python.langchain.com/v0.1/docs/modules/agents/agent_types/structured_chat/
    llm = ChatOllama(model="llama3")
    tools = [TavilySearchResults(max_results=1)]
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = create_structured_chat_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_attempts=1
    )

    continue_chat = True
    while continue_chat:
        human_msg, aimsg, continue_chat = chatbot(agent_executor, human_msg, aimsg)
        print(f'\nChampi: {aimsg.content}')


if __name__ == "__main__":
    main()
