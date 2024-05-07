import sys
import uuid
from pinecone import ServerlessSpec, Pinecone
import os
from dotenv import load_dotenv
import time
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
from loader import load_docs, split_docs, transform_text
from langchain_pinecone import PineconeVectorStore

PATH = "data/plan/3dayworkoutroutineanddietforbeginners.pdf"

load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key, pool_threads=30)

index_name = 'champi'  # Certifique-se de que index_name é uma string
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
# a metrica que este usa é o dotproduct
if index_name not in existing_indexes:
    # if does not exist, create index
    # nao sei como reduzir, problema pra mais tarde
    pc.create_index(
        name=index_name,    # index_name é passado corretamente como uma string
        dimension=4096,     # dimensionality of ada 002
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

for index in pc.list_indexes():
    print(f'index name: {index['name']}')


# connect to index
a = index = pc.Index(index_name)
print(f'index: {a}')
time.sleep(1)
# view index stats
b = index.describe_index_stats()
print(f'index stats: {b}')

embeddings = OllamaEmbeddings(model="llama3")

pages = load_docs()
documents = split_docs(pages)


vectors_dict = {}
vector = []
vector_count=0
for doc in documents:
    for line in doc.page_content.split('\n'):
        transformed_line = transform_text(line)
        print(transformed_line)
        vector.append(transformed_line)  # Adiciona o transformed_line ao vetor

        # Verifica se o vetor excede o limite de tamanho
        if len(vector) >= 4096:
            vector_count += 1
            print(f"Vector size reached or exceeded 4096")
            # Adiciona o vetor completo ao dicionário
            vectors_dict[vector_count] = vector
            # Reinicia o vetor para continuar a adição de novos transformed_line
            vector = []
            print(f"Vector has been reset.")
    
    # Adiciona o vetor final ao dicionário, caso ainda exista
    if vector:
        vectors_dict[vector_count] = vector

sys.exit()

print(f'len(vectors_dict.items()): {len(vectors_dict.items())}')


embedded_vectors = []
for vector in vectors_dict.items():
    vector = vector[1]
    print(f'\nvector: {vector} \nlen(vector): {len(vector)} \ntype(vector): {type(vector)}')

    # Embed the vector
    embedded_vector = embeddings.embed_documents(vector)
    print(f'\nembedded_vector: {embedded_vector}')
    embedded_vectors.append(embedded_vector)

# Flatten embedded_vectors if it's a list of lists
flattened_vectors = [item for sublist in embedded_vectors for item in sublist]
print(f'flattened_vectors: {flattened_vectors}')

embedded_ids = []
for vec in range(embedded_vector):
    id = str(str(uuid.uuid4()))
    embedded_ids.append(id)

# Combine vectors into the required structure
vectors_to_upsert = [{"id": embedded_ids, "values": vector} for vector in flattened_vectors]
print(f'vectors_to_upsert: {vectors_to_upsert}')

# Now, you can use 'vectors_to_upsert' with your index.upsert method
index.upsert(vectors=vectors_to_upsert)




print(f'\n------after inserting data------')
print(index.describe_index_stats())


print(f'-------------------TESTAR------------------')

#text_field = None
# initialize the vector store object
# vectorstore = Pinecone(index, embed_model.embed_query, text_field)
vectorstore = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

print(f'\nvectorstore: {vectorstore}')

query = "Give me a workout plan"
vectorstore.similarity_search(query, k=3)

sys.exit()

chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

chain.invoke("What is Hollywood going to start doing?")

def augment_prompt(query: str):
    # get top 3 results from knowledge base
    results = vectorstore.similarity_search(query, k=3)
    print(f'result')
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt

print(f'\naugment_prompt(query): {augment_prompt(query)}')