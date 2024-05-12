import re
import sys
import uuid
from pinecone import ServerlessSpec, Pinecone
import os
from dotenv import load_dotenv
import time
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
from loader import load_docs, split_docs, transform_text, extract_metadata
from langchain_pinecone import PineconeVectorStore
import json



paths = ["data/plan/3daybeginners.pdf"]

        #  "data/plan/3dayworkoutroutineanddietforbeginners.pdf",
        #  "data/plan/8weekbeginnerfatlossworkoutforwomen_0.pdf",
        #  "data/plan/8weekbeginnerworkoutforwomen.pdf",
        #  "data/plan/12weekfullbodyworkoutroutineforbeginners.pdf",
        #  "data/plan/startfromscratch.pdf",
        #  "data/plan/thebest15minutewarmups.pdf"


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

# index.delete(delete_all=True)

# view index stats
b = index.describe_index_stats()
print(f'index stats: {b}')

embeddings = OllamaEmbeddings(model="llama3")

pages = load_docs(paths)
documents = split_docs(pages)


vectors_dict = {}
vector = []
vector_count=0
for doc in documents:
    for line in doc.page_content.split('\n'):

        line = (str(line)).lower()  # Converter para minúsculas
        line = (re.sub(r'[^\w\s]', '', line)).strip()

        vector.append(line)  # Adiciona o transformed_line ao vetor

        # Verifica se o vetor excede o limite de tamanho
        if len(vector) >= 4096:
            vector_count += 1
            print(f"Vector size reached 4096")
            # Adiciona o vetor completo ao dicionário
            vectors_dict[vector_count] = vector
            # Reinicia o vetor para continuar a adição de novos transformed_line
            vector = []
            print(f"Vector has been reset.")

    # Adiciona o vetor final ao dicionário, caso ainda exista
    if vector:
        vectors_dict[vector_count] = vector


print(f'len(vectors_dict.items()): {len(vectors_dict.items())}')


# print(f'vectors_dict: {vectors_dict}')
# print(f'vectors_dict.items(): {vectors_dict.items()}')


# embedded_vectors = []
# for _, vector in vectors_dict.items():
#     print(f'Vector: {vector}')
#     print(f'Length of vector: {len(vector)}')

#     # Embed the vector
#     embedded_vector = embeddings.embed_documents(vector)
#     # print(f'Embedded vector: {embedded_vector}')
#     embedded_vectors.append(embedded_vector)
# print(f'len(embedded_vectors): {len(embedded_vectors)}')

# # Save the embedded vectors to a JSON file
# with open('embedded_vectors.json', 'w') as f:
#     json.dump(embedded_vectors, f)
# print("Embedded vectors have been saved to 'embedded_vectors.json'.")


# Open the JSON file
with open('embedded_vectors.json', 'r') as f:
    # Load the JSON data from the file
    embedded_vectors_data = json.load(f)
# Now, embedded_vectors_data contains the list of embedded vectors
# You can assign it to your variable
embedded_vectors = embedded_vectors_data


# Flatten embedded_vectors if it's a list of lists
flattened_vectors = [item for sublist in embedded_vectors for item in sublist]
# print(f'Flattened vectors: {flattened_vectors}')

embedded_ids = [str(uuid.uuid4()) for _ in range(len(flattened_vectors))]
# print(f'Embedded IDs: {embedded_ids}')

metadata = []
for doc in vectors_dict.items():
    mtdt = extract_metadata(doc)
    metadata.append(mtdt)

print(f'\n\nmetadata: {metadata}')

# Combine vectors into the required structure
vectors_to_upsert = [{"id": id, "values": vector, "metadata": metadt} for id, vector, metadt in zip(embedded_ids, flattened_vectors, metadata)]
# print(f'Vectors to upsert: {vectors_to_upsert}')

# Now, you can use 'vectors_to_upsert' with your index.upsert method
index.upsert(vectors=vectors_to_upsert)




print(f'\n------after inserting data------')
print(index.describe_index_stats())


print(f'\n-------------------TESTAR SIMILAR SEARCH------------------')

#text_field = None
# initialize the vector store object
# vectorstore = Pinecone(index, embed_model.embed_query, text_field)

vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
# vectorstore = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

# vectorstore = Pinecone(
#     index_name=index_name,
#     embedding=embeddings
# )

print(f'vectorstore: {vectorstore}')

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