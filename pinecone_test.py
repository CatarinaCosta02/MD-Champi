from pinecone import ServerlessSpec, Pinecone
import os
from dotenv import load_dotenv
import time
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

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
        name=index_name,  # index_name é passado corretamente como uma string
        dimension=4096,  # dimensionality of ada 002
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
    print(index['name'])


# connect to index
a = index = pc.Index(index_name)
print(a)
time.sleep(1)
# view index stats
b = index.describe_index_stats()
print(b)

embeddings = OllamaEmbeddings(model="llama3")

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]

res = embeddings.embed_documents(texts)
print(len(res), len(res[0]))


