# from gensim.models import Word2Vec
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from langchain_community.embeddings import OllamaEmbeddings
import re
import sys

def transform_text(text):    
    # Transformar o texto
    text = text.lower()  # Converter para minúsculas
    
    # # Procurar por padrões que indicam o número de 'sets' e 'reps'
    # pattern_set = r'\b\d+\b'
    # match_set = re.search(pattern_set, text)
    # pattern_reps = r'(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))'
    # match_reps = re.search(pattern_reps, text)
    
    # # Verifica se existem números separados por vírgulas
    # if match_reps:
    #     # Construir a nova linha com "set" antes do primeiro número e "reps" após o primeiro número
    #     if match_set:
    #         indice_inicio = text.find(match_set.group())
    #         if indice_inicio!= -1:
    #             text = text[:indice_inicio]
    #         else:
    #             print(f'O caractere "{match_set.group()}" não foi encontrado na string.')
    #         text = f"{text} set {match_set.group()} reps {match_reps.group(0)}"
    
    # Pré-processar o texto
    # Remover caracteres especiais e pontuação
    text = re.sub(r'[^\w\s]', '', text)
    return text

vector = []
transformed_vector = []

# Exemplo de texto
text = 'Monday - Chest & Triceps\nExercise Sets Reps\nChest\nDumbbell Bench Press 4 12, 10, 10, 10\nIncline Bench Press 2 10\nTriceps\nTricep Dip 3 Failure\nLying Tricep Extension 3 10\nNotes\n• Have a 10 min warmup before you begin your workout.\n• Have your bench at a 30 degree angle for incline bench press.\n• Make sure you lean forward to focus the work on your lower chest. Use assisted dip         \nmachine if you cannot do bodyweight.\n• Light weights only for skullcrushers, focus on form.'
# print(f'original text: {text}\n')

# text = 'Dumbbell Bench Press 4 12, 10, 10, 10'
for line in text.split('\n'):
    transformed_line = transform_text(line)
    vector.append(transformed_line)
print(vector)


# sys.exit()


embeddings = OllamaEmbeddings(model="llama3")
# print(f'embeddings: {embeddings}\n')

# Converter o texto pré-processado em vetores
embedded_text = embeddings.embed_documents([vector])
print(f'vector: {vector}\n')

print(f'vector size: {len(embedded_text)}\n')

print('TEST DONE!!!!')