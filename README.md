# Data Mining - Champi

Champi is training assistant chatbot that creates personalized training plans for beginners. It also suggest exercises and challenges according to user's age, weight, goals, etc.

This chatbot was develop as a data mining project with the aim of learning how to create and evaluate a Large Language Model.

## Data Sources

For this project we decided to gather data from various formats:

- 2 **CSV** datasets taken form kaggle with fitness and gym exercises;
- **PDFs** of beginner workout plans taken from M&S website for text data;

## Table of Contents

| data        | folder with the data used                                    |
| ----------- | ------------------------------------------------------------ |
| `datasets`  | Contains the datasets that were used to train the chatbot    |
| `plan`      | Contains the group of beginner workout plans used as text data to train the chatbot |
| `exercises` | Contains the group of exercises descriptions                 |

| Presentations | Contains both presentations |
| ------------- | --------------------------- |

| src                   | folder that contains the source code files of the project    |
| --------------------- | ------------------------------------------------------------ |
| `dataset_T.ipynb`     | Jupyter Notebook for exploring and prepare the datasets      |
| `app.py`              | Makes the connection between the back-end and the front-end  |
| `chat.py`             | Contains the core chatbot implementation, using llama3 as LLM. It is also responsible for the questions and responses |
| `loader.py`           | Contains the script responsible for loading the pdfs documents and csv, making the splits and extracting the metadata. It also collects text from PDFs and performs minor transformations on it |
| `pinecone_handler.py` | Contains the code responsible for inserting the vector embeddings into the database, trough the functions in the `loader.py` script |
| `champiResponses.txt` | Contains some responses from a chat with the chatbot         |

## How to run the project

To run the project first it is needed the following requirements:

- `pip install virtualenv`

- `.\myenv\Scripts\activate` 

- Python 3.12.3

- run `pip install -r requirements.txt`

- `ollama start`

- `ollama pull llama3`

  

To run the app:

    python3 pinecone_handler.py

Then

    python3 chat.py

## References

Datasets:

https://www.kaggle.com/datasets/omarxadel/fitness-exercises-dataset

https://www.kaggle.com/datasets/niharika41298/gym-exercise-data

Pdfs:

https://www.muscleandstrength.com/workout-routines

## Team Members

  - Catarina Costa - pg52676

  - Marta Aguiar - pg52694

  - Rita Dantas - pg51605

    