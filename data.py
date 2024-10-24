import os
import json
import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Document

from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env (not included in repo)

import gdown

data = None
def get_data(download=False):
    global data
    if data is None:
        data = Data(download)
    return data

class Data:
    def __init__(self, download=False):
        print("Initializing Data...")
        print(f"Download: {download}")
        self.client = None
        self.collection = None
        self.index = None
        if download:
            self.download_data()
        self.load_data()

    def download_data(self):
        # Download the already indexed data
        if not os.path.exists("./chroma_db"):
            try: 
                print("Downloading data...")
                file_id = "1JvYQ9E5zDBKRCUKkxejDvp7UGwzxDAUW"
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                output = "chroma_db.zip"
                gdown.download(url, output, quiet=False)
                print("Unzipping data...")
                os.system("unzip chroma_db.zip")
            except Exception as e:
                print(f"Error downloading data: {e}")

        return os.path.exists("./chroma_db")

    def load_data(self):
        print("Loading data...")
    
        with open('data/train-v1.1.json', 'r') as f:
            raw_data = json.load(f)  
            
        raw_documents = []
        documents = []

        for data in raw_data['data']:
            title = data['title']
            for par in data['paragraphs']:
                context = par['context']
                for qa in par['qas']:
                    question = qa['question']
                    answers = []
                    for ans in qa['answers']:
                        if ans['text'] not in answers:
                            answers.append(ans['text'])
                    for answer in answers:
                        raw_documents.append([title, context, question, answer])
                    
                    doc = f"""
                        Title: {title}
                        Context: {context}
                        Question: {question}
                        Acceptable Answers:
                        {[f"{i+1}. {ans}" for i, ans in enumerate(answers)]}
                    """
                    # Remove padding on each line
                    doc = "\n".join([line.strip() for line in doc.split("\n")])
                    documents.append(doc)

        self.df = pd.DataFrame(raw_documents, columns=["Title", "Context", "Question", "Answer"])
        self.documents = [Document(text=t) for t in documents]

        print("Raw Data loaded")

        if not os.path.exists("./chroma_db"):
            # Attempt to generate an index from the raw data
            print("Creating Chroma DB...")
            # initialize client, setting path to save data
            self.client = chromadb.PersistentClient(path="./chroma_db")

            # create collection
            self.collection = self.client.get_or_create_collection("simple_index")

            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # create your index
            self.index = VectorStoreIndex.from_documents(
                self.documents, storage_context=storage_context
            )
            print("Chroma DB created")
        else:
            print("Chroma DB already exists")

        print("Loading index...")
        # initialize client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # get collection
        self.collection = self.client.get_or_create_collection("simple_index")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # load your index from stored vectors
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        print("Index loaded")