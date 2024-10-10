import os
import json
import chromadb
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Document

from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env (not included in repo)

class Data:
    def __init__(self):
        self.client = None
        self.collection = None
        self.index = None
        self.load_data()

    def load_data(self):
        print("Loading data...")
        with open('data/train-v1.1.json', 'r') as f:
            raw_data = json.load(f)        

        extracted_question = []
        extracted_answer = []

        for data in raw_data['data']:
            for par in data['paragraphs']:
                for qa in par['qas']:
                    for ans in qa['answers']:
                        extracted_question.append(qa['question'])
                        extracted_answer.append(ans['text'])

        documents = []
        for i in range(len(extracted_question)):
            documents.append(f"Question: {extracted_question[i]} \nAnswer: {extracted_answer[i]}")

        self.documents = [Document(text=t) for t in documents]
        self.extracted_question = extracted_question
        self.extracted_answer = extracted_answer

        print("Raw Data loaded")

        if not os.path.exists("./chroma_db"):
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