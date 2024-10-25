from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
import Stemmer
import os
import chromadb

def delete_files_in_directory(directory_path: str):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError as e:
        print(f"Error occurred: {e}")


def save_chromadb(nodes: list, 
                  db_name: str, 
                  collection_name: str = "default", 
                  save_dir: str = "./") -> None:
    
    print("-:-:-:- ChromaDB [Vector Database] creating ... -:-:-:-")

    # Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Path to save the database file
    save_pth = os.path.join(save_dir, db_name)

    # Initializing Vector Database
    db = chromadb.PersistentClient(path=save_pth)

    # Creating Collection
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, embed_model=embed_model
    )

    print("-:-:-:- ChromaDB [Vector Database] saved -:-:-:-")


def save_BM25(nodes: list, 
              save_dir: str = "./", 
              db_name: str = "none") -> None:
    
    print("-:-:-:- BM25 [TF_IDF Database] creating ... -:-:-:-")

    # Initializing BM25
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=8,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Path to save BM25
    save_pth = os.path.join(save_dir, db_name)

    # Saving BM25
    bm25_retriever.persist(save_pth)

    print("-:-:-:- BM25 [TF_IDF Database] saved -:-:-:-")


def create_and_save_db(
        data_dir: str, 
        collection_name : str, 
        save_dir: str, 
        db_name: str = "default",
        chunk_size: int = 512, 
        chunk_overlap: int =20
        ) -> None:
    
    # Path directory to data storage 
    DATA_DIR = data_dir

    # Hyperparameters for text splitting
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    # Reading documents
    reader = SimpleDirectoryReader(input_dir=DATA_DIR)
    documents = reader.load_data()

    original_document_content = ""
    for page in documents:
        original_document_content += page.text

    # Initializing text splitter
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
    )

    # Splitting documents to Nodes [text chunks]
    nodes = splitter.get_nodes_from_documents(documents)

    vectordb_name = db_name + "_vectordb"
    bm25db_name = db_name + "_bm25"
    
    # Saving the Vector Database and BM25 Database
    save_chromadb(nodes=nodes, 
                  save_dir=save_dir, 
                  db_name=vectordb_name, 
                  collection_name=collection_name)
    
    save_BM25(nodes=nodes, 
              save_dir=save_dir, 
              db_name=bm25db_name)
    
    delete_files_in_directory(DATA_DIR)