from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever
import chromadb
import Stemmer
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

class SemanticBM25Retriever(BaseRetriever):
    def __init__(self, db_name: str, collection_name: str = "default", mode: str = "OR") -> None:

        self._mode = mode

        # Path to database directories
        VECTOR_DB_PATH = os.path.join(os.getenv("VECTOR_DB_PATH"), f'{db_name}_vectordb')
        BM25_DB_PATH = os.path.join(os.getenv("BM25_DB_PATH"), f'{db_name}_bm25')

        # Embedding Model
        self._embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Read stored Vector Database
        self._vectordb = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _chroma_collection = self._vectordb.get_or_create_collection(collection_name)
        self._vector_store = ChromaVectorStore(chroma_collection=_chroma_collection)
        self._index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            embed_model=self._embed_model,
        )

        self._chromadb_retriever = self._index.as_retriever()

        # Read stored BM25 Database
        self._bm25_retriever = BM25Retriever.from_persist_dir(BM25_DB_PATH)


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:

        # Retrieving Nodes from Database
        vector_nodes = self._chromadb_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        bm25_ids = {n.node.node_id for n in bm25_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in bm25_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(bm25_ids)
        else:
            retrieve_ids = vector_ids.union(bm25_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]

        return retrieve_nodes


if __name__ == "__main__":

    db = SemanticBM25Retriever(collection_name="jax")

    res = db.retrieve("what are Code generation flags ?")

    print(len(res))

