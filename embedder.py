from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.embeddings.embeddings import Embeddings
import os
from sentence_transformers import SentenceTransformer
import logging
from typing import List
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores.faiss import FAISS

class FastEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-small', device="mps")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode("query: " + t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode("query: " + text).tolist()

def check_preload():
    if not os.path.isfile("fais_db/index.faiss"):

        loader = PyMuPDFLoader("data/d2l-en.pdf")
        pages = loader.load_and_split()
        embeds = FastEmbeddings()
        chunker = SemanticChunker(embeds,
                                  breakpoint_threshold_type="percentile")
         
        chunks = chunker.split_documents(pages)
        db = FAISS.from_documents(chunks, embeds)
        print(db.index.ntotal)
        db.save_local("fais_db/", "base_science.index") 
    return True

def embed_docs(docs):
    embeds = FastEmbeddings()
    chunker = SemanticChunker(embeds,
                              breakpoint_threshold_type="percentile")
    
    chunks = chunker.split_documents(docs)
    db = FAISS.from_documents(chunks, embeds)
    return db

def download_papers(ids:list[str]):
    docs = []
    for id_ in ids:
        doc = ArxivLoader(query=id_, load_max_docs=2).load()
        docs.append(doc[0])
    return docs

def show_outline():
    pass
