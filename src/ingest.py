from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma


def create_db():
    # Load PDF
    loader = PyPDFLoader("data/support.pdf")
    docs = loader.load()

    print("DEBUG docs loaded:", len(docs))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    print("DEBUG chunks:", len(chunks))

    # Embeddings
    embeddings = FakeEmbeddings(size=384)

    # Create DB
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    db.persist()
    print("✅ DB Created Successfully!")


if __name__ == "__main__":
    create_db()