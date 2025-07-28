from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from getpass import getpass
import os
import pickle
import faiss
import boto3

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

# # Simplified URL structure for initial testing
# SHOPIFY_URLS = [
#     "https://shopify.dev/api/admin-rest",
#     "https://shopify.dev/api/admin-graphql"
# ]

# SHOPIFY_URLS = {
#     "api_docs": [
        # "https://shopify.dev/api/admin-rest",
        # "https://shopify.dev/api/admin-graphql",
        # "https://shopify.dev/api/storefront-rest",
        # "https://shopify.dev/api/storefront-graphql",
        # "https://shopify.dev/api/checkout-extensibility"
#     ],
#     "developer_docs": [
        # "https://shopify.dev/themes",
        # "https://shopify.dev/apps",
        # "https://shopify.dev/apps/auth",
        # "https://shopify.dev/apps/webhooks"
#     ],
#     "reference_docs": [
        # "https://shopify.github.io/liquid/",
        # "https://polaris.shopify.com/"
#     ]
# }

SHOPIFY_URLS = [
        "https://shopify.dev/api/admin-rest",
        "https://shopify.dev/api/admin-graphql",
        "https://shopify.dev/api/storefront-rest",
        "https://shopify.dev/api/storefront-graphql",
        "https://shopify.dev/api/checkout-extensibility",
        "https://shopify.dev/themes",
        "https://shopify.dev/apps",
        "https://shopify.dev/apps/auth",
        "https://shopify.dev/apps/webhooks",
        "https://shopify.github.io/liquid/",
        "https://polaris.shopify.com/"
]

def load_and_process_docs():
    """Load and process documents with proper error handling"""
    try:
        print("Loading documents...")
        loader = WebBaseLoader(SHOPIFY_URLS)
        docs = loader.load()
        # print("docs:", docs)
        for doc in docs:
            if "404" in doc.metadata["title"]:
                print(doc.metadata["source"])

        print(f"Loaded {len(docs)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print(f"Created {len(splits)} text splits")

        return splits

    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return None

def initialize_vectorstore(splits):
    """Initialize vector store and upload both vectorstore and retriever to S3."""
    try:
        print("Initializing embeddings...")
        embeddings = OpenAIEmbeddings()

        print("Creating vector store...")
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )

        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Save FAISS index
        faiss_index_file = "faiss_index.bin"
        faiss.write_index(vectorstore.index, faiss_index_file)

        # Save retriever metadata (search settings)
        retriever_metadata = {
            "search_type": "similarity",
            "search_kwargs": {"k": 5},
        }
        retriever_metadata_file = "retriever_metadata.pkl"
        with open(retriever_metadata_file, "wb") as f:
            pickle.dump(retriever_metadata, f)

        # Upload to S3
        print("Uploading vectorstore and retriever to S3...")
        store_to_s3("shopibot-docs", "faiss_index.bin", faiss_index_file)
        store_to_s3("shopibot-docs", "retriever_metadata.pkl", retriever_metadata_file)
        print("Upload completed!")

        return vectorstore, retriever

    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return None, None

def store_to_s3(bucket_name, object_name, file_path):
    """Upload a file to S3"""
    s3 = boto3.client('s3',
                      aws_access_key_id='aws_access_key_id',
                      aws_secret_access_key='aws_secret_access_key',
                      region_name='region_name'
                      )

    with open(file_path, "rb") as f:
        s3.put_object(Bucket=bucket_name, Key=object_name, Body=f)

    print(f"Uploaded {object_name} to {bucket_name}")

def save_vectorstore_to_s3(vectorstore, bucket_name, faiss_key, metadata_key):
    """Save vectorstore to S3 by separating the FAISS index and metadata."""
    try:
        # Save FAISS index to a binary file
        faiss.write_index(vectorstore.index, faiss_key)

        # Save metadata (docstore and index_to_docstore_id)
        metadata = {
            "docstore": vectorstore.docstore,
            "index_to_docstore_id": vectorstore.index_to_docstore_id,
        }
        with open(metadata_key, "wb") as f:
            pickle.dump(metadata, f)

        # Upload files to S3
        store_to_s3(bucket_name, faiss_key, faiss_key)
        store_to_s3(bucket_name, metadata_key, metadata_key)
        print("Vectorstore saved to S3 successfully!")

    except Exception as e:
        print(f"Error saving vectorstore to S3: {e}")

# In the main function, you call `initialize_vectorstore` as before.
def main():
    # Load and process documents
    splits = load_and_process_docs()
    if splits is None:
        raise ValueError("Failed to load and process documents")

    # Initialize vector store
    vectorstore, retriever = initialize_vectorstore(splits)
    save_vectorstore_to_s3(vectorstore, "shopibot-docs", "faiss_index.bin", "vectorstore_metadata.pkl")

if __name__ == "__main__":
    main()
