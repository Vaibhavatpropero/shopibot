from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from shopibotStream.logger import logger
import boto3
import faiss
import pickle
import json
import os

def load_vectorstore_from_s3(bucket_name, faiss_key, metadata_key):
    """Load vectorstore from S3 by reconstructing the FAISS index and metadata."""
    s3 = boto3.client(
        's3',
        # config=boto3.session.Config(s3={'use_accelerate_endpoint': True}),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_REGION_NAME"]
    )

    try:
        s3.download_file(bucket_name, faiss_key, faiss_key)
        index = faiss.read_index(faiss_key)
        print("faiss downloaded")

        # Download metadata
        s3.download_file(bucket_name, metadata_key, metadata_key)
        print("metadata downlaoded")
        with open(metadata_key, "rb") as f:
            metadata = pickle.load(f)

        # Reconstruct FAISS vectorstore
        vectorstore = FAISS(
            index=index,
            docstore=metadata["docstore"],
            index_to_docstore_id=metadata["index_to_docstore_id"],
            embedding_function=OpenAIEmbeddings()
        )
        return vectorstore

    except Exception as e:
        raise e

def setup_rag_chain(retriever):
    """Set up the RAG chain with the working retriever"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question about Shopify development,
            formulate a standalone question which can be understood without the chat history."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are a Shopify development expert assistant. Use the following context to answer the question.

Context: {context}

Answer the questions only based on the context provided. If you don't know the answer or can't find it in the context, say so.
And never mention anything about 'context' in your response, that you've taken information or if the information is not available in the 'context'.

If there is anything outside the context provided"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

# def get_ai_response_2(payload):
#     # os.environ['OPENAI_API_KEY'] = getpass("Enter you openai api key: ")
#     vectorstore = load_vectorstore_from_s3("shopibot-docs", "faiss_index.bin", "vectorstore_metadata.pkl")
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#     if vectorstore is None or retriever is None:
#         raise ValueError("Failed to initialize vector store or retriever")

#     rag_chain = setup_rag_chain(retriever)
#     if rag_chain is None:
#         raise ValueError("Failed to set up RAG chain")

#     # Ensure the stream method is asynchronous. You may need to change `rag_chain.stream()` if it's not async.
#     result_generator = rag_chain.stream({
#         "input": payload["query"],
#         "chat_history": payload["history"]
#     })

#     for chunk in result_generator:  # Using async for to consume the asynchronous generator
#         if "answer" in chunk:
#             yield chunk["answer"]

# def truncate_history(history, size):
#     if len(history) > size:
#         history = history[-size:]
#     return history

# @csrf_exempt
# def shopibot_dev(request):
#     # Initialize chat history from session if available
#     chat_history = request.session.get("chat_history", [])

#     if request.method == "POST":
#         message_data = json.loads(request.body.decode("utf-8"))
#         user_message = message_data["message"]

#         # Append user's message to chat history
#         chat_history.append({"role": "user", "content": user_message})

#         message = f"{user_message}\nPlease provide the response in Markdown format with headings, bullet points, and code blocks."
#         payload = {
#             "query": message,
#             "history": chat_history
#         }
#         response_chunks = []
        
#         def stream_and_store_response():
#             nonlocal response_chunks
#             for chunk in get_ai_response_2(payload):
#                 response_chunks.append(chunk)
#                 yield chunk
#             assistant_message = "".join(response_chunks)
#             chat_history.append({"role": "assistant", "content": assistant_message})
#             request.session["chat_history"] = chat_history
#             request.session.modified = True

#         response = StreamingHttpResponse(stream_and_store_response(), status=200, content_type="text/plain")

#         return response

#     return render(request, "shopibot-stream-output.html")

def truncate_history(history, size=10):
    if len(history) > size:
        history = history[-size:]
    return history

def get_ai_response_2(payload):
    """
    Function to generate AI responses using RAG and a retriever.
    """
    vectorstore = load_vectorstore_from_s3("shopibot-docs", "faiss_index.bin", "vectorstore_metadata.pkl")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    if vectorstore is None or retriever is None:
        raise ValueError("Failed to initialize vector store or retriever")

    rag_chain = setup_rag_chain(retriever)
    if rag_chain is None:
        raise ValueError("Failed to set up RAG chain")

    # Generate and stream the response
    result_generator = rag_chain.stream({
        "input": payload["query"],
        "chat_history": payload["history"]
    })

    for chunk in result_generator:
        if "answer" in chunk:
            yield chunk["answer"]

@csrf_exempt
def shopibot_dev(request):
    """
    Django view to handle incoming chat requests and stream AI responses.
    """
    # Chat history sent from the frontend or initialize an empty list
    chat_history = []

    if request.method == "POST":
        # Parse incoming request data
        message_data = json.loads(request.body.decode("utf-8"))
        user_message = message_data["message"]
        frontend_history = message_data.get("history", [])

        # Combine frontend history with the current message
        chat_history = truncate_history(frontend_history)

        # Append the user's new message
        chat_history.append({"role": "user", "content": user_message})

        # Construct the payload for RAG
        query_message = f"{user_message}\nPlease provide the response in Markdown format with headings, bullet points, and code blocks."
        payload = {
            "query": query_message,
            "history": chat_history
        }

        response_chunks = []

        # Stream the response
        def stream_and_store_response():
            nonlocal response_chunks
            for chunk in get_ai_response_2(payload):
                response_chunks.append(chunk)
                yield chunk  # Stream the current chunk

            # Final assistant message
            assistant_message = "".join(response_chunks)
            chat_history.append({"role": "assistant", "content": assistant_message})

        # Return a streaming HTTP response
        response = StreamingHttpResponse(stream_and_store_response(), status=200, content_type="text/plain")

        return response

    return render(request, "shopibot-stream-output.html")