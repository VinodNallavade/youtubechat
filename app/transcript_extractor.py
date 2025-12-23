from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os
from dotenv import load_dotenv
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


#setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
transcriptApi = YouTubeTranscriptApi()

load_dotenv()
token_provider= get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

def extract_transcript(video_id):
    """
    Extracts the transcript from a YouTube video given its video ID.

    Args:
        video_id (str): The YouTube video ID.
    """
    try:
       
        transcript_list = transcriptApi.fetch(video_id=video_id,languages=['en'],preserve_formatting=True)
        
        transcript = " ".join([entry.text for entry in transcript_list.snippets])
        logger.info(f"Transcript successfully extracted for video ID: {video_id}")
        return transcript
    except Exception as e:
        logger.error(f"Error extracting transcript for video ID {video_id}: {e}")
        return None
    

def fetch_available_languages(video_id):
    """
    Fetches the available transcript languages for a YouTube video given its video ID.
    Args:
        video_id (str): The YouTube video ID.
    """
    transcriptlist= transcriptApi.list(video_id=video_id)
    for transcript in transcriptlist:
        logger.info(f"Language Code: {transcript.language_code}, Language: {transcript.language}, Is Generated: {transcript.is_generated}")
    return transcriptlist

def translate_transcript(video_id, target_language):
    """
    Translates the transcript of a YouTube video to a target language.
    Args:
        video_id (str): The YouTube video ID.
        target_language (str): The target language code (e.g., 'en' for English).
    """
    try:
        transcript_list = fetch_available_languages(video_id)
        translated_script = transcript_list.find_transcript(["hi"]).translate(target_language)        
        logger.info(f"Transcript successfully translated to {target_language} for video ID: {video_id}")
        return translated_script
    except Exception as e:
        logger.error(f"Error translating transcript for video ID {video_id} to {target_language}: {e}")


def fetch_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)


def text_splitter(transcript, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.create_documents([transcript])
    return texts


def generate_embeddings(texts):
    """
    Generates embeddings for the given texts.
    
    Args:
        texts (list): List of text chunks.
    """
    embedding_model= AzureOpenAIEmbeddings(
    api_version="2024-12-01-preview",
    azure_endpoint="https://az-genai-3016-resource.openai.azure.com/",
    azure_ad_token_provider= token_provider,
    model= "text-embedding-ada-002"
    )
    store = FAISS.from_documents(texts, embedding_model)
    return store


def upload_to_vector_store(texts, embeddings):
    """
    Uploads texts and their embeddings to a FAISS vector store.
    
    Args:
        texts (list): List of text chunks.
        embeddings (list): List of corresponding embeddings.
    """
    
    vector_store = FAISS.add_embeddings(text_embeddings=embeddings)
    logger.info("Successfully uploaded texts and embeddings to the vector store.")
    return vector_store



if __name__ == "__main__":
    # Example usage
    video_id = "j0wJBEZdwLs"  
    # extract transcript
    transcript = extract_transcript(video_id)   
    #Chunking
    texts = text_splitter(transcript)
    print(f"Number of chunks created: {len(texts)}")
    # Generate embeddings
    store = generate_embeddings(texts)
    store.save_local("faiss_index")
    print(f"Vector store contains {store.index} vectors.")
    retriver = store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    while input("Do you want to ask a question about the video? (yes/no): ").lower() == "yes":
        question = input("Enter your question: ")
        docs = retriver.get_relevant_documents(question)
        for doc in docs:
            print(f"Relevant chunk: {doc.page_content}\n")
       
    




    """
    transcript = extract_transcript(video_id)   
    if transcript:
        print(transcript)   
    
    fetch_available_languages(video_id)
    translate_transcript(video_id, target_language='en')
    """
