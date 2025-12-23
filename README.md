# YouTube Chat RAG Application

A Retrieval-Augmented Generation (RAG) application that extracts YouTube video transcripts, chunks them, generates embeddings, and enables semantic search with LLM-powered responses.

## Features

- Extract transcripts from YouTube videos
- Support for multiple languages with translation capability
- Text chunking with customizable parameters
- Vector embeddings using Azure OpenAI
- FAISS vector store for efficient similarity search
- Interactive Q&A interface over video content

## Architecture

```
YouTube Video → Transcript Extraction → Text Chunking → Embeddings → Vector DB → Retriever → LLM → Response
```

## Installation

```bash
pip install youtube-transcript-api langchain-text-splitters langchain-openai azure-identity faiss-cpu python-dotenv
```

## Configuration

Create a `.env` file with:
```
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=your_endpoint
```

## Usage

```python
from transcript_extractor import extract_transcript, text_splitter, generate_embeddings

# Extract and process video
video_id = "j0wJBEZdwLs"
transcript = extract_transcript(video_id)
texts = text_splitter(transcript)
store = generate_embeddings(texts)

# Query
question = "What is the main topic?"
docs = store.as_retriever(search_kwargs={"k": 2}).get_relevant_documents(question)
```

## Key Functions

- `extract_transcript()` - Get video transcript
- `fetch_available_languages()` - List available languages
- `translate_transcript()` - Translate to target language
- `text_splitter()` - Chunk text with overlap
- `generate_embeddings()` - Create vector embeddings
- `upload_to_vector_store()` - Store vectors
