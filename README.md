
# Retrieval-Augmented Generation (RAG) App for Document Search

This project demonstrates the development of a **Retrieval-Augmented Generation (RAG)** system that utilizes two approaches: one using **llama.cpp** and the other with **Ollama**. The application is designed to retrieve relevant information from local documents, combining the strengths of large language models and vector-based document retrieval.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Method 1: Using llama.cpp](#method-1-using-llamacpp)
- [Method 2: Using Ollama](#method-2-using-ollama)
- [Usage](#usage)
- [Contributing](#contributing)
## Introduction

This project focuses on building a RAG system that retrieves relevant documents and generates answers based on that context. The system supports PDF file uploads and Wikipedia searches. Two different approaches are implemented:

1. **llama.cpp**: A lightweight, local LLM inference using GGUF format models.
2. **Ollama**: A versatile cloud-based model hosting platform that supports LLM inference and embeddings.

The project aims to provide flexibility for NLP-based applications with support for both local and cloud environments.

## Installation

### Prerequisites
- Python 3.10+
- Dependencies listed in `requirements.txt`
- Models from HuggingFace or Ollama

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ibrahimgamal99/Mini_RAG.git
   cd rag-app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download models**:
   - For Ollama, pull the required models using:
     ```bash
     ollama pull llama3.2
     ollama pull nomic-embed-text
     ```
   - For `llama.cpp`, download models from HuggingFace and ensure the format is `.gguf`.
   - OR you can download the model using Ollama, then copy the model file from Ollama’s model path and change the file extension to `.gguf`. 
     - **Windows**: `C:\Users\USERNAME\.ollama\models\blobs`
     - **Linux**: `/usr/share/ollama/.ollama/models/blobs`


## Method 1: Using llama.cpp

In this approach, we use **llama-cpp-python** to run the RAG pipeline locally.

```python
# Load Llama model using llama.cpp
llm = LlamaCpp(model_path="/path/to/gguf/model", verbose=False, n_ctx=4096)
```


## Method 2: Using Ollama

This method involves using **Ollama's Llama 3.2** model along with **nomic-embed-text** for embeddings.

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2")

# Create Chroma vector database using Ollama embeddings
vector_db_Chroma = Chroma.from_documents(
    documents=tokens_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True), # nomic-embed-text embeddings model
    collection_name="rag",
)

```

## Usage

### Running the App

To get started with the RAG app, follow these steps:

## Prerequisites

Ensure you have the required models downloaded. You can obtain them from either HuggingFace or Ollama.

## Instructions

1. **Open the appropriate Jupyter notebook** depending on your task:
   - For working with Ollama model, open `chat_with_pdf.ipynb`.
   - For working with LLama.cpp python, open `chat_with_WiKi.ipynb`.

2. **Run the cells in the notebook step-by-step**. 

3. **Enter your query** when prompted. The system will retrieve relevant documents and generate a context-aware response.

## Note

Make sure to follow the instructions in the notebooks for any additional setup or configurations needed.


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

