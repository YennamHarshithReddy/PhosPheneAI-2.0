# PhosPhene AIWelcome to PhosPhene AI, a cutting-edge AI-powered news question answering system designed to help you explore and understand the latest developments in artificial intelligence. This project leverages Retrieval-Augmented Generation (RAG) techniques enhanced with large language models (LLMs) for precise and contextually grounded answers.

## Features- Robust ingestion pipeline for heterogeneous news articles (HTML, TXT) extracted from a ZIP archive
- Advanced cleaning and parsing using BeautifulSoup to extract meaningful content and metadata
- Customized chunking strategies to balance retrieval quality and performance
- High-quality embeddings using HuggingFace models with FAISS for efficient vector search
- Pluggable LLM component supporting both local models and cloud inference
- Rich, context-aware answers grounded in retrieved news documents
- Automatic citation generation linking answers to their sources with supporting snippets
- User-friendly Streamlit web app interface for easy interaction

## Getting Started### Prerequisites- Python 3.10 or higher
- pip installed

### Installation & Setup1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/phosphene-ai.git
    cd phosphene-ai
    ```

2. Create and activate your virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac```  .venv\Scripts\activate   # Windows
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your news dataset ZIP file in the `data/news/` directory.

5. Configure your API tokens securely:

    - Create a `.streamlit/secrets.toml` file
    - Add your `HF_TOKEN` (HuggingFace API token) inside it:

        ```toml
        HF_TOKEN = "your_huggingface_token_here"```      ```

### Running the Application1. Preprocess and build the index by running:

    ```bash
    python ingest.py
    ```

2. Launch the web app:

    ```bash
    streamlit run app.py
    ```

3. Use the sidebar to choose configurations and try sample questions.

## Project Structure```
/PhosPheneAI
├── /data/news/               # Raw news dataset ZIP and extracted files```─ /faiss_index_base/        # Persisted FAISS vector index (base config)
├── /faiss_index_alt/         # Persisted FAISS vector```dex (alt config)
├── /configs/                 # Chunking and retrieval configs
│   ├── base.yaml            # Base chunking configuration
│   └── alt.yaml             # Alternative chunking configuration
├── /.streamlit/secrets.toml  # API tokens and private secrets
├── ingest.py                 # Data```gestion and indexing pipeline
├── app.py                   # Streamlit application frontend
├── eval.py                   # Evaluation scripts```r QA and metrics
├── requirements.txt          # Python dependencies```─ README.md                 # Project```cumentation
```

## How It WorksThe system works in two main phases:

1. **Indexing Phase**: The ingestion script parses raw news articles, cleans and normalizes text, removes duplicates, chunks the content, and creates vector embeddings stored in FAISS for fast retrieval.

2. **Query Phase**: When you ask a question, the system retrieves the most relevant chunks from the index and uses a large language model (Llama-3.1-8B via HuggingFace) to generate precise, context-aware answers with proper citations.

## Configuration OptionsThe system supports two different chunking strategies:

- **Base Configuration**: 800 token chunks with 100 token overlap, optimized for precision
- **Alt Configuration**: 1500 token chunks with 200 token overlap, using MMR retrieval for diversity

You can switch between configurations in the sidebar to compare performance and answer quality.

## Example Questions & AnswersTry asking questions like:

1. **What actions has the CMA taken regarding AI?**
   - The CMA outlined five principles on safe, fair, and accountable AI development.

2. **Who coined the term 'foundation models'?**
   - Stanford University's Human-Centered Artificial Intelligence Center coined it in 2021.

3. **What did the AI white paper published in March propose?**
   - It promotes cautious AI development with light governance, avoiding bespoke regulation.

4. **How is the EU planning to regulate AI?**
   - The EU is creating fixed regulatory rules applicable to generative AI.

5. **What are foundation models?**
   - AI models trained on massive data and adaptable across multiple tasks.

## Tips for Extension- Add advanced rerankers (e.g., cross-encoders) for improved retrieval quality
- Incorporate temporal filtering for time-based queries
- Enhance UI with document preview and export features
- Experiment with different embedding models and LLM providers
- Add conversation memory for follow-up questions

## Security Notes- API tokens are stored securely in `.streamlit/secrets.toml`
- Never commit secrets to version control
- The `.streamlit` folder should be added to your `.gitignore`

## LicenseThis project is released under the MIT License.

## AcknowledgmentsThanks to the open-source communities behind HuggingFace, LangChain, FAISS, and Streamlit for making this project possible.

***

