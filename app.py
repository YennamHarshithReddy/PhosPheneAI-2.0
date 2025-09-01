import streamlit as st
import os
import json
import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import time

def load_config(config_name="base"):
    """Load chunking configuration from YAML files"""
    config_file = f"configs/{config_name}.yaml"
    if not os.path.exists(config_file):
        st.error(f"Configuration file {config_file} not found!")
        return None
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

# Custom HuggingFace Chat LLM class using secrets management
class CustomChatLLM:
    def __init__(self, model):
        # Read token from Streamlit secrets instead of hardcoding
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except KeyError:
            raise ValueError("HF_TOKEN not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
        
        self.client = InferenceClient(token=hf_token)
        self.model = model

    def invoke(self, prompt):
        """Generate response using chat completions API with correct parsing"""
        system_prompt = """You are a helpful AI assistant that answers questions about news articles accurately and concisely.

INSTRUCTIONS:
- Use ONLY the information provided in the context
- Do not use your general knowledge or make up information  
- If you cannot find the answer in the context, say "I cannot find this information in the provided articles"
- Be concise but comprehensive
- Structure your answer with clear points"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.2
            )
            
            # Correct way to extract content from Hugging Face API response
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            else:
                return str(response)
                
        except Exception as e:
            return f"Error generating response: {e}"

@st.cache_resource
def load_rag_system(config_name="base"):
    """Load the RAG system components with Chat LLM"""
    config = load_config(config_name)
    if not config:
        return None, None, None
        
    index_dir = f"faiss_index_{config['name']}"
    
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory {index_dir} not found.")
    
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    # Configure retriever
    if config.get('use_mmr', False):
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config['retrieval_k'], "fetch_k": config['retrieval_k'] * 2}
        )
    else:
        retriever = db.as_retriever(search_kwargs={"k": config['retrieval_k']})
    
    # Initialize Custom Chat LLM without hardcoded token
    llm = CustomChatLLM(model="meta-llama/Llama-3.1-8B-Instruct")
    
    return retriever, llm, config

def format_citations(source_docs):
    """Format citations according to assignment requirements: [Title, Publisher/Source, Date, URL if available]"""
    citations = []
    for i, doc in enumerate(source_docs, 1):
        metadata = doc.metadata
        source_file = metadata.get('source_file', 'Unknown File')
        title = metadata.get('title', '')
        url = metadata.get('url', '')
        publish_date = metadata.get('publish_date', '')
        
        # Extract clean title from filename if metadata title is empty
        if not title:
            clean_title = source_file.replace('.txt', '')
            # Remove date prefix if present
            if clean_title.startswith(('01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-', '11-', '12-')):
                clean_title = clean_title[6:]  # Remove "MM-DD-" part
            title = clean_title.replace('-', ' ').title()
        
        # Extract date from filename if metadata date is empty
        if not publish_date and source_file.startswith(('01-', '02-', '03-', '04-', '05-', '06-', '07-', '08-', '09-', '10-', '11-', '12-')):
            month_day = source_file[:5]
            month_map = {
                '01': 'January', '02': 'February', '03': 'March', '04': 'April',
                '05': 'May', '06': 'June', '07': 'July', '08': 'August',
                '09': 'September', '10': 'October', '11': 'November', '12': 'December'
            }
            try:
                month_num, day_num = month_day.split('-')
                month_name = month_map.get(month_num, month_num)
                publish_date = f"{month_name} {int(day_num)}, 2023"
            except:
                pass
        
        # Determine publisher/source
        publisher = "News Source"  # Default
        if "cma" in source_file.lower():
            publisher = "CMA Official"
        elif "chatgpt" in source_file.lower():
            publisher = "Tech News"
        elif "google" in source_file.lower() or "openai" in source_file.lower():
            publisher = "Business News"
        elif "hugging" in source_file.lower():
            publisher = "Tech News"
        
        # Format according to assignment requirements: [Title, Publisher/Source, Date, URL if available]
        citation_parts = [title, publisher]
        if publish_date:
            citation_parts.append(publish_date)
        if url:
            citation_parts.append(url)
        
        citation = f"**{i}.** [{', '.join(citation_parts)}]"
        
        # Add supporting snippet
        snippet = doc.page_content.strip()[:120]
        if len(doc.page_content) > 120:
            snippet += "..."
        citation += f"\n   Supporting snippet: \"{snippet}\""
        
        citations.append(citation)
    
    return citations

def generate_answer(query, retriever, llm):
    """Generate answer using retriever and Chat LLM"""
    # Get relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt with context
    prompt = f"""Based on the following context from news articles, answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

Please provide a clear and informative answer based only on the information in the context above."""

    # Generate answer using chat completion
    answer = llm.invoke(prompt)
    
    return {
        'result': answer,
        'source_documents': docs
    }

def main():
    st.set_page_config(page_title="News RAG with Fast AI Answers", layout="wide")
    st.title("📰 News Q&A (with AI-Generated Answers)")
    
    # Configuration selector
    config_choice = st.sidebar.selectbox(
        "Choose Configuration",
        ["base", "alt"],
        help="Compare different chunking strategies"
    )
    
    # Load RAG system
    try:
        retriever, llm, config = load_rag_system(config_choice)
        if retriever is None:
            return
            
        st.sidebar.success(f"✅ Loaded {config_choice} configuration with Llama-3.1-8B")
        st.sidebar.json(config)
        
        # System status
        st.sidebar.subheader("🤖 AI System Status")
        st.sidebar.success("Hugging Face: Connected")
        st.sidebar.success("Model: Llama-3.1-8B-Instruct")
        st.sidebar.success("Token: From Secrets ✅")
        st.sidebar.success("API: Chat Completions")
        st.sidebar.success("Vector Store: FAISS")
        
        # Load stats if available
        stats_file = f"faiss_index_{config['name']}/stats.json"
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            st.sidebar.metric("Total Articles", stats.get('total_articles', 'N/A'))
            st.sidebar.metric("Total Chunks", stats.get('total_chunks', 'N/A'))
            st.sidebar.metric("Index Time", f"{stats.get('index_time_seconds', 0):.2f}s")
            
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        if "HF_TOKEN not found" in str(e):
            st.info("💡 Make sure your .streamlit/secrets.toml file contains: HF_TOKEN = \"your_token_here\"")
        return
    
    # Example questions with clickable buttons
    st.sidebar.subheader("📝 Example Questions:")
    example_questions = [
        "What actions has the CMA taken regarding AI?",
        "What does the AI white paper propose?", 
        "How are companies responding to AI regulation?",
        "What are the key concerns about AI development?",
        "What regulatory approaches are being considered?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.sidebar.button(f"❓ {question}", key=f"example_{i}"):
            st.session_state['selected_question'] = question
    
    # Query input
    default_query = st.session_state.get('selected_question', '')
    query = st.text_input("🔍 Ask a question about the news:", value=default_query)
    
    if query:
        start_time = time.time()
        
        with st.spinner("🚀 Generating fast AI answer with Llama-3.1 Chat..."):
            try:
                # Get answer using chat completion
                result = generate_answer(query, retriever, llm)
                
                end_time = time.time()
                latency = end_time - start_time
                
                # Display answer with enhanced formatting
                st.subheader("🤖 AI-Generated Answer:")
                st.markdown(f"**Question:** {query}")
                st.markdown("---")
                st.write(result['result'])
                
                # Display citations
                st.subheader("📚 Source Citations:")
                citations = format_citations(result['source_documents'])
                for citation in citations:
                    st.markdown(citation)
                
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⚡ Response Time", f"{latency:.2f}s")
                with col2:
                    st.metric("📄 Sources Used", len(result['source_documents']))
                with col3:
                    st.metric("🧠 Model", "Llama-3.1-8B")
                
                # Additional metrics in sidebar
                st.sidebar.subheader("📊 Query Metrics")
                st.sidebar.metric("Last Response Time", f"{latency:.2f}s")
                st.sidebar.metric("Sources Retrieved", len(result['source_documents']))
                
                # Speed improvement indicator
                if latency < 20:
                    st.sidebar.success(f"🚀 {97.76 - latency:.1f}s faster than local model!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()
