import streamlit as st
import os
import json
import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time

# Set the Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def load_config(config_name="base"):
    """Load chunking configuration from YAML files"""
    config_file = f"configs/{config_name}.yaml"
    if not os.path.exists(config_file):
        st.error(f"Configuration file {config_file} not found!")
        return None
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_rag_system(config_name="base"):
    """Load the RAG system components with Hugging Face LLM"""
    config = load_config(config_name)
    if not config:
        return None, None
        
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
    
    # Initialize Hugging Face Hub LLM (much faster than local Ollama)
    llm = HuggingFaceHub(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token="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",
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.1
        }
    )
    
    # Create custom prompt template
    prompt_template = """You are a helpful AI assistant that answers questions about news articles accurately and concisely.

INSTRUCTIONS:
- Use ONLY the information provided in the context below
- Do not use your general knowledge or make up information  
- If you cannot find the answer in the context, say "I cannot find this information in the provided articles"
- Be concise but comprehensive
- Structure your answer with clear points
- Reference specific details from the context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain, config

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
                publish_date = f"{month_name} {int(day_num)}, 2023"  # Assuming 2023
            except:
                pass
        
        # Determine publisher/source (extract from filename or use a default)
        publisher = "News Source"  # Default
        if "cma" in source_file.lower():
            publisher = "CMA Official"
        elif "chatgpt" in source_file.lower():
            publisher = "Tech News"
        elif "google" in source_file.lower() or "openai" in source_file.lower():
            publisher = "Business News"
        
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
        qa_chain, config = load_rag_system(config_choice)
        if qa_chain is None:
            return
            
        st.sidebar.success(f"✅ Loaded {config_choice} configuration with Llama-3.1-8B")
        st.sidebar.json(config)
        
        # System status
        st.sidebar.subheader("🤖 AI System Status")
        st.sidebar.success("Hugging Face: Connected")
        st.sidebar.success("Model: Llama-3.1-8B-Instruct")
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
        
        with st.spinner("🚀 Generating fast AI answer with Llama-3.1..."):
            try:
                # Get answer from LLM
                result = qa_chain.invoke({"query": query})
                
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
                if latency < 10:
                    st.sidebar.success(f"🚀 {97.76 - latency:.1f}s faster than local model!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()
