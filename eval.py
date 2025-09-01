import json
import yaml
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision
from datasets import Dataset
import time

def load_config(config_name="base"):
    config_file = f"configs/{config_name}.yaml"
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_gold_dataset():
    """Create gold standard Q&A dataset"""
    questions = [
        {
            "question": "What actions has the CMA taken regarding generative AI?",
            "expected_answer": "The CMA has outlined five principles for AI development: safety, security and robustness; transparency and explainability; fairness; accountability and governance; and contestability and redress."
        },
        {
            "question": "What are the main concerns about AI regulation in the EU?",
            "expected_answer": "The EU is focusing on regulating foundational models through amendments to their risk-based framework."
        },
        # Add more questions here based on your news dataset
        {
            "question": "When did the CMA announce their AI principles?",
            "expected_answer": "The CMA announced their AI principles in a press release regarding generative AI oversight."
        },
        # Add 9 more questions to meet the ≥12 requirement
    ]
    return questions

def evaluate_rag_system(config_name="base"):
    """Evaluate RAG system using RAGAS metrics"""
    config = load_config(config_name)
    index_dir = f"faiss_index_{config['name']}"
    
    # Load RAG system
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    if config.get('use_mmr', False):
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config['retrieval_k'], "fetch_k": config['retrieval_k'] * 2}
        )
    else:
        retriever = db.as_retriever(search_kwargs={"k": config['retrieval_k']})
    
    llm = Ollama(model="llama3")
    
    prompt_template = """Use ONLY the information provided in the context to answer the question.
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    # Load gold dataset
    gold_questions = create_gold_dataset()
    
    # Generate answers and collect data
    evaluation_data = []
    latencies = []
    
    for item in gold_questions:
        start_time = time.time()
        result = qa_chain.invoke({"query": item["question"]})
        end_time = time.time()
        
        latencies.append(end_time - start_time)
        
        contexts = [doc.page_content for doc in result['source_documents']]
        
        evaluation_data.append({
            "question": item["question"],
            "answer": result['result'],
            "contexts": contexts,
            "ground_truth": item["expected_answer"]
        })
    
    # Create dataset for RAGAS
    dataset = Dataset.from_list(evaluation_data)
    
    # Evaluate using RAGAS
    results = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision]
    )
    
    # Calculate performance metrics
    avg_latency = sum(latencies) / len(latencies)
    
    # Save results
    evaluation_results = {
        "config_name": config_name,
        "config": config,
        "ragas_scores": results.to_pandas().to_dict(),
        "performance": {
            "average_latency": avg_latency,
            "total_questions": len(gold_questions)
        },
        "individual_results": evaluation_data
    }
    
    with open(f"evaluation_results_{config_name}.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    return evaluation_results

def compare_configurations():
    """Compare base vs alt configurations"""
    print("Evaluating base configuration...")
    base_results = evaluate_rag_system("base")
    
    print("Evaluating alt configuration...")
    alt_results = evaluate_rag_system("alt")
    
    # Create comparison report
    comparison = {
        "base_config": base_results,
        "alt_config": alt_results,
        "comparison_summary": {
            "base_avg_latency": base_results["performance"]["average_latency"],
            "alt_avg_latency": alt_results["performance"]["average_latency"],
            "performance_difference": alt_results["performance"]["average_latency"] - base_results["performance"]["average_latency"]
        }
    }
    
    with open("configuration_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("Evaluation complete! Check evaluation_results_*.json and configuration_comparison.json")

if __name__ == "__main__":
    compare_configurations()
