import json
import yaml
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
import time
import os
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
try:
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness, context_precision
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("RAGAS not available. Install with: pip install ragas datasets")
    RAGAS_AVAILABLE = False
def load_config(config_name="base"):
    config_file = f"configs/{config_name}.yaml"
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)
class CustomChatLLM:
    def __init__(self, model):
        try:
            try:
                hf_token = st.secrets["HF_TOKEN"]
            except:
                hf_token = os.environ.get("HF_TOKEN")
                if not hf_token:
                    raise ValueError("HF_TOKEN not found in environment or Streamlit secrets")
        except:
            raise ValueError("HF_TOKEN not found. Set environment variable or Streamlit secrets")
        self.client = InferenceClient(token=hf_token)
        self.model = model
    def invoke(self, prompt):
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
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            else:
                return str(response)
        except Exception as e:
            return f"Error generating response: {e}"
def create_gold_dataset():
    questions = [
        {
            "question": "Who announced the AI principles for generative AI oversight?",
            "expected_answer": "The CMA (Competition and Markets Authority) announced AI principles for generative AI oversight.",
            "category": "who",
            "difficulty": "easy"
        },
        {
            "question": "Who are the key investors mentioned in recent AI startup funding rounds?",
            "expected_answer": "Key investors include Sequoia Capital, Andreessen Horowitz, Thrive, and K2 Global.",
            "category": "who", 
            "difficulty": "medium"
        },
        {
            "question": "What are the main AI regulation approaches being considered?",
            "expected_answer": "Main approaches include risk-based frameworks, principles-based oversight, and specific requirements for foundational models.",
            "category": "what",
            "difficulty": "medium"
        },
        {
            "question": "What investment amount did OpenAI receive recently?",
            "expected_answer": "OpenAI received significant investment funding, including mentions of multi-billion dollar valuations and funding rounds.",
            "category": "what",
            "difficulty": "easy"
        },
        {
            "question": "When did the major AI investment announcements occur?",
            "expected_answer": "Major AI investment announcements occurred in 2023, with specific dates mentioned in May and other months.",
            "category": "when",
            "difficulty": "medium"
        },
        {
            "question": "When was the AI white paper or regulatory framework announced?",
            "expected_answer": "AI regulatory frameworks and white papers were announced in 2023, with specific timing varying by jurisdiction.",
            "category": "when",
            "difficulty": "medium"
        },
        {
            "question": "Where are the main AI regulation efforts taking place?",
            "expected_answer": "Main AI regulation efforts are taking place in the UK, EU, and US, with different approaches in each jurisdiction.",
            "category": "where",
            "difficulty": "easy"
        },
        {
            "question": "How are companies responding to AI regulation proposals?",
            "expected_answer": "Companies are engaging with regulators, adjusting development practices, and providing input on regulatory frameworks.",
            "category": "how",
            "difficulty": "medium"
        },
        {
            "question": "How do the proposed AI regulations differ across jurisdictions?",
            "expected_answer": "Regulations differ in scope, enforcement mechanisms, and focus areas, with some emphasizing principles while others focus on specific requirements.",
            "category": "how",
            "difficulty": "hard"
        },
        {
            "question": "Compare the AI investment trends mentioned across different articles.",
            "expected_answer": "AI investments show consistent growth with increasing valuations, diverse investor participation, and focus on generative AI capabilities.",
            "category": "comparison",
            "difficulty": "hard"
        },
        {
            "question": "What are the different perspectives on AI safety mentioned in the articles?",
            "expected_answer": "Perspectives range from emphasis on rapid innovation to calls for careful oversight and risk management.",
            "category": "comparison", 
            "difficulty": "hard"
        },
        {
            "question": "How have AI regulatory discussions evolved over time according to the articles?",
            "expected_answer": "Regulatory discussions have evolved from general AI concerns to specific frameworks for generative AI and foundational models.",
            "category": "temporal",
            "difficulty": "hard"
        },
        {
            "question": "What are the key concerns about AI development mentioned in the articles?",
            "expected_answer": "Key concerns include safety, transparency, fairness, accountability, and potential market concentration.",
            "category": "what",
            "difficulty": "medium"
        },
        {
            "question": "Which AI companies are mentioned as receiving significant funding?",
            "expected_answer": "Companies mentioned include OpenAI, Hugging Face, Anthropic, and various AI startups.",
            "category": "who",
            "difficulty": "easy"
        },
        {
            "question": "What role does the CMA play in AI oversight according to the articles?",
            "expected_answer": "The CMA plays a regulatory oversight role, establishing principles and frameworks for AI governance.",
            "category": "what",
            "difficulty": "easy"
        }
    ]
    print(f"Created gold dataset with {len(questions)} questions")
    print(f"Categories: {set([q['category'] for q in questions])}")
    print(f"Difficulty levels: {set([q['difficulty'] for q in questions])}")
    return questions
def load_rag_system(config_name="base"):
    config = load_config(config_name)
    index_dir = f"faiss_index_{config['name']}"
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Index directory {index_dir} not found. Run ingest.py first.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    if config.get('use_mmr', False):
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config['retrieval_k'], "fetch_k": config['retrieval_k'] * 2}
        )
    else:
        retriever = db.as_retriever(search_kwargs={"k": config['retrieval_k']})
    llm = CustomChatLLM(model="meta-llama/Llama-3.1-8B-Instruct")
    return retriever, llm, config
def generate_answer(query, retriever, llm):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Based on the following context from news articles, answer the question accurately and concisely.
CONTEXT:
{context}
QUESTION: {query}
Please provide a clear and informative answer based only on the information in the context above."""
    answer = llm.invoke(prompt)
    return {
        'result': answer,
        'source_documents': docs
    }
def simple_faithfulness_check(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    answer_lower = answer.lower()
    context_text = " ".join(contexts).lower()
    answer_words = set(answer_lower.split())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    answer_words = answer_words - stop_words
    if not answer_words:
        return 0.5
    found_words = sum(1 for word in answer_words if word in context_text)
    return found_words / len(answer_words)
def evaluate_rag_system(config_name="base"):
    print(f"\n=== Evaluating {config_name.upper()} Configuration ===")
    try:
        retriever, llm, config = load_rag_system(config_name)
        print(f"✓ Loaded {config_name} configuration successfully")
    except Exception as e:
        print(f"✗ Failed to load {config_name} configuration: {e}")
        return None
    gold_questions = create_gold_dataset()
    evaluation_data = []
    latencies = []
    token_counts = []
    print(f"Running evaluation on {len(gold_questions)} questions...")
    for i, item in enumerate(gold_questions, 1):
        print(f"Processing question {i}/{len(gold_questions)}: {item['question'][:50]}...")
        start_time = time.time()
        try:
            result = generate_answer(item["question"], retriever, llm)
            end_time = time.time()
            latency = end_time - start_time
            contexts = [doc.page_content for doc in result['source_documents']]
            estimated_tokens = len(result['result'].split()) + sum(len(ctx.split()) for ctx in contexts)
            evaluation_data.append({
                "question": item["question"],
                "answer": result['result'],
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "num_sources": len(result['source_documents']),
                "latency": latency,
                "estimated_tokens": estimated_tokens
            })
            latencies.append(latency)
            token_counts.append(estimated_tokens)
        except Exception as e:
            print(f"✗ Error processing question {i}: {e}")
            evaluation_data.append({
                "question": item["question"],
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": item["expected_answer"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "num_sources": 0,
                "latency": 0,
                "estimated_tokens": 0
            })
    performance_metrics = {
        "average_latency": np.mean(latencies) if latencies else 0,
        "median_latency": np.median(latencies) if latencies else 0,
        "min_latency": np.min(latencies) if latencies else 0,
        "max_latency": np.max(latencies) if latencies else 0,
        "total_questions": len(gold_questions),
        "successful_questions": len([d for d in evaluation_data if not d["answer"].startswith("ERROR")]),
        "average_tokens": np.mean(token_counts) if token_counts else 0,
        "total_estimated_tokens": sum(token_counts)
    }
    ragas_scores = {}
    if RAGAS_AVAILABLE and evaluation_data:
        try:
            print("Running RAGAS evaluation...")
            valid_data = [d for d in evaluation_data if not d["answer"].startswith("ERROR") and d["contexts"]]
            if valid_data:
                dataset = Dataset.from_list(valid_data)
                ragas_result = evaluate(
                    dataset,
                    metrics=[answer_relevancy, faithfulness, context_precision]
                )
                ragas_scores = ragas_result.to_dict()
                print("✓ RAGAS evaluation completed")
            else:
                print("✗ No valid data for RAGAS evaluation")
        except Exception as e:
            print(f"✗ RAGAS evaluation failed: {e}")
            ragas_scores = {"error": str(e)}
    else:
        print("Running simple faithfulness check...")
        faithfulness_scores = []
        for item in evaluation_data:
            if not item["answer"].startswith("ERROR") and item["contexts"]:
                score = simple_faithfulness_check(item["answer"], item["contexts"])
                faithfulness_scores.append(score)
        ragas_scores = {
            "simple_faithfulness_mean": np.mean(faithfulness_scores) if faithfulness_scores else 0,
            "simple_faithfulness_std": np.std(faithfulness_scores) if faithfulness_scores else 0
        }
    evaluation_results = {
        "config_name": config_name,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "performance_metrics": performance_metrics,
        "ragas_scores": ragas_scores,
        "detailed_results": evaluation_data,
        "category_breakdown": {
            category: len([d for d in evaluation_data if d["category"] == category])
            for category in set([d["category"] for d in evaluation_data])
        },
        "difficulty_breakdown": {
            difficulty: len([d for d in evaluation_data if d["difficulty"] == difficulty])
            for difficulty in set([d["difficulty"] for d in evaluation_data])
        }
    }
    results_file = f"evaluation_results_{config_name}.json"
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    print(f"✓ Results saved to {results_file}")
    print(f"✓ Average latency: {performance_metrics['average_latency']:.2f}s")
    print(f"✓ Success rate: {performance_metrics['successful_questions']}/{performance_metrics['total_questions']}")
    return evaluation_results
def compare_configurations():
    print("\n" + "="*60)
    print("STARTING COMPREHENSIVE RAG EVALUATION")
    print("="*60)
    print("\n🔄 Evaluating BASE configuration...")
    base_results = evaluate_rag_system("base")
    print("\n🔄 Evaluating ALT configuration...")
    alt_results = evaluate_rag_system("alt")
    if not base_results or not alt_results:
        print("❌ Evaluation failed for one or both configurations")
        return
    comparison = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "base_config": base_results,
        "alt_config": alt_results,
        "comparison_summary": {
            "base_avg_latency": base_results["performance_metrics"]["average_latency"],
            "alt_avg_latency": alt_results["performance_metrics"]["average_latency"],
            "latency_difference": alt_results["performance_metrics"]["average_latency"] - base_results["performance_metrics"]["average_latency"],
            "latency_improvement_pct": ((base_results["performance_metrics"]["average_latency"] - alt_results["performance_metrics"]["average_latency"]) / base_results["performance_metrics"]["average_latency"] * 100) if base_results["performance_metrics"]["average_latency"] > 0 else 0,
            "base_avg_tokens": base_results["performance_metrics"]["average_tokens"],
            "alt_avg_tokens": alt_results["performance_metrics"]["average_tokens"],
            "token_difference": alt_results["performance_metrics"]["average_tokens"] - base_results["performance_metrics"]["average_tokens"],
            "base_success_rate": base_results["performance_metrics"]["successful_questions"] / base_results["performance_metrics"]["total_questions"],
            "alt_success_rate": alt_results["performance_metrics"]["successful_questions"] / alt_results["performance_metrics"]["total_questions"],
            "chunk_size_difference": alt_results["config"]["chunk_size"] - base_results["config"]["chunk_size"],
            "overlap_difference": alt_results["config"]["chunk_overlap"] - base_results["config"]["chunk_overlap"],
            "retrieval_k_difference": alt_results["config"]["retrieval_k"] - base_results["config"]["retrieval_k"],
            "mmr_comparison": f"Base: {base_results['config'].get('use_mmr', False)}, Alt: {alt_results['config'].get('use_mmr', False)}"
        },
        "analysis": {
            "better_latency": "base" if base_results["performance_metrics"]["average_latency"] < alt_results["performance_metrics"]["average_latency"] else "alt",
            "better_success_rate": "base" if (base_results["performance_metrics"]["successful_questions"] / base_results["performance_metrics"]["total_questions"]) > (alt_results["performance_metrics"]["successful_questions"] / alt_results["performance_metrics"]["total_questions"]) else "alt",
            "configuration_trade_offs": {
                "base": "Smaller chunks (800), lower overlap (100), similarity search, k=5",
                "alt": "Larger chunks (1500), higher overlap (200), MMR search, k=4"
            }
        }
    }
    if base_results.get("ragas_scores") and alt_results.get("ragas_scores"):
        if "faithfulness" in str(base_results["ragas_scores"]) and "faithfulness" in str(alt_results["ragas_scores"]):
            comparison["ragas_comparison"] = {
                "base_ragas": base_results["ragas_scores"],
                "alt_ragas": alt_results["ragas_scores"]
            }
    comparison_file = "comprehensive_evaluation_report.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    generate_summary_report(comparison)
    print(f"\n✅ EVALUATION COMPLETE!")
    print(f"📊 Detailed results: {comparison_file}")
    print(f"📄 Summary report: evaluation_summary_report.md")
    print("\n" + "="*60)
def generate_summary_report(comparison):
    base_perf = comparison["base_config"]["performance_metrics"]
    alt_perf = comparison["alt_config"]["performance_metrics"]
    summary = comparison["comparison_summary"]
    
    improvement_word = 'improvement' if summary["latency_improvement_pct"] > 0 else 'degradation'
    base_search = 'MMR' if comparison["base_config"]["config"].get("use_mmr") else 'Similarity'
    alt_search = 'MMR' if comparison["alt_config"]["config"].get("use_mmr") else 'Similarity'
    better_speed_config = 'BASE' if comparison["analysis"]["better_latency"] == 'base' else 'ALT'
    better_reliability_config = 'BASE' if comparison["analysis"]["better_success_rate"] == 'base' else 'ALT'
    ragas_status = 'RAGAS metrics included' if 'ragas_comparison' in comparison else 'Simple faithfulness checking used'
    
    report = f"""# RAG System Evaluation Report
*Generated on {comparison["evaluation_timestamp"]}*

## Executive Summary
This report evaluates two RAG configurations on a news article dataset with {base_perf["total_questions"]} test questions spanning multiple categories (who/what/when/where/how, comparisons, temporal queries).

## Configuration Comparison
| Metric | BASE Config | ALT Config | Difference |
|--------|-------------|------------|------------|
| **Chunk Size** | {comparison["base_config"]["config"]["chunk_size"]} | {comparison["alt_config"]["config"]["chunk_size"]} | {summary["chunk_size_difference"]:+d} |
| **Chunk Overlap** | {comparison["base_config"]["config"]["chunk_overlap"]} | {comparison["alt_config"]["config"]["chunk_overlap"]} | {summary["overlap_difference"]:+d} |
| **Retrieval K** | {comparison["base_config"]["config"]["retrieval_k"]} | {comparison["alt_config"]["config"]["retrieval_k"]} | {summary["retrieval_k_difference"]:+d} |
| **Search Type** | {base_search} | {alt_search} | {summary["mmr_comparison"]} |

## Performance Results
### Latency Analysis
- **BASE Average Latency**: {base_perf["average_latency"]:.2f}s
- **ALT Average Latency**: {alt_perf["average_latency"]:.2f}s
- **Difference**: {summary["latency_difference"]:+.2f}s ({summary["latency_improvement_pct"]:+.1f}% {improvement_word})

### Success Rates
- **BASE Success Rate**: {summary["base_success_rate"]:.1%} ({base_perf["successful_questions"]}/{base_perf["total_questions"]})
- **ALT Success Rate**: {summary["alt_success_rate"]:.1%} ({alt_perf["successful_questions"]}/{alt_perf["total_questions"]})

### Token Usage
- **BASE Average Tokens**: {base_perf["average_tokens"]:.0f}
- **ALT Average Tokens**: {alt_perf["average_tokens"]:.0f}
- **Difference**: {summary["token_difference"]:+.0f} tokens

## Question Categories Analysis
### BASE Configuration"""
    
    for category, count in comparison["base_config"]["category_breakdown"].items():
        report += f"\n- **{category.title()}**: {count} questions"
    
    report += "\n\n### ALT Configuration"
    for category, count in comparison["alt_config"]["category_breakdown"].items():
        report += f"\n- **{category.title()}**: {count} questions"
    
    report += f"""

## Key Findings
### Performance Winner
- **Latency**: {comparison["analysis"]["better_latency"].upper()} configuration performed better
- **Success Rate**: {comparison["analysis"]["better_success_rate"].upper()} configuration performed better

### Configuration Trade-offs
- **BASE**: {comparison["analysis"]["configuration_trade_offs"]["base"]}
- **ALT**: {comparison["analysis"]["configuration_trade_offs"]["alt"]}

## Recommendations
Based on the evaluation results:
1. **For Speed**: Use {better_speed_config} configuration for faster response times
2. **For Reliability**: Use {better_reliability_config} configuration for higher success rates
3. **For Resource Efficiency**: Consider token usage differences when choosing configuration

## Technical Notes
- Evaluation conducted with Llama-3.1-8B-Instruct model
- {base_perf["total_questions"]} gold standard questions across multiple difficulty levels
- FAISS vector store with sentence-transformers/all-MiniLM-L6-v2 embeddings
- {ragas_status}

## Data Files
- `evaluation_results_base.json`: Detailed BASE configuration results
- `evaluation_results_alt.json`: Detailed ALT configuration results  
- `comprehensive_evaluation_report.json`: Complete comparison data"""

    with open("evaluation_summary_report.md", 'w') as f:
        f.write(report)
        
if __name__ == "__main__":
    if not os.path.exists("faiss_index_base") or not os.path.exists("faiss_index_alt"):
        print("❌ Required FAISS indices not found!")
        print("Please run the following commands first:")
        print("1. python ingest.py  # (with base config)")
        print("2. Edit ingest.py to use 'base' config and run again")
        print("   OR create both indices as needed")
        exit(1)
    compare_configurations()
