# RAG System Evaluation Report
*Generated on 2025-09-02T10:27:40.314393*

## Executive Summary
This report evaluates two RAG configurations on a news article dataset with 15 test questions spanning multiple categories (who/what/when/where/how, comparisons, temporal queries).

## Configuration Comparison
| Metric | BASE Config | ALT Config | Difference |
|--------|-------------|------------|------------|
| **Chunk Size** | 800 | 1500 | +700 |
| **Chunk Overlap** | 100 | 200 | +100 |
| **Retrieval K** | 5 | 4 | -1 |
| **Search Type** | Similarity | MMR | Base: False, Alt: True |

## Performance Results
### Latency Analysis
- **BASE Average Latency**: 0.85s
- **ALT Average Latency**: 0.45s
- **Difference**: -0.40s (+46.8% improvement)

### Success Rates
- **BASE Success Rate**: 100.0% (15/15)
- **ALT Success Rate**: 100.0% (15/15)

### Token Usage
- **BASE Average Tokens**: 609
- **ALT Average Tokens**: 751
- **Difference**: +143 tokens

## Question Categories Analysis
### BASE Configuration
- **When**: 2 questions
- **Where**: 1 questions
- **Who**: 3 questions
- **Temporal**: 1 questions
- **How**: 2 questions
- **What**: 4 questions
- **Comparison**: 2 questions

### ALT Configuration
- **When**: 2 questions
- **Where**: 1 questions
- **Who**: 3 questions
- **Temporal**: 1 questions
- **How**: 2 questions
- **What**: 4 questions
- **Comparison**: 2 questions

## Key Findings
### Performance Winner
- **Latency**: ALT configuration performed better
- **Success Rate**: ALT configuration performed better

### Configuration Trade-offs
- **BASE**: Smaller chunks (800), lower overlap (100), similarity search, k=5
- **ALT**: Larger chunks (1500), higher overlap (200), MMR search, k=4

## Recommendations
Based on the evaluation results:
1. **For Speed**: Use ALT configuration for faster response times
2. **For Reliability**: Use ALT configuration for higher success rates
3. **For Resource Efficiency**: Consider token usage differences when choosing configuration

## Technical Notes
- Evaluation conducted with Llama-3.1-8B-Instruct model
- 15 gold standard questions across multiple difficulty levels
- FAISS vector store with sentence-transformers/all-MiniLM-L6-v2 embeddings
- RAGAS metrics included

## Data Files
- `evaluation_results_base.json`: Detailed BASE configuration results
- `evaluation_results_alt.json`: Detailed ALT configuration results  
- `comprehensive_evaluation_report.json`: Complete comparison data