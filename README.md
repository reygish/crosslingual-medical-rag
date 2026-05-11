# Local LLM Medical RAG Evaluation System

A research project comparing Retrieval-Augmented Generation (RAG) vs non-RAG approaches on local (Gemma) models for answering medical questions using Indonesian pharmaceutical data.

## Project Overview

This project evaluates the performance of two approaches for medical question answering:
- **RAG (Retrieval-Augmented Generation)**: Uses retrieved context from medical knowledge base to answer questions
- **No-RAG**: Uses only the language model's internal knowledge to answer questions

The evaluation uses RAGAS (Retrieval-Augmented Generation Assessment) metrics to measure accuracy, faithfulness, and relevance.

## Directory Structure

```
├── rm.ipynb                                    # Main research methodology notebook
├── medDataset_symptoms_treatment.csv           # Test dataset (symptoms & treatment Q&A)
├── med_rag_results.json                        # Generated RAG vs No-RAG responses
├── experiments/                                # Evaluation results
│   ├── rag_eval.csv                            # Full RAG evaluation results
│   ├── norag_eval.csv                          # Full No-RAG evaluation results
│   ├── rag_eval_metrics.csv                    # RAG metrics only
│   ├── norag_eval_metrics.csv                  # No-RAG metrics only
│   ├── norag_scores_all_questions.png          # No-RAG scores for all questions
│   ├── rag_scores_all_questions.png            # RAG scores for all questions
└── Indonesian Pharmaceutical/
    ├── Disease/
    │   └── processed_data_penyakit.csv         # Indonesian disease data
    └── Drug/
        └── processed_data_obat.csv             # Indonesian drug/medication data
```

## Setup

### Prerequisites
- Python 3.14.4
- Ollama (for local LLM inference)
- CUDA or MPS (for GPU acceleration)

## Usage

### 1. Document Preparation
The notebook loads Indonesian pharmaceutical data (diseases and drugs) and transforms them into structured documents with metadata.

### 2. Vector Store Setup
- Uses BAAI/bge-m3 embeddings for vector representation
- Stores embeddings in Chroma vector database
- Splits documents into 500-token chunks with 100-token overlap

### 3. Retrieval System
- Dense retriever: Semantic similarity search via vector embeddings
- BM25 retriever: Keyword-based search
- Ensemble retriever: Combines both with 50/50 weights
- Cross-encoder reranker: BAAI/bge-reranker-v2-m3 for result ranking

### 4. Generate Q&A Responses
Using gemma3:4b to generate answers from the questions in the test dataset provided.

### 5. Evaluate Results
The notebook computes metrics for both approaches:

**RAG Metrics:**
- Faithfulness: How consistent answer is with retrieved context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: Quality of retrieved context
- Context Recall: Coverage of retrieved context
- Answer Correctness: Accuracy of the answer

**No-RAG Metrics:**
- Answer Relevancy
- Answer Correctness

## Evaluation Results

Results are saved to:
- `experiments/rag_eval_metrics.csv`
- `experiments/norag_eval_metrics..csv`

## Configuration

### Models
- **RAG Model**: `gemma3:4b` (4B parameters)
- **Evaluator Model**: `gemma4:31b` (31B parameters)
- **Embedding Model**: `BAAI/bge-m3`
- **Reranker Model**: `BAAI/bge-reranker-v2-m3`

### Batch Settings
- Batch size: 10 samples per evaluation batch
- Max workers: 1 (to prevent rate limiting)
- Request timeout: 180 seconds

### Device
- Uses MPS (Metal Performance Shaders) on macOS
- Falls back to CPU if unavailable

## Dataset

- **Source**: MedQuad-MedicalQnADataset
- **Test Set**: 100 questions (50 symptoms + 50 treatments)
- **Format**: user_input, response, reference, retrieved_contexts

## Key Files

| File | Purpose |
|------|---------|
| rm.ipynb | Main research notebook with complete pipeline |
| med_rag_results.json | Raw generated responses for both approaches |
| rag_eval.csv | Complete RAG evaluation results |
| norag_eval.csv | Complete No-RAG evaluation results |

## Notes

- Evaluation is cached to avoid redundant API calls
- Batch processing helps manage rate limits and memory usage
- Cross-encoding reranking improves retrieval quality
- Results are incrementally saved during batch processing for fault tolerance

## Future Work

- Evaluate with different embedding models
- Compare with other LLMs (Claude, GPT-4, etc.)
- Test different retrieval strategies
- Fine-tune reranker for medical domain
- Expand to multilingual support
- More datasets for retrieval
