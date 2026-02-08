# EVA-Pharma-AI-Customer-Support-Agent
## RAG-Optimized Customer Support Agent

An automated customer support solution utilizing Llama-3-8B and FAISS-based Retrieval Augmented Generation (RAG) to categorize, prioritize, and respond to customer inquiries with grounded citations.

Key Features
- RAG Integration: Uses a Knowledge Base (historical tickets and articles) to provide context-aware responses.
- Batch Processing: Optimized `ask_llama_batch` function reducing inference time by ~13% compared to serial processing.
- Hardware-Aware Design: Built for NVIDIA T4 GPUs using 4-bit quantization and smart padding/sorting strategies.
- Citations: Automatic extraction of `article_id` and `ticket_id` to ensure response auditability.

Requirements
- `torch` & `transformers`
- `sentence-transformers` (for RAG embeddings)
- `faiss-cpu` / `faiss-gpu`
- `bitsandbytes` (for 4-bit quantization)

File Structure
- `CH_03_MuhammedAhmedEsmail_Solution.ipynb`: Main execution notebook.
- `/data/kb_articles.jsonl`: Given knowledge base articles.
- `/data/tickets.csv`: Given ticket inputs.
- `/data/tickets_labeled.csv`: Given ticket labeles.
- `/output/predictions.jsonl`: Final generated responses in the required schema.

Optimization Settings
To achieve maximum performance on a T4 GPU, the following settings are pre-configured:
- Batch Size: [8 - 24]
- Quantization: 4-bit (NF4)
- Tokenization: Left-padding
- Data Pipeline: Tickets are sorted by length before processing to minimize padding overhead.

Usage
1. Load your training and test datasets.
2. Initialize the FAISS index using the provided `RAG_model` (SentenceTransformer).
3. Run `process_tickets_optimized` to generate responses.

Expected Runtime: >10s per ticket, ~137s for a batch of 16 tickets. 

Notes:
- Make sure to select T4 GPU from runtime -> change runtime type to not receive a "cuda compilation" error.
- Follow the correct schema as defined in the challenge statement.
- Do not run the model loading multiple types as it will drain your Google Colab resources unnecessarily.
