# CaLM - Caregiving Large Language Model

An intelligent RAG (Retrieval-Augmented Generation) system designed to assist caregivers of patients with Alzheimer's Disease and Related Dementias (ADRD). This system leverages local LLMs through Ollama and Langchain to provide contextual, accurate, and helpful information for family caregivers.

![Rag workflow](./public/calm-workflow.png)

## 🌟 Features

### AI Memory
Memory are categoriezed into two level Long term memory (LTM) and short term memory (STM), Bio, job, social relationship, relationship with care recipient, topics if interest to user etc.,are considered as global level memory which is LTM and stored in the database. Preferences, answer tone, language, etc., are considered as short term memory. They will be stored in runtime variable. Some of them are loaded from disk while others are inferred from rounds of converation.

All memory will be generated and express as a sentences.

## 🚀 Getting Started

### Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd LangchainRag_Ollama
```

2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Ollama service locally

## 💡 Usage

1. **Data Ingestion**:
   - Place your knowledge base documents in `data/raw_content/`
   - Run the vectorization script to process documents
   - Use PubMed auto-search for latest research updates

2. **Query the System**:
   - Use the main notebook or Python interface
   - Get contextual responses based on the knowledge base
   - Access medical information and caregiving advice

## 🔒 Privacy & Security

- All processing is done locally
- No sensitive data is sent to external services
- Secure storage of medical and personal information

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Healthcare professionals and caregivers who provided domain expertise
- Open-source community for tools and libraries
- Research papers and medical resources that form our knowledge base
