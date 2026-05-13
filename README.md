# Gemini Embeddings 2

This project demonstrates how to generate and query multimodal embeddings using the `gemini-embedding-2` model via the Google GenAI SDK. It supports generating embeddings for various file types (images, audio, PDF, text) and searching them using cosine similarity.

## Prerequisites
- Python 3.8+

## Setup Instructions

1. **Clone the repository** (if you haven't already) and navigate to the project directory:
   ```bash
   cd path/to/gemini-embeddings-2
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the environment variables**:
   Copy the `.env.example` file to a new file named `.env`:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your Google Gemini API key:
   ```env
   API_KEY=your_gemini_api_key_here
   ```
Get your API_KEY here https://aistudio.google.com/app/api-keys 


## Running the Project

The project workflow consists of two main steps: data ingestion (generating embeddings) and querying.

### 1. Ingestion (Generating Embeddings)
First, you need to generate embeddings for the dataset. The `ingestion.py` script reads files from the `dataset/` directory, sends them to the Gemini API, and saves the resulting embeddings into a local file named `embeddings.json`.

Run the ingestion script:
```bash
python ingestion.py
```
*Wait for the script to finish. It will create or update the `embeddings.json` file in the project root.*

### 2. Querying the Dataset
Once `embeddings.json` has been generated, you can query the dataset. The `query.py` script takes a text query, generates an embedding for it using the Gemini model, and compares it against your dataset using cosine similarity to find the best matches.

Run the query script with the `--q` flag to specify your search query:
```bash
python query.py --q="a cute dog"
```

The script will output the top 5 closest matches from your dataset along with their similarity scores.
