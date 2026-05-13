import argparse
import json
import os
import math
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser(description="Query embeddings dataset.")
    parser.add_argument("--q", required=True, type=str, help="The query text")
    args = parser.parse_args()

    query_text = args.q
    
    with open("embeddings.json", "r") as f:
            dataset_embeddings = json.load(f)

    # Generate embedding for the query
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query_text
    )
    query_embedding = response.embeddings[0].values


    results = []
    for data, embedding in dataset_embeddings.items():
        sim = cosine_similarity(query_embedding, embedding)
        results.append((data, sim))

    # Sort results by highest similarity
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Matches:")
    for data, sim in results[:5]:
        print(f"[{sim:.4f}] {data}")


if __name__ == "__main__":
    main()
