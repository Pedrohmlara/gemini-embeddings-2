import os
import json
import mimetypes
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("API_KEY"))

def get_part(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    
    mime_mappings = {
        ".png": "image/png",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".avif": "image/avif",
        ".pdf": "application/pdf",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mp4": "video/mp4",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
    }
    
    mime_type = mime_mappings.get(ext)
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(file_path)
        
    if not mime_type:
        raise ValueError(f"Unsupported or unknown file type for: {file_path}")
        
    with open("dataset/" + file_path, "rb") as f:
        data = f.read()

    if mime_type == "text/plain":
        return data

    return types.Part.from_bytes(
        data=data, 
        mime_type=mime_type
    )


def gen_embeddings(*items):
    contents = list(items) if len(items) > 1 else items[0]
    print("CONTENT", contents)
    
    return client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents
    )


def embed():
    dataset = [
        "baby.png",
        "black_dog.png",
        "dogs_loyalty.wav",
        "real_state.pdf",
        "san_francisco.png",
        "shiba.png",
        "shiba.txt",
        "ultrassound.png",
        "white_dog.png",
        "zion.png",
    ]

    embeddings = {}
    for data in dataset:
        embeddings[data] = gen_embeddings(get_part(data)).embeddings[0].values

    with open("embeddings.json", "w") as f:
        json.dump(embeddings, f)


if __name__ == "__main__":
    embed()