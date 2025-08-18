import os
import nltk
from typing import List

nltk.download('punkt')

def read_and_chunk_sentences(
    file_path: str,
    chunk_size: int = 40,
    overlap: int = 10
) -> List[str]:
    """
    Reads a text file, splits it into sentences, and chunks them with overlap.

    Args:
        file_path (str): Path to the text file.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between chunks.

    Returns:
        List[str]: List of sentence chunks as strings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text, language='spanish')
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i:i+chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # Example usage
    # Ensure you have a text file named 'cv1.txt' in the same directory
    # with some content to test the function.
    
    # Uncomment the following lines to test the function:
    chunks = read_and_chunk_sentences("cv1.txt", chunk_size=4, overlap=1)
    
    for i, c in enumerate(chunks):
        print(f"Numero {i}: " + c)
        print(50 * "*")
    print(50*"-")