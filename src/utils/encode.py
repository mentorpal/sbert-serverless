import numpy as np
import ollama
import requests


def cos_sim_weight(a, b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)

    if a.ndim == 1:
        a = np.expand_dims(a, axis=0)
    if b.ndim == 1:
        b = np.expand_dims(b, axis=0)

    return (
        a
        @ b.T
        / (
            np.linalg.norm(a, axis=1, keepdims=True)
            @ np.linalg.norm(b, axis=1, keepdims=True).T
        )
    )


def encode(sentence):
    return ollama.embeddings(model="mxbai-embed-large", prompt=sentence)["embedding"]


def encode_batch(sentences, batch_size=32, for_testing=False):

    embeddings = []
    if not for_testing:
        for i in range(0, len(sentences) - batch_size + 1, batch_size):
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={
                    "model": "mxbai-embed-large",
                    "input": sentences[i : i + batch_size],
                },
            )
            data = response.json()["embeddings"]
            embeddings.extend(data)
        return embeddings
    else:
        size = len(sentences) // batch_size
        for i in range(0, len(sentences) - batch_size + 1, batch_size):
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={
                    "model": "mxbai-embed-large",
                    "input": sentences[i : i + batch_size],
                },
            )
            data = response.json()["embeddings"]
            embeddings.append(data)
        return embeddings, size
