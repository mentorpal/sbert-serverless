import numpy as np
import ollama


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
