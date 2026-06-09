#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

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
