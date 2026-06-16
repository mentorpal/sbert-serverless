#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import numpy as np
from typing import List
from openai import OpenAI


def cos_sim_weight(a: List[List[float]], b: List[List[float]]) -> np.ndarray:

    a_array = np.array(a)

    b_array = np.array(b)

    if a_array.ndim == 1:
        a_array = np.expand_dims(a_array, axis=0)
    if b_array.ndim == 1:
        b_array = np.expand_dims(b_array, axis=0)

    a_norms = np.linalg.norm(a_array, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b_array, axis=1, keepdims=True)

    norm_product = a_norms @ b_norms.T
    return a_array @ b_array.T / norm_product


def encode(sentences):
    client = OpenAI()
    response = client.embeddings.create(model="text-embedding-3-large", input=sentences)
    if isinstance(sentences, str):
        return response.data[0].embedding
    else:
        return [emb.embedding for emb in response.data]
