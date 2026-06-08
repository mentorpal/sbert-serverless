import torch, json, queue, requests
from typing import List
from src.utils.encode import cos_sim_weight, encode
from torch import topk
from src.utils.http_utils import create_json_response


def paraphrase_mining(
    # self,
    sentences: List[str],
    batch_size: int = 32,
    query_chunk_size: int = 5000,
    corpus_chunk_size: int = 100000,
    max_pairs: int = 500000,
    top_k: int = 100,
):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param model: SentenceTransformer model for embedding computation
    :param sentences: A list of strings (texts or sentences)
    :param batch_size: Number of texts that are encoded simultaneously by the model
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease, to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :return: Returns a list of triplets with the format [score, id1, id2]
    """

    # Compute embedding for the sentences

    embeddings = []
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

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs: queue.PriorityQueue = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = cos_sim_weight(
                embeddings[
                    query_start_idx : query_start_idx + query_chunk_size  # noqa: E203
                ],
                embeddings[
                    corpus_start_idx : corpus_start_idx  # noqa: E203
                    + corpus_chunk_size
                ],
            )

            scores_top_k_values, scores_top_k_idx = topk(
                torch.from_numpy(scores),
                min(top_k, len(scores[0])),
                dim=1,
                largest=True,
                sorted=False,
            )
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def paraphrase_mining_handler(event, context):
    body = event.get("body")
    if not body:
        return create_json_response(
            status=400, data={"error": "missing body for POST method"}, event=event
        )

    try:
        data = json.loads(body)
    except:
        return create_json_response(
            status=400, data={"error": "invalid JSON body"}, event=event
        )

    if "sentence" not in data:
        return create_json_response(
            status=400, data={"error": "sentence not provided"}, event=event
        )

    if "topk" not in data:
        return create_json_response(
            status=400, data={"error": "topk not provided"}, event=event
        )

    sentences = data["sentence"]
    result = paraphrase_mining(sentences=sentences, top_k=data["topk"])

    return create_json_response(
        status=200, data={"query": data["sentence"], "encoding": result}, event=event
    )
