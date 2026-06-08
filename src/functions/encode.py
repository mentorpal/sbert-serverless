import json, logging
from typing import Dict, Any
from src.utils.http_utils import create_json_response
from src.utils.encode import cos_sim_weight, encode


def encode_handler(event, context):
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

    result = encode(data["sentence"])
    return create_json_response(
        status=200, data={"query": data["sentence"], "encoding": result}, event=event
    )


def multiple_encode_handler(event, context):
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

    if "sentences" not in data:
        return create_json_response(
            status=400, data={"error": "sentences not provided"}, event=event
        )

    sentences = data["sentences"]
    result = list(
        map(
            lambda sentence: {"original": sentence, "encoded": encode(sentence)},
            sentences,
        )
    )
    logging.info(
        f"Input length: {len(sentences)}, number of encoded result: {len(result)} "
    )
    return create_json_response(
        status=200, data={"query": data["sentences"], "encoding": result}, event=event
    )


def cos_sim_weight_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    body = event.get("body")

    if body is None:
        return create_json_response(
            status=400,
            data={"error": "Missing request body for POST method"},
            event=event,
        )

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return create_json_response(
            status=400, data={"error": "Invalid JSON body"}, event=event
        )

    if "a" not in data or "b" not in data:
        return create_json_response(
            status=400, data={"error": "Both a and b are required"}, event=event
        )

    a = data["a"]
    b = data["b"]

    try:
        similarity_matrix = cos_sim_weight(a, b)

    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"{e}"})}

    # return {"statusCode":200, "body": json.dumps({"similarity":float(similarity)})}
    return create_json_response(
        status=200, data={"score": similarity_matrix.tolist()}, event=event
    )
