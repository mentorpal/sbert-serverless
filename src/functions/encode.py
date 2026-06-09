#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import json
from typing import Dict, Any
from src.utils.http_utils import create_json_response
from src.utils.encode import cos_sim_weight, encode, encode_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def encode_handler(event, context):
    body = event.get("body")
    if not body:
        return create_json_response(
            status=400, data={"error": "missing body for POST method"}, event=event
        )

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
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
    logger.info("starting encode batch")

    body = event.get("body")
    if not body:
        return create_json_response(
            status=400, data={"error": "missing body for POST method"}, event=event
        )

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return create_json_response(
            status=400, data={"error": "invalid JSON body"}, event=event
        )

    if "sentences" not in data:
        return create_json_response(
            status=400, data={"error": "sentences not provided"}, event=event
        )

    if "batch_size" not in data:
        return create_json_response(
            status=400, data={"error": "batch_size not provided"}, event=event
        )

    sentences = data["sentences"]
    batch_size = data["batch_size"]
    result = encode_batch(sentences, batch_size=batch_size)

    logger.debug("batch size=%s", batch_size)

    logger.info(
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

    return create_json_response(
        status=200, data={"score": similarity_matrix.tolist()}, event=event
    )
