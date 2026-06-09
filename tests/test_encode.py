#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from src.functions.encode import (
    encode_handler,
    cos_sim_weight_handler,
    multiple_encode_handler,
)
from src.utils.encode import cos_sim_weight, encode, encode_batch


import numpy as np
import json


def test_cosine_sim():

    response = cos_sim_weight_handler(
        {"body": json.dumps({"a": [1, 2, 3], "b": [1, 2, 3]})}, None
    )
    assert response["statusCode"] == 200
    assert json.loads(response["body"])["data"]["score"] == [[1.0]]

    response = cos_sim_weight_handler(
        {"body": json.dumps({"a": [1, 2, 3], "b": [-1, -2, -3]})}, None
    )
    assert response["statusCode"] == 200
    assert json.loads(response["body"])["data"]["score"] == [[-1.0]]

    response = cos_sim_weight_handler(
        {"body": json.dumps({"p": [1, 2, 3], "b": [-1, -2, -3]})}, None
    )
    assert response["statusCode"] == 400

    response = cos_sim_weight_handler({"body": None}, None)
    assert response["statusCode"] == 400


def test_encode():
    response = encode_handler({"body": json.dumps({"sentence": "Hello world"})}, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["data"]["query"] == "Hello world"
    assert len(json.loads(response["body"])["data"]["encoding"]) > 0

    response = encode_handler({"body": None}, None)
    assert response["statusCode"] == 400

    response = encode_handler(
        {"body": json.dumps({"not sentence": "Hello world"})}, None
    )
    assert response["statusCode"] == 400


def test_multiple_encode_handler():
    response = multiple_encode_handler(
        {
            "body": json.dumps(
                {
                    "sentences": ["Hello world", "This is a test sentence"],
                    "batch_size": 1,
                }
            )
        },
        None,
    )

    assert response["statusCode"] == 200
    assert len(json.loads(response["body"])["data"]["query"]) == 2
    assert len(json.loads(response["body"])["data"]["encoding"]) == 2

    assert json.loads(response["body"])["data"]["query"] == [
        "Hello world",
        "This is a test sentence",
    ]

    response = multiple_encode_handler({"body": None}, None)
    assert response["statusCode"] == 400

    response = multiple_encode_handler(
        {"body": json.dumps({"not sentences": "Hello world"})}, None
    )
    assert response["statusCode"] == 400


def test_encode_and_cos_sim_weight():
    embedding1 = encode("Hello this is Syn")
    embedding2 = encode("Hi this is Syn")
    assert np.linalg.norm(cos_sim_weight(embedding1, embedding2)) >= 0.9


def test_encode_batch():
    texts = [
        "Hello",
        "Hi",
        "Hi there",
        "Good morning",
        "Good night",
        "Thank you",
        "Thanks a lot",
        "You are welcome",
        "How are you",
        "What is your name",
        "My name is Alex",
        "Nice to meet you",
        "See you later",
        "Talk to you soon",
        "Have a good day",
        "I need help",
        "Can you help me",
        "Please wait",
        "One moment please",
        "I do not understand",
        "Can you repeat that",
        "What do you mean",
        "That makes sense",
        "I agree with you",
        "I disagree",
        "I am not sure",
        "Maybe later",
        "Let me check",
        "I found the answer",
        "This is correct",
        "That is wrong",
        "Try again",
        "Start over",
        "Stop now",
        "Cancel this request",
        "Delete my account",
        "Update my profile",
        "Change my password",
        "Reset my password",
        "I forgot my password",
        "Log me in",
        "Sign me out",
        "Create a new account",
        "Verify my email",
        "Send me a code",
        "I did not receive it",
        "The code expired",
        "Payment failed",
        "Payment successful",
        "Refund my order",
        "Cancel my order",
        "Track my order",
        "Where is my package",
        "The package is late",
        "I want a refund",
        "I want to return this",
        "The item is damaged",
        "The item is missing",
        "I received the wrong item",
        "Contact customer support",
        "Talk to an agent",
        "Schedule an appointment",
        "Reschedule my appointment",
        "Cancel my appointment",
        "Book a meeting",
        "Join the meeting",
        "Send an invitation",
        "Check my calendar",
        "What time is it",
        "What is today",
        "Set a reminder",
        "Remind me tomorrow",
        "Open the file",
        "Save the document",
        "Upload the image",
        "Download the report",
        "Search the database",
        "Run the test",
        "Fix the bug",
        "Deploy the app",
        "Restart the server",
        "Check the logs",
        "The server is down",
        "The app crashed",
        "The request timed out",
        "Invalid input",
        "Missing required field",
        "Access denied",
        "Permission granted",
        "Connection failed",
        "Connection successful",
        "The result is similar",
        "The result is different",
        "Compare these sentences",
        "Encode this text",
        "Calculate cosine similarity",
        "Generate an embedding",
        "This sentence is short",
        "This phrase is meaningful",
        "The model works well",
        "The model performs poorly",
        "We need a better fallback",
    ]

    batch_size = 32
    embeddings, size = encode_batch(
        sentences=texts, batch_size=batch_size, for_testing=True
    )

    assert len(embeddings) == size
    assert len(embeddings) == len(texts) // batch_size

    batch_size = 1
    embeddings, size = encode_batch(
        sentences=texts, batch_size=batch_size, for_testing=True
    )

    assert len(embeddings) == size
    assert len(embeddings) == len(texts) // batch_size
