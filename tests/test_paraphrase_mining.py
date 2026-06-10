#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

from src.functions.paraphrase import paraphrase_mining_handler
import json
import pytest
import os


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skip this test in GitHub Actions"
)
def test_paraphrase_mining():
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
    response = paraphrase_mining_handler(
        {"body": json.dumps({"sentence": texts, "topk": 10})}, None
    )

    assert response["statusCode"] == 200

    response = paraphrase_mining_handler(
        {"body": json.dumps({"sentence": texts, "topk": 20})}, None
    )

    assert response["statusCode"] == 200
