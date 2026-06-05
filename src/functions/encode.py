import numpy as np
import json, ollama, logging
from typing import Dict, Any
from src.utils.http_utils import create_json_response


def _encode(sentence):
    return ollama.embeddings(model='mxbai-embed-large',prompt=sentence)['embedding']

def encode_handler(event, context):
    body = event.get('body')
    if not body: 
        return create_json_response(status=400, data={"error": "missing body for POST method"}, event=event)
    
    try:
        data = json.loads(body)
    except:
        return create_json_response(status=400, data={"error": "invalid JSON body"}, event=event)
    
    if 'sentence'not in data:
        return create_json_response(status=400, data={"error": "sentence not provided"}, event=event)


    result = _encode(data['sentence']) 
    return create_json_response(status=200, data={"query":data['sentence'], "encoding":result}, event=event)



def multiple_encode_handler(event, context):
    body = event.get('body')
    if not body: 
        return create_json_response(status=400, data={"error": "missing body for POST method"}, event=event)
    
    try:
        data = json.loads(body)
    except:
        return create_json_response(status=400, data={"error": "invalid JSON body"}, event=event)
    
    if 'sentences'not in data:
        return create_json_response(status=400, data={"error": "sentences not provided"}, event=event)

    sentences = data['sentences']
    result = list(map(lambda sentence: {"original":sentence,'encoded':_encode(sentence)},sentences))
    logging.info(f"Input length: {len(sentences)}, number of encoded result: {len(result)} ")
    return create_json_response(status=200, data={"query":data['sentences'], "encoding":result}, event=event)

    

def _cos_sim_weight(a,b):
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)

    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))


def cos_sim_weight_handler(event: Dict[str, Any], context:Any) -> Dict[str, Any]:
    body = event.get('body')

    if body is None:
        return {'statusCode': 400,
                'body':json.dumps({"error": "Missing request body for POST method"})}
    
    try:
        data= json.loads(body)
    except json.JSONDecodeError:
        return {
            'statusCode':400,
            "body": json.dumps({"error": "Invalid JSON body"})
        }
    
    if 'a' not in data or 'b' not in data:
        return {
            'statusCode':400,
            "body": json.dumps({"error": "both a and b are required"})
        }
    
    a = data['a']
    b = data['b']

    try:
        similarity = _cos_sim_weight(a,b)

    except Exception as e:
        return {
            'statusCode':400,
            "body": json.dumps({"error": f"{e}"})
        }
    
    # return {"statusCode":200, "body": json.dumps({"similarity":float(similarity)})}
    return create_json_response(status=200, data={"score": float(similarity)}, event=event)