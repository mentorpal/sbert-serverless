from src.functions.encode import *
import pytest

def test_cosine_sim():

    response = cos_sim_weight_handler({'body': json.dumps({'a':[1,2,3], 'b': [1,2,3]})}, None)
    assert response['statusCode']==200
    assert json.loads(response['body'])['data']['score'] == pytest.approx(1.0)

    response = cos_sim_weight_handler({'body': json.dumps({'a':[1,2,3], 'b': [-1,-2,-3]})}, None)
    assert response['statusCode']==200
    assert json.loads(response['body'])['data']['score'] == pytest.approx(-1.0)

    response = cos_sim_weight_handler({'body': json.dumps({'p':[1,2,3], 'b': [-1,-2,-3]})}, None)
    assert response['statusCode']==400

    response = cos_sim_weight_handler({'body': None}, None)
    assert response['statusCode']==400

def test_encode():
    response = encode_handler({'body': json.dumps({'sentence':'Hello world'})}, None)
    
    assert response['statusCode']==200
    assert json.loads(response['body'])['data']['query'] == "Hello world"
    assert len(json.loads(response['body'])['data']['encoding']) > 0

    response = encode_handler({'body': None}, None)
    assert response['statusCode']==400

    response = encode_handler({'body': json.dumps({'not sentence':'Hello world'})}, None)
    assert response['statusCode']==400



def test_multiple_encode():
    response = multiple_encode_handler({'body': json.dumps({'sentences':['Hello world','This is a test sentence']})}, None)
    
    assert response['statusCode']==200
    assert len(json.loads(response['body'])['data']['query']) == 2
    assert len(json.loads(response['body'])['data']['encoding']) == 2

    assert json.loads(response['body'])['data']['query'] == ['Hello world','This is a test sentence']

    response = multiple_encode_handler({'body': None}, None)
    assert response['statusCode']==400

    response = multiple_encode_handler({'body': json.dumps({'not sentences':'Hello world'})}, None)
    assert response['statusCode']==400


