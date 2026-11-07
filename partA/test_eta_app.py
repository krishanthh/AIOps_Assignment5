"""Required real-model contract tests; models are generated with eta_data.py first."""
from uuid import UUID
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from partA import eta_app
from eta_contracts import FEATURES,example

@pytest.fixture
def client():
    with TestClient(eta_app.app,raise_server_exceptions=False) as c: yield c

def send(client,rows=None,**kwargs):
    return client.post('/v1/models/eta:predict',json={'instances':[example()] if rows is None else rows},**kwargs)

def test_valid_prediction_and_rounding(client):
    response=send(client);assert response.status_code==200
    body=response.json();assert body['model_version']=='v1'
    assert len(body['predictions'])==1 and round(body['predictions'][0],1)==body['predictions'][0]
    assert str(UUID(body['request_id']))==response.headers['X-Request-ID']

def test_batch_order_matches_model(client):
    rows=[{**example(),'distance_km':70.0},{**example(),'distance_km':5.0}]
    expected=np.round(eta_app.model.predict(pd.DataFrame(rows)[FEATURES]),1).tolist()
    assert send(client,rows).json()['predictions']==expected

@pytest.mark.parametrize('change',[{'distance_km':0},{'hour':'not-a-number'},{'traffic_index':1.2},{'rain':2},{'weight_kg':0},{'n_stops':61},{'weekday':7}])
def test_invalid_ranges_or_types(client,change):
    response=send(client,[{**example(),**change}]);assert response.status_code==422
    body=response.json();assert body['error']=='validation_failed' and body['details']
    assert set(body['details'][0])=={'field','msg'}

def test_missing_feature(client):
    row=example();del row['weight_kg'];assert send(client,[row]).status_code==422

def test_infeasible_cross_field_rule(client):
    response=send(client,[{**example(),'distance_km':151,'n_stops':21}])
    assert response.status_code==422 and 'infeasible' in response.text.lower()

@pytest.mark.parametrize('size',[0,501])
def test_batch_limits_rejected(client,size): assert send(client,[example()]*size).status_code==422

def test_batch_upper_boundary_500(client):
    response=send(client,[example()]*500);assert response.status_code==200 and len(response.json()['predictions'])==500

def test_request_id_echo_success(client):
    response=send(client,headers={'X-Request-ID':'student-request-42'})
    assert response.headers['X-Request-ID']==response.json()['request_id']=='student-request-42'

def test_generated_request_id_on_validation_error(client):
    response=send(client,[]);assert str(UUID(response.headers['X-Request-ID']))==response.json()['request_id']

def test_request_id_echo_on_error(client):
    response=send(client,[],headers={'X-Request-ID':'bad-request'})
    assert response.json()['request_id']==response.headers['X-Request-ID']=='bad-request'

def test_metrics_histogram_and_self_exclusion(client):
    send(client);first=client.get('/metrics');second=client.get('/metrics')
    assert first.status_code==200 and first.headers['X-Request-ID']
    assert 'eta_requests_total{path="/v1/models/eta:predict",status="200"}' in first.text
    for name in ['eta_request_latency_seconds_bucket','le="+Inf"','eta_request_latency_seconds_sum','eta_request_latency_seconds_count']:
        assert name in first.text
    assert 'path="/metrics"' not in second.text
    assert first.text==second.text

def test_generic_500_does_not_leak_internals(client,monkeypatch):
    def fail(*args,**kwargs): raise RuntimeError('secret-model-internals')
    monkeypatch.setattr(eta_app.model,'predict',fail)
    response=send(client,headers={'X-Request-ID':'failure-case'})
    assert response.status_code==500 and 'secret-model-internals' not in response.text
    assert response.json()['request_id']==response.headers['X-Request-ID']=='failure-case'

def test_health(client):
    response=client.get('/healthz');assert response.status_code==200
    assert response.json()['status']=='ok' and response.json()['model_version']=='v1'

def test_unknown_path_carries_request_id(client):
    response=client.get('/missing');assert response.status_code==404
    assert response.headers['X-Request-ID']==response.json()['request_id']

def test_valid_range_boundaries(client):
    row=dict(distance_km=500,n_stops=20,hour=23,weekday=6,traffic_index=1,rain=1,weight_kg=1000)
    assert send(client,[row]).status_code==200
