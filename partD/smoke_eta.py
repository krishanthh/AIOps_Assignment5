"""Release smoke gate: actual HTTP checks, exit 0 only when every check passes."""
import argparse
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import requests
from eta_contracts import example

def smoke(url,expect_version,max_p95_ms=80):
    suffix='/v1/models/eta:predict';base=url[:-len(suffix)] if url.endswith(suffix) else url.rstrip('/')
    session=requests.Session();checks=[]
    def add(name,condition,**details): checks.append({'check':name,'passed':bool(condition),**details})
    def predict(rows,headers=None):
        response=session.post(base+suffix,json={'instances':rows},headers=headers or {},timeout=10)
        response.raise_for_status();return response
    row=example();health=session.get(base+'/healthz',timeout=10)
    add('health and expected model version',health.status_code==200 and health.json().get('model_version')==expect_version)
    first=predict([row]);body=first.json()
    add('finite non-negative ETA and prediction contract',len(body['predictions'])==1 and np.isfinite(body['predictions'][0]) and body['predictions'][0]>=0 and body.get('model_version')==expect_version)
    distance=predict([{**row,'distance_km':80}]).json()['predictions'][0]
    add('model sanity: increasing distance does not shorten ETA',distance>=body['predictions'][0])
    low=predict([{**row,'traffic_index':.1}]).json()['predictions'][0]
    high=predict([{**row,'traffic_index':.9}]).json()['predictions'][0]
    add('model sanity: heavier traffic does not shorten ETA',high>=low)
    invalid=session.post(base+suffix,json={'instances':[{**row,'distance_km':-1}]},timeout=10)
    add('invalid input rejected with 422',invalid.status_code==422)
    batch=predict([row,{**row,'distance_km':80}]).json()['predictions']
    add('batch count and order',batch==[body['predictions'][0],distance])
    token='smoke-'+str(uuid4());echo=predict([row],{'X-Request-ID':token})
    add('request ID round trip',echo.headers.get('X-Request-ID')==echo.json().get('request_id')==token)
    times=[]
    for _ in range(50):
        start=time.perf_counter();predict([row]);times.append((time.perf_counter()-start)*1000)
    p95=float(np.percentile(times,95))
    add('HTTP prediction p95 under release threshold',p95<=max_p95_ms,p95_ms=p95,threshold_ms=max_p95_ms,observations=50)
    return {'passed':all(c['passed'] for c in checks),'url':base,'expect_version':expect_version,'checks':checks,'p95_ms':p95}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--expect-version',required=True);parser.add_argument('--max-p95-ms',type=float,default=80);args=parser.parse_args()
    try: report=smoke(args.url,args.expect_version,args.max_p95_ms)
    except Exception as error: report={'passed':False,'error':str(error)}
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report['passed'] else 1)
