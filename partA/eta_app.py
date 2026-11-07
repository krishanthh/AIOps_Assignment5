"""Run from the submission root: python -m uvicorn partA.eta_app:app."""
import logging
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,Response
from prometheus_client import CollectorRegistry,Counter,Histogram,generate_latest,CONTENT_TYPE_LATEST
from starlette.exceptions import HTTPException as StarletteHTTPException
from eta_contracts import FEATURES,PredictRequest

path=Path(os.getenv('ETA_MODEL','data/eta_v1.joblib'))
if not path.is_absolute() and not path.exists(): path=ROOT/path
bundle=joblib.load(path)
model=bundle['model']  # Public module-level object, including for failure injection tests.
model_version=bundle['version']
registry=CollectorRegistry()
REQUESTS=Counter('eta_requests_total','HTTP requests excluding the metrics endpoint.',['path','status'],registry=registry)
LATENCY=Histogram('eta_request_latency_seconds','HTTP request latency excluding metrics.',buckets=(.001,.0025,.005,.01,.02,.04,.08,.16,.32,1,5),registry=registry)
app=FastAPI(title='SwiftParcel ETA',version='1.0')

@app.middleware('http')
async def request_context(request:Request,call_next):
    supplied=request.headers.get('X-Request-ID')
    request.state.request_id=supplied if supplied is not None else str(uuid4())
    start=time.perf_counter()
    try:
        response=await call_next(request)
    except Exception:
        logging.exception('Request failed: %s',request.state.request_id)
        response=JSONResponse({'error':'internal_server_error','request_id':request.state.request_id},status_code=500)
    response.headers['X-Request-ID']=request.state.request_id
    if request.url.path!='/metrics':
        REQUESTS.labels(path=request.url.path,status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter()-start)
    return response

@app.exception_handler(RequestValidationError)
async def validation_error(request:Request,error:RequestValidationError):
    details=[{'field':'.'.join(str(p) for p in item['loc'] if p!='body'),'msg':item['msg']} for item in error.errors()]
    return JSONResponse({'error':'validation_failed','details':details,'request_id':request.state.request_id},status_code=422)

@app.exception_handler(StarletteHTTPException)
async def http_error(request:Request,error:StarletteHTTPException):
    return JSONResponse({'error':str(error.detail),'request_id':request.state.request_id},status_code=error.status_code)

@app.post('/v1/models/eta:predict')
def predict(body:PredictRequest,request:Request):
    frame=pd.DataFrame([row.model_dump() for row in body.instances],columns=FEATURES)
    predictions=np.asarray(model.predict(frame),dtype=float)
    if not np.isfinite(predictions).all(): raise RuntimeError('Non-finite inference result')
    return {'predictions':[round(float(p),1) for p in predictions],'model_version':model_version,'request_id':request.state.request_id}

@app.get('/healthz')
def healthz(request:Request):
    return {'status':'ok','model_version':model_version,'request_id':request.state.request_id}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(registry),headers={'Content-Type':CONTENT_TYPE_LATEST})
