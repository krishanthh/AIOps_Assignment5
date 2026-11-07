"""Two-CPU Ray Serve app with stable per-customer canary routing."""
import hashlib
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import joblib
import pandas as pd
from fastapi import FastAPI,Request
from ray import serve
from eta_contracts import CanaryRequest,FEATURES

def bucket(customer_id):
    return int.from_bytes(hashlib.sha256(customer_id.encode('utf-8')).digest()[:8],'big')

def choose_arm(customer_id,weight):
    return 'v2' if bucket(customer_id)<int(weight*(1<<64)) else 'v1'

@serve.deployment(ray_actor_options={'num_cpus':0.25},max_ongoing_requests=16)
class EtaModel:
    def __init__(self,bundle_path,extra_work_ms=0.0):
        self.bundle=joblib.load(bundle_path)  # Once for each replica, never per request.
        self.extra_work_ms=extra_work_ms

    def predict(self,rows):
        if self.extra_work_ms:
            end=time.perf_counter()+self.extra_work_ms/1000
            while time.perf_counter()<end: pass  # Explicit optional CPU work for experiments.
        frame=pd.DataFrame(rows,columns=FEATURES)
        result=self.bundle['model'].predict(frame)
        return {'predictions':[round(float(p),1) for p in result],'model_version':self.bundle['version'],'served_by':str(os.getpid())}

api=FastAPI(title='SwiftParcel sticky canary')

@serve.deployment(num_replicas=1,ray_actor_options={'num_cpus':0.1},max_ongoing_requests=32)
@serve.ingress(api)
class EtaRouter:
    def __init__(self,prod,candidate,weight):
        self.prod=prod;self.candidate=candidate;self.weight=weight

    @api.post('/v1/models/eta:predict')
    async def predict(self,body:CanaryRequest,request:Request):
        arm='v2' if request.headers.get('X-Beta')=='1' else choose_arm(body.customer_id,self.weight)
        handle=self.candidate if arm=='v2' else self.prod
        output=await handle.predict.remote([row.model_dump() for row in body.instances])
        return {**output,'arm':arm,'customer_id':body.customer_id}

    @api.get('/healthz')
    async def health(self):
        return {'status':'ok','canary_weight':self.weight,'routing':'SHA256 customer bucket','prod_replicas':2,'candidate_min_replicas':1,'candidate_max_replicas':3}

def build(weight=0.1,extra_work_ms=0.0):
    if not math.isfinite(weight) or not 0<=weight<=1 or not math.isfinite(extra_work_ms) or extra_work_ms<0:
        raise ValueError('weight must be in [0,1]; extra_work_ms must be finite and non-negative')
    prod=EtaModel.options(name='eta-prod',num_replicas=2).bind(str(ROOT/'data/eta_v1.joblib'),extra_work_ms)
    candidate=EtaModel.options(name='eta-candidate',autoscaling_config={'min_replicas':1,'max_replicas':3,'target_ongoing_requests':4,'metrics_interval_s':1,'look_back_period_s':5,'upscale_delay_s':3,'downscale_delay_s':20}).bind(str(ROOT/'data/eta_v2.joblib'),extra_work_ms)
    return EtaRouter.options(name='eta-router').bind(prod,candidate,weight)
