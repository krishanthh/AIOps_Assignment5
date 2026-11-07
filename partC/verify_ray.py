"""Execute all five required canary checks against a genuine 2-CPU Ray Serve app."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import ray
from ray import serve
import requests
from partC.eta_ray import build
from eta_contracts import example

def verify(url='http://127.0.0.1:8055',keep_alive=False):
    context=ray.init(num_cpus=2,include_dashboard=True,dashboard_host='127.0.0.1',dashboard_port=8265,object_store_memory=100663296,_memory=805306368,log_to_driver=False)
    assert ray.cluster_resources()['CPU']==2
    serve.start(http_options={'host':'127.0.0.1','port':8055})
    ids=[f'customer-{i:04d}' for i in range(1000)]
    def request(customer,beta=False):
        r=requests.post(url+'/v1/models/eta:predict',json={'customer_id':customer,'instances':[example()]},headers={'X-Beta':'1'} if beta else {},timeout=30)
        r.raise_for_status();body=r.json()
        assert body['arm']==body['model_version'] and len(body['predictions'])==1
        return body['arm']
    def calls(customers):
        with ThreadPoolExecutor(max_workers=4) as pool: return dict(zip(customers,pool.map(request,customers)))
    report={'ray_cpus':ray.cluster_resources()['CPU'],'customers':1000,'checks':[]}
    try:
        print('Deploying weight 0.2 on 2 CPUs',flush=True)
        serve.run(build(weight=.2),name='eta',route_prefix='/')
        initial=calls(ids);share=sum(a=='v2' for a in initial.values())/len(ids)
        assert .16<=share<=.24,share
        report['candidate_share_at_0_2']=share
        report['checks'].append({'check':'1000 distinct customers: share within 0.20 +/- 0.04','passed':True,'actual_share':share})
        repeated=calls(ids[:200]);assert all(initial[k]==v for k,v in repeated.items())
        report['checks'].append({'check':'200 customers sticky on second call','passed':True})
        print('Initial share and repeat-call stickiness passed',flush=True)
        serve.delete('eta')
        serve.run(build(weight=.2),name='eta',route_prefix='/')
        redeployed=calls(ids[:200]);assert all(initial[k]==v for k,v in redeployed.items())
        report['checks'].append({'check':'200 customers sticky after deleting and redeploying identical app','passed':True})
        print('Fresh-replica redeployment stickiness passed',flush=True)
        serve.run(build(weight=.5),name='eta',route_prefix='/')
        expanded=calls(ids)
        assert all(expanded[k]=='v2' for k,v in initial.items() if v=='v2')
        report['candidate_share_at_0_5']=sum(a=='v2' for a in expanded.values())/1000
        report['checks'].append({'check':'No prior candidate customer returns to prod after raising weight to 0.5','passed':True,'customers_checked':1000})
        prod_ids=[customer for customer,arm in expanded.items() if arm=='v1'][:50]
        assert len(prod_ids)==50
        for customer in prod_ids: assert request(customer,beta=True)=='v2'
        for customer in prod_ids: assert request(customer)=='v1'
        report['checks'].append({'check':'X-Beta override forces candidate for 50 normally-prod customers, without changing later assignment','passed':True})
        # Leave the documented production canary default for the video/manual review.
        serve.run(build(weight=.1),name='eta',route_prefix='/')
        report['final_weight']=.1
        report['deployments']=str(serve.status())
        report['max_reserved_cpus']=2*.25+3*.25+.1
        report['passed']=True
        (ROOT/'partC/verify_output.txt').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if keep_alive:
            print('VERIFICATION PASSED; app remains on port 8055, dashboard on 8265.',flush=True)
            stopped=threading.Event()
            signal.signal(signal.SIGTERM,lambda *_:stopped.set())
            signal.signal(signal.SIGINT,lambda *_:stopped.set())
            stopped.wait()
        return report
    finally:
        serve.shutdown();ray.shutdown()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--keep-alive',action='store_true');args=parser.parse_args()
    verify(keep_alive=args.keep_alive)
