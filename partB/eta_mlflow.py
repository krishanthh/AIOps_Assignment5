"""MLflow registry, multi-criteria promotion decision and paired shadow comparison."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import joblib
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error
from eta_data import FEATURES,make_data

class BundleModel(mlflow.pyfunc.PythonModel):
    def load_context(self,context):
        self.bundle=joblib.load(context.artifacts['bundle'])

    def predict(self,context,model_input,params=None):
        frame=pd.DataFrame(model_input)[self.bundle['feature_columns']]
        values=np.asarray(self.bundle['model'].predict(frame),dtype=float)
        return pd.DataFrame({'eta_minutes':np.round(values,1),'model_version':[self.bundle['version']]*len(frame)})

def configure():
    uri=os.getenv('MLFLOW_TRACKING_URI','sqlite:///'+str(ROOT/'mlflow.db'))
    mlflow.set_tracking_uri(uri);mlflow.set_registry_uri(uri)
    return MlflowClient()

def register():
    client=configure();name='swiftparcel-eta'
    experiment=client.get_experiment_by_name(name)
    if experiment is None:
        client.create_experiment(name,artifact_location=(ROOT/'mlartifacts').as_uri())
    mlflow.set_experiment(name)
    sample=make_data(5,seed=42)[FEATURES]
    requirements=[f'{name}=={importlib.metadata.version(name)}' for name in ['mlflow','scikit-learn','pandas','numpy','joblib']]
    table=[]
    for version,alias in [('v1','prod'),('v2','candidate')]:
        path=ROOT/'data'/f'eta_{version}.joblib'
        bundle=joblib.load(path)
        output=pd.DataFrame({'eta_minutes':np.round(bundle['model'].predict(sample),1),'model_version':[version]*len(sample)})
        with mlflow.start_run(run_name='eta-'+version) as run:
            mlflow.log_params({'bundle_version':version,'training_seed':1,'fresh_seed':99,'algorithm':type(bundle['model']).__name__})
            info=mlflow.pyfunc.log_model(artifact_path='model',python_model=BundleModel(),artifacts={'bundle':str(path)},signature=infer_signature(sample,output),input_example=sample,pip_requirements=requirements,code_paths=[str(ROOT/'partB'),str(ROOT/'eta_data.py')])
            registered=mlflow.register_model(info.model_uri,'eta-model')
            client.set_registered_model_alias('eta-model',alias,registered.version)
            table.append({'model':'eta-model','bundle':version,'registry_version':registered.version,'alias':alias,'run_id':run.info.run_id})
    return table

def decide(prod,cand,min_rel_gain=0.03,max_p95_ms=10.0,max_slice_regression=0.05):
    keys=['mae','mae_rush','mae_rain','p95_ms']
    for metrics in [prod,cand]:
        if any(not math.isfinite(float(metrics[k])) or float(metrics[k])<0 for k in keys):
            raise ValueError('Metrics must be finite and non-negative')
    if not 0<=min_rel_gain<=1 or max_p95_ms<0 or max_slice_regression<0:
        raise ValueError('Invalid gate thresholds')
    tolerance=1e-12
    criteria={
        'mae_gain':bool(cand['mae']<=prod['mae']*(1-min_rel_gain)+tolerance and (prod['mae']>0 or min_rel_gain==0)),
        'p95_latency':bool(cand['p95_ms']<=max_p95_ms+tolerance),
        'rush_slice':bool(cand['mae_rush']<=prod['mae_rush']*(1+max_slice_regression)+tolerance),
        'rain_slice':bool(cand['mae_rain']<=prod['mae_rain']*(1+max_slice_regression)+tolerance),
    }
    gain=(prod['mae']-cand['mae'])/prod['mae'] if prod['mae'] else 0.0
    return {'criteria':criteria,'promote':all(criteria.values()),'relative_mae_gain':gain}

def promotion_gate(**thresholds):
    client=configure();fresh=make_data(2000,seed=99);x=fresh[FEATURES]
    masks={'mae_rush':fresh.hour.isin([7,8,9,16,17,18,19]),'mae_rain':fresh.rain==1}
    metrics={}
    for alias in ['prod','candidate']:
        model=mlflow.pyfunc.load_model(f'models:/eta-model@{alias}')
        predicted=model.predict(x).eta_minutes.to_numpy(dtype=float)
        measured={'mae':float(mean_absolute_error(fresh.eta_minutes,predicted))}
        for name,mask in masks.items(): measured[name]=float(mean_absolute_error(fresh.loc[mask,'eta_minutes'],predicted[mask]))
        model.predict(x.iloc[:1])  # One untimed warm-up, then exactly 200 observations.
        times=[]
        for i in range(200):
            start=time.perf_counter();model.predict(x.iloc[i:i+1]);times.append((time.perf_counter()-start)*1000)
        measured['p95_ms']=float(np.percentile(times,95))
        measured['single_row_latencies_ms']=times
        measured['registry_version']=client.get_model_version_by_alias('eta-model',alias).version
        metrics[alias]=measured
    return {**decide(metrics['prod'],metrics['candidate'],**thresholds),'thresholds':{'min_rel_gain':.03,'max_p95_ms':10.,'max_slice_regression':.05,**thresholds},'metrics':metrics,'fresh_rows':2000,'seed':99,'timing_observations_per_model':200}

def shadow(prod_url='http://127.0.0.1:5101/invocations',candidate_url='http://127.0.0.1:5102/invocations'):
    rows=pd.read_csv(ROOT/'data/eta_fresh.csv')[FEATURES].head(200).to_dict(orient='records')
    sessions=[requests.Session(),requests.Session()];urls=[prod_url,candidate_url]
    values=[[],[]];times=[[],[]]
    for session,url in zip(sessions,urls):
        response=session.post(url,json={'dataframe_records':[rows[0]]},timeout=20);response.raise_for_status()
    for row in rows:
        for i,(session,url) in enumerate(zip(sessions,urls)):
            start=time.perf_counter();response=session.post(url,json={'dataframe_records':[row]},timeout=20);elapsed=(time.perf_counter()-start)*1000
            response.raise_for_status();prediction=response.json()['predictions'][0]
            assert prediction['model_version']==['v1','v2'][i]
            values[i].append(float(prediction['eta_minutes']));times[i].append(elapsed)
    delta=np.abs(np.array(values[0])-np.array(values[1]))
    return {'paired_requests':200,'source':'First 200 feature rows of data/eta_fresh.csv; same rows, same order for both servers. One excluded warm-up each.','mean_absolute_difference_minutes':float(delta.mean()),'percent_differing_over_10_minutes':float(100*np.mean(delta>10)),'prod_p95_ms':float(np.percentile(times[0],95)),'candidate_p95_ms':float(np.percentile(times[1],95)),'prod_latencies_ms':times[0],'candidate_latencies_ms':times[1],'first_five_predictions':{'prod':values[0][:5],'candidate':values[1][:5]}}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','gate','shadow']);parser.add_argument('--min-rel-gain',type=float,default=.03);args=parser.parse_args()
    result=register() if args.action=='register' else promotion_gate(min_rel_gain=args.min_rel_gain) if args.action=='gate' else shadow()
    print(json.dumps(result,indent=2),flush=True)
