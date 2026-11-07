"""Boundary-aware pure promotion-gate tests, independent of the registry."""
import math
import pytest
from partB.eta_mlflow import decide

PROD=dict(mae=10.,mae_rush=12.,mae_rain=15.,p95_ms=2.)
def candidate(**kwargs): return dict(mae=9.,mae_rush=11.,mae_rain=14.,p95_ms=3.,**kwargs) if not kwargs else {**dict(mae=9.,mae_rush=11.,mae_rain=14.,p95_ms=3.),**kwargs}
def test_all_good(): assert decide(PROD,candidate())['promote'] is True
def test_gain_exact_boundary(): assert decide(PROD,candidate(mae=9.7))['promote'] is True
def test_gain_just_outside(): assert decide(PROD,candidate(mae=9.7001))['promote'] is False
def test_latency_exact_boundary(): assert decide(PROD,candidate(p95_ms=10.))['promote'] is True
def test_latency_outside(): assert decide(PROD,candidate(p95_ms=10.001))['criteria']['p95_latency'] is False
def test_rush_exact_boundary(): assert decide(PROD,candidate(mae_rush=12.6))['promote'] is True
def test_rush_outside(): assert decide(PROD,candidate(mae_rush=12.601))['criteria']['rush_slice'] is False
def test_rain_exact_boundary(): assert decide(PROD,candidate(mae_rain=15.75))['promote'] is True
def test_rain_outside(): assert decide(PROD,candidate(mae_rain=15.751))['criteria']['rain_slice'] is False
def test_strict_gain(): assert decide(PROD,candidate(),min_rel_gain=.60)['promote'] is False
def test_perfect_prod_cannot_improve():
    zero=dict(mae=0.,mae_rush=0.,mae_rain=0.,p95_ms=1.)
    assert decide(zero,zero)['promote'] is False
def test_nonfinite_metric_rejected():
    with pytest.raises(ValueError): decide(PROD,candidate(mae=math.nan))
