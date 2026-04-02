"""
GETTER One vs Established Causal Inference Methods
====================================================

合成データ（ground truth付き）でGETTER Oneの因果推定性能を
VAR Granger, Transfer Entropy, PCMCI+, GraphicalLasso, EventXCorr
と比較する。

Usage:
  pip install getter-one
  python getter_one_benchmark.py

Built with 💕 by Masamichi & Tamaki
"""

import math
import warnings
import time
import warnings

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import precision_score, recall_score, f1_score

# Optional imports
_HAS_STATSMODELS = False
try:
    from statsmodels.tsa.api import VAR
    _HAS_STATSMODELS = True
except ImportError:
    pass

_HAS_PYINFORM = False
try:
    from pyinform.transferentropy import transfer_entropy as _pyinform_transfer_entropy
    _HAS_PYINFORM = True
except ImportError:
    pass

_HAS_TIGRAMITE = False
try:
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    _HAS_TIGRAMITE = True
except ImportError:
    pass

from getter_one.pipeline import PipelineConfig, run as getter_run
from getter_one.data.loader import from_dataframe


# =====================================================================
# Data structures (from benchmark framework)
# =====================================================================

@dataclass
class ScenarioConfig:
    T: int = 400
    burn_in: int = 100
    n_series: int = 3
    max_lag: int = 8
    noise_scale: float = 0.35
    regime_split: int = 200
    event_percentile: float = 97.0
    zero_inflation_prob: float = 0.88
    seed: int = 42

@dataclass
class GroundTruth:
    names: List[str]
    adjacency: np.ndarray
    lag_matrix: Optional[np.ndarray] = None
    sign_matrix: Optional[np.ndarray] = None
    notes: str = ""
    regime_boundaries: Optional[List[int]] = None
    adjacency_by_regime: Optional[List[np.ndarray]] = None
    lag_by_regime: Optional[List[np.ndarray]] = None
    sign_by_regime: Optional[List[np.ndarray]] = None
    low_corr_edges: Optional[List[Tuple[int, int]]] = None
    forbidden_edges: Optional[List[Tuple[int, int]]] = None

    def undirected_adjacency(self) -> np.ndarray:
        return ((self.adjacency + self.adjacency.T) > 0).astype(int)

@dataclass
class MethodOutput:
    method_name: str
    names: List[str]
    adjacency_scores: Optional[np.ndarray] = None
    adjacency_bin: Optional[np.ndarray] = None
    directed_support: bool = True
    lag_support: bool = False
    sign_support: bool = False
    regime_support: bool = False
    lag_matrix: Optional[np.ndarray] = None
    sign_matrix: Optional[np.ndarray] = None
    regime_boundaries: Optional[List[int]] = None
    adjacency_by_regime: Optional[List[np.ndarray]] = None
    lag_by_regime: Optional[List[np.ndarray]] = None
    sign_by_regime: Optional[List[np.ndarray]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def undirected_bin(self) -> Optional[np.ndarray]:
        if self.adjacency_bin is None:
            return None
        return ((self.adjacency_bin + self.adjacency_bin.T) > 0).astype(int)


# =====================================================================
# Utility functions
# =====================================================================

def _rng(seed): return np.random.default_rng(seed)

def _diff_events(x, percentile=97.0):
    d = np.diff(x, prepend=x[0])
    thr = np.percentile(np.abs(d), percentile)
    return (np.abs(d) > thr).astype(int)

def _shifted_overlap(a, b, lag):
    if lag > 0: return a[:-lag], b[lag:]
    if lag < 0: return a[-lag:], b[:lag]
    return a, b

def _quantile_discretize(x, n_bins=4):
    x = np.asarray(x)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmin == xmax: return np.zeros_like(x, dtype=int)
        edges = np.linspace(xmin, xmax, n_bins + 1)
    return np.digitize(x, edges[1:-1], right=False).astype(int)

def _te_discrete(source, target, lag=1, n_bins=4):
    if lag < 1: raise ValueError("lag must be >= 1")
    x = _quantile_discretize(source, n_bins)
    y = _quantile_discretize(target, n_bins)
    t0 = max(lag, 1)
    y_t, y_prev, x_lag = y[t0:], y[t0-1:-1], x[t0-lag:len(x)-lag]
    n = min(len(y_t), len(y_prev), len(x_lag))
    y_t, y_prev, x_lag = y_t[:n], y_prev[:n], x_lag[:n]
    if n <= 5: return 0.0
    c_xyz = Counter(zip(x_lag.tolist(), y_prev.tolist(), y_t.tolist()))
    c_xy = Counter(zip(x_lag.tolist(), y_prev.tolist()))
    c_yz = Counter(zip(y_prev.tolist(), y_t.tolist()))
    c_y = Counter(y_prev.tolist())
    te = 0.0
    for (xs, yp, yt), c in c_xyz.items():
        p_xyz = c / n
        p1 = c / c_xy[(xs, yp)]
        p2 = c_yz[(yp, yt)] / c_y[yp]
        if p1 > 0 and p2 > 0: te += p_xyz * math.log(p1 / p2)
    return float(te)

def _te_pyinform(source, target, lag=1, n_bins=4, k=1):
    if lag < 1: raise ValueError
    if not _HAS_PYINFORM: return _te_discrete(source, target, lag, n_bins)
    xs, ys = _shifted_overlap(np.asarray(source), np.asarray(target), lag)
    xs = _quantile_discretize(xs, n_bins).astype(int).tolist()
    ys = _quantile_discretize(ys, n_bins).astype(int).tolist()
    if len(xs) <= max(k+2, 8): return 0.0
    try:
        te = _pyinform_transfer_entropy(xs, ys, k=k)
        return 0.0 if (np.isnan(te) or np.isinf(te)) else float(te)
    except: return 0.0

def _event_sync_score(a_events, b_events, lag):
    aa, bb = _shifted_overlap(a_events, b_events, lag)
    return float(np.mean(aa * bb)) if len(aa) else 0.0

def _binarize_adjacency_from_scores(scores, threshold=None, percentile=None, symmetric=False):
    s = np.asarray(scores).copy()
    np.fill_diagonal(s, 0.0)
    vals = s[~np.eye(s.shape[0], dtype=bool)]
    if threshold is None:
        percentile = percentile or 75.0
        threshold = float(np.percentile(vals, percentile)) if len(vals) else 0.0
    out = (s >= threshold).astype(int)
    np.fill_diagonal(out, 0)
    if symmetric:
        out = ((out + out.T) > 0).astype(int)
        np.fill_diagonal(out, 0)
    return out

def _precision_recall_f1(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def _boundary_f1(true_b, pred_b, tolerance=5):
    true_b, pred_b = list(true_b or []), list(pred_b or [])
    if not true_b and not pred_b: return 1.0
    if not true_b or not pred_b: return 0.0
    mt, mp = set(), set()
    for i, tb in enumerate(true_b):
        for j, pb in enumerate(pred_b):
            if abs(tb - pb) <= tolerance and j not in mp:
                mt.add(i); mp.add(j); break
    tp = len(mt); fp = len(pred_b) - tp; fn = len(true_b) - tp
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0


# =====================================================================
# Scenario generators
# =====================================================================

def generate_null_independent(cfg, seed):
    rng = _rng(seed); n = cfg.n_series; T = cfg.T + cfg.burn_in
    x = np.zeros((T, n))
    for t in range(1, T): x[t] = 0.65*x[t-1] + rng.normal(scale=cfg.noise_scale, size=n)
    x = x[cfg.burn_in:]
    names = [f"X{i+1}" for i in range(n)]
    return pd.DataFrame(x, columns=names), GroundTruth(names=names, adjacency=np.zeros((n,n),dtype=int), lag_matrix=np.zeros((n,n),dtype=int), sign_matrix=np.zeros((n,n),dtype=int), notes="Null")

def generate_delayed_directional(cfg, seed):
    rng = _rng(seed); T = cfg.T+cfg.burn_in; lag=3; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.75*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        eff=1.15*x[t-lag,0] if t-lag>=0 else 0.0
        x[t,1]=0.60*x[t-1,1]+eff+0.15*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=lag
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Delayed A->B")

def generate_asymmetric_coupling(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,1]=0.45*x[t-1,1]+1.35*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,0]+=0.08*x[t-1,1]
        x[t,2]=0.60*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=1
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Asymmetric A->B")

def generate_event_driven_delayed(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; lag=3; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        if rng.random()<0.04: x[t,0]+=rng.choice([-1,1])*rng.uniform(2.0,4.0)
        x[t,1]=0.60*x[t-1,1]+rng.normal(scale=cfg.noise_scale)
        if t-lag>=1:
            a_jump=abs(x[t-lag,0]-x[t-lag-1,0])
            if a_jump>1.5: x[t,1]+=0.8*np.sign(x[t-lag,0]-x[t-lag-1,0])*rng.uniform(1.5,3.0)
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=lag
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Event-driven delayed A->B")

def generate_event_driven_asymmetric(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        if rng.random()<0.05: x[t,0]+=rng.choice([-1,1])*rng.uniform(2.5,5.0)
        x[t,1]=0.50*x[t-1,1]+rng.normal(scale=cfg.noise_scale)
        if t>=2:
            a_jump=abs(x[t-1,0]-x[t-2,0])
            if a_jump>1.5: x[t,1]+=2.5*np.sign(x[t-1,0]-x[t-2,0])*rng.uniform(0.8,1.2)
        if t>=2:
            b_jump=abs(x[t-1,1]-x[t-2,1])
            if b_jump>1.5: x[t,0]+=0.05*np.sign(x[t-1,1]-x[t-2,1])
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=1
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Event asymmetric A->B")

def generate_confounder(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,2]=0.80*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        x[t,0]=0.50*x[t-1,0]+1.20*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        x[t,1]=0.55*x[t-1,1]+(-1.05*x[t-2,2] if t-2>=0 else 0.0)+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[2,0]=1; adj[2,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[2,0]=1; lagm[2,1]=2
    signm=np.zeros((3,3),dtype=int); signm[2,0]=1; signm[2,1]=-1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,forbidden_edges=[(0,1),(1,0)],notes="Confounder C->A,B")

SCENARIOS = {
    "S0_null": generate_null_independent,
    "S1_delayed": generate_delayed_directional,
    "S2_asymmetric": generate_asymmetric_coupling,
    "S5_confounder": generate_confounder,
    "S7_event_delayed": generate_event_driven_delayed,
    "S8_event_asym": generate_event_driven_asymmetric,
}


# =====================================================================
# GETTER One Adapter (Full Pipeline!)
# =====================================================================

class GetterOneAdapter:
    method_name = "GETTER_One"

    def __init__(self, sync_threshold=0.3, causal_threshold=0.25, max_lag=8):
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag

    def fit(self, df, cfg):
        names = list(df.columns)
        n = len(names)

        # GETTER One フルパイプライン実行！
        config = PipelineConfig(
            window_steps=min(24, max(5, len(df) // 10)),
            enable_boundary=True,
            enable_topology=True,
            enable_anomaly=True,
            enable_network=True,
            sync_threshold=self.sync_threshold,
            causal_threshold=self.causal_threshold,
            max_lag=self.max_lag,
            enable_confidence=False,
            enable_report=False,
            verbose=False,
        )

        dataset = from_dataframe(df, normalize="none")
        pipeline_result = getter_run(dataset, config=config)
        result = pipeline_result.network

        if result is None:
            return MethodOutput(
                method_name=self.method_name, names=names,
                adjacency_scores=np.zeros((n, n)),
                adjacency_bin=np.zeros((n, n), dtype=int),
                directed_support=True, lag_support=True, sign_support=True,
            )

        # MethodOutput形式に変換
        scores = np.abs(result.sync_matrix) + np.abs(result.causal_matrix)
        np.fill_diagonal(scores, 0.0)

        adjacency = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int)
        signm = np.zeros((n, n), dtype=int)

        for link in result.sync_network:
            adjacency[link.from_dim, link.to_dim] = 1
            adjacency[link.to_dim, link.from_dim] = 1
            signm[link.from_dim, link.to_dim] = int(np.sign(link.correlation))
            signm[link.to_dim, link.from_dim] = int(np.sign(link.correlation))

        for link in result.causal_network:
            adjacency[link.from_dim, link.to_dim] = 1
            lagm[link.from_dim, link.to_dim] = link.lag
            signm[link.from_dim, link.to_dim] = int(np.sign(link.correlation))

        np.fill_diagonal(adjacency, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            regime_support=False,
            lag_matrix=lagm,
            sign_matrix=signm,
            meta={
                "pattern": result.pattern,
                "hubs": result.hub_names,
                "n_boundaries": len(pipeline_result.structural_boundaries.get("boundary_locations", [])),
            },
        )


# =====================================================================
# Other Adapters (from benchmark framework)
# =====================================================================

class VARGrangerAdapter:
    method_name = "VAR_Granger"
    def __init__(self, maxlags=8, alpha=0.05): self.maxlags=maxlags; self.alpha=alpha
    def fit(self, df, cfg):
        names=list(df.columns); n=len(names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model=VAR(df.copy()); results=model.fit(maxlags=min(self.maxlags,max(1,len(df)//10)),ic='aic')
        scores=np.zeros((n,n)); adj=np.zeros((n,n),dtype=int); lagm=np.zeros((n,n),dtype=int); signm=np.zeros((n,n),dtype=int)
        params=results.params.copy()
        for i,src in enumerate(names):
            for j,dst in enumerate(names):
                if i==j: continue
                try: test=results.test_causality(caused=dst,causing=[src],kind='f'); pval=float(test.pvalue)
                except: pval=1.0
                scores[i,j]=-math.log10(max(pval,1e-12)); adj[i,j]=int(pval<self.alpha)
                cv,cl=[],[]
                for lag in range(1,results.k_ar+1):
                    rn=f"L{lag}.{src}"
                    if rn in params.index and dst in params.columns: cv.append(float(params.loc[rn,dst])); cl.append(lag)
                if cv: idx=int(np.argmax(np.abs(cv))); lagm[i,j]=cl[idx]; signm[i,j]=int(np.sign(cv[idx]))
        np.fill_diagonal(scores,0); np.fill_diagonal(adj,0)
        return MethodOutput(method_name=self.method_name,names=names,adjacency_scores=scores,adjacency_bin=adj,directed_support=True,lag_support=True,sign_support=True,lag_matrix=lagm,sign_matrix=signm)

class TransferEntropyAdapter:
    method_name = "TransferEntropy"
    def __init__(self, max_lag=8, alpha=0.05, n_perm=30): self.max_lag=max_lag; self.alpha=alpha; self.n_perm=n_perm
    def fit(self, df, cfg):
        names=list(df.columns); n=len(names); x=df.to_numpy(); rng=_rng(12345)
        scores=np.zeros((n,n)); adj=np.zeros((n,n),dtype=int); lagm=np.zeros((n,n),dtype=int)
        for i in range(n):
            for j in range(n):
                if i==j: continue
                best_te,best_lag=-np.inf,1
                for lag in range(1,min(self.max_lag,len(df)-2)+1):
                    te=_te_pyinform(x[:,i],x[:,j],lag=lag) if _HAS_PYINFORM else _te_discrete(x[:,i],x[:,j],lag=lag)
                    if te>best_te: best_te=te; best_lag=lag
                null=[(_te_pyinform if _HAS_PYINFORM else _te_discrete)(rng.permutation(x[:,i]),x[:,j],lag=best_lag) for _ in range(self.n_perm)]
                pval=(1+np.sum(np.array(null)>=best_te))/(self.n_perm+1)
                scores[i,j]=max(best_te,0); lagm[i,j]=best_lag; adj[i,j]=int(pval<self.alpha)
        np.fill_diagonal(scores,0); np.fill_diagonal(adj,0)
        return MethodOutput(method_name=self.method_name,names=names,adjacency_scores=scores,adjacency_bin=adj,directed_support=True,lag_support=True,sign_support=False,lag_matrix=lagm)

class GraphicalLassoAdapter:
    method_name = "GraphLasso"
    def __init__(self, edge_threshold=0.03): self.edge_threshold=edge_threshold
    def fit(self, df, cfg):
        names=list(df.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model=GraphicalLassoCV(); model.fit(StandardScaler().fit_transform(df.to_numpy()))
        p=np.abs(model.precision_); np.fill_diagonal(p,0)
        adj=((p>=self.edge_threshold).astype(int)+((p>=self.edge_threshold).astype(int)).T>0).astype(int)
        np.fill_diagonal(adj,0)
        return MethodOutput(method_name=self.method_name,names=names,adjacency_scores=p,adjacency_bin=adj,directed_support=False,lag_support=False,sign_support=False)

class EventXCorrAdapter:
    method_name = "EventXCorr"
    def __init__(self, max_lag=8, n_perm=30): self.max_lag=max_lag; self.n_perm=n_perm
    def fit(self, df, cfg):
        names=list(df.columns); n=len(names); x=df.to_numpy(); rng=_rng(54321)
        events=np.column_stack([_diff_events(x[:,i]) for i in range(n)])
        scores=np.zeros((n,n)); adj=np.zeros((n,n),dtype=int); lagm=np.zeros((n,n),dtype=int)
        for i in range(n):
            for j in range(n):
                if i==j: continue
                best_s,best_l=-np.inf,1
                for lag in range(1,min(self.max_lag,len(df)-2)+1):
                    s=_event_sync_score(events[:,i],events[:,j],lag)
                    if s>best_s: best_s=s; best_l=lag
                null=[_event_sync_score(rng.permutation(events[:,i]),events[:,j],lag=best_l) for _ in range(self.n_perm)]
                pval=(1+np.sum(np.array(null)>=best_s))/(self.n_perm+1)
                scores[i,j]=max(best_s,0); lagm[i,j]=best_l; adj[i,j]=int(pval<0.05)
        np.fill_diagonal(scores,0); np.fill_diagonal(adj,0)
        return MethodOutput(method_name=self.method_name,names=names,adjacency_scores=scores,adjacency_bin=adj,directed_support=True,lag_support=True,sign_support=False,lag_matrix=lagm)


class PCMCIPlusAdapter:
    method_name = "PCMCIPlus"

    def __init__(self, tau_max=8, pc_alpha=0.05, verbosity=0):
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.verbosity = verbosity

    def fit(self, df, cfg):
        if not _HAS_TIGRAMITE:
            raise RuntimeError("tigramite is not installed")

        names = list(df.columns)
        x = df.to_numpy(dtype=float)
        n = len(names)

        tg_df = pp.DataFrame(x, var_names=names)
        pcmci = PCMCI(dataframe=tg_df, cond_ind_test=ParCorr(significance='analytic'), verbosity=self.verbosity)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.pc_alpha)

        graph = results.get("graph")
        val_matrix = results.get("val_matrix")

        adjacency = np.zeros((n, n), dtype=int)
        scores = np.zeros((n, n), dtype=float)
        lagm = np.zeros((n, n), dtype=int)
        signm = np.zeros((n, n), dtype=int)

        if graph is not None:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    best_abs, best_tau, best_sign, found = 0.0, 0, 0, False
                    for tau in range(1, min(self.tau_max, graph.shape[2] - 1) + 1):
                        g = graph[i, j, tau]
                        if isinstance(g, bytes):
                            g = g.decode()
                        if g is None:
                            g = ""
                        if str(g).strip() != "":
                            found = True
                            val = 0.0
                            if val_matrix is not None:
                                try:
                                    val = float(val_matrix[i, j, tau])
                                except Exception:
                                    val = 0.0
                            if abs(val) >= best_abs:
                                best_abs = abs(val)
                                best_tau = tau
                                best_sign = int(np.sign(val)) if val != 0 else 0
                    adjacency[i, j] = int(found)
                    scores[i, j] = best_abs
                    lagm[i, j] = best_tau
                    signm[i, j] = best_sign

        np.fill_diagonal(adjacency, 0)
        np.fill_diagonal(scores, 0.0)

        return MethodOutput(
            method_name=self.method_name, names=names,
            adjacency_scores=scores, adjacency_bin=adjacency,
            directed_support=True, lag_support=True, sign_support=True,
            lag_matrix=lagm, sign_matrix=signm,
            meta={"pc_alpha": self.pc_alpha, "tau_max": self.tau_max},
        )


# =====================================================================
# Evaluation
# =====================================================================

def evaluate(output, gt):
    metrics = {}
    n = len(gt.names); mask = ~np.eye(n, dtype=bool)

    if output.adjacency_bin is not None:
        # Undirected
        pred_u = output.undirected_bin()
        true_u = gt.undirected_adjacency()
        prf = _precision_recall_f1(true_u[mask], pred_u[mask])
        metrics["edge_f1_undir"] = prf["f1"]

        # Directed
        if output.directed_support:
            prf = _precision_recall_f1(gt.adjacency[mask], output.adjacency_bin[mask])
            metrics["edge_f1_dir"] = prf["f1"]
            metrics["edge_prec_dir"] = prf["precision"]
            metrics["edge_rec_dir"] = prf["recall"]
        else:
            metrics["edge_f1_dir"] = np.nan

        # Lag accuracy
        if output.lag_support and output.lag_matrix is not None and gt.lag_matrix is not None:
            true_edges = np.argwhere(gt.adjacency == 1)
            errs = []
            for src, dst in true_edges:
                if output.adjacency_bin[src, dst] == 1:
                    errs.append(abs(int(output.lag_matrix[src, dst]) - int(gt.lag_matrix[src, dst])))
            metrics["lag_mae"] = float(np.mean(errs)) if errs else np.nan
        else:
            metrics["lag_mae"] = np.nan

        # Sign accuracy
        if output.sign_support and output.sign_matrix is not None and gt.sign_matrix is not None:
            true_edges = np.argwhere(gt.adjacency == 1)
            hits = []
            for src, dst in true_edges:
                if output.adjacency_bin[src, dst] == 1:
                    hits.append(int(np.sign(output.sign_matrix[src, dst]) == np.sign(gt.sign_matrix[src, dst])))
            metrics["sign_acc"] = float(np.mean(hits)) if hits else np.nan
        else:
            metrics["sign_acc"] = np.nan

        # Forbidden edges
        if gt.forbidden_edges:
            fp = [int(output.adjacency_bin[s,d]==1) for s,d in gt.forbidden_edges]
            metrics["spurious_rate"] = float(np.mean(fp))
        else:
            metrics["spurious_rate"] = np.nan

    return metrics


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark(n_repeats=20, save_csv=True):
    print("=" * 70)
    print("  GETTER One vs Established Causal Inference Methods")
    print("  Synthetic Benchmark with Ground Truth")
    print(f"  Repeats: {n_repeats}")
    print("  Built with 💕 by Masamichi & Tamaki")
    print("=" * 70)

    cfg = ScenarioConfig()

    methods = [
        GetterOneAdapter(sync_threshold=0.3, causal_threshold=0.25, max_lag=8),
        VARGrangerAdapter(maxlags=8),
        TransferEntropyAdapter(max_lag=8, n_perm=50),
        GraphicalLassoAdapter(),
        EventXCorrAdapter(max_lag=8, n_perm=50),
    ]
    if _HAS_TIGRAMITE:
        methods.append(PCMCIPlusAdapter(tau_max=8))
    else:
        print("  ⚠️ tigramite not installed - skipping PCMCI+")

    all_rows = []

    for scenario_name, gen_fn in SCENARIOS.items():
        print(f"\n{'─'*60}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'─'*60}")

        for repeat in range(n_repeats):
            seed = cfg.seed + repeat * 7  # 素数ステップでseed分散
            df, gt = gen_fn(cfg, seed)
            if repeat % 5 == 0:
                print(f"  [{gt.notes}] repeat={repeat}/{n_repeats}")

            for adapter in methods:
                t0 = time.time()
                try:
                    output = adapter.fit(df, cfg)
                    elapsed = time.time() - t0
                    metrics = evaluate(output, gt)
                    metrics["scenario"] = scenario_name
                    metrics["method"] = adapter.method_name
                    metrics["repeat"] = repeat
                    metrics["time_s"] = elapsed
                    all_rows.append(metrics)
                except Exception as e:
                    print(f"    ⚠️ {adapter.method_name}: {e}")

    # Aggregate
    results_df = pd.DataFrame(all_rows)
    metric_cols = [c for c in results_df.columns if c not in ("scenario", "method", "repeat", "time_s")]

    print("\n" + "=" * 70)
    print(f"  RESULTS: Per-Method Average (n={n_repeats} repeats)")
    print("=" * 70)

    summary_mean = results_df.groupby("method")[metric_cols + ["time_s"]].mean()
    summary_std = results_df.groupby("method")[metric_cols].std()
    print(summary_mean.to_string(float_format="%.3f"))

    # Mean ± Std table
    print("\n" + "=" * 70)
    print("  RESULTS: Mean ± Std")
    print("=" * 70)
    for method in summary_mean.index:
        print(f"\n  {method}:")
        for col in metric_cols:
            m = summary_mean.loc[method, col]
            s = summary_std.loc[method, col]
            if not np.isnan(m):
                print(f"    {col:>20s}: {m:.3f} ± {s:.3f}")

    # Composite score
    print("\n" + "=" * 70)
    print("  COMPOSITE SCORE (higher = better)")
    print("=" * 70)

    composite_scores = {}
    for method in summary_mean.index:
        row = summary_mean.loc[method]
        f1d = row.get("edge_f1_dir", 0) if not np.isnan(row.get("edge_f1_dir", np.nan)) else 0
        f1u = row.get("edge_f1_undir", 0) if not np.isnan(row.get("edge_f1_undir", np.nan)) else 0
        lag = row.get("lag_mae", np.nan)
        lag_score = 1/(1+lag) if not np.isnan(lag) else 0
        sign = row.get("sign_acc", np.nan)
        sign_score = sign if not np.isnan(sign) else 0
        spur = row.get("spurious_rate", np.nan)
        spur_score = (1-spur) if not np.isnan(spur) else 0.5
        time_s = row.get("time_s", 1)

        composite = (f1d * 1.5 + f1u * 1.0 + lag_score * 1.0 + sign_score * 0.8 + spur_score * 0.8) / 5.1
        composite_scores[method] = composite
        print(f"  {method:20s}: {composite:.3f}  (F1d={f1d:.3f} F1u={f1u:.3f} Lag={lag_score:.3f} Sign={sign_score:.3f} Spur={spur_score:.3f} | {time_s:.2f}s)")

    # Rank
    print("\n  🏆 Ranking:")
    for rank, (method, score) in enumerate(sorted(composite_scores.items(), key=lambda x: x[1], reverse=True), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"    {medal} {rank}. {method}: {score:.3f}")

    # Per-scenario breakdown
    print("\n" + "=" * 70)
    print("  PER-SCENARIO F1 (directed) — Mean")
    print("=" * 70)

    pivot = results_df.groupby(["scenario", "method"])["edge_f1_dir"].mean().unstack(fill_value=np.nan)
    print(pivot.to_string(float_format="%.3f"))

    print("\n" + "=" * 70)
    print("  PER-SCENARIO F1 (directed) — Std")
    print("=" * 70)

    pivot_std = results_df.groupby(["scenario", "method"])["edge_f1_dir"].std().unstack(fill_value=np.nan)
    print(pivot_std.to_string(float_format="%.3f"))

    # Save CSV
    if save_csv:
        csv_path = f"benchmark_results_{n_repeats}repeats.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  📄 Raw results saved: {csv_path}")

    return results_df


if __name__ == "__main__":
    import argparse
    import logging
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="GETTER One Causal Inference Benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=20, help="Number of repeats (default: 20)")
    parser.add_argument("--no-csv", action="store_true", help="Don't save CSV results")
    args = parser.parse_args()

    results = run_benchmark(n_repeats=args.repeats, save_csv=not args.no_csv)
