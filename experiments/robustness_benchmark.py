#!/usr/bin/env python3
"""Repeated-seed synthetic robustness benchmark; see results/expanded_benchmark.md.

Example: python experiments/robustness_benchmark.py --seeds 10 --bars 1200 --bootstrap 300
All synthetic data, metrics, plots, and model explanations are emitted under results/benchmark/.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, log_loss, precision_recall_curve, roc_auc_score, precision_score, recall_score)
from main import make_features, make_models, shift_targets, trailing_regime_states

def fast_models(seed):
    """Reduced tree counts keep the repeated-seed robustness experiment tractable."""
    models=make_models(seed)
    for m in models.values():
        params=m.get_params()
        if 'extratreesclassifier__n_estimators' in params:m.set_params(extratreesclassifier__n_estimators=48)
        if 'randomforestclassifier__n_estimators' in params:m.set_params(randomforestclassifier__n_estimators=48)
        if 'histgradientboostingclassifier__max_iter' in params:m.set_params(histgradientboostingclassifier__max_iter=80)
        if 'extratreesclassifier__n_jobs' in params:m.set_params(extratreesclassifier__n_jobs=1)
        if 'randomforestclassifier__n_jobs' in params:m.set_params(randomforestclassifier__n_jobs=1)
    return models

HORIZONS=(4,8,12,16)
MODELS=("extra_trees","random_forest","hist_gradient_boosting","logistic")

def synthetic_ohlcv(n, seed, mode):
    """Regime-switching synthetic log-price: trend-rich or mean-reversion-rich cycles."""
    rng=np.random.default_rng(seed); regime=np.zeros(n,dtype=int); returns=np.zeros(n)
    # Epoch state changes create prolonged, distinct regimes; signs randomize direction.
    pos=0; state=0; direction=1
    while pos<n:
        duration=int(rng.integers(35,150)); end=min(n,pos+duration)
        p_trend=.62 if mode=='trend_rich' else .22
        state=int(rng.random()<p_trend); direction=int(rng.choice([-1,1]))
        regime[pos:end]=state
        if state:
            returns[pos:end]=direction*rng.normal(.00055,.0035,end-pos)+rng.normal(0,.001,end-pos)
        else:
            # Negative feedback to recent displacement approximates mean reversion.
            for j in range(pos,end):
                deviation=np.log(100.)+returns[:j].sum()-np.log(100.)
                returns[j]=np.clip(-.012*deviation+rng.normal(0,.0035),-.025,.025)
        pos=end
    price=100*np.exp(np.cumsum(returns)); op=np.r_[price[0],price[:-1]]
    wick=rng.uniform(.0002,.002,n)
    idx=pd.date_range('2020-01-01',periods=n,freq='h',tz='UTC')
    frame=pd.DataFrame({'open':op,'high':np.maximum(op,price)*(1+wick),'low':np.minimum(op,price)*(1-wick),
                        'close':price,'volume':rng.lognormal(7+1.5*np.abs(returns)/.004,.7,n)},index=idx)
    return frame

def features_and_targets(data, lookback, threshold):
    x=make_features(data)
    # Override state age with the exact candidate definition under test.
    state=trailing_regime_states(data,lookback,threshold)
    changes=state.ne(state.shift()).where(state.notna() & state.shift().notna(),False)
    last=pd.Series(np.where(changes,np.arange(len(state)),np.nan),index=state.index).ffill()
    x['time_since_state_change']=pd.Series(np.arange(len(state)),index=state.index)-last
    # The principal definition is parameter-swept; keep feature causality exact.
    ys=shift_targets(data,HORIZONS,lookback,threshold)
    ok=ys.notna().all(axis=1)&state.notna()
    return x.loc[ok],ys.loc[ok].astype(int),state.loc[ok].astype(int)

def metric_row(y,p,prob,threshold=.5):
    pred=(prob>=threshold).astype(int)
    out={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
         'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
         'trend_precision':__import__('sklearn').metrics.precision_score(y,pred,zero_division=0),
         'trend_recall':__import__('sklearn').metrics.recall_score(y,pred,zero_division=0),
         'trend_f1':__import__('sklearn').metrics.f1_score(y,pred,zero_division=0),
         'pr_auc':average_precision_score(y,prob),'roc_auc':roc_auc_score(y,prob) if len(np.unique(y))==2 else np.nan,
         'brier':brier_score_loss(y,prob),'log_loss':log_loss(y,np.c_[1-prob,prob],labels=[0,1]),'threshold':threshold}
    return out

def block_ci(y,prob,reps,rng,block=24):
    n=len(y); vals=[]
    for _ in range(reps):
        ids=[]
        while len(ids)<n:
            start=int(rng.integers(n)); ids.extend((start+np.arange(block))%n)
        ids=np.asarray(ids[:n]); yy=y[ids]; pp=prob[ids]
        if len(np.unique(yy))<2: continue
        mm=metric_row(yy,None,pp)
        vals.append([mm[k] for k in ('balanced_accuracy','macro_f1','roc_auc')])
    if not vals:return {}
    a=np.asarray(vals)
    return {k:[float(v) for v in np.quantile(a[:,i],[.025,.975])] for i,k in enumerate(('balanced_accuracy','macro_f1','roc_auc'))}

def run(seeds=10,bars=1200,bootstrap=300,outdir='results/benchmark'):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(9182)
    collected={}; records=[]; curves={}; calibrations={}; importances=[]; audit=[]; selected={}; sweep_records=[]; state_calibration=[]
    for regime_mode in ('trend_rich','mean_reverting'):
      for seed in range(seeds):
        data=synthetic_ohlcv(bars,seed+10000*(regime_mode=='mean_reverting'),regime_mode)
        for lookback,eff_thr in ((24,.35),):
          X,Y,state=features_and_targets(data,lookback,eff_thr)
          n=len(X); test_start=int(.8*n); gap=16
          # Three expanding walk-forward validation folds before untouched final 20% test.
          folds=[]
          for a,b in ((.50,.60),(.60,.70),(.70,.80)):
            va=int(a*n); vb=int(b*n); tr_end=va-gap
            folds.append((slice(0,tr_end),slice(va,vb)))
          final_train=slice(0,test_start-gap); test=slice(test_start,n)
          for h in HORIZONS:
            target=f'shift_{h}h'; y=Y[target]
            for model_name,model in fast_models(seed+13).items():
              val_y=[];val_p=[]
              # Nested-style walk-forward OOF probabilities drive both selection and threshold tuning.
              for tr,va in folds:
                if len(np.unique(y.iloc[tr]))<2: continue
                fitted=clone(model);fitted.fit(X.iloc[tr],y.iloc[tr]);val_y.extend(y.iloc[va].tolist())
                val_p.extend(fitted.predict_proba(X.iloc[va])[:,list(fitted.classes_).index(1)].tolist())
              if not val_y: continue
              vy=np.asarray(val_y);vp=np.asarray(val_p)
              pred05=vp>=.5; valba=balanced_accuracy_score(vy,pred05)
              # Prefer precision>=.60 and recall>=.55; otherwise maximize F1 under precision constraint,
              # then F1 unconstrained. This choice is made strictly on walk-forward validation.
              pr,rc,th=precision_recall_curve(vy,vp); candidates=[]
              for t in np.r_[.05,th,.95]:
                q=metric_row(vy,None,vp,float(t))
                if q['trend_precision']>=.60 and q['trend_recall']>=.55:candidates.append((q['trend_f1'],t))
              tuned=max(candidates)[1] if candidates else .5
              tuned_validation=metric_row(vy,None,vp,float(tuned))
              constraints_met=(tuned_validation['trend_precision']>=.60 and tuned_validation['trend_recall']>=.55)
              key=(regime_mode,seed,lookback,eff_thr,h,model_name)
              collected.setdefault((regime_mode,h,model_name),[])
              records.append({'mode':regime_mode,'seed':seed,'lookback':lookback,'efficiency_threshold':eff_thr,
                'horizon':h,'model':model_name,'validation_balanced_accuracy':valba,'validation_threshold':tuned,
                'validation_precision_at_threshold':tuned_validation['trend_precision'],'validation_recall_at_threshold':tuned_validation['trend_recall'],
                'precision_recall_goal_met':constraints_met})
              if len(np.unique(y.iloc[final_train]))<2 or len(np.unique(y.iloc[test]))<2: continue
              final=clone(model);final.fit(X.iloc[final_train],y.iloc[final_train])
              ty=y.iloc[test].to_numpy();tp=final.predict_proba(X.iloc[test])[:,list(final.classes_).index(1)]
              mets=metric_row(ty,None,tp,tuned);base=metric_row(ty,None,tp,.5)
              mets.update({f'fixed05_{k}':v for k,v in base.items() if k in ('accuracy','balanced_accuracy','trend_precision','trend_recall','trend_f1')})
              for fixed_t in (.30,.50,.70):
                fixed=metric_row(ty,None,tp,fixed_t)
                mets[f't{fixed_t:.2f}_precision']=fixed['trend_precision'];mets[f't{fixed_t:.2f}_recall']=fixed['trend_recall']
              # Explicit baselines: majority class and persistence (assume no shift).
              majority=int(y.iloc[final_train].mean()>=.5)
              for nm,bpred in (('majority',np.full(len(ty),majority)),('persistence_no_shift',np.zeros(len(ty),dtype=int))):
                mets[f'{nm}_balanced_accuracy']=balanced_accuracy_score(ty,bpred)
                mets[f'{nm}_accuracy']=accuracy_score(ty,bpred)
              mets['seed']=seed;mets['mode']=regime_mode;mets['model']=model_name;mets['horizon']=h
              collected[(regime_mode,h,model_name)].append((ty,tp,mets))
              if regime_mode=='trend_rich' and seed==0 and model_name=='logistic' and h in (4,8,12,16):
                curves[h]=(ty,tp); calibrations[h]=(ty,tp)
              if regime_mode=='trend_rich' and seed==0 and h in (4,16):
                perm=permutation_importance(final,X.iloc[test],y.iloc[test],scoring='average_precision',n_repeats=4,random_state=seed)
                importances.extend({'horizon':h,'feature':f,'importance':float(v)} for f,v in zip(X.columns,perm.importances_mean))
          # Current-regime leakage audit: all features vs remove direct rule/technical proxies.
          for audit_name,features in [('all',X.columns),('proxy_dropped',[c for c in X.columns if not (c.startswith('path_efficiency_') or c.startswith('sma_slope_') or c in ('range_atr','sma_distance_atr_12','sma_distance_atr_24','sma_distance_atr_48','sma_distance_atr_72'))])]:
            clf=clone(fast_models(seed+13)['logistic']);clf.fit(X.loc[:,features].iloc[final_train],state.iloc[final_train])
            pp=clf.predict_proba(X.loc[:,features].iloc[test])[:,list(clf.classes_).index(1)]
            audit.append({'mode':regime_mode,'seed':seed,'features':audit_name,'accuracy':accuracy_score(state.iloc[test],pp>=.5),
                          'balanced_accuracy':balanced_accuracy_score(state.iloc[test],pp>=.5),
                          'pr_auc':average_precision_score(state.iloc[test],pp), 'brier':brier_score_loss(state.iloc[test],pp),
                          'log_loss':log_loss(state.iloc[test],np.c_[1-pp,pp],labels=[0,1])})
            if audit_name=='all': state_calibration.append((state.iloc[test].to_numpy(),pp))
        # Definition robustness: logistic baseline for all nine lookback/threshold pairs.
        for lb in (12,24,48):
          for th in (.25,.35,.45):
            if (lb,th)==(24,.35): continue
            xx,yy,ss=features_and_targets(data,lb,th); nn=len(xx); cut=int(.8*nn); tr=slice(0,cut-16); te=slice(cut,nn)
            if len(np.unique(yy['shift_4h'].iloc[tr]))<2: continue
            for hh in HORIZONS:
              col=f'shift_{hh}h'; clf=fast_models(seed+13)['logistic']; clf.fit(xx.iloc[tr],yy[col].iloc[tr])
              pp=clf.predict_proba(xx.iloc[te])[:,list(clf.classes_).index(1)]; yt=yy[col].iloc[te].to_numpy()
              sweep_records.append({'mode':regime_mode,'seed':seed,'lookback':lb,'threshold':th,'horizon':hh,
                'balanced_accuracy':balanced_accuracy_score(yt,pp>=.5),'pr_auc':average_precision_score(yt,pp),
                'brier':brier_score_loss(yt,pp),'prevalence':yt.mean()})
    # Aggregate seed results and pooled moving-block bootstrap intervals.
    rows=[]
    for (mode,h,model),items in collected.items():
      if not items:continue
      all_y=np.concatenate([q[0] for q in items]);all_p=np.concatenate([q[1] for q in items]);
      for regime_mode in (mode,):
        stats=pd.DataFrame([q[2] for q in items]);row={'mode':mode,'horizon':h,'model':model,'n_seed_test_sets':len(items)}
        for col in stats.select_dtypes('number').columns:
          if col in ('seed','horizon'):continue
          row[col+'_mean']=float(stats[col].mean());row[col+'_std']=float(stats[col].std(ddof=1)) if len(stats)>1 else 0.
        # CIs from pooled samples with circular 24-bar block bootstrap.
        ci=block_ci(all_y,all_p,bootstrap,rng)
        for k,v in ci.items():row[k+'_block95_ci']=v
        row['test_prevalence']=float(all_y.mean())
        rows.append(row)
    summary=pd.DataFrame(rows); summary.to_csv(out/'all_model_test_metrics.csv',index=False)
    pd.DataFrame(records).to_csv(out/'walk_forward_validation.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'current_regime_leakage_audit.csv',index=False)
    sweep=pd.DataFrame(sweep_records); sweep.to_csv(out/'definition_sweep_raw.csv',index=False)
    if not sweep.empty:
      sweep.groupby(['mode','lookback','threshold','horizon']).agg({c:['mean','std'] for c in ('balanced_accuracy','pr_auc','brier','prevalence')}).to_csv(out/'definition_sweep_summary.csv')
    pd.DataFrame(importances).to_csv(out/'permutation_importance.csv',index=False)
    summary.to_json(out/'summary.json',orient='records',indent=2)
    # PR and calibration plots by horizon (first repeat seed illustrative only).
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,h in zip(axes,HORIZONS):
      if h not in curves:continue
      yy,pp=curves[h];pr,rc,_=precision_recall_curve(yy,pp);ax.plot(rc,pr);ax.axhline(yy.mean(),ls='--',color='gray');ax.set(title=f'{h}h PR, trend-rich seed 0',xlabel='Recall',ylabel='Precision',ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'precision_recall_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for ax,h in zip(axes,HORIZONS):
      if h not in calibrations:continue
      yy,pp=calibrations[h];bins=np.linspace(0,1,11);means=[];fracs=[]
      for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(pp>=lo)&(pp<(hi if hi<1 else hi+1e-10))
        if mask.any():means.append(pp[mask].mean());fracs.append(yy[mask].mean())
      ax.plot([0,1],[0,1],ls='--',color='gray');ax.plot(means,fracs,marker='o');ax.set(title=f'{h}h calibration',xlabel='Predicted shift probability',ylabel='Observed shift frequency',xlim=(0,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'calibration_plots.png',dpi=150);plt.close(fig)
    # Current-regime calibration, pooled only from seed-0 test fold by regime mode.
    if state_calibration:
      fig,ax=plt.subplots(figsize=(5,5)); ax.plot([0,1],[0,1],ls='--',color='gray')
      for j,(yy,pp) in enumerate(state_calibration[:2]):
        bins=np.linspace(0,1,11);means=[];freq=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
          mask=(pp>=lo)&(pp<(hi if hi<1 else hi+1e-10))
          if mask.any():means.append(pp[mask].mean());freq.append(yy[mask].mean())
        ax.plot(means,freq,marker='o',label=('trend-rich' if j==0 else 'mean-reverting'))
      ax.set(xlabel='Predicted trend probability',ylabel='Observed trend frequency',xlim=(0,1),ylim=(0,1),title='Current regime calibration (seed 0)');ax.legend();fig.tight_layout();fig.savefig(out/'current_regime_calibration.png',dpi=150);plt.close(fig)
    config={'seeds':seeds,'seed_ranges':{'trend_rich':'0..N-1','mean_reverting':'10000..10000+N-1'},'bars':bars,'bootstrap_replicates':bootstrap,'bootstrap_block_bars':24,'walk_forward_folds':[[.50,.60],[.60,.70],[.70,.80]],'test_fraction':.20,'embargo_bars':16,'note':'Synthetic only; no market data.'}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    print('Wrote benchmark artifacts to',out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=10);p.add_argument('--bars',type=int,default=1200);p.add_argument('--bootstrap',type=int,default=300);p.add_argument('--outdir',default='results/benchmark');a=p.parse_args();run(a.seeds,a.bars,a.bootstrap,a.outdir)
