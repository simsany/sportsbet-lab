"""Five-family, time-ordered multi-sport modelling engine.

Canonical data schema: date, home_team, away_team, home_score, away_score.
Features are pre-event only. No artifact is emitted until all five families train.
"""
from __future__ import annotations
import os, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar
from scipy.stats import poisson, norm
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'processed'; ART=ROOT/'models'
SPORTS=['football','tennis','basketball','baseball','nfl']
MIN_ROWS=int(os.getenv('MIN_TRAINING_ROWS','350'))
_ARTIFACT_CACHE={}
FEATURES=['elo_diff','home_elo','away_elo','home_scored_avg5','home_allowed_avg5','away_scored_avg5','away_allowed_avg5','home_win_pct5','away_win_pct5','home_games','away_games','rest_diff_days','home_rest_days','away_rest_days']


def _features(df):
    ratings={}; history={}; last_date={}; rows=[]
    for r in df.itertuples(index=False):
        h=str(r.home_team); a=str(r.away_team); date=pd.Timestamp(r.date)
        rh=ratings.get(h,1500.); ra=ratings.get(a,1500.)
        hh=history.get(h,[]); ah=history.get(a,[])
        def rest(team):
            prev=last_date.get(team)
            return float(np.clip((date-prev).total_seconds()/86400,0,30)) if prev is not None else 7.0
        hr=rest(h); ar=rest(a)
        def avg(items,idx): return float(np.mean([v[idx] for v in items[-5:]])) if items else 0.0
        rows.append([rh-ra,rh,ra,avg(hh,0),avg(hh,1),avg(ah,0),avg(ah,1),avg(hh,2),avg(ah,2),len(hh),len(ah),hr-ar,hr,ar])
        hs=float(r.home_score); aws=float(r.away_score)
        expected=1/(1+10**((ra-rh)/400)); actual=1. if hs>aws else (0.5 if hs==aws else 0.)
        k=24.; ratings[h]=rh+k*(actual-expected); ratings[a]=ra-k*(actual-expected)
        history.setdefault(h,[]).append((hs,aws,1. if hs>aws else (0.5 if hs==aws else 0.)))
        history.setdefault(a,[]).append((aws,hs,1. if aws>hs else (0.5 if aws==hs else 0.)))
        last_date[h]=date; last_date[a]=date
    return np.asarray(rows,dtype=float),ratings,history,last_date


def _target(df,sport):
    if sport=='football': return np.where(df.home_score>df.away_score,0,np.where(df.home_score==df.away_score,1,2)).astype(int)
    return (df.home_score>df.away_score).astype(int)


def _softmax_temperature(p,t):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1.0)
    z=np.log(p)/max(float(t),0.05); z-=z.max(axis=1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=1,keepdims=True)


def _fit_temperature(p,y):
    if len(y)<30:return 1.0
    def loss(t):
        q=_softmax_temperature(p,t)
        return log_loss(y,q,labels=list(range(q.shape[1])))
    try:return float(minimize_scalar(loss,bounds=(0.5,3.0),method='bounded').x)
    except Exception:return 1.0


def _elo_probs(x,sport,temp=1.0):
    d=np.clip(x[:,0]/400,-8,8); ph=1/(1+10**(-d))
    if sport=='football':
        draw=np.clip(0.25*np.exp(-np.abs(d)),0.05,0.32)
        p=np.column_stack([ph*(1-draw),draw,(1-ph)*(1-draw)])
    else:p=np.column_stack([ph,1-ph])
    return _softmax_temperature(p,temp)


def _poisson_grid(lh,la,sport,rho=0.0,margin_sd=None):
    if sport in ('football','baseball'):
        maxv=max(12,int(max(lh,la)*3+10)); vals=np.arange(maxv+1)
        mat=np.outer(poisson.pmf(vals,max(.05,float(lh))),poisson.pmf(vals,max(.05,float(la))))
        if sport=='football':
            mat[0,0]*=max(.01,1-lh*la*rho); mat[0,1]*=max(.01,1+lh*rho); mat[1,0]*=max(.01,1+la*rho); mat[1,1]*=max(.01,1-rho)
        home=np.tril(mat,-1).sum(); draw=np.trace(mat); away=np.triu(mat,1).sum()
        if sport=='football':
            p=np.array([home,draw,away]); return p/p.sum()
        p=np.array([home,away]); return p/p.sum()
    # Basketball/NFL: sport-adapted Gaussian score-margin distribution; avoids a
    # nonsensical Poisson assumption for high-scoring point totals.
    sd=max(1.0,float(margin_sd or 12.0)); p=float(norm.cdf((lh-la)/sd))
    return np.array([p,1-p])


def _align_proba(model,x,n):
    raw=model.predict_proba(x); out=np.zeros((len(x),n))
    for j,c in enumerate(model.classes_):
        if int(c)<n:out[:,int(c)]=raw[:,j]
    out=np.clip(out,1e-8,1); return out/out.sum(axis=1,keepdims=True)


def _make_models(classes):
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    common=dict(n_estimators=110,max_depth=4,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=2,n_jobs=2,random_state=42)
    xgb_args=dict(common); xgb_args.update(objective='multi:softprob' if classes==3 else 'binary:logistic',eval_metric='mlogloss' if classes==3 else 'logloss')
    if classes==3:xgb_args['num_class']=classes
    xgb=XGBClassifier(**xgb_args)
    cat=CatBoostClassifier(iterations=110,depth=5,learning_rate=.04,loss_function='MultiClass' if classes==3 else 'Logloss',verbose=False,random_seed=42,thread_count=2,allow_writing_files=False)
    lgb_args=dict(n_estimators=110,max_depth=4,learning_rate=.04,num_leaves=20,verbosity=-1,n_jobs=2,random_state=42,objective='multiclass' if classes==3 else 'binary')
    if classes==3:lgb_args['num_class']=classes
    lgb=LGBMClassifier(**lgb_args)
    return {'xgboost':xgb,'catboost':cat,'lightgbm':lgb}


def _metrics(y,p):
    p=np.clip(p,1e-8,1);p=p/p.sum(axis=1,keepdims=True)
    one=np.eye(p.shape[1])[y]
    return {'accuracy':round(float(accuracy_score(y,p.argmax(axis=1))),5),'log_loss':round(float(log_loss(y,p,labels=list(range(p.shape[1])))),5),'brier_score':round(float(np.mean(np.sum((p-one)**2,axis=1))),5),'holdout_rows':int(len(y))}


def train_sport(sport):
    path=DATA/f'{sport}.csv'
    if not path.exists():return {'sport':sport,'status':'no_data','rows':0}
    df=pd.read_csv(path)
    required=['date','home_team','away_team','home_score','away_score']
    missing=[c for c in required if c not in df]
    if missing:return {'sport':sport,'status':'invalid_schema','missing':missing}
    df['date']=pd.to_datetime(df.date,utc=True,errors='coerce')
    for c in ['home_score','away_score']:df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=required).sort_values('date').drop_duplicates(['date','home_team','away_team']).reset_index(drop=True)
    df=df[(df.home_score>=0)&(df.away_score>=0)]
    if len(df)<MIN_ROWS:return {'sport':sport,'status':'insufficient_data','rows':len(df),'minimum':MIN_ROWS}
    X,ratings,history,last_date=_features(df); y=_target(df,sport); nclass=3 if sport=='football' else 2
    if len(np.unique(y))<nclass:return {'sport':sport,'status':'insufficient_classes','rows':len(df)}
    # 70% fit, 10% calibration, 20% untouched chronological test.
    n=len(df); fit_end=int(n*.70); cal_end=int(n*.80)
    if n-cal_end<30 or cal_end-fit_end<30:return {'sport':sport,'status':'insufficient_validation_rows','rows':n}
    Xfit,Xcal,Xtest=X[:fit_end],X[fit_end:cal_end],X[cal_end:]
    yfit,ycal,ytest=y[:fit_end],y[fit_end:cal_end],y[cal_end:]
    if len(np.unique(yfit))<nclass or len(np.unique(ycal))<nclass:return {'sport':sport,'status':'insufficient_classes_in_time_split','rows':n}
    models=_make_models(nclass)
    for m in models.values():m.fit(Xfit,yfit)
    raw={k:_align_proba(m,Xcal,nclass) for k,m in models.items()}
    temps={k:_fit_temperature(v,ycal) for k,v in raw.items()}
    # Elo calibration uses the same earlier calibration partition, never final holdout.
    elo_temp=_fit_temperature(_elo_probs(Xcal,sport),ycal)
    # Score-family model: goals/runs for football/baseball, score margin for basketball/NFL/tennis.
    if sport in ('football','baseball'):
        sh=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(Xfit,df.home_score.iloc[:fit_end])
        sa=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(Xfit,df.away_score.iloc[:fit_end])
        lh=np.clip(sh.predict(Xcal),.05,20); la=np.clip(sa.predict(Xcal),.05,20)
        rho=0.0
        if sport=='football':
            def objective(r):
                vals=[]
                for a,b,hg,ag in zip(lh,la,df.home_score.iloc[fit_end:cal_end],df.away_score.iloc[fit_end:cal_end]):
                    tau=1.
                    if hg==0 and ag==0:tau=1-a*b*r
                    elif hg==0 and ag==1:tau=1+a*r
                    elif hg==1 and ag==0:tau=1+b*r
                    elif hg==1 and ag==1:tau=1-r
                    vals.append(-np.log(max(1e-10,tau*poisson.pmf(hg,a)*poisson.pmf(ag,b))))
                return float(np.mean(vals))
            rho=float(minimize_scalar(objective,bounds=(-.12,.12),method='bounded').x)
        cal_score=np.asarray([_poisson_grid(a,b,sport,rho) for a,b in zip(lh,la)])
        score_temp=_fit_temperature(cal_score,ycal)
        # Refit score models on fit+calibration only.
        score_home=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(X[:cal_end],df.home_score.iloc[:cal_end])
        score_away=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(X[:cal_end],df.away_score.iloc[:cal_end])
        margin_sd=None
    else:
        margin=(df.home_score-df.away_score).iloc[:fit_end].to_numpy()
        score_home=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(Xfit,margin)
        residual=margin-score_home.predict(Xfit); margin_sd=max(float(np.std(residual)),1.0)
        cal_margin=score_home.predict(Xcal)
        cal_score=np.asarray([_poisson_grid(float(v),0,sport,margin_sd=margin_sd) for v in cal_margin])
        score_temp=_fit_temperature(cal_score,ycal); score_away=None; rho=0.0
        score_home=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X[:cal_end],(df.home_score-df.away_score).iloc[:cal_end])
    # Evaluate all five models on untouched final 20% holdout.
    evals={}
    evals['elo']=_metrics(ytest,_elo_probs(Xtest,sport,elo_temp))
    if sport in ('football','baseball'):
        shv=np.clip(score_home.predict(Xtest),.05,20); sav=np.clip(score_away.predict(Xtest),.05,20)
        ps=np.asarray([_poisson_grid(a,b,sport,rho) for a,b in zip(shv,sav)])
    else:
        mv=score_home.predict(Xtest); ps=np.asarray([_poisson_grid(float(v),0,sport,margin_sd=margin_sd) for v in mv])
    evals['poisson']=_metrics(ytest,_softmax_temperature(ps,score_temp))
    for k,m in models.items():evals[k]=_metrics(ytest,_softmax_temperature(_align_proba(m,Xtest,nclass),temps[k]))
    # Refit score-distribution family on all completed historical rows after holdout scoring.
    if sport in ('football','baseball'):
        score_home=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(X,df.home_score)
        score_away=make_pipeline(StandardScaler(),PoissonRegressor(alpha=.7,max_iter=500)).fit(X,df.away_score)
    else:
        all_margin=(df.home_score-df.away_score).to_numpy()
        score_home=make_pipeline(StandardScaler(),Ridge(alpha=10.0)).fit(X,all_margin)
        margin_sd=max(float(np.std(all_margin-score_home.predict(X))),1.0)
    # Final deployed classifiers are refit on all available history; calibration parameters
    # remain learned only on the earlier calibration window, and final holdout stays untouched.
    final_models=_make_models(nclass)
    for m in final_models.values():m.fit(X,y)
    artifact={'sport':sport,'trained_at':datetime.now(timezone.utc).isoformat(),'rows':n,'fit_rows':fit_end,'calibration_rows':cal_end-fit_end,'holdout_rows':n-cal_end,
      'feature_names':FEATURES,'ratings':ratings,'history':history,'last_date':{k:v.isoformat() for k,v in last_date.items()},
      'score_home':score_home,'score_away':score_away,'rho':rho,'margin_sd':margin_sd,'models':final_models,
      'temperatures':{**temps,'elo':elo_temp,'poisson':score_temp},'evaluation':evals,'classes':nclass}
    ART.mkdir(parents=True,exist_ok=True); tmp=ART/f'{sport}.joblib.tmp';joblib.dump(artifact,tmp,compress=3);tmp.replace(ART/f'{sport}.joblib')
    _ARTIFACT_CACHE[sport]=artifact
    return {'sport':sport,'status':'trained','rows':n,'holdout_rows':n-cal_end,'evaluation':evals}


def train_all():
    results=[]
    for sport in SPORTS:
        try:results.append(train_sport(sport))
        except Exception as e:results.append({'sport':sport,'status':'training_error','error':str(e)[:700]})
    return results


def _fixture_features(a,home,away,event_date=None):
    h=str(home);w=str(away);ratings=a['ratings'];hist=a['history'];last=a['last_date']
    rh=ratings.get(h,1500.);ra=ratings.get(w,1500.);hh=hist.get(h,[]);ah=hist.get(w,[])
    now=pd.Timestamp(event_date) if event_date else pd.Timestamp.now(tz='UTC')
    if now.tzinfo is None:now=now.tz_localize('UTC')
    else:now=now.tz_convert('UTC')
    def rest(t):
        prev=pd.Timestamp(last[t]) if t in last else now-pd.Timedelta(days=7)
        return float(np.clip((now-prev).total_seconds()/86400,0,30))
    def avg(v,i):return float(np.mean([x[i] for x in v[-5:]])) if v else 0.
    hr=rest(h);ar=rest(w)
    return np.asarray([[rh-ra,rh,ra,avg(hh,0),avg(hh,1),avg(ah,0),avg(ah,1),avg(hh,2),avg(ah,2),len(hh),len(ah),hr-ar,hr,ar]],dtype=float)


def _known_key(name, known):
    import re
    norm=lambda v: re.sub(r'[^a-z0-9]','',str(v).lower())
    target=norm(name)
    for key in known:
        if norm(key)==target:return key
    return None

def predict(sport,home,away,event_date=None):
    path=ART/f'{sport}.joblib'
    if not path.exists():return None
    a=_ARTIFACT_CACHE.get(sport)
    if a is None:a=joblib.load(path);_ARTIFACT_CACHE[sport]=a
    hk=_known_key(home,a['ratings']); ak=_known_key(away,a['ratings'])
    if hk is None or ak is None:return None
    x=_fixture_features(a,hk,ak,event_date); n=a['classes']; raw={}
    raw['elo']=_elo_probs(x,sport,a['temperatures']['elo'])[0]
    if sport in ('football','baseball'):
        lh=float(a['score_home'].predict(x)[0]);la=float(a['score_away'].predict(x)[0]); pp=_poisson_grid(lh,la,sport,a['rho'])
    else:
        margin=float(a['score_home'].predict(x)[0]);pp=_poisson_grid(margin,0,sport,margin_sd=a['margin_sd'])
    raw['poisson']=_softmax_temperature(pp.reshape(1,-1),a['temperatures']['poisson'])[0]
    for k,m in a['models'].items():raw[k]=_softmax_temperature(_align_proba(m,x,n),a['temperatures'][k])[0]
    arr=np.asarray(list(raw.values()));avg=arr.mean(axis=0)
    return {'probabilities':avg.tolist(),'model_probabilities':{k:v.tolist() for k,v in raw.items()},'agreement':int(np.max(np.bincount(arr.argmax(axis=1),minlength=n))),'uncertainty':float(np.std(arr,axis=0).mean()*100),'models_used':5,'training_rows':a['rows'],'trained_at':a['trained_at'],'evaluation':a['evaluation']}


def model_status():
    out=[]
    for s in SPORTS:
        p=ART/f'{s}.joblib'
        if not p.exists():out.append({'sport':s,'status':'not_trained','rows':0});continue
        try:
            a=_ARTIFACT_CACHE.get(s)
            if a is None:a=joblib.load(p);_ARTIFACT_CACHE[s]=a
            out.append({'sport':s,'status':'trained','rows':a['rows'],'trained_at':a['trained_at'],'holdout_rows':a['holdout_rows'],'evaluation':a['evaluation']})
        except Exception as e:out.append({'sport':s,'status':'artifact_error','error':str(e)[:300]})
    return out
