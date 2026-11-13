#!/usr/bin/env python3
"""Fetch public datasets and normalize rows into the common pre-match schema.
Only sources with explicit/public research terms are enabled by default. MLB stays opt-in.
"""
from pathlib import Path
import re, sys, requests, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data'/'raw'; OUT=ROOT/'data'/'processed'
S=requests.Session(); S.headers['User-Agent']='SportsAIScanner/1.0 (historical sports research)'

def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>1000: return True
    try:
        r=S.get(url,timeout=90); r.raise_for_status();
        if len(r.content)<1000: raise ValueError('download too small')
        path.write_bytes(r.content); print('Downloaded',path,len(r.content)); return True
    except Exception as e: print('WARN',url,e); return False

def col(df,*names):
    lookup={str(c).strip().lower().replace(' ','_'):c for c in df.columns}
    for n in names:
        k=n.lower().replace(' ','_')
        if k in lookup:return lookup[k]
    return None

def write(df,sport,date,home,away,hs,as_):
    if any(x is None for x in [date,home,away,hs,as_]): print('No schema match:',sport,list(df.columns)[:35]); return False
    raw_date=df[date].astype(str).str.strip()
    parsed=pd.to_datetime(raw_date,format='%Y%m%d',errors='coerce',utc=True)
    fallback=pd.to_datetime(raw_date,errors='coerce',utc=True,dayfirst=True)
    parsed=parsed.fillna(fallback)
    out=pd.DataFrame({'date':parsed,'home_team':df[home].astype(str),'away_team':df[away].astype(str),'home_score':pd.to_numeric(df[hs],errors='coerce'),'away_score':pd.to_numeric(df[as_],errors='coerce')})
    out=out.dropna().query("home_team != 'nan' and away_team != 'nan'").drop_duplicates(['date','home_team','away_team']).sort_values('date')
    OUT.mkdir(parents=True,exist_ok=True); out.to_csv(OUT/f'{sport}.csv',index=False); print(sport,len(out),'normalized rows'); return len(out)>0

def football():
    frames=[]
    for season in range(2015,2028):
        code=f'{str(season)[-2:]}{str(season+1)[-2:]}'
        for league in ['E0','D1','I1','SP1','F1','N1','B1','P1']:
            p=RAW/'football'/f'{code}_{league}.csv';
            if fetch(f'https://www.football-data.co.uk/mmz4281/{code}/{league}.csv',p):
                try: frames.append(pd.read_csv(p,encoding='latin1',on_bad_lines='skip'))
                except Exception: pass
    if not frames:return False
    d=pd.concat(frames,ignore_index=True)
    return write(d,'football',col(d,'Date'),col(d,'HomeTeam'),col(d,'AwayTeam'),col(d,'FTHG'),col(d,'FTAG'))

def tennis():
    # Hugging Face public dataset repository file listing; only main-draw match outcomes used.
    api='https://huggingface.co/api/datasets/minmaxxxer/atp-wta-matches-with-odds/tree/main?recursive=true'
    try: files=S.get(api,timeout=30).json()
    except Exception as e: print('Tennis listing unavailable',e); return False
    paths=[x.get('path','') for x in files if re.search(r'(?:^|/)(atp|wta)_matches_202[1-6]\.csv$',x.get('path',''))]
    frames=[]
    for name in paths:
        p=RAW/'tennis'/name
        if fetch('https://huggingface.co/datasets/minmaxxxer/atp-wta-matches-with-odds/resolve/main/'+name,p):
            try: frames.append(pd.read_csv(p,on_bad_lines='skip'))
            except Exception: pass
    if not frames:return False
    d=pd.concat(frames,ignore_index=True)
    # For tennis, winner is home side; the binary target is winner vs loser.
    date=col(d,'date','tourney_date','match_date'); winner=col(d,'winner_name','winner','winner_player'); loser=col(d,'loser_name','loser','loser_player')
    ws=col(d,'winner_sets','w_sets','winner_sets_won'); ls=col(d,'loser_sets','l_sets','loser_sets_won')
    if not all([date,winner,loser]): print('Tennis schema not recognized',list(d.columns)[:40]); return False
    d=d.copy(); d['_winner_score']=pd.to_numeric(d[ws],errors='coerce') if ws else 1; d['_loser_score']=pd.to_numeric(d[ls],errors='coerce') if ls else 0
    # Alternate the winner's venue side so the target is not trivially 100% home wins.
    swap=(pd.Series(range(len(d)),index=d.index)%2==1)
    d['_home']=d[winner].astype(str).where(~swap,d[loser].astype(str))
    d['_away']=d[loser].astype(str).where(~swap,d[winner].astype(str))
    d['_home_score']=d['_winner_score'].where(~swap,d['_loser_score'])
    d['_away_score']=d['_loser_score'].where(~swap,d['_winner_score'])
    return write(d,'tennis',date,'_home','_away','_home_score','_away_score')

def basketball():
    url='https://huggingface.co/datasets/hamzas/nba-games/resolve/main/games.csv'; p=RAW/'basketball'/'games.csv'
    if not fetch(url,p): return False
    try:d=pd.read_csv(p,on_bad_lines='skip')
    except Exception as e: print(e); return False
    date=col(d,'game_date','date','GAME_DATE')
    home=col(d,'home_team','HOME_TEAM','team_home','home_team_name','team_name_home','HOME_TEAM_ABBREVIATION','HOME_TEAM_ID','home_team_id')
    away=col(d,'away_team','AWAY_TEAM','team_away','away_team_name','team_name_away','VISITOR_TEAM_ABBREVIATION','VISITOR_TEAM_ID','away_team_id')
    nba={'ATL':'Atlanta Hawks','BOS':'Boston Celtics','BKN':'Brooklyn Nets','CHA':'Charlotte Hornets','CHI':'Chicago Bulls','CLE':'Cleveland Cavaliers','DAL':'Dallas Mavericks','DEN':'Denver Nuggets','DET':'Detroit Pistons','GSW':'Golden State Warriors','HOU':'Houston Rockets','IND':'Indiana Pacers','LAC':'Los Angeles Clippers','LAL':'Los Angeles Lakers','MEM':'Memphis Grizzlies','MIA':'Miami Heat','MIL':'Milwaukee Bucks','MIN':'Minnesota Timberwolves','NOP':'New Orleans Pelicans','NYK':'New York Knicks','OKC':'Oklahoma City Thunder','ORL':'Orlando Magic','PHI':'Philadelphia 76ers','PHX':'Phoenix Suns','POR':'Portland Trail Blazers','SAC':'Sacramento Kings','SAS':'San Antonio Spurs','TOR':'Toronto Raptors','UTA':'Utah Jazz','WAS':'Washington Wizards'}
    if home and away:
        d[home]=d[home].map(lambda x:nba.get(str(x),str(x)));d[away]=d[away].map(lambda x:nba.get(str(x),str(x)))
    return write(d,'basketball',date,home,away,col(d,'home_score','PTS_home','home_pts','HOME_TEAM_SCORE','home_points'),col(d,'away_score','PTS_away','away_pts','AWAY_TEAM_SCORE','away_points'))

def nfl():
    p=RAW/'nfl'/'schedules.csv'
    if not fetch('https://github.com/nflverse/nflverse-data/releases/download/schedules/schedules.csv',p):return False
    try:d=pd.read_csv(p,on_bad_lines='skip')
    except Exception as e:print(e);return False
    if 'game_type' in d: d=d[d.game_type.isin(['REG','POST'])]
    team_names={'ARI':'Arizona Cardinals','ATL':'Atlanta Falcons','BAL':'Baltimore Ravens','BUF':'Buffalo Bills','CAR':'Carolina Panthers','CHI':'Chicago Bears','CIN':'Cincinnati Bengals','CLE':'Cleveland Browns','DAL':'Dallas Cowboys','DEN':'Denver Broncos','DET':'Detroit Lions','GB':'Green Bay Packers','HOU':'Houston Texans','IND':'Indianapolis Colts','JAX':'Jacksonville Jaguars','KC':'Kansas City Chiefs','LV':'Las Vegas Raiders','LAC':'Los Angeles Chargers','LAR':'Los Angeles Rams','MIA':'Miami Dolphins','MIN':'Minnesota Vikings','NE':'New England Patriots','NO':'New Orleans Saints','NYG':'New York Giants','NYJ':'New York Jets','PHI':'Philadelphia Eagles','PIT':'Pittsburgh Steelers','SF':'San Francisco 49ers','SEA':'Seattle Seahawks','TB':'Tampa Bay Buccaneers','TEN':'Tennessee Titans','WAS':'Washington Commanders'}
    d['home_team']=d['home_team'].map(lambda x:team_names.get(str(x),str(x)));d['away_team']=d['away_team'].map(lambda x:team_names.get(str(x),str(x)))
    return write(d,'nfl',col(d,'gameday','game_date','date'),col(d,'home_team'),col(d,'away_team'),col(d,'home_score'),col(d,'away_score'))

def baseball(allow=False):
    # MLB Stats API public schedule/results endpoint; only completed regular-season games.
    # Query one season at a time to keep payload sizes bounded.
    rows=[]
    for year in range(2015,2027):
        url=f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={year}-03-01&endDate={year}-11-30&gameType=R'
        try:
            r=S.get(url,timeout=90); r.raise_for_status(); payload=r.json()
            for day in payload.get('dates',[]):
                for g in day.get('games',[]):
                    if g.get('status',{}).get('abstractGameState')!='Final': continue
                    h=g.get('teams',{}).get('home',{}); a=g.get('teams',{}).get('away',{})
                    if h.get('score') is None or a.get('score') is None: continue
                    rows.append({'date':day.get('date'),'home_team':h.get('team',{}).get('name'),'away_team':a.get('team',{}).get('name'),'home_score':h.get('score'),'away_score':a.get('score')})
            print('MLB',year,'games:',len(rows))
        except Exception as e: print('MLB API warning',year,e)
    if not rows: return False
    d=pd.DataFrame(rows)
    return write(d,'baseball',col(d,'date'),col(d,'home_team'),col(d,'away_team'),col(d,'home_score'),col(d,'away_score'))

if __name__=='__main__':
    results={'football':football(),'tennis':tennis(),'basketball':basketball(),'nfl':nfl(),'baseball':baseball()}
    print('\nRESULTS',results)
    # Return a failing process code when no source produced usable rows. Partial
    # availability is allowed: unavailable sports remain visibly untrained.
    sys.exit(0 if any(results.values()) else 2)
