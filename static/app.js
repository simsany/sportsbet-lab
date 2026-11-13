const $=s=>document.querySelector(s);
const notice=$("#notice");
function config(){return {
 sports:[...document.querySelectorAll('[name=sports]:checked')].map(x=>x.value),
 markets:[...document.querySelectorAll('[name=markets]:checked')].map(x=>x.value),
 sport_keys:[...document.querySelectorAll('[name=sport_keys]:checked')].map(x=>x.value),
 min_odds:Number($("#minOdds").value),max_picks:Number($("#maxPicks").value),
 sort_by:$("#sortBy").value,regions:[$("#region").value],include_inplay:$("#inplay").checked
};}
async function save(){const r=await fetch("/api/config",{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(config())});const d=await r.json();if(!r.ok)throw Error(d.error||"Mentési hiba");notice.textContent="Beállítások elmentve.";notice.className="notice ok";return d;}
$("#save").onclick=async()=>{try{await save()}catch(e){notice.textContent=e.message;notice.className="notice error"}};
async function checkApi(){try{const r=await fetch("/health");const d=await r.json();$("#apiStatus").textContent=d.odds_api_key_configured?"● ODDS_API_KEY beállítva":"! Hiányzó ODDS_API_KEY";$("#apiStatus").style.color=d.odds_api_key_configured?"#58d6ad":"#ff9b9b"}catch(e){$("#apiStatus").textContent="API státusz nem elérhető"}}
function pct(v){return v==null?"—":`${Number(v).toFixed(2)}%`}
function render(data){
 const out=$("#results");$("#summary").textContent=`${data.total_found} odds-szelekció · ${Object.entries(data.events_by_sport||{}).map(([k,v])=>`${k}: ${v} esemény`).join(" · ")}`;
 const qs=(data.quota||[]).filter(Boolean);const last=qs[qs.length-1];if(last)$("#quota").textContent=`Kvóta: ${last["x-requests-remaining"]??"?"} maradt · ${last["x-requests-used"]??"?"} felhasználva`;
 $("#modelNotice").textContent=data.notice||"";$("#modelNotice").classList.remove("hidden");
 if(data.errors?.length){const e=document.createElement("div");e.className="errors";e.innerHTML="<b>Adatforrás-jelzések</b><br>"+data.errors.map(x=>escapeHtml(x)).join("<br>");out.innerHTML="";out.appendChild(e)}
 if(!data.picks?.length){if(!data.errors?.length)out.innerHTML='<div class="empty">A megadott feltételekkel most nem érkezett elérhető odds.</div>';return}
 const wrap=document.createElement("div");wrap.className="table-wrap";
 const table=document.createElement("table");
 table.innerHTML="<thead><tr><th>Mérkőzés</th><th>Piac / választás</th><th>Odds</th><th>Implied prob.</th><th>Modell prob.</th><th>Modell-egyezés</th><th>Modell EV</th><th>Bizonytalanság</th><th>Piaci prob.</th><th>Bookmaker</th><th>Adatforrás</th><th>Kezdés</th></tr></thead>";
 const tbody=document.createElement("tbody");
 data.picks.forEach(p=>{const tr=document.createElement("tr");const ev=Number(p.market_consensus_ev);const date=p.commence_time?new Date(p.commence_time).toLocaleString():"—";
 const mev=p.model_ev==null?null:Number(p.model_ev);
 tr.innerHTML=`<td><strong>${escapeHtml(p.home_team)} – ${escapeHtml(p.away_team)}</strong><small>${escapeHtml(p.sport_title||p.sport_key||"")}</small></td><td><span class="pill">${escapeHtml(p.market)}</span><br>${escapeHtml(p.selection)}</td><td><strong>${Number(p.best_odds).toFixed(2)}</strong></td><td>${pct(p.implied_probability)}</td><td>${pct(p.model_probability)}</td><td>${escapeHtml(p.model_agreement||"—")}</td><td class="${mev==null?"":mev>=0?"positive":"negative"}">${mev==null?"—":mev.toFixed(2)+"%"}</td><td>${p.uncertainty==null?"—":Number(p.uncertainty).toFixed(2)+"%"}</td><td>${pct(p.market_probability)}</td><td>${escapeHtml(p.best_bookmaker)}<small>${p.bookmaker_count} bookmaker</small></td><td>${escapeHtml(p.data_source||"—")}</td><td>${escapeHtml(date)}</td>`;
 tbody.appendChild(tr)});
 table.appendChild(tbody);wrap.appendChild(table);out.appendChild(wrap);
 const foot=document.createElement("p");foot.className="result-foot";foot.textContent="A modell EV kizárólag az ötmodell-ensemble valószínűségéből számolódik. Ahol nincs betanított sportági modell vagy az adott piacra nincs modell, ott a modellmezők üresek. A piaci valószínűség odds-alapú konszenzus.";out.appendChild(foot);
}
function escapeHtml(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
$("#scan").onclick=async()=>{const b=$("#scan");b.disabled=true;b.textContent="Oddsok lekérése…";$("#results").innerHTML='<div class="empty">Kapcsolódás az Odds API-hoz…</div>';try{await save();const r=await fetch("/api/scan",{method:"POST"});const d=await r.json();if(!r.ok)throw Error(d.error||"Lekérdezési hiba");render(d);notice.textContent="Oddslekérés befejezve.";notice.className="notice ok"}catch(e){$("#results").innerHTML=`<div class="errors">${escapeHtml(e.message)}</div>`;notice.textContent=e.message;notice.className="notice error"}finally{b.disabled=false;b.textContent="Valós oddsok lekérése és elemzés →"}};
checkApi();

async function refreshModels(){try{const r=await fetch('/api/models');const d=await r.json();const states=d.sports||[];states.forEach(x=>{});const trained=states.filter(x=>x.status==='trained');const msg=d.training?.status==='running'?'Adatok letöltése és modellek tanítása folyamatban…':`${trained.length}/5 sportágban elérhető betanított modell`;document.querySelectorAll('.model-state').forEach(el=>el.textContent=trained.length?`Betanítva: ${trained.map(x=>x.sport).join(', ')}`:'Történeti adatok / betanítás szükséges');$('#trainingStatus').textContent=msg;$('#trainingStatus').className='notice '+(d.training?.status==='failed'?'error':'');}catch(e){$('#trainingStatus').textContent='Modellstátusz nem elérhető'}}
const trainButton=$('#train');if(trainButton)trainButton.onclick=async()=>{trainButton.disabled=true;$('#trainingStatus').textContent='Betanítás indítása…';try{const r=await fetch('/api/train',{method:'POST'});const d=await r.json();if(!r.ok)throw Error(d.error||'Betanítás nem indult');$('#trainingStatus').textContent='Betanítás elindult. Frissülő státusz…';}catch(e){$('#trainingStatus').textContent=e.message;}finally{trainButton.disabled=false;}};
refreshModels();setInterval(refreshModels,15000);

let savedLeagueKeys=[];
async function loadLeagues(){const box=$('#leagueList');try{const [r,c]=await Promise.all([fetch('/api/sports'),fetch('/api/config')]);const d=await r.json();const cfg=await c.json();savedLeagueKeys=cfg.sport_keys||[];const selectedSports=cfg.sports||[];const leagues=(d.available||[]).filter(x=>x.active&&selectedSports.includes(x.sport));if(!leagues.length){box.textContent='Nincs elérhető aktív liga ehhez a sportágválasztáshoz.';return;}box.innerHTML='';leagues.forEach(l=>{const label=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.name='sport_keys';input.value=l.key;input.checked=savedLeagueKeys.length===0||savedLeagueKeys.includes(l.key);const span=document.createElement('span');span.textContent=`${l.title} (${l.sport})`;label.append(input,span);box.appendChild(label);});}catch(e){box.textContent='A ligalista nem tölthető be: '+e.message;}}
document.querySelectorAll('[name=sports]').forEach(el=>el.addEventListener('change',()=>{savedLeagueKeys=[];loadLeagues();}));loadLeagues();
