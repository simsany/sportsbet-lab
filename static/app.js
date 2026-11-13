
const $ = (selector) => document.querySelector(selector);
const notice = $("#notice");
let availableLeagues = [];
let savedLeagueKeys = [];
let leagueSelectionInitialized = false;

function selectedValues(name) {
  return [...document.querySelectorAll(`[name="${name}"]:checked`)].map((el) => el.value);
}

function config() {
  return {
    sports: selectedValues("sports"),
    markets: selectedValues("markets"),
    sport_keys: selectedValues("sport_keys"),
    min_odds: Number($("#minOdds").value),
    max_picks: Number($("#maxPicks").value),
    sort_by: $("#sortBy").value,
    regions: [$("#region").value],
    include_inplay: $("#inplay").checked
  };
}

async function save() {
  const response = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config())
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Mentési hiba");
  notice.textContent = "Beállítások elmentve.";
  notice.className = "notice ok";
  return data;
}

$("#save").onclick = async () => {
  try {
    await save();
  } catch (error) {
    notice.textContent = error.message;
    notice.className = "notice error";
  }
};

async function checkApi() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    $("#apiStatus").textContent = data.odds_api_key_configured
      ? "● ODDS_API_KEY beállítva"
      : "! Hiányzó ODDS_API_KEY";
    $("#apiStatus").style.color = data.odds_api_key_configured ? "#58d6ad" : "#ff9b9b";
  } catch (_) {
    $("#apiStatus").textContent = "API státusz nem elérhető";
  }
}

function pct(value) {
  return value == null ? "—" : `${Number(value).toFixed(2)}%`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[char]));
}

function render(data) {
  const out = $("#results");
  out.innerHTML = "";

  $("#summary").textContent =
    `${data.total_found} odds-szelekció · ` +
    Object.entries(data.events_by_sport || {})
      .map(([key, count]) => `${key}: ${count} esemény`)
      .join(" · ");

  const quotas = (data.quota || []).filter(Boolean);
  const lastQuota = quotas[quotas.length - 1];

  if (lastQuota) {
    $("#quota").textContent =
      `Kvóta: ${lastQuota["x-requests-remaining"] ?? "?"} maradt · ` +
      `${lastQuota["x-requests-used"] ?? "?"} felhasználva`;
  }

  $("#modelNotice").textContent = data.notice || "";
  $("#modelNotice").classList.remove("hidden");

  if (data.errors?.length) {
    const errorBox = document.createElement("div");
    errorBox.className = "errors";
    errorBox.innerHTML =
      "<b>Adatforrás-jelzések</b><br>" +
      data.errors.map(escapeHtml).join("<br>");
    out.appendChild(errorBox);
  }

  if (!data.picks?.length) {
    if (!data.errors?.length) {
      out.insertAdjacentHTML(
        "beforeend",
        '<div class="empty">A megadott feltételekkel most nem érkezett elérhető odds.</div>'
      );
    }
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";

  const table = document.createElement("table");
  table.innerHTML =
    "<thead><tr>" +
    "<th>Mérkőzés</th><th>Piac / választás</th><th>Odds</th>" +
    "<th>Implied prob.</th><th>Modell prob.</th><th>Modell-egyezés</th>" +
    "<th>Modell EV</th><th>Bizonytalanság</th><th>Piaci prob.</th>" +
    "<th>Bookmaker</th><th>Adatforrás</th><th>Kezdés</th>" +
    "</tr></thead>";

  const tbody = document.createElement("tbody");

  data.picks.forEach((pick) => {
    const row = document.createElement("tr");
    const modelEv = pick.model_ev == null ? null : Number(pick.model_ev);
    const date = pick.commence_time
      ? new Date(pick.commence_time).toLocaleString()
      : "—";

    row.innerHTML =
      `<td><strong>${escapeHtml(pick.home_team)} – ${escapeHtml(pick.away_team)}</strong>` +
      `<small>${escapeHtml(pick.sport_title || pick.sport_key || "")}</small></td>` +
      `<td><span class="pill">${escapeHtml(pick.market)}</span><br>${escapeHtml(pick.selection)}</td>` +
      `<td><strong>${Number(pick.best_odds).toFixed(2)}</strong></td>` +
      `<td>${pct(pick.implied_probability)}</td>` +
      `<td>${pct(pick.model_probability)}</td>` +
      `<td>${escapeHtml(pick.model_agreement || "—")}</td>` +
      `<td class="${modelEv == null ? "" : modelEv >= 0 ? "positive" : "negative"}">` +
      `${modelEv == null ? "—" : modelEv.toFixed(2) + "%"}</td>` +
      `<td>${pick.uncertainty == null ? "—" : Number(pick.uncertainty).toFixed(2) + "%"}</td>` +
      `<td>${pct(pick.market_probability)}</td>` +
      `<td>${escapeHtml(pick.best_bookmaker)}` +
      `<small>${Number(pick.bookmaker_count || 0)} bookmaker</small></td>` +
      `<td>${escapeHtml(pick.data_source || "—")}</td>` +
      `<td>${escapeHtml(date)}</td>`;

    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  wrap.appendChild(table);
  out.appendChild(wrap);

  const foot = document.createElement("p");
  foot.className = "result-foot";
  foot.textContent =
    "A modell EV kizárólag az ötmodell-ensemble valószínűségéből számolódik. " +
    "Ahol nincs betanított sportági modell vagy az adott piacra nincs modell, " +
    "ott a modellmezők üresek. A piaci valószínűség külön odds-alapú mező.";
  out.appendChild(foot);
}

$("#scan").onclick = async () => {
  const button = $("#scan");
  button.disabled = true;
  button.textContent = "Oddsok lekérése…";
  $("#results").innerHTML =
    '<div class="empty">Kapcsolódás az Odds API-hoz…</div>';

  try {
    await save();
    const response = await fetch("/api/scan", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Lekérdezési hiba");

    render(data);
    notice.textContent = "Oddslekérés befejezve.";
    notice.className = "notice ok";
  } catch (error) {
    $("#results").innerHTML =
      `<div class="errors">${escapeHtml(error.message)}</div>`;
    notice.textContent = error.message;
    notice.className = "notice error";
  } finally {
    button.disabled = false;
    button.textContent = "Valós oddsok lekérése és elemzés →";
  }
};

async function refreshModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    const states = data.sports || [];
    const trained = states.filter((item) => item.status === "trained");

    const message = data.training?.status === "running"
      ? "Adatok letöltése és modellek tanítása folyamatban…"
      : `${trained.length}/5 sportágban elérhető betanított modell`;

    document.querySelectorAll(".model-state").forEach((element) => {
      element.textContent = trained.length
        ? `Betanítva: ${trained.map((item) => item.sport).join(", ")}`
        : "Történeti adatok / betanítás szükséges";
    });

    $("#trainingStatus").textContent = message;
    $("#trainingStatus").className =
      "notice " + (data.training?.status === "failed" ? "error" : "");
  } catch (_) {
    $("#trainingStatus").textContent = "Modellstátusz nem elérhető";
  }
}

const trainButton = $("#train");

if (trainButton) {
  trainButton.onclick = async () => {
    trainButton.disabled = true;
    $("#trainingStatus").textContent = "Betanítás indítása…";

    try {
      const response = await fetch("/api/train", { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Betanítás nem indult");
      }
      $("#trainingStatus").textContent =
        "Betanítás elindult. Frissülő státusz…";
    } catch (error) {
      $("#trainingStatus").textContent = error.message;
    } finally {
      trainButton.disabled = false;
    }
  };
}

/* LIGAVÁLASZTÓ */

function renderLeagues() {
  const box = $("#leagueList");
  box.innerHTML = "";

  const selectedSports = selectedValues("sports");
  const visible = availableLeagues.filter((league) => league.active);

  if (!visible.length) {
    box.textContent =
      "Az Odds API jelenleg nem adott vissza aktív versenysorozatot.";
    return;
  }

  const labels = {
    football: "⚽ FOOTBALL",
    tennis: "🎾 TENNIS",
    basketball: "🏀 BASKETBALL",
    baseball: "⚾ BASEBALL",
    nfl: "🏈 NFL"
  };

  const groups = ["football", "tennis", "basketball", "baseball", "nfl"];

  groups.forEach((sport) => {
    const leagues = visible.filter((league) => league.sport === sport);
    if (!leagues.length) return;

    const heading = document.createElement("div");
    heading.className = "league-group-title";
    heading.textContent = `${labels[sport] || sport.toUpperCase()} · ${leagues.length} liga`;
    box.appendChild(heading);

    const group = document.createElement("div");
    group.className = "league-group";

    leagues.forEach((league) => {
      const label = document.createElement("label");
      label.className = "league-option";

      const input = document.createElement("input");
      input.type = "checkbox";
      input.name = "sport_keys";
      input.value = league.key;

      input.checked = leagueSelectionInitialized
        ? (
            savedLeagueKeys.length === 0
              ? selectedSports.includes(sport)
              : savedLeagueKeys.includes(league.key)
          )
        : selectedSports.includes(sport);

      const span = document.createElement("span");
      span.textContent = league.title;

      input.addEventListener("change", () => {
        const chosen = selectedValues("sport_keys");

        const sameSportChosen = availableLeagues.some(
          (item) => item.sport === sport && chosen.includes(item.key)
        );

        const sportInput = document.querySelector(
          `[name="sports"][value="${sport}"]`
        );

        if (sportInput) sportInput.checked = sameSportChosen;
        savedLeagueKeys = chosen;
      });

      label.append(input, span);
      group.appendChild(label);
    });

    box.appendChild(group);
  });
}

async function loadLeagues() {
  const box = $("#leagueList");
  box.textContent = "Versenysorozatok betöltése…";

  try {
    const [sportsResponse, configResponse] = await Promise.all([
      fetch("/api/sports"),
      fetch("/api/config")
    ]);

    if (!sportsResponse.ok) {
      throw new Error("A sportlista lekérése sikertelen");
    }

    const sportsData = await sportsResponse.json();
    const configData = await configResponse.json();

    availableLeagues = sportsData.available || [];
    savedLeagueKeys = configData.sport_keys || [];
    leagueSelectionInitialized = true;

    renderLeagues();
  } catch (error) {
    box.textContent = "A ligalista nem tölthető be: " + error.message;
  }
}

document.querySelectorAll('[name="sports"]').forEach((element) => {
  element.addEventListener("change", renderLeagues);
});

checkApi();
refreshModels();
setInterval(refreshModels, 15000);
loadLeagues();
