let catalogue = [];
let savedLeagues = [];

const $ = (id) => document.getElementById(id);

async function api(url, opts = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...opts
    });

    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
        ? await response.json()
        : { error: await response.text() };

    if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
    }
    return data;
}

function show(id, data) {
    $(id).textContent = typeof data === "string"
        ? data
        : JSON.stringify(data, null, 2);
}

function setMessage(id, message) {
    $(id).textContent = message;
}

async function init() {
    try {
        const [catalogueData, leaguesData] = await Promise.all([
            api("/api/catalogue"),
            api("/api/leagues")
        ]);

        catalogue = catalogueData;
        savedLeagues = leaguesData;

        const countries = [...new Set(catalogue.map(x => x.country))].sort();
        $("country").innerHTML = countries
            .map(country => `<option value="${escapeHtml(country)}">${escapeHtml(country)}</option>`)
            .join("");

        fillLeagues();
        populateSavedLeagues();
    } catch (error) {
        setMessage("createResult", `Inicializálási hiba: ${error.message}`);
        setMessage("trainResult", `A ligák betöltése sikertelen: ${error.message}`);
    }
}

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[char]);
}

function fillLeagues() {
    const country = $("country").value;
    const items = catalogue.filter(item => item.country === country);

    $("league").innerHTML = items.map(item =>
        `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} (${item.min_year}–)</option>`
    ).join("");

    if (items[0]) {
        $("year").value = Math.max(Number(items[0].min_year), 2018);
        $("leagueId").value = `${country}-${items[0].name}`;
    } else {
        $("leagueId").value = "";
    }
}

function populateSavedLeagues(selectedId = "") {
    const select = $("trainLeague");
    const previous = selectedId || select.value;

    if (!savedLeagues.length) {
        select.innerHTML = '<option value="">Nincs még letöltött liga</option>';
        $("predLeague").value = "";
        return;
    }

    select.innerHTML =
        '<option value="">Válassz egy mentett ligát</option>' +
        savedLeagues.map(league => {
            const label = `${league.country} · ${league.name} (${league.id})`;
            return `<option value="${escapeHtml(league.id)}">${escapeHtml(label)}</option>`;
        }).join("");

    if (previous && savedLeagues.some(league => league.id === previous)) {
        select.value = previous;
    } else {
        select.selectedIndex = 1;
    }

    syncSelectedLeague();
}

function syncSelectedLeague() {
    const leagueId = $("trainLeague").value;
    $("predLeague").value = leagueId;
}

function selectLeague(id) {
    $("trainLeague").value = id;
    syncSelectedLeague();
    $("toast").textContent = `Kiválasztott liga: ${id}`;
}

async function createLeague() {
    const statsColumns = $("stats").value
        .split(",")
        .map(value => value.trim())
        .filter(Boolean);

    if (!statsColumns.length) {
        setMessage("createResult", "Adj meg legalább egy statisztikai oszlopot.");
        return;
    }

    try {
        setMessage("createResult", "Liga letöltése folyamatban…");
        const result = await api("/api/leagues", {
            method: "POST",
            body: JSON.stringify({
                country: $("country").value,
                name: $("league").value,
                start_year: Number($("year").value),
                league_id: $("leagueId").value.trim(),
                stats_columns: statsColumns,
                match_history_window: Number($("window").value)
            })
        });

        savedLeagues = await api("/api/leagues");
        populateSavedLeagues(result.id);
        show("createResult", result);
        $("toast").textContent = `Liga sikeresen letöltve: ${result.id}`;
    } catch (error) {
        setMessage("createResult", `Liga letöltése sikertelen: ${error.message}`);
    }
}

async function trainModel() {
    const leagueId = $("trainLeague").value.trim();
    const modelId = $("modelId").value.trim();

    if (!leagueId) {
        setMessage("trainResult", "Válassz ki egy már letöltött ligát a tanítás előtt.");
        return;
    }

    if (!modelId) {
        setMessage("trainResult", "A Model ID megadása kötelező.");
        return;
    }

    const testSize = Number($("testSize").value);
    if (!Number.isFinite(testSize) || testSize <= 0 || testSize >= 1) {
        setMessage("trainResult", "A Test size értéke 0 és 1 közé essen (például 0.2).");
        return;
    }

    try {
        setMessage("trainResult", "Modell tanítása folyamatban…");
        const result = await api(
            `/api/leagues/${encodeURIComponent(leagueId)}/train`,
            {
                method: "POST",
                body: JSON.stringify({
                    model_id: modelId,
                    target: $("target").value,
                    classifier: $("classifier").value,
                    test_size: testSize,
                    seed: Number($("seed").value)
                })
            }
        );
        show("trainResult", result);
        $("toast").textContent = `A modell elkészült: ${modelId}`;
    } catch (error) {
        setMessage("trainResult", `Modell tanítása sikertelen: ${error.message}`);
    }
}

async function predict() {
    const leagueId = $("predLeague").value.trim();
    const modelId = $("predModel").value.trim();

    if (!leagueId || !modelId) {
        setMessage("predictionResult", "Válassz ligát és adj meg Model ID-t.");
        return;
    }

    try {
        const result = await api(
            `/api/leagues/${encodeURIComponent(leagueId)}/predict`,
            {
                method: "POST",
                body: JSON.stringify({
                    model_id: modelId,
                    home: $("home").value.trim(),
                    away: $("away").value.trim(),
                    odd_1: Number($("odd1").value),
                    odd_x: Number($("oddx").value),
                    odd_2: Number($("odd2").value)
                })
            }
        );
        show("predictionResult", result);
    } catch (error) {
        setMessage("predictionResult", `Predikció sikertelen: ${error.message}`);
    }
}

$("trainLeague").addEventListener("change", syncSelectedLeague);
init();
