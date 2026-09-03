let ws;
const API_BASE = "http://localhost:8044/api";
const WS_URL = "ws://localhost:8044/ws";

// UI Elements
const els = {
    total: document.getElementById('stat-total'),
    done: document.getElementById('stat-done'),
    pending: document.getElementById('stat-pending'),
    
    progFill: document.getElementById('progress-fill'),
    progPct: document.getElementById('progress-pct'),
    
    logBody: document.getElementById('log-body'),
    
    btnStart: document.getElementById('btn-start'),
    btnPause: document.getElementById('btn-pause'),
    btnResume: document.getElementById('btn-resume'),
    
    modal: document.getElementById('start-modal')
};

// All 30 categories
const categories = [
    "extrusion_based", "droplet_based_inkjet", "light_based_vat", "laser_assisted", "in_situ_bioprinting", "4d_bioprinting",
    "natural_polymers", "synthetic_polymers", "dmatrix_bioinks", "nanomaterial_composite", "hydrogel_rheology", "sacrificial_support",
    "stem_cells", "primary_cells", "multicellular_co_culture", "spheroids_organoids", "cell_viability_proliferation", "vascularization",
    "tissue_engineering", "disease_modeling", "drug_screening_discovery", "anatomical_models", "conductive_electronics",
    "process_calibration", "bayesian_optimization", "computer_vision_monitoring", "deep_learning_models", "large_language_models", "finite_element_analysis", "open_source_software"
];

let currentStatus = { total: 0, processed: 0 };
let isRunning = false;

function initWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => console.log("WebSocket connected.");
    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "status_update") {
            updateStats(msg.data);
        } else if (msg.type === "log") {
            addLog(msg.message, msg.level);
        } else if (msg.type === "progress") {
            const tags = msg.data.tags.length > 0 ? msg.data.tags.join(", ") : "None";
            addLog(`Tagged ${msg.data.filename}: [${tags}]`, 'info');
        }
    };
    ws.onclose = () => {
        console.log("WebSocket disconnected. Reconnecting in 3s...");
        setTimeout(initWebSocket, 3000);
    };
}

function updateStats(data) {
    currentStatus = data;
    els.total.textContent = data.total;
    els.done.textContent = data.processed;
    els.pending.textContent = data.remaining;
    
    // Update all 30 category counts
    categories.forEach(cat => {
        const el = document.getElementById(`stat-${cat}`);
        if (el) {
            el.textContent = data.counts[cat] || 0;
            // Highlight row briefly if count increased
            if (parseInt(el.textContent) > 0) {
                el.parentElement.style.color = "var(--text-main)";
            }
        }
    });
    
    els.progFill.style.width = `${data.pct}%`;
    els.progPct.textContent = `${data.pct}%`;
    
    if (data.processed > 0 && data.remaining > 0 && isRunning) {
        updateButtons("running");
    } else if (data.processed > 0 && data.remaining > 0 && !isRunning) {
        updateButtons("paused");
    } else if (data.processed === data.total && data.total > 0) {
        updateButtons("done");
        isRunning = false;
    } else {
        updateButtons("idle");
    }
}

function addLog(message, level = "info") {
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-msg ${level}">${message}</span>`;
    els.logBody.appendChild(div);
    els.logBody.scrollTop = els.logBody.scrollHeight;
}

function updateButtons(state) {
    if (state === "idle") {
        els.btnStart.style.display = 'inline-flex';
        els.btnPause.style.display = 'none';
        els.btnResume.style.display = 'none';
    } else if (state === "running") {
        els.btnStart.style.display = 'none';
        els.btnResume.style.display = 'none';
        els.btnPause.style.display = 'inline-flex';
        els.btnPause.disabled = false;
    } else if (state === "paused") {
        els.btnStart.style.display = 'none';
        els.btnPause.style.display = 'none';
        els.btnResume.style.display = 'inline-flex';
        els.btnResume.disabled = false;
    } else if (state === "done") {
        els.btnStart.style.display = 'none';
        els.btnPause.style.display = 'none';
        els.btnResume.style.display = 'none';
    }
}

function openStartModal() {
    els.modal.style.display = 'flex';
    document.getElementById('modal-concurrency').innerText = document.getElementById('concurrency-input').value;
}

function closeStartModal() {
    els.modal.style.display = 'none';
}

async function startProcessing() {
    closeStartModal();
    const concurrency = parseInt(document.getElementById('concurrency-input').value) || 20;
    const model = document.getElementById('model-select').value;
    const apiKey = document.getElementById('api-key-input').value;
    
    try {
        const res = await fetch(`${API_BASE}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: apiKey, model: model, delay_ms: 0, concurrency: concurrency })
        });
        const data = await res.json();
        if (data.status === "started") {
            isRunning = true;
            updateButtons("running");
            addLog(`⚡ Parallel Paper Tagging started (concurrency=${concurrency})...`, "success");
        } else {
            addLog(data.message || "Failed to start", "danger");
        }
    } catch (e) {
        addLog("Network error: " + e.message, "danger");
    }
}

async function pauseProcessing() {
    try {
        const res = await fetch(`${API_BASE}/pause`, { method: 'POST' });
        const data = await res.json();
        if (data.status === "paused") {
            isRunning = false;
            updateButtons("paused");
            addLog("Requested pause. Will pause after current papers finish.", "warning");
        }
    } catch (e) {
        addLog("Network error: " + e.message, "danger");
    }
}

async function resetAllData() {
    if (!confirm("Are you sure you want to reset all classification progress? This cannot be undone.")) return;
    try {
        const res = await fetch(`${API_BASE}/reset`, { method: 'POST' });
        const data = await res.json();
        if (data.status === "reset") {
            isRunning = false;
            addLog("All progress reset.", "danger");
            els.logBody.innerHTML = '';
        }
    } catch (e) {
        addLog("Network error: " + e.message, "danger");
    }
}

async function exportData() {
    try {
        addLog("Exporting to master JSON and 30 category MD files...", "info");
        const res = await fetch(`${API_BASE}/export`);
        const data = await res.json();
        if (data.status === "success") {
            addLog(data.message, "success");
            alert("Export complete! Files saved to the output/ folder.");
        } else {
            addLog(data.message || "Export failed", "danger");
        }
    } catch (e) {
        addLog("Network error: " + e.message, "danger");
    }
}

// Init
initWebSocket();
