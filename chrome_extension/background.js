// =============================================================================
// AI Text Detector — Chrome Extension Background Service Worker (Manifest V3)
// =============================================================================

const API_BASE = "http://localhost:8000";

// Register right-click context menu
chrome.runtime.onInstalled.addListener(() => {
    setupContextMenu();
});

chrome.runtime.onStartup.addListener(() => {
    setupContextMenu();
});

function setupContextMenu() {
    chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({
            id: "analyze-ai-text-selection",
            title: "⚡ Detect AI in Selection",
            contexts: ["selection"]
        });
    });
}

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "analyze-ai-text-selection") {
        const selectedText = (info.selectionText || "").trim();
        if (!selectedText) return;

        if (tab && tab.id) {
            // Inject loading widget
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: injectLoadingWidget
            }).catch(() => {
                fallbackNotify("AI Detector", "Analyzing selected text...");
            });

            // Run backend inference
            processTextInference(selectedText, tab.id);
        }
    }
});

// Primary analysis flow
async function processTextInference(text, tabId) {
    try {
        const resp = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text, fast: true })
        });

        if (!resp.ok) {
            throw new Error(`API returned HTTP ${resp.status}`);
        }

        const data = await resp.json();
        const taskId = data.task_id;

        // Poll for completion
        const result = await pollForCompletion(taskId);

        // Inject result widget into page
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: injectResultWidget,
            args: [result]
        }).catch(() => {
            // Fallback to desktop notification if page blocks DOM injection (e.g. PDF viewer)
            const label = result.label || result.prediction;
            const conf = ((result.confidence || 0) * 100).toFixed(1);
            fallbackNotify(`AI Verdict: ${label}`, `Confidence: ${conf}% | DeBERTa: ${((result.metrics?.deberta_conf || 0) * 100).toFixed(0)}% | RF: ${((result.metrics?.rf_conf || 0) * 100).toFixed(0)}%`);
        });

    } catch (err) {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: injectErrorWidget,
            args: [err.message || "Failed to reach server"]
        }).catch(() => {
            fallbackNotify("AI Detector Offline", "Ensure FastAPI server is running on localhost:8000.");
        });
    }
}

// Fast polling loop
async function pollForCompletion(taskId) {
    let attempts = 0;
    while (attempts < 50) {
        const res = await fetch(`${API_BASE}/predict/result/${taskId}`);
        if (res.ok) {
            const data = await res.json();
            if (data.status === "SUCCESS") return data.result;
            if (data.status === "FAILURE") throw new Error("Server inference failed.");
        }
        await new Promise(r => setTimeout(r, 120));
        attempts++;
    }
    throw new Error("Prediction timed out.");
}

// System desktop notification fallback
function fallbackNotify(title, message) {
    if (chrome.notifications) {
        chrome.notifications.create({
            type: "basic",
            iconUrl: chrome.runtime.getURL("icons/icon48.png"),
            title: title,
            message: message,
            priority: 2
        });
    }
}

// =============================================================================
// INJECTED DOM WIDGETS (Run inside the web page with complete CSS isolation)
// =============================================================================

function injectLoadingWidget() {
    let host = document.getElementById("ai-detector-extension-host");
    if (!host) {
        host = document.createElement("div");
        host.id = "ai-detector-extension-host";
        host.style.cssText = "all:initial;position:fixed;top:24px;right:24px;z-index:2147483647;pointer-events:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;";
        document.documentElement.appendChild(host);
    }

    host.innerHTML = `
        <div style="width:310px;background:rgba(15,23,42,0.96);backdrop-filter:blur(16px);border:1px solid rgba(99,102,241,0.4);border-radius:12px;padding:14px 16px;box-shadow:0 12px 32px rgba(0,0,0,0.45);color:#f8fafc;box-sizing:border-box;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:14px;">⚡</span>
                    <strong style="font-size:13px;font-weight:700;color:#e2e8f0;letter-spacing:-0.2px;">AI Deepfake Detector</strong>
                </div>
                <button id="ai-ext-close" style="background:none;border:none;color:#94a3b8;font-size:16px;cursor:pointer;line-height:1;padding:0;">&times;</button>
            </div>
            <div style="display:flex;align-items:center;gap:10px;font-size:12.5px;color:#94a3b8;">
                <div style="width:14px;height:14px;border:2px solid rgba(99,102,241,0.3);border-top-color:#818cf8;border-radius:50%;animation:aiExtSpin 0.8s linear infinite;"></div>
                <span>Analyzing linguistic patterns...</span>
            </div>
        </div>
        <style>@keyframes aiExtSpin { to { transform: rotate(360deg); } }</style>
    `;

    document.getElementById("ai-ext-close")?.addEventListener("click", () => host.remove());
}

function injectResultWidget(result) {
    let host = document.getElementById("ai-detector-extension-host");
    if (!host) {
        host = document.createElement("div");
        host.id = "ai-detector-extension-host";
        host.style.cssText = "all:initial;position:fixed;top:24px;right:24px;z-index:2147483647;pointer-events:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;";
        document.documentElement.appendChild(host);
    }

    const label = result.label || result.prediction;
    const conf = ((result.confidence || 0) * 100).toFixed(1);
    const m = result.metrics || {};

    let theme = {
        color: "#fbbf24",
        bg: "rgba(245,158,11,0.15)",
        border: "rgba(245,158,11,0.35)",
        bar: "#f59e0b",
        icon: "🔄"
    };

    if (label.includes("FULLY AI")) {
        theme = {
            color: "#f87171",
            bg: "rgba(239,68,68,0.15)",
            border: "rgba(239,68,68,0.35)",
            bar: "#ef4444",
            icon: "🤖"
        };
    } else if (label.includes("HUMAN-WRITTEN")) {
        theme = {
            color: "#34d399",
            bg: "rgba(16,185,129,0.15)",
            border: "rgba(16,185,129,0.35)",
            bar: "#10b981",
            icon: "👤"
        };
    }

    const debConf = m.deberta_conf ? (m.deberta_conf * 100).toFixed(0) : "--";
    const rfConf = m.rf_conf ? (m.rf_conf * 100).toFixed(0) : "--";
    const ppl = m.perplexity ? m.perplexity.toFixed(1) : "--";
    const burst = m.burstiness !== undefined ? m.burstiness.toFixed(2) : "--";
    const ttr = m.ttr !== undefined ? (m.ttr * 100).toFixed(0) + "%" : "--";

    host.innerHTML = `
        <div id="ai-ext-card" style="width:320px;background:rgba(15,23,42,0.96);backdrop-filter:blur(18px);border:1px solid ${theme.border};border-radius:12px;padding:14px 16px;box-shadow:0 14px 36px rgba(0,0,0,0.5);color:#f8fafc;box-sizing:border-box;animation:aiExtSlide 0.25s ease-out;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:13px;">⚡</span>
                    <strong style="font-size:12.5px;color:#94a3b8;font-weight:600;">AI Text Detector</strong>
                </div>
                <button id="ai-ext-close" style="background:none;border:none;color:#64748b;font-size:16px;cursor:pointer;line-height:1;padding:0;">&times;</button>
            </div>

            <div style="background:${theme.bg};border:1px solid ${theme.border};border-radius:8px;padding:10px 12px;margin-bottom:10px;text-align:center;">
                <div style="font-size:13.5px;font-weight:700;color:${theme.color};margin-bottom:3px;display:flex;align-items:center;justify-content:center;gap:6px;">
                    <span>${theme.icon}</span>
                    <span>${label}</span>
                </div>
                <div style="font-size:11.5px;color:#cbd5e1;">Confidence: <b>${conf}%</b></div>
            </div>

            <div style="height:5px;background:#0f172a;border-radius:3px;overflow:hidden;margin-bottom:12px;">
                <div style="height:100%;width:${conf}%;background:${theme.bar};border-radius:3px;"></div>
            </div>

            <div style="display:flex;justify-content:space-between;background:#0f172a;padding:6px 10px;border-radius:6px;font-size:10.5px;color:#cbd5e1;margin-bottom:8px;">
                <span>DeBERTa: <b>${debConf}%</b></span>
                <span style="color:#64748b;">|</span>
                <span>Random Forest: <b>${rfConf}%</b></span>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-size:10px;text-align:center;">
                <div style="background:#0f172a;padding:5px;border-radius:5px;">
                    <div style="color:#64748b;font-size:9px;">PPL</div>
                    <div style="font-weight:600;color:#e2e8f0;">${ppl}</div>
                </div>
                <div style="background:#0f172a;padding:5px;border-radius:5px;">
                    <div style="color:#64748b;font-size:9px;">BURST</div>
                    <div style="font-weight:600;color:#e2e8f0;">${burst}</div>
                </div>
                <div style="background:#0f172a;padding:5px;border-radius:5px;">
                    <div style="color:#64748b;font-size:9px;">TTR</div>
                    <div style="font-weight:600;color:#e2e8f0;">${ttr}</div>
                </div>
            </div>
        </div>
        <style>
            @keyframes aiExtSlide {
                from { opacity: 0; transform: translateY(-8px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    `;

    document.getElementById("ai-ext-close")?.addEventListener("click", () => host.remove());

    // Auto-dismiss after 15 seconds
    setTimeout(() => {
        if (host && host.parentNode) host.remove();
    }, 15000);
}

function injectErrorWidget(errMsg) {
    let host = document.getElementById("ai-detector-extension-host");
    if (!host) {
        host = document.createElement("div");
        host.id = "ai-detector-extension-host";
        host.style.cssText = "all:initial;position:fixed;top:24px;right:24px;z-index:2147483647;pointer-events:auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;";
        document.documentElement.appendChild(host);
    }

    host.innerHTML = `
        <div style="width:310px;background:rgba(15,23,42,0.96);backdrop-filter:blur(16px);border:1px solid rgba(239,68,68,0.4);border-radius:12px;padding:14px 16px;box-shadow:0 12px 32px rgba(0,0,0,0.45);color:#f8fafc;box-sizing:border-box;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:6px;">
                    <span style="font-size:14px;color:#ef4444;">⚠️</span>
                    <strong style="font-size:13px;font-weight:700;color:#f87171;">Connection Failed</strong>
                </div>
                <button id="ai-ext-close" style="background:none;border:none;color:#94a3b8;font-size:16px;cursor:pointer;line-height:1;padding:0;">&times;</button>
            </div>
            <div style="font-size:11.5px;color:#cbd5e1;line-height:1.4;">
                Could not reach API server. Ensure <code>uvicorn main:app</code> is running on localhost:8000.
            </div>
        </div>
    `;

    document.getElementById("ai-ext-close")?.addEventListener("click", () => host.remove());
}