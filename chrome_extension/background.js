// Create the right-click context menu
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "analyze-ai-text",
        title: "Check if AI Generated",
        contexts: ["selection"]
    });
});

// Listen for the context menu click
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "analyze-ai-text") {
        const selectedText = info.selectionText;
        
        // Inject the loading UI into the webpage
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: showLoadingPopup
        });

        // Start the analysis chain
        analyzeText(selectedText, tab.id);
    }
});

// Function to call FastAPI and poll the Celery task
async function analyzeText(text, tabId) {
    try {
        const res = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        
        const data = await res.json();
        const taskId = data.task_id;
        
        // Poll for result
        const result = await pollResult(taskId);
        
        // Inject the result UI into the webpage
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: showResultPopup,
            args: [result]
        });

    } catch (e) {
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            func: showErrorPopup,
            args: [e.toString()]
        });
    }
}

async function pollResult(taskId) {
    while (true) {
        const res = await fetch(`http://localhost:8000/predict/result/${taskId}`);
        const data = await res.json();
        if (data.status === "SUCCESS") return data.result;
        if (data.status === "FAILURE") throw new Error("Analysis failed on server.");
        await new Promise(r => setTimeout(r, 1000));
    }
}

// ---- INJECTED SCRIPTS (These run inside the actual webpage) ----

function showLoadingPopup() {
    let popup = document.getElementById("ai-detector-ext-popup");
    if (!popup) {
        popup = document.createElement("div");
        popup.id = "ai-detector-ext-popup";
        popup.style.cssText = "position:fixed;top:20px;right:20px;width:300px;background:#1e293b;color:#f8fafc;border-radius:12px;padding:16px;box-shadow:0 10px 25px rgba(0,0,0,0.5);z-index:999999;font-family:sans-serif;border:1px solid #334155;transition:all 0.3s;";
        document.body.appendChild(popup);
    }
    popup.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <strong style="color:#a78bfa;">AI Text Detector</strong>
            <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;color:#94a3b8;cursor:pointer;">✕</button>
        </div>
        <div style="font-size:14px;color:#94a3b8;display:flex;align-items:center;gap:8px;">
            <div style="width:12px;height:12px;border:2px solid #3b82f6;border-top-color:transparent;border-radius:50%;animation:spin 1s linear infinite;"></div>
            Analyzing (Celery MapReduce)...
        </div>
        <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
    `;
}

function showResultPopup(result) {
    const popup = document.getElementById("ai-detector-ext-popup");
    if (!popup) return;
    
    let color = "#fbbf24"; // yellow (mixed)
    if (result.label.includes("AI")) color = "#f87171"; // red
    if (result.label.includes("HUMAN")) color = "#34d399"; // green
    
    popup.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <strong style="color:#a78bfa;">AI Text Detector</strong>
            <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;color:#94a3b8;cursor:pointer;">✕</button>
        </div>
        <div style="background:rgba(0,0,0,0.2);border:1px solid ${color};border-radius:8px;padding:12px;text-align:center;">
            <div style="font-size:16px;font-weight:bold;color:${color};margin-bottom:4px;">${result.label}</div>
            <div style="font-size:13px;color:#cbd5e1;">Ensemble Confidence: <b>${(result.confidence * 100).toFixed(1)}%</b></div>
        </div>
        <div style="margin-top:12px;font-size:12px;color:#94a3b8;display:flex;justify-content:space-between;">
            <span>DeBERTa: ${(result.metrics.deberta_conf * 100).toFixed(1)}%</span>
            <span>RF: ${(result.metrics.rf_conf * 100).toFixed(1)}%</span>
        </div>
    `;
}

function showErrorPopup(err) {
    const popup = document.getElementById("ai-detector-ext-popup");
    if (popup) {
        popup.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <strong style="color:#ef4444;">Error</strong>
                <button onclick="this.parentElement.parentElement.remove()" style="background:none;border:none;color:#94a3b8;cursor:pointer;">✕</button>
            </div>
            <div style="font-size:13px;color:#cbd5e1;margin-top:8px;">Make sure your FastAPI Docker server is running on localhost:8000.</div>
        `;
    }
}
