const API_BASE = "http://localhost:8000";

const PRESET_SAMPLES = {
  ai: "In contemporary discourse, artificial intelligence represents a transformative paradigm across digital ecosystems. By leveraging deep contextual architectures and iterative gradient updates, modern neural models synthesize information with unprecedented efficiency and mathematical precision.",
  human: "Hey guys! I was thinking about grabbing some coffee after work today around five, let me know if anyone wants to join or if you're all too swamped with project deadlines. Either way catch you tomorrow!",
  paraphrased: "Artificial intelligence is fundamentally shaking up enterprise software, but let's be real: without human judgment guiding the workflow, purely automated pipelines end up creating as many headaches as they solve."
};

document.addEventListener("DOMContentLoaded", () => {
  const textInput = document.getElementById("text-input");
  const charCount = document.getElementById("char-count");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnGrab = document.getElementById("btn-grab");
  const btnLabel = document.getElementById("btn-label");
  const spinner = document.getElementById("spinner");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const errorBox = document.getElementById("error-box");
  const resultBox = document.getElementById("result-box");
  const latencyTag = document.getElementById("latency-tag");

  // Ping backend status
  checkBackendHealth();

  // Presets
  document.getElementById("preset-ai").addEventListener("click", () => {
    textInput.value = PRESET_SAMPLES.ai;
    updateCharCount();
  });
  document.getElementById("preset-human").addEventListener("click", () => {
    textInput.value = PRESET_SAMPLES.human;
    updateCharCount();
  });
  document.getElementById("preset-paraphrase").addEventListener("click", () => {
    textInput.value = PRESET_SAMPLES.paraphrased;
    updateCharCount();
  });

  // Character counter
  textInput.addEventListener("input", updateCharCount);

  function updateCharCount() {
    charCount.textContent = `${textInput.value.length} chars`;
  }

  // Grab text from current active webpage
  btnGrab.addEventListener("click", async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab || !tab.id) return;

      const injection = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection().toString()
      });

      if (injection && injection[0] && injection[0].result) {
        const selected = injection[0].result.trim();
        if (selected) {
          textInput.value = selected;
          updateCharCount();
          analyzeText(selected);
        } else {
          showError("No text is currently highlighted on this webpage. Select some text first!");
        }
      } else {
        showError("Could not grab text. Ensure the active webpage allows script injection.");
      }
    } catch (err) {
      showError("Cannot access this tab (restricted Chrome/PDF page).");
    }
  });

  // Analyze button click
  btnAnalyze.addEventListener("click", () => {
    const text = textInput.value.trim();
    if (!text) {
      showError("Please enter or select some text to analyze.");
      return;
    }
    analyzeText(text);
  });

  async function checkBackendHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: "GET", cache: "no-store" });
      if (res.ok) {
        statusDot.classList.add("online");
        statusText.textContent = "API: Online";
        statusText.style.color = "#34d399";
      } else {
        throw new Error();
      }
    } catch {
      statusDot.classList.remove("online");
      statusText.textContent = "API: Offline";
      statusText.style.color = "#f87171";
    }
  }

  async function analyzeText(text) {
    hideError();
    setLoading(true);
    const startTime = performance.now();

    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, fast: true })
      });

      if (!resp.ok) {
        throw new Error(`Server returned status ${resp.status}`);
      }

      const initData = await resp.json();
      const taskId = initData.task_id;

      // Poll task
      const result = await pollResult(taskId);
      const latency = Math.round(performance.now() - startTime);

      renderResult(result, latency);
    } catch (err) {
      showError(`Analysis failed: ${err.message}. Make sure FastAPI is running on localhost:8000.`);
    } finally {
      setLoading(false);
    }
  }

  async function pollResult(taskId) {
    let attempts = 0;
    while (attempts < 60) {
      const res = await fetch(`${API_BASE}/predict/result/${taskId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === "SUCCESS") return data.result;
        if (data.status === "FAILURE") throw new Error("Backend model inference failed.");
      }
      await new Promise(r => setTimeout(r, 120));
      attempts++;
    }
    throw new Error("Request timed out waiting for prediction.");
  }

  function renderResult(result, latency) {
    resultBox.style.display = "block";
    latencyTag.textContent = `⚡ ${latency}ms`;

    const verdictBadge = document.getElementById("verdict-badge");
    const confValue = document.getElementById("conf-value");
    const progressFill = document.getElementById("progress-fill");

    const label = result.label || result.prediction;
    const confPct = (result.confidence * 100).toFixed(1);

    verdictBadge.textContent = label;
    confValue.textContent = `${confPct}% Conf`;
    progressFill.style.width = `${confPct}%`;

    // Apply color schemes
    verdictBadge.className = "verdict-badge";
    if (label.includes("FULLY AI")) {
      verdictBadge.classList.add("badge-ai");
      progressFill.style.background = "#ef4444";
    } else if (label.includes("HUMAN-WRITTEN")) {
      verdictBadge.classList.add("badge-human");
      progressFill.style.background = "#10b981";
    } else {
      verdictBadge.classList.add("badge-paraphrased");
      progressFill.style.background = "#f59e0b";
    }

    // Models breakdown
    const m = result.metrics || {};
    document.getElementById("score-deberta").textContent = m.deberta_conf ? `${(m.deberta_conf * 100).toFixed(1)}%` : "--";
    document.getElementById("score-rf").textContent = m.rf_conf ? `${(m.rf_conf * 100).toFixed(1)}%` : "--";

    // Diagnostic features
    document.getElementById("val-ppl").textContent = m.perplexity ? m.perplexity.toFixed(1) : "--";
    document.getElementById("val-burst").textContent = m.burstiness !== undefined ? m.burstiness.toFixed(3) : "--";
    document.getElementById("val-ttr").textContent = m.ttr !== undefined ? `${(m.ttr * 100).toFixed(1)}%` : "--";
    
    const sentScore = m.sentiment !== undefined ? m.sentiment.toFixed(2) : "0.0";
    const entScore = m.entropy !== undefined ? m.entropy.toFixed(1) : "0.0";
    document.getElementById("val-sent").textContent = `Sent: ${sentScore} | H: ${entScore}`;
  }

  function setLoading(loading) {
    btnAnalyze.disabled = loading;
    btnGrab.disabled = loading;
    if (loading) {
      spinner.style.display = "block";
      btnLabel.textContent = "Analyzing...";
    } else {
      spinner.style.display = "none";
      btnLabel.textContent = "Analyze Text";
    }
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = "block";
  }

  function hideError() {
    errorBox.style.display = "none";
  }
});