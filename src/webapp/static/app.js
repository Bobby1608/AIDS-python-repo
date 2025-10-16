// --- SSE connection ---
const statusEl = document.getElementById("status");
const banner = document.getElementById("alert-banner");
const tableBody = document.querySelector("#eventsTable tbody");

// Data for charts
const classNames = window.CLASS_NAMES || [];
const counts = {};
classNames.forEach(c => counts[c] = 0);

let confLabels = [];
let confData = [];

function showBanner(isAttack) {
  if (isAttack) {
    banner.classList.remove("hidden");
    // auto-hide after a couple seconds
    setTimeout(() => banner.classList.add("hidden"), 2000);
  }
}

// --- Charts ---
const countCtx = document.getElementById("countChart").getContext("2d");
const countChart = new Chart(countCtx, {
  type: "bar",
  data: {
    labels: classNames,
    datasets: [{
      label: "Count",
      data: classNames.map(c => counts[c]),
    }]
  },
  options: {
    responsive: true,
    animation: false,
    scales: { y: { beginAtZero: true } }
  }
});

const confCtx = document.getElementById("confChart").getContext("2d");
const confChart = new Chart(confCtx, {
  type: "line",
  data: {
    labels: confLabels,
    datasets: [{
      label: "Primary confidence",
      data: confData,
    }]
  },
  options: {
    responsive: true,
    animation: false,
    scales: { y: { min: 0, max: 1 } }
  }
});

function updateCharts() {
  // counts
  countChart.data.datasets[0].data = classNames.map(c => counts[c]);
  countChart.update();
  // confidence
  confChart.data.labels = confLabels;
  confChart.data.datasets[0].data = confData;
  confChart.update();
}

function addRow(idx, item) {
  const tr = document.createElement("tr");
  const isAttack = item.predicted !== "Normal";

  tr.innerHTML = `
    <td>${idx}</td>
    <td class="${isAttack ? "attack" : "normal"}">${item.predicted}</td>
    <td>${item.confidence.toFixed(3)}</td>
    <td>${item.ground_truth !== undefined ? item.ground_truth : "-"}</td>
    <td>${item.rf_pred}</td>
    <td>${item.mlp_pred}</td>
  `;

  tableBody.prepend(tr); // newest on top
  // limit rows to 200
  while (tableBody.rows.length > 200) {
    tableBody.deleteRow(tableBody.rows.length - 1);
  }

  if (isAttack) {
    showBanner(true);
  }
}

function handleBatch(payload) {
  if (!payload.items) return;

  payload.items.forEach((item, i) => {
    // update counts
    counts[item.predicted] = (counts[item.predicted] || 0) + 1;

    // update confidence series (cap to last 50)
    confLabels.push("");
    confData.push(item.confidence);
    if (confData.length > 50) {
      confData.shift();
      confLabels.shift();
    }

    // add row to table
    addRow(payload.batch_start + i + 1, item);
  });

  updateCharts();
}

// Connect SSE
let es;
function connect() {
  es = new EventSource("/events");
  statusEl.textContent = "Streaming Connected";
  statusEl.className = "status ok";

  es.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      if (data.heartbeat) return;
      handleBatch(data);
    } catch (e) {
      console.error("Bad SSE payload:", e);
    }
  };

  es.onerror = () => {
    statusEl.textContent = "Disconnected - retrying…";
    statusEl.className = "status warn";
    es.close();
    setTimeout(connect, 1500);
  };
}

connect();
