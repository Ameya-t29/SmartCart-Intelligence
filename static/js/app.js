let currentModel = "kmeans";
let currentK = 4;
let lastData = null;

const $ = (id) => document.getElementById(id);

function money(v) {
  return "₹" + Number(v || 0).toLocaleString("en-IN", {maximumFractionDigits: 0});
}

function showToast(message) {
  const t = $("toast");
  t.textContent = message;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2600);
}

async function loadDashboard() {
  $("refreshBtn").disabled = true;
  $("refreshBtn").querySelector("span:first-child").textContent = "Running…";

  try {
    const res = await fetch(`/api/dashboard?model=${currentModel}&k=${currentK}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    lastData = data;
    renderMetrics(data);
    renderScatter(data);
    renderClusterBar(data);
    renderElbow(data);
    renderSilhouette(data);
    renderSummary(data);

    $("mModel").textContent = currentModel === "kmeans" ? "K-Means" : "Agglomerative";
  } catch (err) {
    showToast(err.message);
  } finally {
    $("refreshBtn").disabled = false;
    $("refreshBtn").querySelector("span:first-child").textContent = "Run analysis";
  }
}

function renderMetrics(d) {
  $("mCustomers").textContent = d.total_rows.toLocaleString();
  $("mClusters").textContent = d.k;
  $("mSilhouette").textContent = Number(d.silhouette_score).toFixed(3);
  const pca = d.pca_variance.reduce((a,b)=>a+b,0) * 100;
  $("mPca").textContent = pca.toFixed(1) + "%";
}

const baseLayout = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: {family:"Inter", color:"#8f9ab0", size:10},
  margin: {l:0,r:0,t:10,b:0},
  showlegend: true,
  legend: {font:{size:10}, bgcolor:"rgba(0,0,0,0)"},
  hoverlabel: {bgcolor:"#101727", font:{color:"#fff"}}
};

const clusterColors = ["#5ee7ff","#9b7cff","#6dffb8","#ffbb6b","#ff6b91","#70a7ff","#f4e26d","#d58cff","#8fe9df","#ff8fca"];

function renderScatter(d) {
  const traces = [];
  for (let c=0;c<d.k;c++) {
    const points = d.records.filter(r => Number(r.cluster) === c);
    traces.push({
      x: points.map(r=>r.PCA1),
      y: points.map(r=>r.PCA2),
      z: points.map(r=>r.PCA3),
      text: points.map(r => `Customer ${r.ID}<br>Income: ${money(r.Income)}<br>Spending: ${money(r.Total_Spending)}<br>Recency: ${r.Recency}`),
      hovertemplate: "%{text}<extra>Cluster " + c + "</extra>",
      mode:"markers",
      type:"scatter3d",
      name:"Cluster " + c,
      marker:{size:4, color:clusterColors[c], opacity:.82}
    });
  }

  Plotly.newPlot("scatter3d", traces, {
    ...baseLayout,
    scene:{
      bgcolor:"rgba(0,0,0,0)",
      xaxis:{title:"PCA 1",gridcolor:"rgba(255,255,255,.06)",zerolinecolor:"rgba(255,255,255,.05)"},
      yaxis:{title:"PCA 2",gridcolor:"rgba(255,255,255,.06)",zerolinecolor:"rgba(255,255,255,.05)"},
      zaxis:{title:"PCA 3",gridcolor:"rgba(255,255,255,.06)",zerolinecolor:"rgba(255,255,255,.05)"}
    }
  }, {responsive:true, displaylogo:false});
}

function renderClusterBar(d) {
  const labels = Object.keys(d.cluster_counts).map(c => "Cluster " + c);
  const values = Object.values(d.cluster_counts);
  Plotly.newPlot("clusterBar", [{
    x:labels, y:values, type:"bar",
    marker:{color:values.map((_,i)=>clusterColors[i])},
    hovertemplate:"%{x}<br>%{y} customers<extra></extra>"
  }], {
    ...baseLayout,
    margin:{l:45,r:10,t:10,b:45},
    xaxis:{gridcolor:"rgba(255,255,255,.04)"},
    yaxis:{gridcolor:"rgba(255,255,255,.04)"}
  }, {responsive:true,displaylogo:false});
}

function renderElbow(d) {
  Plotly.newPlot("elbow", [{
    x:d.wcss_k,y:d.wcss,type:"scatter",mode:"lines+markers",
    line:{color:"#5ee7ff",width:3},marker:{color:"#5ee7ff",size:7}
  }], {
    ...baseLayout,
    margin:{l:50,r:15,t:15,b:45},
    xaxis:{title:"K",gridcolor:"rgba(255,255,255,.04)"},
    yaxis:{title:"WCSS",gridcolor:"rgba(255,255,255,.04)"},
    shapes:[{
      type:"line",x0:d.elbow_k,x1:d.elbow_k,y0:0,y1:1,
      yref:"paper",line:{color:"#9b7cff",dash:"dot"}
    }],
    annotations:[{
      x:d.elbow_k,y:1,yref:"paper",text:`Suggested elbow: ${d.elbow_k}`,
      showarrow:false,font:{size:10,color:"#b9adff"},yshift:10
    }]
  }, {responsive:true,displaylogo:false});
}

function renderSilhouette(d) {
  Plotly.newPlot("silhouette", [{
    x:d.silhouette_k,y:d.silhouette,type:"scatter",mode:"lines+markers",
    line:{color:"#6dffb8",width:3},marker:{color:"#6dffb8",size:7}
  }], {
    ...baseLayout,
    margin:{l:50,r:15,t:15,b:45},
    xaxis:{title:"Number of clusters",gridcolor:"rgba(255,255,255,.04)"},
    yaxis:{title:"Silhouette score",gridcolor:"rgba(255,255,255,.04)"}
  }, {responsive:true,displaylogo:false});
}

function renderSummary(d) {
  $("summaryBody").innerHTML = d.summary.map(s => `
    <tr>
      <td>Cluster ${s.cluster}</td>
      <td>${s.customers.toLocaleString()}</td>
      <td>${money(s.avg_income)}</td>
      <td>${money(s.avg_spending)}</td>
      <td>${s.avg_recency.toFixed(1)} days</td>
      <td>${s.avg_age.toFixed(1)} yrs</td>
    </tr>
  `).join("");
}

async function inspectCustomer() {
  const id = $("customerId").value.trim();
  if (!id) return showToast("Enter a customer ID first.");

  try {
    const res = await fetch(`/api/customer/${id}?model=${currentModel}&k=${currentK}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.error);

    const c = data.customer;
    $("customerPanel").classList.remove("hidden");
    $("customerAvatar").textContent = String(c.ID).slice(-2);
    $("customerTitle").textContent = `Customer #${c.ID}`;
    $("customerSegment").textContent = `Cluster ${c.cluster} • ${c.Education} • ${c.Living_With}`;

    $("customerStats").innerHTML = `
      <div class="stat"><b>${money(c.Income)}</b><span>INCOME</span></div>
      <div class="stat"><b>${money(c.Total_Spending)}</b><span>TOTAL SPENDING</span></div>
      <div class="stat"><b>${c.Age}</b><span>AGE</span></div>
      <div class="stat"><b>${c.Recency}</b><span>RECENCY</span></div>
      <div class="stat"><b>${c.NumWebPurchases}</b><span>WEB PURCHASES</span></div>
      <div class="stat"><b>${c.NumStorePurchases}</b><span>STORE PURCHASES</span></div>
    `;

    $("clusterContext").textContent =
      `This customer belongs to Cluster ${c.cluster}. The cluster contains ${data.cluster_stats.customers} customers, ` +
      `with average income ${money(data.cluster_stats.avg_income)}, average spending ${money(data.cluster_stats.avg_spending)}, ` +
      `and average recency ${data.cluster_stats.avg_recency.toFixed(1)} days.`;

    $("customerPanel").scrollIntoView({behavior:"smooth",block:"center"});
  } catch (err) {
    showToast(err.message);
  }
}

document.querySelectorAll(".model-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".model-btn").forEach(b=>b.classList.remove("active"));
    btn.classList.add("active");
    currentModel = btn.dataset.model;
  });
});

$("kSlider").addEventListener("input", e => {
  currentK = Number(e.target.value);
  $("kValue").textContent = currentK;
});

$("refreshBtn").addEventListener("click", loadDashboard);
$("inspectBtn").addEventListener("click", inspectCustomer);
$("customerId").addEventListener("keydown", e => {
  if (e.key === "Enter") inspectCustomer();
});

loadDashboard();
