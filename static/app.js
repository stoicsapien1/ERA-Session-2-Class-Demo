// static/app.js
(() => {
    const $ = (id) => document.getElementById(id);
  
    const inH = $("inH"), inW = $("inW"), inC = $("inC");
    const layerType = $("layerType");
    const k = $("k"), s = $("s"), p = $("p"), d = $("d"), outC = $("outC");
    const addLayerBtn = $("addLayer");
    const archTbody = $("archTbody");
    const clearBtn = $("clearLayers");
    const exampleBtn = $("exampleNet");
    const exportBtn = $("exportJson");
    const importInput = $("importJson");
  
    let layers = [];
  
    function asInt(el) { return Math.max( parseInt(el.value || "0", 10), 0 ); }
    function asIntMin1(el) { return Math.max( parseInt(el.value || "1", 10), 1 ); }
  
    function compute() {
      // Read input image
      let H = asIntMin1(inH);
      let W = asIntMin1(inW);
      let C = asIntMin1(inC);
  
      // RF and jumps per axis
      let rfH = 1, rfW = 1;
      let jH = 1, jW = 1;
  
      const rows = [];
  
      for (let i = 0; i < layers.length; i++) {
        const L = layers[i];
        let note = "";
  
        if (L.type === "conv") {
          const kEff = L.d * (L.k - 1) + 1;
          const Hout = Math.floor((H + 2*L.p - kEff) / L.s) + 1;
          const Wout = Math.floor((W + 2*L.p - kEff) / L.s) + 1;
          if (Hout <= 0 || Wout <= 0) note = "⚠️ output <= 0 (check k/p/s/d)";
          // RF, jump update
          const rfHn = rfH + (kEff - 1) * jH;
          const rfWn = rfW + (kEff - 1) * jW;
          const jHn = jH * L.s, jWn = jW * L.s;
  
          H = Hout; W = Wout; C = L.outC;
          rfH = rfHn; rfW = rfWn; jH = jHn; jW = jWn;
  
        } else if (L.type === "maxpool" || L.type === "avgpool") {
          const kEff = L.d * (L.k - 1) + 1;
          const Hout = Math.floor((H + 2*L.p - kEff) / L.s) + 1;
          const Wout = Math.floor((W + 2*L.p - kEff) / L.s) + 1;
          if (Hout <= 0 || Wout <= 0) note = "⚠️ output <= 0 (check k/p/s/d)";
          // Pooling keeps channels, grows RF like a conv with no params
          const rfHn = rfH + (kEff - 1) * jH;
          const rfWn = rfW + (kEff - 1) * jW;
          const jHn = jH * L.s, jWn = jW * L.s;
  
          H = Hout; W = Wout; // C unchanged
          rfH = rfHn; rfW = rfWn; jH = jHn; jW = jWn;
  
        } else if (L.type === "fc") {
          // Flatten HxW -> 1x1, assume dense connects to all positions
          const rfHn = rfH + (H - 1) * jH;
          const rfWn = rfW + (W - 1) * jW;
          H = 1; W = 1; C = L.outC; // units
          rfH = rfHn; rfW = rfWn; /* jumps irrelevant after 1x1 but keep */
        }
  
        rows.push({
          idx: i+1,
          type: L.type,
          k: L.k, s: L.s, p: L.p, d: L.d,
          outShape: `${H}×${W}×${C}`,
          rf: `${rfH}×${rfW}`,
          jump: `${jH}×${jW}`,
          note,
        });
      }
  
      renderTable(rows);
    }
  
    function renderTable(rows) {
      archTbody.innerHTML = "";
      rows.forEach((r, i) => {
        const tr = document.createElement("tr");
        tr.className = "border-b hover:bg-slate-50";
        tr.innerHTML = `
          <td class="p-2">${r.idx}</td>
          <td class="p-2">${prettyType(layers[i])}</td>
          <td class="p-2">${fmtKSPL(layers[i])}</td>
          <td class="p-2 font-mono">${r.outShape}</td>
          <td class="p-2 font-mono">${r.rf}</td>
          <td class="p-2 font-mono">${r.jump}</td>
          <td class="p-2">${r.note}</td>
          <td class="p-2">
            <button data-idx="${i}" class="del px-2 py-1 rounded-lg bg-white border shadow text-xs hover:bg-slate-100">Remove</button>
          </td>
        `;
        archTbody.appendChild(tr);
      });
  
      // Hook up delete buttons
      document.querySelectorAll("button.del").forEach(btn => {
        btn.onclick = () => {
          const idx = parseInt(btn.getAttribute("data-idx"), 10);
          layers.splice(idx, 1);
          compute();
        };
      });
    }
  
    function prettyType(L) {
      if (L.type === "conv") return `Conv2D → ${L.outC}`;
      if (L.type === "maxpool") return `MaxPool2D`;
      if (L.type === "avgpool") return `AvgPool2D`;
      if (L.type === "fc") return `FC → ${L.outC}`;
      return L.type;
    }
    function fmtKSPL(L) {
      if (L.type === "fc") return "—";
      return `k=${L.k} / s=${L.s} / p=${L.p} / d=${L.d}`;
    }
  
    // Event: add layer
    addLayerBtn.onclick = () => {
      const t = layerType.value;
      const layer = {
        type: t,
        k: asIntMin1(k),
        s: asIntMin1(s),
        p: asInt(p),
        d: asIntMin1(d),
        outC: asIntMin1(outC),
      };
      // For Pool, outC unused
      if (t === "maxpool" || t === "avgpool") {
        layer.outC = null;
      }
      // For FC, k/s/p/d irrelevant
      if (t === "fc") {
        layer.k = 1; layer.s = 1; layer.p = 0; layer.d = 1;
      }
      layers.push(layer);
      compute();
    };
  
    // Event: input image changes trigger recompute
    [inH, inW, inC].forEach(el => el.addEventListener("input", compute));
  
    // Clear
    clearBtn.onclick = () => { layers = []; compute(); };
  
    // Example net (like quick VGG-ish)
    exampleBtn.onclick = () => {
      layers = [
        { type: "conv", k:3, s:1, p:1, d:1, outC:64 },
        { type: "conv", k:3, s:1, p:1, d:1, outC:64 },
        { type: "maxpool", k:2, s:2, p:0, d:1, outC:null },
        { type: "conv", k:3, s:1, p:1, d:1, outC:128 },
        { type: "conv", k:3, s:1, p:1, d:1, outC:128 },
        { type: "maxpool", k:2, s:2, p:0, d:1, outC:null },
        { type: "fc",   k:1, s:1, p:0, d:1, outC:256 },
        { type: "fc",   k:1, s:1, p:0, d:1, outC:10 },
      ];
      compute();
    };
  
    // Export/Import JSON
    exportBtn.onclick = () => {
      const payload = {
        input: { H: asIntMin1(inH), W: asIntMin1(inW), C: asIntMin1(inC) },
        layers
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type:"application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "cnn_architecture.json";
      a.click();
      URL.revokeObjectURL(a.href);
    };
  
    importInput.onchange = (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const obj = JSON.parse(reader.result);
          if (obj.input) {
            inH.value = obj.input.H;
            inW.value = obj.input.W;
            inC.value = obj.input.C;
          }
          if (Array.isArray(obj.layers)) {
            layers = obj.layers;
          }
          compute();
        } catch (err) {
          alert("Invalid JSON");
        }
      };
      reader.readAsText(file);
      // reset input
      importInput.value = "";
    };
  
    // Initial render
    compute();
  })();