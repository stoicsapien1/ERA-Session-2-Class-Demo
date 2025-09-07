# Demo ERA v4 DEMO S2

# Git/GitHub/Flask/EC2 Instructions

* A single-page app (served by Flask) with:

  * Inputs for **image size** and **channels**.
  * An **“Add layer”** panel (Conv2D, MaxPool2D, AvgPool2D, Fully-Connected).
  * A **live table** showing for each step:

    * Output shape (H×W×C)
    * Receptive field (RF<sub>H</sub> × RF<sub>W</sub>)
    * Effective stride (“jump”) per axis
  * Export/Import architecture as JSON.
  * Upload to GitHub
  * Set up EC2 
  * Upload code to EC2
  * Enjoy!
  

The RF math supports kernel/stride/padding/dilation. (Details and formulas are explained near the end.)

---

# Prerequisites

* Python 3.9+
* Basic terminal comfort

---

# 1) Project setup

```bash
uv init s2_class_demo

uv add Flask
```

Project structure:

```
s2_class_demo/
├─ app.py
├─ static/
│  ├─ app.js
│  └─ styles.css        # optional (we'll mostly use Tailwind CDN)
└─ templates/
   └─ index.html
```

---

# 2) Flask backend (`app.py`)

This serves one page. All computation happens client-side for instant feedback.

```python
# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Single-page UI

if __name__ == "__main__":
    app.run(debug=True)
```

---

# 3) Minimal modern UI (`templates/index.html`)

Uses Tailwind via CDN for a clean look. The page hosts:

* Dataset image settings
* “Add Layer” controls
* Architecture table
* Export/Import buttons

```html
<!-- templates/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CNN Receptive Field Designer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- Tailwind (CDN) -->
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
  <div class="max-w-6xl mx-auto p-6 space-y-6">
    <header class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold">CNN Receptive Field Designer</h1>
      <div class="flex gap-2">
        <button id="exportJson" class="px-3 py-2 rounded-xl bg-slate-900 text-white shadow hover:opacity-90">Export JSON</button>
        <label class="px-3 py-2 rounded-xl bg-white border shadow cursor-pointer hover:bg-slate-100">
          Import JSON
          <input id="importJson" type="file" accept="application/json" class="hidden" />
        </label>
      </div>
    </header>

    <!-- Image Settings -->
    <section class="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div class="bg-white rounded-2xl p-4 shadow border">
        <h2 class="font-medium mb-3">Input Image</h2>
        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="text-sm">Height</label>
            <input id="inH" type="number" min="1" value="224" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Width</label>
            <input id="inW" type="number" min="1" value="224" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Channels</label>
            <input id="inC" type="number" min="1" value="3" class="w-full mt-1 rounded-lg border p-2">
          </div>
        </div>
        <p class="text-xs text-slate-500 mt-2">Change these and the whole stack recomputes.</p>
      </div>

      <!-- Add Layer -->
      <div class="bg-white rounded-2xl p-4 shadow border md:col-span-2">
        <h2 class="font-medium mb-3">Add Layer</h2>
        <div class="grid md:grid-cols-8 grid-cols-2 gap-3 items-end">
          <div class="md:col-span-2">
            <label class="text-sm">Type</label>
            <select id="layerType" class="w-full mt-1 rounded-lg border p-2">
              <option value="conv">Conv2D</option>
              <option value="maxpool">MaxPool2D</option>
              <option value="avgpool">AvgPool2D</option>
              <option value="fc">Fully Connected</option>
            </select>
          </div>
          <div>
            <label class="text-sm">Kernel</label>
            <input id="k" type="number" min="1" value="3" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Stride</label>
            <input id="s" type="number" min="1" value="1" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Padding</label>
            <input id="p" type="number" min="0" value="1" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Dilation</label>
            <input id="d" type="number" min="1" value="1" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div>
            <label class="text-sm">Out Channels / Units</label>
            <input id="outC" type="number" min="1" value="64" class="w-full mt-1 rounded-lg border p-2">
          </div>
          <div class="md:col-span-2">
            <button id="addLayer" class="w-full px-3 py-2 rounded-xl bg-blue-600 text-white shadow hover:bg-blue-700">Add Layer</button>
          </div>
        </div>
        <p class="text-xs text-slate-500 mt-2">For Pool layers, "Out Channels" is ignored. For FC, H×W collapses to 1×1.</p>
      </div>
    </section>

    <!-- Architecture Table -->
    <section class="bg-white rounded-2xl p-4 shadow border">
      <div class="flex items-center justify-between mb-3">
        <h2 class="font-medium">Architecture</h2>
        <div class="flex gap-2">
          <button id="clearLayers" class="px-3 py-2 rounded-xl bg-white border shadow hover:bg-slate-100">Clear</button>
          <button id="exampleNet" class="px-3 py-2 rounded-xl bg-white border shadow hover:bg-slate-100">Load Example</button>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead class="text-left bg-slate-100">
            <tr>
              <th class="p-2">#</th>
              <th class="p-2">Layer</th>
              <th class="p-2">k / s / p / d</th>
              <th class="p-2">Out Shape (H×W×C)</th>
              <th class="p-2">RF (H×W)</th>
              <th class="p-2">Jump (H×W)</th>
              <th class="p-2">Notes</th>
              <th class="p-2"></th>
            </tr>
          </thead>
          <tbody id="archTbody"></tbody>
        </table>
      </div>
    </section>

    <!-- Footer -->
    <footer class="text-xs text-slate-500">
      Tip: RF grows by <code>(d·(k−1)+1 − 1) × jump</code> per layer along each axis; jump multiplies by stride.
    </footer>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
```

---

# 4) Client logic (`static/app.js`)

This is the heart: it tracks layers, recomputes **output sizes**, **receptive field (height & width)**, and **jump** after every edit.

**Formulas used** (per axis, H and W separately):

* Effective kernel:
  `k_eff = d * (k - 1) + 1`
* Output size:
  `n_out = floor((n_in + 2p - k_eff) / s) + 1`
* Receptive field update:
  `rf_out = rf_in + (k_eff - 1) * jump_in`
* Jump (effective stride) update:
  `jump_out = jump_in * s`
* For **Fully Connected (FC)** (after flattening H×W): we assume each unit sees *all* spatial locations, so:
  `rfH_out = rfH_in + (H_in - 1) * jumpH_in`
  `rfW_out = rfW_in + (W_in - 1) * jumpW_in`
  and the spatial output becomes `1 × 1`.

```javascript
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
```

> You can skip `styles.css`—Tailwind already gives a nice modern look.

---

# 5) Run it

```bash
python app.py
# Flask will say: Running on http://127.0.0.1:5000
```

Open the URL in our browser.

* Set **Input Image** (e.g., 224×224×3).
* Add **Conv/Pool/FC** layers.
* Watch **Out Shape**, **RF (H×W)**, and **Jump** update instantly.
* Try “Load Example” to see a quick stack.

---

# 6) How the receptive field math works (brief & practical)

Track the following **per axis** (do it for H and W separately):

* `k_eff = d * (k - 1) + 1` (dilated kernel)
* Output size:

  $$
    n_\text{out} = \left\lfloor \frac{n_\text{in} + 2p - k_\text{eff}}{s} \right\rfloor + 1
  $$
* **Jump** (a.k.a. effective stride from input to this layer):

  $$
    j_\text{out} = j_\text{in} \cdot s
  $$
* **Receptive Field**:

  $$
    \text{RF}_\text{out} = \text{RF}_\text{in} + (k_\text{eff} - 1)\cdot j_\text{in}
  $$
* Initialize: `RF = 1`, `jump = 1`, `n = input_size`.
* **Pooling** behaves like a conv (no params) for RF purposes (it increases RF the same way, and multiplies jump by stride).
* **Fully-Connected** after flatten: each unit “sees” all spatial positions → the RF spans the full feature map:

  $$
    \text{RF}_H \leftarrow \text{RF}_H + (H_\text{in}-1)\cdot j_H,\quad
    \text{RF}_W \leftarrow \text{RF}_W + (W_\text{in}-1)\cdot j_W
  $$

  and spatial size becomes `1×1`.

> Tip: If you ever see `Out Shape <= 0`, our kernel/padding/stride combo is invalid for that input—reduce kernel or stride, or increase padding.

---

# 7) Nice extensions (optional)

* **“same” padding** toggle (compute `p` so that `n_out = ceil(n_in / s)`).
* **Per-axis kernels**: (kH, kW), (sH, sW), etc.
* **ConvTranspose2D** support (upsampling + RF math).
* **Latency estimator** (rough MACs/params).
* **Shareable links** (encode JSON in a URL).

---

# 8) Quick sanity check

With input **224×224×3** and a **Conv3×3 s=1 p=1 d=1**, RF grows from `1×1` to `3×3`, jump stays `1×1`, and output stays `224×224×C`.
Add **MaxPool2×2 s=2 p=0** next: output halves (112), RF grows by `(2-1)*1 = 1` per axis → becomes `4×4`, jump doubles to `2×2`.

---




# PART 2: GitHub and EC2

______________________________________________________

Boom—Let’s get it live on an AWS EC2 free-tier box. We’ll assume our app folder is `s2_class_demo/` with `app.py`, `templates/`, and `static/` as we built. Follow this exactly once; you’ll be done in \~15–25 minutes.

---

# A) Create & log in (Mumbai region recommended for India)

1. Go to **aws.amazon.com** → **Sign in** (or Create account).
2. After login, top-right **Region** → pick **Asia Pacific (Mumbai) ap-south-1** (or your preferred).
3. (Optional but good practice) Set up MFA on our account (IAM → Users → our user → Security credentials → MFA). But can do later, not required for this handson

---

# B) Launch an EC2 instance (free tier)

1. In the search bar, open **EC2** → **Instances** → **Launch instance**.
2. Name: `s2_class_demo`.
3. **AMI**: choose **Ubuntu Server 22.04 LTS (free tier eligible)**.
4. **Instance type**: `t2.micro` or `t3.micro` (free tier).
5. **Key pair**: Create new → give it a name → **Download .pem** (keep it safe!).
6. **Network settings** → “Allow HTTP (80)” and “Allow HTTPS (443)”.

   * SSH (22): restrict to “My IP” if possible.
7. **Launch instance**.

**Grab public IPv4**: EC2 → Instances → our instance → copy “Public IPv4 address”.

*(Optional but recommended)* Allocate an **Elastic IP** (EC2 → Elastic IPs → Allocate → Associate to our instance) so our IP doesn’t change.


---

# 0) Prep our code → GitHub
_I am using Windows, if you're using something else, ask ChatGPT for your OS commands_

**In PowerShell (Windows), inside our project folder**

```powershell
cd "I:\TSAI\2025\ERA V4\Course Content\Session 2\ins_code\s2_class_demo"

# Optional: create a nice .gitignore
@"
.venv/
__pycache__/
*.pyc
.env
.DS_Store
"@ | Out-File -Encoding utf8 .gitignore

git init
git add .
git commit -m "Initial commit: S2 Demo Code (Flask)"
```

**Create an empty repo on GitHub** (e.g., `s2_class_demo`) → copy its **HTTPS** URL, something like:

```
https://github.com/theschoolofai/s2_class_demo.git
```

**Connect and push**

```powershell
git remote add origin https://github.com/theschoolofai/s2_class_demo.git
git branch -M main
git push -u origin main
```

> Tip: If prompted for login, use a **Personal Access Token** (classic) as the password.



## 1) SSH into EC2 (Windows PowerShell)

```powershell
cd "I:\TSAI\2025\ERA V4\Course Content\Session 2\ins_code\s2_class_demo"
icacls .\s2_class_demo.pem /inheritance:r
icacls .\s2_class_demo.pem /grant:r "$($env:USERNAME):(R)"
icacls .\s2_class_demo.pem /remove "Authenticated Users" "BUILTIN\Users" "Everyone"

ssh -i .\s2_class_demo.pem ubuntu@X.X.X.X
```

---

## 2) Install Python

```bash
sudo apt update && sudo apt -y install python3-venv python3-pip git
```

---

## 3) Clone our repo

```bash
git clone https://github.com/<your-username>/s2_class_demo.git
cd s2_class_demo
python3 -m venv .venv
source .venv/bin/activate
pip install flask
```

---

## 4) Run Flask directly

Inside our repo folder (`s2_class_demo/`), run:

```bash
python3 app.py
```

---

## 5) Open in browser

Go to:

```
http://X.X.X.X:5000
```

> ⚠️ Oh! We had to **allow port 5000** in our **EC2 security group**.
> In AWS console → EC2 → our instance → **Security** → Inbound rules →
> **Add rule**:

* Type: Custom TCP
* Port: 5000
* Source: My IP (or 0.0.0.0/0 just for testing)

---

Let's fix this. 

## 1) Edit locally (Windows)

Open our project and change the last line of `app.py`:

```python
# BEFORE
app.run(debug=True)

# AFTER
app.run(host="0.0.0.0", port=5000, debug=True)
```

Save.

---

## 2) Commit & push to GitHub (PowerShell in our project folder)

```powershell
cd "I:\TSAI\2025\ERA V4\Course Content\Session 2\ins_code\s2_class_demo"

git status
git add app.py
git commit -m "Bind Flask to 0.0.0.0:5000 for EC2"
git push
```

(If prompted for GitHub auth, use our PAT.)

---

## 3) Pull on AWS and run

SSH in:

```powershell
ssh -i "I:\TSAI\2025\ERA V4\Course Content\Session 2\ins_code\s2_class_demo.pem" ubuntu@X.X.X.X
```

Then on the server:

```bash
cd ~/s2_class_demo   # or /opt/s2_class_demo if that's where you cloned, but you probably are area already there, so no need to run this command
git pull
source .venv/bin/activate    # if you created a venv here earlier, but you may be already activated, so not required
# ensure Flask is installed if this is a fresh shell
pip install flask #already installed

# run the app
python3 app.py
# You should now see: Running on http://0.0.0.0:5000
```

Open in our browser:

```
http://X.X.X.X:5000
```

> If it still times out, make sure our **EC2 Security Group** has an **inbound rule** allowing TCP **5000** from **our IP** (or 0.0.0.0/0 temporarily).

---

## How to exit SSH?
```bash
exit
```
