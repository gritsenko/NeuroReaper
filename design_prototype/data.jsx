/* data.jsx — model graph, per-node telemetry, seeded fields + color ramps */

/* ---- seeded PRNG ---- */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

/* smooth scalar field on an NxN grid, values ~[0,1]; box-blurred random noise */
function smoothField(seed, N) {
  const rnd = mulberry32(seed);
  const raw = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) raw[i] = rnd();
  // a couple low-freq sine humps for structure
  const fx = 0.4 + rnd() * 1.6, fy = 0.4 + rnd() * 1.6, ph = rnd() * 6.28;
  const out = new Float32Array(N * N);
  let mn = Infinity, mx = -Infinity;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let s = 0, c = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= N || yy >= N) continue;
      s += raw[yy * N + xx]; c++;
    }
    let v = s / c;
    v = 0.62 * v + 0.38 * (0.5 + 0.5 * Math.sin(fx * x / N * 6.28 + fy * y / N * 6.28 + ph));
    out[y * N + x] = v;
    if (v < mn) mn = v; if (v > mx) mx = v;
  }
  const r = mx - mn || 1;
  for (let i = 0; i < N * N; i++) out[i] = (out[i] - mn) / r;
  return out;
}

/* INT8-quantize a field: stair-step to `levels` + small per-node divergence offset */
function quantizeField(field, N, levels, diffScale, seed) {
  const rnd = mulberry32(seed ^ 0x9e3779b9);
  const out = new Float32Array(N * N);
  for (let i = 0; i < N * N; i++) {
    const q = Math.round(field[i] * (levels - 1)) / (levels - 1);
    const jitter = (rnd() - 0.5) * 2 * diffScale;
    out[i] = Math.min(1, Math.max(0, q + jitter));
  }
  return out;
}

/* ---- color ramps ---- */
function lerp(a, b, t) { return a + (b - a) * t; }
function mix(c1, c2, t) { return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)]; }
function rgbStr(c) { return `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`; }

/* inferno-ish ramp for activation feature maps (dark -> purple -> amber -> white) */
const INFERNO = [[5, 4, 18], [40, 11, 70], [101, 21, 110], [159, 42, 99], [212, 72, 66], [241, 130, 38], [250, 196, 80], [252, 255, 220]];
/* "ice" alt ramp (dark -> teal -> cyan -> white) */
const ICE = [[6, 10, 22], [12, 44, 74], [16, 86, 110], [26, 138, 140], [70, 190, 170], [150, 224, 210], [225, 250, 248]];
/* diff ramp (black -> deep red -> amber -> white hot) */
const DIFFRAMP = [[8, 6, 10], [70, 14, 18], [150, 30, 22], [212, 90, 28], [245, 170, 70], [255, 240, 200]];

function sampleRamp(ramp, t) {
  t = Math.min(1, Math.max(0, t));
  const n = ramp.length - 1; const f = t * n; const i = Math.min(n - 1, Math.floor(f));
  return mix(ramp[i], ramp[i + 1], f - i);
}

/* divergence heatmap stops as RGB (kept in sync w/ css vars conceptually) */
const HEAT_RAMPS = {
  thermal: [[58, 196, 122], [150, 214, 70], [220, 196, 60], [226, 132, 40], [214, 64, 46]], // green->lime->amber->red
  magma:   [[40, 20, 70], [120, 36, 110], [200, 70, 90], [240, 140, 60], [252, 220, 130]],
  ice2red: [[40, 130, 175], [110, 180, 180], [210, 200, 90], [226, 120, 50], [214, 60, 46]],
};
function heatColor(t, ramp) {
  t = Math.min(1, Math.max(0, t));
  const stops = HEAT_RAMPS[ramp] || HEAT_RAMPS.thermal;
  const n = stops.length - 1; const f = t * n; const i = Math.min(n - 1, Math.floor(f));
  return rgbStr(mix(stops[i], stops[i + 1], f - i));
}

/* ---- model graph (resnet18, ONNX export) ---- */
/* Top level is the FOLDED hierarchical view: each `group` node is a torch
   module (layerN) that, in the raw ONNX graph, expands to many machine-named
   Conv/Add/Relu/downsample nodes. `diff` on a node is the MAX abs diff seen
   anywhere in its subgraph, so the heatmap reads as "worst drift per stage".
   Global peak (0.0419) lives INSIDE the folded layer3 group — at
   /layer3/layer3.1/conv2/Conv — which is the whole point of expand-on-click. */
const GLOBAL_MAX = 0.0419;

/* children carry real ONNX-style names; shape/grid/quant inherit from parent */
const NODES = [
  { id: "input",   op: "Input",             name: "input",   shape: "1×3×224×224",  out: "FP32 image",     diff: 0.0000, quant: false, grid: 16 },
  { id: "conv1",   op: "Conv 7×7 /2",       name: "conv1",   shape: "1×64×112×112", out: "Conv output",    diff: 0.0009, quant: true,  grid: 16 },
  { id: "bn1",     op: "BatchNorm",         name: "bn1",     shape: "1×64×112×112", out: "Normalized",     diff: 0.0012, quant: true,  grid: 16 },
  { id: "relu",    op: "Relu",              name: "relu",    shape: "1×64×112×112", out: "Activation",     diff: 0.0010, quant: true,  grid: 16 },
  { id: "maxpool", op: "MaxPool 3×3 /2",    name: "maxpool", shape: "1×64×56×56",   out: "Pooled",         diff: 0.0014, quant: true,  grid: 14 },

  { id: "layer1",  op: "Bottleneck ×2", name: "layer1", shape: "1×64×56×56",  out: "Residual block", diff: 0.0046, quant: true, grid: 14, group: true, nInner: 14, children: [
    { id: "/layer1/layer1.0/conv1/Conv", op: "Conv 3×3", diff: 0.0021 },
    { id: "/layer1/layer1.0/conv2/Conv", op: "Conv 3×3", diff: 0.0034 },
    { id: "/layer1/layer1.0/Add",        op: "Add (skip)", diff: 0.0029 },
    { id: "/layer1/layer1.1/conv1/Conv", op: "Conv 3×3", diff: 0.0038 },
    { id: "/layer1/layer1.1/conv2/Conv", op: "Conv 3×3", diff: 0.0046, hot: true },
    { id: "/layer1/layer1.1/Relu_1",     op: "Relu",     diff: 0.0041 },
  ]},
  { id: "layer2",  op: "Bottleneck ×2", name: "layer2", shape: "1×128×28×28", out: "Residual block", diff: 0.0123, quant: true, grid: 14, group: true, nInner: 16, children: [
    { id: "/layer2/layer2.0/conv1/Conv",            op: "Conv 3×3 /2", diff: 0.0058 },
    { id: "/layer2/layer2.0/downsample/downsample.0/Conv", op: "Conv 1×1 /2", diff: 0.0083 },
    { id: "/layer2/layer2.0/conv2/Conv",            op: "Conv 3×3", diff: 0.0097 },
    { id: "/layer2/layer2.0/Add",                   op: "Add (skip)", diff: 0.0090 },
    { id: "/layer2/layer2.1/conv1/Conv",            op: "Conv 3×3", diff: 0.0111 },
    { id: "/layer2/layer2.1/conv2/Conv",            op: "Conv 3×3", diff: 0.0123, hot: true },
    { id: "/layer2/layer2.1/Relu_1",                op: "Relu", diff: 0.0116 },
  ]},
  { id: "layer3",  op: "Bottleneck ×2", name: "layer3", shape: "1×256×14×14", out: "Residual block", diff: 0.0419, quant: true, grid: 14, group: true, hot: true, nInner: 16, children: [
    { id: "/layer3/layer3.0/conv1/Conv",            op: "Conv 3×3 /2", diff: 0.0204 },
    { id: "/layer3/layer3.0/downsample/downsample.0/Conv", op: "Conv 1×1 /2", diff: 0.0288 },
    { id: "/layer3/layer3.0/conv2/Conv",            op: "Conv 3×3", diff: 0.0311 },
    { id: "/layer3/layer3.0/Add",                   op: "Add (skip)", diff: 0.0301 },
    { id: "/layer3/layer3.1/conv1/Conv",            op: "Conv 3×3", diff: 0.0357 },
    { id: "/layer3/layer3.1/conv2/Conv",            op: "Conv 3×3", diff: 0.0419, hot: true },
    { id: "/layer3/layer3.1/Add",                   op: "Add (skip)", diff: 0.0392 },
    { id: "/layer3/layer3.1/Relu_1",                op: "Relu", diff: 0.0388 },
  ]},
  { id: "layer4",  op: "Bottleneck ×2", name: "layer4", shape: "1×512×7×7", out: "Residual block", diff: 0.0388, quant: true, grid: 7, group: true, nInner: 16, children: [
    { id: "/layer4/layer4.0/conv1/Conv",            op: "Conv 3×3 /2", diff: 0.0331 },
    { id: "/layer4/layer4.0/downsample/downsample.0/Conv", op: "Conv 1×1 /2", diff: 0.0352 },
    { id: "/layer4/layer4.0/conv2/Conv",            op: "Conv 3×3", diff: 0.0371 },
    { id: "/layer4/layer4.0/Add",                   op: "Add (skip)", diff: 0.0364 },
    { id: "/layer4/layer4.1/conv1/Conv",            op: "Conv 3×3", diff: 0.0383 },
    { id: "/layer4/layer4.1/conv2/Conv",            op: "Conv 3×3", diff: 0.0388, hot: true },
    { id: "/layer4/layer4.1/Relu_1",                op: "Relu", diff: 0.0380 },
  ]},

  { id: "gap",    op: "GlobalAveragePool", name: "gap",    shape: "1×512×1×1", out: "Pooled vector",  diff: 0.0291, quant: true,  grid: 8 },
  { id: "fc",     op: "Gemm",              name: "fc",     shape: "1×1000",    out: "Logits",         diff: 0.0231, quant: true,  grid: 8 },
  { id: "output", op: "Softmax",           name: "output", shape: "1×1000",    out: "Probabilities",  diff: 0.0074, quant: false, grid: 8 },
];

/* flat lookup including expandable children (children inherit parent fields) */
const NODE_BY_ID = (() => {
  const m = {};
  for (const n of NODES) {
    m[n.id] = n;
    if (n.children) for (const c of n.children) {
      const segs = c.id.split("/").filter(Boolean);
      const short = segs.slice(1, -1).join("/") || segs[segs.length - 1]; // e.g. layer3.1/conv2
      m[c.id] = {
        id: c.id,
        op: c.op,
        name: c.id,            // full ONNX path
        shortName: short,
        shape: c.shape || n.shape,
        out: c.op.startsWith("Add") ? "Residual sum" : c.op.startsWith("Relu") ? "Activation" : "Conv output",
        diff: c.diff,
        quant: c.quant != null ? c.quant : n.quant,
        grid: c.grid || n.grid,
        hot: !!c.hot,
        parent: n.id,
      };
    }
  }
  return m;
})();

/* per-node derived telemetry (deterministic) */
function nodeStats(node) {
  const r = mulberry32(hashStr(node.id));
  const t = node.diff / GLOBAL_MAX;
  const meanF = (r() * 0.4 - 0.05);
  const stdF = (0.3 + r() * 0.5);
  const minF = (-1.2 - r() * 1.8);
  const maxF = (1.4 + r() * 2.6);
  const maxAbs = node.diff;
  const meanAbs = node.diff * (0.12 + r() * 0.06);
  const cosine = node.diff === 0 ? 1 : 1 - (t * t) * (0.0009 + r() * 0.0004);

  /* INT8 candidate: symmetric-ish affine quant over [minF,maxF], 256 levels.
     dequantized stats drift slightly from FP32 reference. */
  const levels = node.quant ? 256 : 0;
  const scale = (maxF - minF) / 255;
  const zeroPoint = Math.max(0, Math.min(255, Math.round(-minF / scale)));
  // dequant range snaps to representable grid; mean/std shift a hair
  const min8 = Math.round(minF / scale) * scale;
  const max8 = Math.round(maxF / scale) * scale;
  const mean8 = meanF + (r() - 0.5) * scale * 1.2;
  const std8 = stdF * (1 - (0.004 + r() * 0.004));

  const f4 = (x) => x.toFixed(4);
  return {
    // FP32 reference
    min: f4(minF), max: f4(maxF), mean: f4(meanF), std: f4(stdF),
    // INT8 candidate (dequantized)
    min8: f4(min8), max8: f4(max8), mean8: f4(mean8), std8: f4(std8),
    scale: scale.toFixed(5), zeroPoint, levels,
    // divergence
    maxAbs: maxAbs.toFixed(4),
    meanAbs: meanAbs.toFixed(4),
    cosine: cosine.toFixed(4),
  };
}

/* histogram bins from a node's value field — returns FP32 (smooth) + INT8 (stepped) */
function histogram(node, bins, seedSalt) {
  const N = 26;
  const f = smoothField((hashStr(node.id) ^ 0x55 ^ ((seedSalt || 0) * 0x9e37)) >>> 0, N);
  const fp = new Array(bins).fill(0);
  for (let i = 0; i < N * N; i++) {
    const v = (f[i] - 0.5);
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v * 0.5 + 0.5) * bins)));
    fp[idx]++;
  }
  // INT8: collapse the same samples onto far fewer representable levels (visible stair-step)
  const q = node.quant ? 18 : bins;
  const int8 = new Array(bins).fill(0);
  for (let i = 0; i < N * N; i++) {
    const v = (f[i] - 0.5);
    const u = (v * 0.5 + 0.5);
    const qu = Math.round(u * (q - 1)) / (q - 1);
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(qu * bins)));
    int8[idx]++;
  }
  const mx = Math.max(Math.max(...fp), Math.max(...int8)) || 1;
  return { fp: fp.map(v => v / mx), int8: int8.map(v => v / mx) };
}

/* ---- latency / cost model ----
   INT8 on the NPU is faster than FP32 on CPU, but the win varies wildly by op:
   big early convs save a lot; tiny tail ops (gap/fc) save almost nothing while
   still drifting. cpuMs = FP32 reference latency, npuMs = INT8 candidate latency,
   tSaved = cpuMs - npuMs. Numbers are per-op wall time for this single inference. */
const COST = {
  input:   { cpu: 0.00, npu: 0.00 },
  conv1:   { cpu: 3.10, npu: 1.58 },   // saved 1.52 — fat 7×7, huge INT8 win
  bn1:     { cpu: 0.30, npu: 0.20 },
  relu:    { cpu: 0.10, npu: 0.06 },
  maxpool: { cpu: 0.14, npu: 0.08 },
  layer1:  { cpu: 2.40, npu: 1.60 },   // saved 0.80
  layer2:  { cpu: 2.05, npu: 1.50 },   // saved 0.55
  layer3:  { cpu: 2.80, npu: 2.05 },   // saved 0.75 (group)
  layer4:  { cpu: 1.95, npu: 1.65 },   // saved 0.30
  gap:     { cpu: 0.18, npu: 0.13 },   // saved 0.05 — tiny win, real drift
  fc:      { cpu: 0.42, npu: 0.30 },   // saved 0.12
  output:  { cpu: 0.05, npu: 0.03 },
  /* the inspected hot child carries its own measured cost */
  "/layer3/layer3.1/conv2/Conv": { cpu: 0.94, npu: 0.63 }, // saved 0.31
};
const MAX_SAVED = 1.52;

/* fusion-break probe penalty: reading an intermediate tensor forces the runtime
   to materialize it, breaking the conv→bn→relu hardware fusion. */
const PROBE_PENALTY_MS = 0.4;

function costOf(node) {
  const c = COST[node.id] || COST[node.parent] || { cpu: 0, npu: 0 };
  const saved = +(c.cpu - c.npu).toFixed(2);
  return { cpu: c.cpu, npu: c.npu, saved, quant: !!node.quant };
}

/* ============================================================================
   UNIVERSAL CONFIG MODEL
   A run is described by a config = { ep, precision, device }. A comparison holds
   one reference config + N candidate configs. "Quantization" and "Backend
   divergence" are just PRESETS over this one model — the engine is identical, only
   the interpretation (inspector, verdict, coloring) changes per mode.
   ============================================================================ */
const BACKENDS  = ["CPU", "CUDA", "TensorRT", "DirectML", "QNN", "OpenVINO"];   // execution providers / EP
const PRECISIONS = ["FP32", "BF16", "FP16", "FP8", "INT8"];
const DEVICES   = ["CPU", "GPU", "NPU"];

/* EP metadata: default device per EP + lifecycle status.
   DirectML is deprecated (superseded by Windows ML) — kept selectable because
   legacy targets still ship it, but flagged in the UI. */
const EP_INFO = {
  CPU:      { device: "CPU" },
  CUDA:     { device: "GPU" },
  TensorRT: { device: "GPU" },
  DirectML: { device: "GPU", legacy: true, note: "deprecated — superseded by Windows ML" },
  QNN:      { device: "NPU" },
  OpenVINO: { device: "CPU" },
};
function epDevice(ep) { return (EP_INFO[ep] || {}).device || "GPU"; }

function cfg(ep, precision, device) { return { ep, precision, device }; }
function sameCfg(a, b) { return a && b && a.ep === b.ep && a.precision === b.precision && a.device === b.device; }
function cfgLabel(c) { return `${c.ep} · ${c.precision}`; }

/* preset templates — the default starting point for each mode */
const PRESETS = {
  quant: {
    label: "Quantization",
    blurb: "same backend · precision only",
    ref: cfg("CPU", "FP32", "CPU"),
    cands: [cfg("CPU", "INT8", "CPU")],     // ONE axis changes: FP32 → INT8
  },
  backend: {
    label: "Backend divergence",
    blurb: "same precision · backend only",
    // gold is a real GPU run (CUDA·FP16), so candidates differ on the BACKEND
    // axis ONLY — same precision, same device. Divergence is then attributable
    // purely to the backend (kernels / fusion / reduction order), not to a
    // precision drop or a host↔device move. Switch ref to CPU·FP32 manually to
    // deliberately mix precision back in — the compare bar will flag it.
    ref: cfg("CUDA", "FP16", "GPU"),
    cands: [
      cfg("DirectML", "FP16", "GPU"),
      cfg("TensorRT", "FP16", "GPU"),
    ],
  },
};

/* execution-provider palette (categorical, for the "Execution provider" color mode) */
const EP_COLORS = {
  CPU:      "oklch(0.66 0.020 250)",
  CUDA:     "oklch(0.80 0.165 150)",
  DirectML: "oklch(0.80 0.120 205)",
  TensorRT: "oklch(0.82 0.150 78)",
  QNN:      "oklch(0.76 0.140 305)",
  OpenVINO: "oklch(0.74 0.130 255)",
};

/* the one node where DirectML's fast-math fusion blows up numerically */
const HOT_ID = "/layer3/layer3.1/conv2/Conv";

/* op-fallback map: certain ops aren't supported on a given EP and silently drop
   back to the CPU EP — a perf cliff (host<->device copy) AND a numerical tell
   (that node ends up bit-near the CPU reference). Keyed by EP. */
function epFallback(ep, id) {
  if (ep === "TensorRT" && id === "gap") return true;     // GlobalAveragePool → CPU
  if (ep === "DirectML" && id === "output") return true;  // Softmax(1000) → CPU
  if (ep === "QNN" && (id === "gap" || id === "output")) return true; // HTP graph splits at the tail → CPU
  return false;
}

/* precision-distance scale: divergence induced purely by changing precision,
   normalized so FP32→INT8 == 1.0 (i.e. == the authored node.diff). */
const PREC_ERR = { FP32: 0, BF16: 0.11, FP16: 0.085, FP8: 0.45, INT8: 1.0 };
function precScale(refP, candP) {
  if (refP === candP) return 0;
  return Math.abs((PREC_ERR[candP] ?? 0.2) - (PREC_ERR[refP] ?? 0.2));
}
/* backend-induced divergence floor: even at equal precision, a different EP
   diverges (TF32 accumulation, fast-math, fusion, reduction order). */
function backendScale(refEp, candEp) {
  if (refEp === candEp) return 0;
  return { CPU: 0.02, CUDA: 0.05, DirectML: 0.07, TensorRT: 0.06, QNN: 0.085, OpenVINO: 0.04 }[candEp] ?? 0.05;
}

/* leaf-node divergence for an arbitrary (ref, cand) pair — works for ANY dropdown
   combination, so the user can move one axis at a time and watch it respond. */
function leafDivergence(node, ref, cand) {
  if (sameCfg(ref, cand)) return 0;
  // fallback to CPU EP ⇒ ran the same path as a CPU reference ⇒ ~0 divergence
  if (epFallback(cand.ep, node.id)) return 0.00002 + (hashStr(node.id + cand.ep) % 9) * 1e-6;
  // authored hotspot: DirectML FP16 fast-math fusion on the peak conv
  if (node.id === HOT_ID && cand.ep === "DirectML" && cand.precision === "FP16") return 0.0312;
  const base = node.diff || 0;             // authored sensitivity proxy
  const ps = precScale(ref.precision, cand.precision);
  const bs = backendScale(ref.ep, cand.ep);
  const scale = ps + bs;
  if (scale === 0) return 0;
  // pure same-EP FP32↔INT8 — the calibrated quantization run: report the authored
  // measurement EXACTLY (no jitter) so the top bar, graph, inspector and scatter agree
  if (bs === 0 && ps === 1.0) return base;
  const j = 0.85 + (hashStr(node.id + cand.ep + cand.precision) % 30) / 100;  // 0.85..1.15
  return +(base * scale * j).toFixed(4);
}

/* divergence of `node` for candidate index `i` (group = max over its subgraph) */
function candDiff(node, cmp, i) {
  const cand = cmp.cands[i];
  if (!cand) return 0;
  if (node.group && node.children) {
    let m = 0;
    for (const ch of node.children) m = Math.max(m, candDiff(NODE_BY_ID[ch.id] || ch, cmp, i));
    return m;
  }
  return leafDivergence(node, cmp.ref, cand);
}
/* divergence used for the heatmap: the focused candidate, or worst-among-candidates */
function nodeDiff(node, cmp) {
  if (cmp.focus === "worst") {
    let m = 0;
    for (let i = 0; i < cmp.cands.length; i++) m = Math.max(m, candDiff(node, cmp, i));
    return m;
  }
  return candDiff(node, cmp, (cmp.focus | 0));
}
/* the candidate index that owns the worst divergence at a node (for "worst" labels) */
function worstCandAt(node, cmp) {
  let bi = 0, bv = -1;
  for (let i = 0; i < cmp.cands.length; i++) { const v = candDiff(node, cmp, i); if (v > bv) { bv = v; bi = i; } }
  return bi;
}
/* global max divergence across all top-level nodes for the current comparison */
function cmpMax(cmp) {
  let m = 0;
  for (const n of NODES) m = Math.max(m, nodeDiff(n, cmp));
  return m || 1;
}

/* which EP a node actually executed on under a candidate (+ fallback flag) */
function nodeEP(node, cand) {
  if (!cand) return { ep: "CPU", fallback: false };
  if (epFallback(cand.ep, node.id)) return { ep: "CPU", fallback: true, intended: cand.ep };
  // a group may *contain* a fallback child even if the group itself didn't fall back
  let partial = false;
  if (node.group && node.children) partial = node.children.some((ch) => epFallback(cand.ep, ch.id));
  return { ep: cand.ep, fallback: false, partial };
}

/* per-config latency (ms) for a node. Reference = CPU FP32 = COST.cpu.
   GPU EPs are much faster; a fallback node pays CPU time + host<->device copy. */
function configLatency(node, cand) {
  const c = COST[node.id] || COST[node.parent] || { cpu: 0 };
  const cpu = c.cpu;
  if (cpu === 0) return 0;
  if (!cand || cand.ep === "CPU") {
    const f = cand && cand.precision === "INT8" ? 0.62 : cand && cand.precision === "FP16" ? 0.9 : 1.0;
    return +(cpu * f).toFixed(2);
  }
  if (epFallback(cand.ep, node.id)) return +(cpu + 0.22).toFixed(2);   // CPU exec + copy back
  let f = { TensorRT: 0.22, CUDA: 0.27, DirectML: 0.40, QNN: 0.30, OpenVINO: 0.62 }[cand.ep] ?? 0.40;
  if (cand.precision === "INT8") f *= 0.8;
  else if (cand.precision === "FP8") f *= 0.85;
  else if (cand.precision === "FP32") f *= 1.3;
  return +(cpu * f).toFixed(2);
}

/* math mode badge per config (what numerics the EP actually used) */
function mathMode(cand) {
  if (!cand) return "—";
  if (cand.ep === "CUDA")     return cand.precision === "FP16" ? "FP16 · TF32 accum" : cand.precision === "FP8" ? "FP8 e4m3 · TC accum" : "TF32";
  if (cand.ep === "DirectML") return cand.precision === "FP16" ? "FP16 fast-math" : cand.precision;
  if (cand.ep === "TensorRT") return cand.precision === "FP16" ? "FP16 · layer fusion" : cand.precision === "FP8" ? "FP8 · layer fusion" : cand.precision;
  if (cand.ep === "QNN")      return cand.precision === "INT8" ? "HTP fixed-point" : cand.precision + " · HTP";
  if (cand.ep === "OpenVINO") return cand.precision + " · oneDNN";
  return cand.precision; // CPU EP — IEEE
}

/* TOLERANCE verdict (backend mode) — NOT glued to latency. Reports whether the
   numerical divergence is acceptable and the probable cause. */
const TOL_LOW = 0.005, TOL_HIGH = 0.020;
function toleranceVerdict(node, cmp, i) {
  const cand = cmp.cands[i];
  const d = candDiff(node, cmp, i);
  const ep = nodeEP(node, cand);
  let cls, short;
  if (d < TOL_LOW)       { cls = "within";      short = "within tolerance"; }
  else if (d < TOL_HIGH) { cls = "investigate"; short = "investigate"; }
  else                   { cls = "out";         short = "out of tolerance"; }
  let cause;
  if (ep.fallback)                                          cause = `op unsupported on ${cand.ep} → ran on CPU fallback (≈ reference, but a perf cliff)`;
  else if (node.id === HOT_ID && cand.ep === "DirectML")    cause = "fast-math + different conv→add fusion on DirectML";
  else if (cand.ep === "TensorRT")                          cause = "aggressive layer fusion changes reduction order";
  else if (cand.ep === "CUDA")                              cause = "TF32→FP16 rounding + reduction order";
  else if (cand.ep === "QNN")                               cause = "fixed-point requantization between HTP ops";
  else if (cand.ep === "OpenVINO")                          cause = "oneDNN kernel selection + reduction order";
  else                                                      cause = "FP16 rounding + reduction order";
  return { cls, short, d, cause, ep: ep.ep, fallback: !!ep.fallback, intended: ep.intended };
}

/* divergence stats (maxAbs / meanAbs / cosine) from a single divergence value —
   used per-candidate in the Divergence section regardless of mode. */
function divStats(node, d) {
  const r = mulberry32(hashStr(node.id + ":" + d.toFixed(5)));
  const t = d / (GLOBAL_MAX || 1);
  const meanAbs = d * (0.12 + r() * 0.06);
  const cosine = d === 0 ? 1 : 1 - (t * t) * (0.0009 + r() * 0.0004);
  return { maxAbs: d.toFixed(4), meanAbs: meanAbs.toFixed(4), cosine: cosine.toFixed(4) };
}

/* top-1 prediction per config (synthetic but monotone: INT8 drops most,
   FP16 backends barely move, fallback-heavy candidates land near reference) */
function predOf(cand) {
  const base = 0.871;
  if (!cand || cand.ep === "CPU" && cand.precision === "FP32") return { cls: "coffee mug", p: base };
  let p = base;
  if (cand.precision === "INT8") p -= 0.037;
  else if (cand.precision === "FP8") p -= 0.021;
  else if (cand.precision === "BF16") p -= 0.004;
  else if (cand.precision === "FP16") p -= (cand.ep === "DirectML" ? 0.012 : cand.ep === "CUDA" ? 0.006 : 0.004);
  if (cand.ep === "QNN") p -= 0.003;
  return { cls: "coffee mug", p: +p.toFixed(3) };
}

/* ---- shape-agnostic tensor viewer: pick a renderer from the tensor shape ----
   1×C×H×W (H,W>1)  → feature-map grid (conv activations)
   Gemm/MatMul      → matrix heatmap (weight matrix)
   heads×seq×seq    → attention map (per-head)
   1×N or 1×C×1×1   → 1-D bar/line (logits / pooled vector / probabilities) */
function parseShape(shape) { return shape.split("×").map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n)); }
function vizKindOf(node) {
  const d = parseShape(node.shape);
  if (node.op === "Gemm" || node.op === "MatMul") return "matrix";
  if (d.length === 3 && d[1] === d[2]) return "attention";      // heads × seq × seq
  if (d.length === 4 && d[2] > 1 && d[3] > 1) return "featuremap"; // 1×C×H×W
  if (d.length === 4 && d[2] === 1 && d[3] === 1) return "vector1d"; // 1×C×1×1 pooled
  if (d.length === 2) return node.op === "Softmax" ? "vector1d" : "vector1d";
  return "featuremap";
}
const VIZ_LABEL = {
  featuremap: "feature maps · 1×C×H×W",
  matrix:     "weight matrix · in × out",
  attention:  "attention · heads × seq × seq",
  vector1d:   "1-D vector",
};

/* normalized [0,1] metric value for heatmap coloring under each COLOR-BY mode.
   `cmp` carries mode/ref/cands/focus; `mode` is the color-by axis. */
function metricT(node, cmp, mode) {
  if (mode === "divergence") {
    return nodeDiff(node, cmp) / cmpMax(cmp);
  }
  if (mode === "latency") {
    // colour by speedup of the focused candidate vs CPU reference: faster = cool,
    // slower-than-CPU (fallback) = hot.
    const i = cmp.focus === "worst" ? 0 : (cmp.focus | 0);
    const c = COST[node.id] || COST[node.parent] || { cpu: 0 };
    if (!c.cpu) return 0;
    const lat = node.group && node.children
      ? node.children.reduce((a, ch) => a + configLatency(NODE_BY_ID[ch.id] || ch, cmp.cands[i]), 0) || configLatency(node, cmp.cands[i])
      : configLatency(node, cmp.cands[i]);
    const saved = (c.cpu - lat) / c.cpu;        // 1 = free, 0 = no gain, <0 = slower
    return Math.min(1, Math.max(0, 1 - saved)); // big save → cool
  }
  if (mode === "tradeoff") {
    const { saved } = costOf(node);
    const ratio = (node.diff || 0) / (saved + 0.04);
    return Math.min(1, ratio / 0.62);
  }
  // ep mode is categorical — handled by the graph directly, not via a heat ramp
  return nodeDiff(node, cmp) / cmpMax(cmp);
}

/* ---- GROUP granularity helpers ----
   A folded group (layerN) is NOT a single tensor: it has no one scale/zero-point,
   and its `diff` is the MAX abs diff of its hottest child, not the block output.
   So group views must show AGGREGATES + an explicit peak node — never a single
   per-tensor scale, and never a verdict that glues block time to peak-node Δ. */
function peakChildOf(node) {
  if (!node.children || !node.children.length) return null;
  let best = node.children[0];
  for (const c of node.children) if (c.diff > best.diff) best = c;
  return NODE_BY_ID[best.id] || best;
}
const GROUP_TOL = 0.030; // divergence tolerance used to count "hot" nodes in a block
function groupAgg(node) {
  const kids = node.children || [];
  const diffs = kids.map((k) => k.diff);
  const meanDiff = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : 0;
  const overT = diffs.filter((d) => d >= GROUP_TOL).length;
  return {
    nNodes: node.nInner || kids.length, // total folded ONNX nodes
    nRaw: kids.length,                  // representative subset shown
    maxDiff: node.diff,                 // = peak child's diff
    meanDiff,
    overT,
    peak: peakChildOf(node),
  };
}
/* block-level verdict — reports block time + WHERE the peak is, and explicitly
   defers the keep-FP32 call to the node (so block time is never glued to one node's Δ) */
function groupVerdict(node) {
  const { saved } = costOf(node);
  const agg = groupAgg(node);
  const peakName = agg.peak ? agg.peak.shortName : "—";
  return {
    cls: "group",
    saved,
    line: `whole block +${saved.toFixed(2)} ms over ${agg.nNodes} ops · peak drift Δ ${agg.maxDiff.toFixed(4)} sits at ${peakName} — drill into that node to decide FP32 vs INT8`,
  };
}

/* verdict: should this node stay FP32 or is INT8 a safe win? */
function verdictOf(node, dOverride, savedOverride) {
  const saved = savedOverride != null ? savedOverride : costOf(node).saved;
  const diff = (dOverride != null ? dOverride : node.diff) || 0;
  const ratio = diff / (saved + 0.04);
  if (!node.quant) return { cls: "na", short: "not quantized", line: "kept FP32 — op not quantizable" };
  if (ratio < 0.04) return { cls: "accept", short: "accept", line: `+${saved.toFixed(2)} ms saved · Δ ${diff.toFixed(4)} cost · clear win — quantize` };
  if (ratio < 0.12) return { cls: "marginal", short: "marginal", line: `+${saved.toFixed(2)} ms saved · Δ ${diff.toFixed(4)} cost · marginal — candidate to keep FP32` };
  return { cls: "reject", short: "keep FP32", line: `+${saved.toFixed(2)} ms saved · Δ ${diff.toFixed(4)} cost · poor trade — keep FP32` };
}

/* SPEED ↔ ACCURACY scatter: x = Δt saved by INT8 (ms), y = divergence Δ.
   class drives color; `hot` marks the inspected outlier. */
const SCATTER = [
  { id: "conv1",                         label: "conv1",          x: 1.52, y: 0.0009, cls: "good" },
  { id: "layer1",                        label: "layer1",         x: 0.80, y: 0.0046, cls: "good" },
  { id: "layer2",                        label: "layer2",         x: 0.55, y: 0.0123, cls: "good" },
  { id: "fc",                            label: "fc",             x: 0.12, y: 0.0231, cls: "marginal" },
  { id: "gap",                           label: "gap",            x: 0.05, y: 0.0291, cls: "reject" },
  { id: "layer4",                        label: "layer4",         x: 0.30, y: 0.0388, cls: "reject" },
  { id: "/layer3/layer3.1/conv2/Conv",   label: "layer3.1/conv2", x: 0.31, y: 0.0419, cls: "reject", hot: true },
];
/* acceptance threshold: divergence you'll tolerate scales with time saved.
   points BELOW/right of the line are worth quantizing; ABOVE/left are not. */
const ACCEPT_SLOPE = 0.026; // Δ per ms

const PREDICTIONS = {
  cpu: { cls: "coffee mug", p: 0.871, runners: [["cup", 0.052], ["espresso", 0.021]] },
  npu: { cls: "coffee mug", p: 0.834, runners: [["cup", 0.069], ["espresso", 0.028]] },
};

Object.assign(window, {
  mulberry32, hashStr, smoothField, quantizeField,
  sampleRamp, heatColor, INFERNO, ICE, DIFFRAMP, HEAT_RAMPS,
  NODES, NODE_BY_ID, GLOBAL_MAX, nodeStats, histogram, PREDICTIONS, rgbStr,
  COST, MAX_SAVED, PROBE_PENALTY_MS, costOf, metricT, verdictOf, SCATTER, ACCEPT_SLOPE,
  peakChildOf, groupAgg, groupVerdict, GROUP_TOL,
  /* universal config model */
  BACKENDS, PRECISIONS, DEVICES, EP_INFO, epDevice, cfg, sameCfg, cfgLabel, PRESETS, EP_COLORS, HOT_ID,
  epFallback, precScale, backendScale, leafDivergence, candDiff, nodeDiff, worstCandAt, cmpMax,
  nodeEP, configLatency, mathMode, TOL_LOW, TOL_HIGH, toleranceVerdict, divStats, predOf,
  parseShape, vizKindOf, VIZ_LABEL,
});
