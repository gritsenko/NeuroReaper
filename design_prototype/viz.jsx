/* viz.jsx — canvas-based visualizations: feature maps, diff tiles, histogram, depth chart */
const { useRef, useEffect } = React;

/* paint an NxN scalar field with a color ramp onto a canvas */
function paintField(canvas, field, N, ramp) {
  const ctx = canvas.getContext("2d");
  canvas.width = N; canvas.height = N;
  const img = ctx.createImageData(N, N);
  for (let i = 0; i < N * N; i++) {
    const c = sampleRamp(ramp, field[i]);
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/* single feature-map channel — heterogeneous "personalities" so dead/saturated
   channels are visibly different (that's the diagnostic value of this view) */
const FMAP_TYPES = ["oriented", "textured", "dead", "sparse", "oriented", "saturated", "textured", "sparse", "oriented", "dead", "textured", "blob"];

function channelField(node, channel, idx, seedSalt, N) {
  const seed = (hashStr(node.id + ":ch" + channel) ^ (seedSalt || 0)) >>> 0;
  const base = smoothField(seed, N);
  const rnd = mulberry32(seed ^ 0x1234);
  const type = FMAP_TYPES[idx % FMAP_TYPES.length];
  const out = new Float32Array(N * N);
  if (type === "dead") {
    for (let i = 0; i < N * N; i++) out[i] = base[i] * 0.05 + 0.015;
  } else if (type === "saturated") {
    for (let i = 0; i < N * N; i++) out[i] = Math.min(1, base[i] * 0.22 + 0.84);
  } else if (type === "sparse") {
    const th = 0.74;
    for (let i = 0; i < N * N; i++) out[i] = base[i] > th ? (base[i] - th) / (1 - th) : base[i] * 0.06;
  } else if (type === "textured") {
    for (let i = 0; i < N * N; i++) { const r = rnd(); out[i] = Math.min(1, Math.max(0, 0.5 * base[i] + 0.5 * r)); }
  } else if (type === "blob") {
    const cx = 0.3 + rnd() * 0.4, cy = 0.3 + rnd() * 0.4, rad = 0.18 + rnd() * 0.12;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const dx = x / N - cx, dy = y / N - cy; const d = Math.sqrt(dx * dx + dy * dy);
      out[y * N + x] = Math.max(0, 1 - d / rad) * 0.9 + base[y * N + x] * 0.1;
    }
  } else { // oriented (stripes)
    for (let i = 0; i < N * N; i++) out[i] = base[i];
  }
  return { field: out, type };
}

function FeatureMap({ node, channel, idx, ramp, seedSalt }) {
  const ref = useRef(null);
  const N = Math.max(7, node.grid);
  const { field, type } = channelField(node, channel, idx, seedSalt, N);
  useEffect(() => {
    if (!ref.current) return;
    paintField(ref.current, field, N, ramp === "ice" ? ICE : INFERNO);
  }, [node.id, channel, ramp, seedSalt, idx]);
  const flag = type === "dead" ? "dead" : type === "saturated" ? "sat" : null;
  return (
    <div className={"fmap" + (flag ? " flag-" + flag : "")}>
      <canvas ref={ref}></canvas>
      <span className="ch">c{String(channel).padStart(3, "0")}</span>
      {flag && <span className={"fflag " + flag}>{flag === "dead" ? "DEAD" : "SAT"}</span>}
    </div>
  );
}

/* divergence trio tile: variant = cpu | npu | diff.
   `mag` optionally overrides the divergence magnitude (per-candidate), so the same
   node can render different |Δ| intensities under different backends. */
function DivTile({ node, variant, seedSalt, mag }) {
  const ref = useRef(null);
  const m = (mag == null ? node.diff : mag);
  useEffect(() => {
    if (!ref.current) return;
    const N = Math.max(7, node.grid);
    const base = smoothField(hashStr(node.id + ":divch") ^ (seedSalt || 0), N);
    if (variant === "cpu") {
      paintField(ref.current, base, N, INFERNO);
    } else if (variant === "npu") {
      const q = quantizeField(base, N, 16, m * 1.4, hashStr(node.id) ^ (seedSalt || 0));
      paintField(ref.current, q, N, INFERNO);
    } else {
      const q = quantizeField(base, N, 16, m * 1.4, hashStr(node.id) ^ (seedSalt || 0));
      const d = new Float32Array(N * N);
      for (let i = 0; i < N * N; i++) d[i] = Math.min(1, Math.abs(base[i] - q[i]) / (m * 2.2 + 0.02));
      paintField(ref.current, d, N, DIFFRAMP);
    }
  }, [node.id, variant, seedSalt, m]);
  return <div className={"div-tile" + (variant === "diff" ? " diff" : "")}><canvas ref={ref}></canvas></div>;
}

/* distribution histogram — FP32 reference (filled, faint) + INT8 candidate (stepped, accent) */
function Histogram({ node, accent, quant, seedSalt }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);
    const { fp, int8 } = histogram(node, 44, seedSalt);
    const bw = W / fp.length;
    // zero line
    const zx = W * 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(zx, 0); ctx.lineTo(zx, H); ctx.stroke();

    // FP32 reference — faint filled area
    ctx.fillStyle = "rgba(190,200,215,0.16)";
    ctx.strokeStyle = "rgba(190,200,215,0.55)"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < fp.length; i++) {
      const x = i * bw + bw / 2; const y = H - fp[i] * (H - 4);
      ctx.lineTo(x, y);
    }
    ctx.lineTo(W, H); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    for (let i = 0; i < fp.length; i++) {
      const x = i * bw + bw / 2; const y = H - fp[i] * (H - 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // INT8 candidate — accent stepped bars
    if (quant) {
      for (let i = 0; i < int8.length; i++) {
        const h = int8[i] * (H - 4);
        const x = i * bw;
        ctx.fillStyle = accent;
        ctx.globalAlpha = 0.30 + 0.5 * int8[i];
        ctx.fillRect(x + 0.5, H - h, Math.max(1, bw - 1), h);
      }
      ctx.globalAlpha = 1;
    }
  }, [node.id, accent, quant, seedSalt]);
  return <canvas ref={ref} style={{ width: "100%", height: "56px", display: "block" }}></canvas>;
}

/* ============================================================================
   SHAPE-AGNOSTIC RENDERERS — the Visualization section dispatches on tensor shape
   ============================================================================ */

/* 2-D weight / activation matrix → heatmap (Gemm / MatMul) */
function MatrixView({ node, seedSalt, ramp }) {
  const ref = useRef(null);
  const dims = parseShape(node.shape);
  // fc: 1×1000 output, weight matrix is in(512) × out(1000) — render a representative block
  const ROWS = 80, COLS = 120;
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const f = smoothField(hashStr(node.id + ":W") ^ (seedSalt || 0), Math.max(ROWS, COLS));
    cv.width = COLS; cv.height = ROWS;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(COLS, ROWS);
    const R = ramp === "ice" ? ICE : INFERNO;
    for (let y = 0; y < ROWS; y++) for (let x = 0; x < COLS; x++) {
      // signed weights: center the field, fold into a diverging look via |v|
      const v = f[(y % 80) * Math.max(ROWS, COLS) + (x % 120)];
      const c = sampleRamp(R, v);
      const i = (y * COLS + x) * 4;
      img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [node.id, seedSalt, ramp]);
  const inDim = dims.length === 2 ? 512 : (dims[1] || 512);
  const outDim = dims.length === 2 ? dims[1] : (dims[1] || 1000);
  return (
    <div className="mtx-wrap">
      <div className="mtx-axis-y mono">in {inDim} ↓</div>
      <div className="mtx-canvas">
        <canvas ref={ref}></canvas>
      </div>
      <div className="mtx-axis-x mono">out {outDim} →</div>
      <div className="mtx-note">weight matrix · row = input feature, col = output logit · brighter = larger |w|</div>
    </div>
  );
}

/* attention map: heads × seq × seq, one head at a time */
function AttentionView({ node, seedSalt, ramp }) {
  const dims = parseShape(node.shape);
  const heads = dims[0] || 8, seq = dims[1] || 64;
  const [head, setHead] = React.useState(0);
  const ref = useRef(null);
  const N = Math.min(seq, 96);
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const rnd = mulberry32(hashStr(node.id + ":att" + head) ^ (seedSalt || 0));
    cv.width = N; cv.height = N;
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(N, N);
    const R = ramp === "ice" ? ICE : INFERNO;
    for (let y = 0; y < N; y++) {
      // softmax-ish row: diagonal + a couple attention sinks
      const sink = Math.floor(rnd() * N);
      let row = new Float32Array(N), sum = 0;
      for (let x = 0; x < N; x++) {
        const diag = Math.exp(-Math.abs(x - y) / (2 + head)); 
        const s = x === sink ? 0.6 : 0;
        row[x] = diag + s + rnd() * 0.05; sum += row[x];
      }
      for (let x = 0; x < N; x++) {
        const v = row[x] / sum * N * 0.5;
        const c = sampleRamp(R, Math.min(1, v));
        const i = (y * N + x) * 4;
        img.data[i] = c[0]; img.data[i + 1] = c[1]; img.data[i + 2] = c[2]; img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [node.id, head, seedSalt, ramp]);
  return (
    <div className="att-wrap">
      <div className="att-heads">
        {Array.from({ length: heads }).map((_, h) => (
          <button key={h} className={"att-head" + (h === head ? " active" : "")} onClick={() => setHead(h)}>h{h}</button>
        ))}
      </div>
      <div className="att-canvas"><canvas ref={ref}></canvas>
        <span className="att-ax q mono">query →</span>
        <span className="att-ax k mono">key ↓</span>
      </div>
      <div className="mtx-note">head {head} / {heads} · row attends over keys · diagonal = local, bright column = attention sink</div>
    </div>
  );
}

/* 1-D vector → bars (logits / probabilities / pooled vector). Highlights top-1. */
function Vector1D({ node, seedSalt }) {
  const dims = parseShape(node.shape);
  const len = dims[dims.length - 1] || (dims[1] || 64);
  const isProb = node.op === "Softmax";
  const SHOWN = Math.min(48, len);
  const rnd = mulberry32(hashStr(node.id + ":vec") ^ (seedSalt || 0));
  // synth a peaked distribution for softmax, signed-ish for logits/pooled
  const vals = [];
  let peakIdx = Math.floor(rnd() * SHOWN * 0.6) + 2;
  for (let i = 0; i < SHOWN; i++) {
    let v;
    if (isProb) { v = Math.exp(-Math.abs(i - peakIdx) / 3) * (0.7 + rnd() * 0.3); }
    else { v = (rnd() - 0.45) * (0.6 + rnd() * 0.9); }
    vals.push(v);
  }
  const mx = Math.max(...vals.map(Math.abs)) || 1;
  const top = vals.indexOf(Math.max(...vals));
  return (
    <div className="vec-wrap">
      <div className={"vec-bars" + (isProb ? " baseline" : " centered")}>
        {vals.map((v, i) => {
          const neg = v < 0;
          // probabilities grow up from a 0 baseline (full height); signed logits
          // diverge from a center line (half height each side, never overflowing)
          const h = isProb ? (Math.abs(v) / mx * 100) : (Math.abs(v) / mx * 50);
          const style = isProb
            ? { height: h + "%", bottom: 0 }
            : { height: h + "%", [neg ? "top" : "bottom"]: "50%" };
          return (
            <div className="vec-col" key={i} title={`[${i}] ${v.toFixed(3)}`}>
              <div className="vec-cell">
                <i className={"vbar" + (i === top ? " top" : "") + (neg ? " neg" : "")}
                   style={style}></i>
              </div>
            </div>
          );
        })}
      </div>
      <div className="vec-foot mono">
        <span>{isProb ? "probabilities" : "values"} · {len} dims ({SHOWN} shown)</span>
        <span className="vec-top">argmax @ idx {top}{isProb ? ` · p ${vals[top].toFixed(3)}` : ""}</span>
      </div>
    </div>
  );
}

/* dispatcher: choose the renderer from tensor shape */
function ShapeViz({ node, channels, ramp, seedSalt, maxCh, chList }) {
  const kind = vizKindOf(node);
  if (kind === "matrix")    return <MatrixView node={node} seedSalt={seedSalt} ramp={ramp} />;
  if (kind === "attention") return <AttentionView node={node} seedSalt={seedSalt} ramp={ramp} />;
  if (kind === "vector1d")  return <Vector1D node={node} seedSalt={seedSalt} />;
  // feature maps (default)
  return (
    <div className="fmap-grid" style={{ gridTemplateColumns: `repeat(${Math.min(4, channels)}, 1fr)` }}>
      {chList.map((ch, i) => (
        <FeatureMap key={i} node={node} channel={ch} idx={i} ramp={ramp} seedSalt={seedSalt} />
      ))}
    </div>
  );
}

Object.assign(window, { FeatureMap, DivTile, Histogram, paintField, MatrixView, AttentionView, Vector1D, ShapeViz });
