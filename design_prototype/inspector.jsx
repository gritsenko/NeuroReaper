/* inspector.jsx — right tensor inspector (mode-aware) + bottom strip */

/* ============================================================================
   TENSOR INSPECTOR — dispatches on group vs leaf, then on preset mode.
   The engine is one config model; only the INTERPRETATION changes per mode.
   ============================================================================ */
function Inspector({ node, cmp, fmapRamp, channels, seedSalt, onSelect }) {
  if (node.group) return <GroupInspector node={node} cmp={cmp} seedSalt={seedSalt} onSelect={onSelect} />;

  const maxCh = parseInt(node.shape.split("×")[1] || "64", 10) || 64;
  const chList = [];
  for (let i = 0; i < channels; i++) chList.push(Math.floor((i + 0.5) * maxCh / channels) % maxCh + (i % 7) * 3);
  const title = node.shortName || node.name;
  const vizKind = vizKindOf(node);

  return (
    <div className="col right" data-screen-label="Tensor inspector">
      <div className="panel-head">
        <span className="title">Tensor Inspector</span>
        <span className="meta">{cmp.mode === "quant" ? "quantization" : "backend divergence"}</span>
      </div>
      <div className="scroll">

        {/* ---- 01 TENSOR ---- */}
        <div className="insp-sec">
          <div className="sec-head">
            <span className="ix">01</span><span className="nm">Tensor</span>
            <span className="pill">{node.op}</span>
          </div>
          <div className="node-title">{title}</div>
          <div className="node-sub">shape {node.shape} · {node.out}</div>
          {node.shortName && <div className="node-path mono">{node.name}</div>}

          {cmp.mode === "quant"
            ? <QuantTensor node={node} seedSalt={seedSalt} />
            : <ExecTable node={node} cmp={cmp} />}
        </div>

        {/* ---- 02 VISUALIZATION (shape-agnostic) ---- */}
        <div className="insp-sec">
          <div className="sec-head">
            <span className="ix">02</span><span className="nm">Visualization</span>
            <span className="pill">{VIZ_LABEL[vizKind]}</span>
          </div>
          <div className="fmap-note">
            {vizKind === "featuremap" && <>activation channels · scan for <b>dead</b> &amp; <b>saturated</b> maps</>}
            {vizKind === "matrix" && <>2-D operand · rendered as a <b>matrix heatmap</b>, not feature maps</>}
            {vizKind === "attention" && <>attention tensor · per-head <b>query × key</b> map</>}
            {vizKind === "vector1d" && <>1-D tensor · rendered as <b>bars</b> (logits / probabilities)</>}
          </div>
          <ShapeViz node={node} channels={channels} ramp={fmapRamp} seedSalt={seedSalt} maxCh={maxCh} chList={chList} />
        </div>

        {/* ---- 03 DIVERGENCE ---- */}
        {cmp.mode === "quant"
          ? <QuantDivergence node={node} cmp={cmp} seedSalt={seedSalt} />
          : <BackendDivergence node={node} cmp={cmp} seedSalt={seedSalt} />}

        {/* ---- 04 COST / LATENCY ---- */}
        {cmp.mode === "quant"
          ? <QuantCost node={node} cmp={cmp} />
          : <LatencySection node={node} cmp={cmp} />}
      </div>
    </div>
  );
}

/* ---------------- QUANT MODE PIECES ---------------- */
function QuantTensor({ node, seedSalt }) {
  const s = nodeStats(node);
  const q = node.quant;
  return (
    <>
      {q && (
        <div className="probe-badge mono">
          <span className="pdot"></span>probe active · fusion broken · +{PROBE_PENALTY_MS.toFixed(1)} ms
        </div>
      )}
      <div className="cmp-head">
        <span className="cmp-c0">statistic</span>
        <span className="cmp-c1 ref">CPU · FP32 <em>ref</em></span>
        <span className="cmp-c2 cand">{q ? <>INT8 <em>cand</em></> : <em className="na">— not quantized</em>}</span>
      </div>
      <div className="cmp-grid">
        {[["min", s.min, s.min8], ["max", s.max, s.max8], ["mean", s.mean, s.mean8], ["std", s.std, s.std8]].map(([k, a, b]) => (
          <div className="cmp-row" key={k}>
            <span className="cmp-k">{k}</span>
            <span className="cmp-v ref mono">{a}</span>
            <span className="cmp-v cand mono">{q ? b : "—"}</span>
          </div>
        ))}
      </div>
      {q && (
        <div className="quant-params mono">
          <span><span className="qk">scale</span> {s.scale}</span>
          <span><span className="qk">zero-pt</span> {s.zeroPoint}</span>
          <span><span className="qk">levels</span> {s.levels}</span>
        </div>
      )}
      <div className="hist-wrap">
        <div className="cap">
          <span className="up">Value distribution</span>
          <span className="hist-legend">
            <span className="hl-ref"><i></i>FP32</span>
            {q && <span className="hl-int8"><i></i>INT8</span>}
          </span>
        </div>
        <Histogram node={node} accent="var(--accent)" quant={q} seedSalt={seedSalt} />
      </div>
    </>
  );
}

/* divergence vs the LIVE candidate config — d comes from candDiff, not the
   static authored stats, so editing the compare bar updates this section too */
function QuantDivergence({ node, cmp, seedSalt }) {
  const cand = cmp.cands[0];
  const d = candDiff(node, cmp, 0);
  const st = divStats(node, d);
  const refLbl = cmp.ref.precision, candLbl = cand ? cand.precision : "INT8";
  return (
    <div className="insp-sec key">
      <div className="sec-head keytitle">
        <span className="ix">03</span><span className="nm">Divergence</span>
        <span className="pill" style={{ borderColor: "var(--accent-line)", color: "var(--accent)" }}>{refLbl} ↔ {candLbl}</span>
      </div>
      <div className="div-pair">
        <div className="div-cell">
          <div className="vlbl cpu"><span className="d"></span>{refLbl}<span className="x">ref</span></div>
          <DivTile node={node} variant="cpu" seedSalt={seedSalt} />
        </div>
        <div className="div-cell">
          <div className="vlbl npu"><span className="d"></span>{candLbl}<span className="x">cand</span></div>
          <DivTile node={node} variant="npu" seedSalt={seedSalt} mag={d} />
        </div>
        <div className="div-cell">
          <div className="vlbl diff"><span className="d"></span>|Δ|<span className="x">abs</span></div>
          <DivTile node={node} variant="diff" seedSalt={seedSalt} mag={d} />
        </div>
      </div>
      <div className="div-nums">
        <div className="n peak"><div className="k">max abs diff</div><div className="v">{st.maxAbs}</div></div>
        <div className="n"><div className="k">mean abs diff</div><div className="v">{st.meanAbs}</div></div>
        <div className="n"><div className="k">cosine sim</div><div className="v">{st.cosine}</div></div>
      </div>
      <div className="footnote">{candLbl === "INT8" ? "INT8 candidate output dequantized before compare" : "candidate output upcast to FP32 before compare"}</div>
    </div>
  );
}

function QuantCost({ node, cmp }) {
  const q = node.quant;
  if (!q) return null;
  const cost = costOf(node);
  const cand = cmp.cands[0];
  const candLbl = cand ? cand.precision : "INT8";
  // default calibrated run (CPU·INT8) uses authored measurements; any other
  // config falls back to the latency model so the bars stay live
  const isDefault = cand && cand.ep === "CPU" && cand.precision === "INT8";
  const candMs = isDefault ? cost.npu : configLatency(node, cand);
  const saved = +(cost.cpu - candMs).toFixed(2);
  const d = candDiff(node, cmp, 0);
  const vd = verdictOf(node, d, saved);
  return (
    <div className="insp-sec cost-sec">
      <div className="sec-head">
        <span className="ix">04</span><span className="nm">Cost</span>
        <span className="pill">Δt saved by {candLbl}</span>
      </div>
      <div className="cost-grid">
        <div className="cost-row">
          <span className="ck ref">{cmp.ref.ep} · {cmp.ref.precision}</span>
          <span className="cbar"><i style={{ width: "100%", background: "var(--tx-1)" }}></i></span>
          <span className="cv mono">{cost.cpu.toFixed(2)} ms</span>
        </div>
        <div className="cost-row">
          <span className="ck cand">{candLbl}</span>
          <span className="cbar"><i style={{ width: (cost.cpu ? (candMs / cost.cpu * 100) : 0) + "%", background: "var(--accent)" }}></i></span>
          <span className="cv mono accent">{candMs.toFixed(2)} ms</span>
        </div>
        <div className="cost-row saved">
          <span className="ck">Δt saved</span>
          <span className="cbar"><i style={{ width: (cost.cpu ? (Math.max(0, saved) / cost.cpu * 100) : 0) + "%", background: "var(--heat-0)" }}></i></span>
          <span className="cv mono pos">{saved.toFixed(2)} ms</span>
        </div>
      </div>
      <div className={"verdict-line " + vd.cls}>
        <span className="vl-tag">Quantize verdict</span>
        <span className="vl-body mono">{vd.line}</span>
      </div>
      <div className="twopass-note mono">
        <span className="tp-dot"></span>
        Δt from the fused timing run · Δ from the instrumented run — two passes, not one
      </div>
    </div>
  );
}

/* ---------------- BACKEND MODE PIECES ---------------- */
/* per-config execution: which EP actually ran the node + math mode */
function ExecTable({ node, cmp }) {
  const rows = [{ cfg: cmp.ref, isRef: true, tag: "ref", i: -1 }]
    .concat(cmp.cands.map((c, i) => ({ cfg: c, isRef: false, tag: CAND_LETTERS[i], i })));
  return (
    <>
      <div className="exec-cap up" style={{ marginTop: 12 }}>Execution · where each config ran this node</div>
      <div className="exec-table">
        {rows.map((r, k) => {
          const e = nodeEP(node, r.cfg);
          const col = r.isRef ? "var(--tx-1)" : (EP_COLORS[r.cfg.ep] || "var(--accent)");
          return (
            <div className={"exec-row" + (e.fallback ? " fb" : "")} key={k}>
              <span className="exec-tag" style={{ color: col }}>{r.tag}</span>
              <span className="exec-cfg mono">{cfgLabel(r.cfg)}</span>
              <span className={"ep-badge mono" + (e.fallback ? " fb" : e.partial ? " partial" : "")} style={!e.fallback ? { borderColor: col, color: col } : null}>
                {e.fallback ? "⚠ CPU fallback" : e.ep}{e.partial ? " ·partial" : ""}
              </span>
              <span className="exec-math mono">{mathMode(r.cfg)}</span>
            </div>
          );
        })}
      </div>
      {rows.some((r) => nodeEP(node, r.cfg).fallback) && (
        <div className="exec-note mono"><span className="tp-dot"></span>fallback = op unsupported on that EP → ran on CPU (≈ reference numerically, but a host↔device perf cliff)</div>
      )}
    </>
  );
}

function CandDivCard({ node, cmp, i, seedSalt }) {
  const cand = cmp.cands[i];
  const d = candDiff(node, cmp, i);
  const st = divStats(node, d);
  const tv = toleranceVerdict(node, cmp, i);
  const col = EP_COLORS[cand.ep] || "var(--accent)";
  return (
    <div className="canddiv" style={{ "--cc": col }}>
      <div className="canddiv-head">
        <span className="cd-dot"></span>
        <span className="cd-name">cand {CAND_LETTERS[i]}</span>
        <span className="cd-cfg mono">{cfgLabel(cand)}</span>
        {tv.fallback && <span className="ep-badge mono fb">⚠ CPU fallback</span>}
      </div>
      <div className="canddiv-body">
        <div className="div-cell">
          <div className="vlbl diff"><span className="d"></span>|Δ|<span className="x">vs ref</span></div>
          <DivTile node={node} variant="diff" seedSalt={seedSalt} mag={d} />
        </div>
        <div className="canddiv-nums">
          <div className="cn"><span className="k">max abs</span><span className="v">{st.maxAbs}</span></div>
          <div className="cn"><span className="k">mean abs</span><span className="v">{st.meanAbs}</span></div>
          <div className="cn"><span className="k">cosine</span><span className="v">{st.cosine}</span></div>
        </div>
      </div>
      <div className={"tol-verdict " + tv.cls}>
        <span className="tv-tag">{tv.short}</span>
        <span className="tv-cause">{tv.cause}</span>
      </div>
    </div>
  );
}

function BackendDivergence({ node, cmp, seedSalt }) {
  return (
    <div className="insp-sec key">
      <div className="sec-head keytitle">
        <span className="ix">03</span><span className="nm">Divergence · vs gold reference</span>
        <span className="pill" style={{ borderColor: "var(--accent-line)", color: "var(--accent)" }}>{cmp.cands.length} cand{cmp.cands.length > 1 ? "s" : ""}</span>
      </div>
      <div className="ref-tile-row">
        <div className="div-cell" style={{ maxWidth: 96 }}>
          <div className="vlbl cpu"><span className="d"></span>{cmp.ref.ep}<span className="x">{cmp.ref.precision} gold</span></div>
          <DivTile node={node} variant="cpu" seedSalt={seedSalt} />
        </div>
        <div className="ref-tile-note mono">
          activation under the <b>gold reference</b> ({cfgLabel(cmp.ref)}). each candidate below shows |Δ| against it — verdict is a <b>tolerance</b> call, not a latency one.
        </div>
      </div>
      {cmp.cands.map((_, i) => (
        <CandDivCard key={i} node={node} cmp={cmp} i={i} seedSalt={seedSalt} />
      ))}
    </div>
  );
}

/* latency by backend — perf only, explicitly decoupled from the divergence verdict */
function LatencySection({ node, cmp }) {
  const refLat = configLatency(node, cmp.ref);
  const cands = cmp.cands.map((c, i) => ({ c, i, lat: configLatency(node, c), fb: nodeEP(node, c).fallback }));
  const all = [refLat].concat(cands.map((x) => x.lat));
  const mx = Math.max(...all, 0.001);
  return (
    <div className="insp-sec lat-sec">
      <div className="sec-head">
        <span className="ix">04</span><span className="nm">Latency by backend</span>
        <span className="pill">perf · not the verdict</span>
      </div>
      <div className="cost-grid">
        <div className="cost-row">
          <span className="ck ref">{cmp.ref.ep} · {cmp.ref.precision}</span>
          <span className="cbar"><i style={{ width: (refLat / mx * 100) + "%", background: "var(--tx-1)" }}></i></span>
          <span className="cv mono">{refLat.toFixed(2)} ms</span>
        </div>
        {cands.map(({ c, i, lat, fb }) => (
          <div className="cost-row" key={i}>
            <span className="ck" style={{ color: EP_COLORS[c.ep] }}>{c.ep} · {c.precision}</span>
            <span className="cbar"><i style={{ width: (lat / mx * 100) + "%", background: EP_COLORS[c.ep] }}></i></span>
            <span className="cv mono" style={{ color: fb ? "var(--heat-4)" : EP_COLORS[c.ep] }}>{lat.toFixed(2)} ms{fb ? " ⚠" : ""}</span>
          </div>
        ))}
      </div>
      <div className="twopass-note mono">
        <span className="tp-dot"></span>
        wall time per node, per EP — a fallback node is <b style={{ color: "var(--heat-4)" }}>slower</b> (host↔device copy) yet numerically near the reference. speed and tolerance are independent signals.
      </div>
    </div>
  );
}

/* ============================================================================
   GROUP INSPECTOR — a folded module is not a tensor: no single scale/zero-point,
   block time never glued to one node's Δ. Aggregates + explicit peak node.
   ============================================================================ */
function GroupInspector({ node, cmp, seedSalt, onSelect }) {
  const agg = groupAgg(node);
  const peakIdx = cmp.focus === "worst" ? worstCandAt(node, cmp) : (cmp.focus | 0);
  // peak child for the focused/worst candidate
  let peak = agg.peak, pv = -1;
  for (const ch of node.children) { const full = NODE_BY_ID[ch.id] || ch; const v = candDiff(full, cmp, peakIdx); if (v > pv) { pv = v; peak = full; } }
  const blockMax = nodeDiff(node, cmp);

  return (
    <div className="col right" data-screen-label="Block inspector">
      <div className="panel-head">
        <span className="title">Block Inspector</span>
        <span className="meta">group · {cmp.mode === "quant" ? "quantization" : "backend"}</span>
      </div>
      <div className="scroll">
        <div className="insp-sec">
          <div className="sec-head">
            <span className="ix">01</span><span className="nm">Group</span>
            <span className="pill">{node.op}</span>
          </div>
          <div className="node-title">{node.name} · Residual block</div>
          <div className="node-sub">output {node.shape} · {node.out}</div>
          <div className="node-path mono">{agg.nNodes} ONNX nodes folded · {agg.nRaw} shown</div>

          <div className="group-disclaim">
            <span className="gd-mark mono">⚠ folded module</span>
            <span className="gd-body">
              No single tensor here — {cmp.mode === "quant"
                ? <><b>scale · zero-point · levels are per-tensor</b> and live on the nodes inside.</>
                : <><b>EP, math mode &amp; tolerance are per-node</b> — a block can mix backends and fallbacks.</>} Drill into a node to inspect it.
            </span>
          </div>

          <button className="peak-callout" onClick={() => onSelect(peak.id)}>
            <span className="pc-l">
              <span className="pc-tag mono">PEAK NODE{cmp.focus !== "worst" && cmp.mode === "backend" ? " · cand " + CAND_LETTERS[peakIdx] : ""}</span>
              <span className="pc-name mono">{peak.shortName}</span>
              <span className="pc-path mono">{peak.name}</span>
            </span>
            <span className="pc-r">
              <span className="pc-diff mono">Δ {pv >= 0 ? pv.toFixed(4) : peak.diff.toFixed(4)}</span>
              <span className="pc-go mono">drill in →</span>
            </span>
          </button>

          <div className="agg-cap up">Block aggregate · across {agg.nRaw} nodes{cmp.mode === "backend" && cmp.focus !== "worst" ? " · cand " + CAND_LETTERS[peakIdx] : cmp.mode === "backend" ? " · worst-of-N" : ""}</div>
          <div className="div-nums agg">
            <div className="n peak"><div className="k">peak Δ in block</div><div className="v">{blockMax.toFixed(4)}</div></div>
            <div className="n"><div className="k">mean Δ</div><div className="v">{agg.meanDiff.toFixed(4)}</div></div>
            <div className="n"><div className="k">nodes Δ ≥ {GROUP_TOL.toFixed(3)}</div><div className="v">{agg.overT}/{agg.nRaw}</div></div>
          </div>
        </div>

        {/* peak preview */}
        <div className="insp-sec key">
          <div className="sec-head keytitle">
            <span className="ix">02</span><span className="nm">Peak node preview</span>
            <span className="pill" style={{ borderColor: "var(--accent-line)", color: "var(--accent)" }}>{peak.shortName}</span>
          </div>
          <div className="fmap-note">divergence at the block's hottest node · <b>drill in</b> for full detail</div>
          <div className="div-pair">
            <div className="div-cell">
              <div className="vlbl cpu"><span className="d"></span>{cmp.mode === "quant" ? "FP32" : cmp.ref.ep}<span className="x">ref</span></div>
              <DivTile node={peak} variant="cpu" seedSalt={seedSalt} />
            </div>
            <div className="div-cell">
              <div className="vlbl npu"><span className="d"></span>{cmp.mode === "quant" ? "INT8" : "cand"}<span className="x">{cmp.mode === "quant" ? "cand" : CAND_LETTERS[peakIdx]}</span></div>
              <DivTile node={peak} variant="npu" seedSalt={seedSalt} mag={cmp.mode === "backend" ? pv : undefined} />
            </div>
            <div className="div-cell">
              <div className="vlbl diff"><span className="d"></span>|Δ|<span className="x">abs</span></div>
              <DivTile node={peak} variant="diff" seedSalt={seedSalt} mag={cmp.mode === "backend" ? pv : undefined} />
            </div>
          </div>
          <div className="footnote">aggregate above is block-level · these tiles are the peak node only</div>
        </div>

        {/* block cost / latency */}
        {cmp.mode === "quant" ? <GroupQuantCost node={node} /> : <GroupLatency node={node} cmp={cmp} />}
      </div>
    </div>
  );
}

function GroupQuantCost({ node }) {
  const cost = costOf(node);
  const vd = groupVerdict(node);
  return (
    <div className="insp-sec cost-sec">
      <div className="sec-head">
        <span className="ix">03</span><span className="nm">Cost · whole block</span>
        <span className="pill">Δt saved by INT8</span>
      </div>
      <div className="cost-grid">
        <div className="cost-row">
          <span className="ck ref">CPU · FP32</span>
          <span className="cbar"><i style={{ width: "100%", background: "var(--tx-1)" }}></i></span>
          <span className="cv mono">{cost.cpu.toFixed(2)} ms</span>
        </div>
        <div className="cost-row">
          <span className="ck cand">INT8</span>
          <span className="cbar"><i style={{ width: (cost.cpu ? (cost.npu / cost.cpu * 100) : 0) + "%", background: "var(--accent)" }}></i></span>
          <span className="cv mono accent">{cost.npu.toFixed(2)} ms</span>
        </div>
        <div className="cost-row saved">
          <span className="ck">Δt saved</span>
          <span className="cbar"><i style={{ width: (cost.cpu ? (cost.saved / cost.cpu * 100) : 0) + "%", background: "var(--heat-0)" }}></i></span>
          <span className="cv mono pos">{cost.saved.toFixed(2)} ms</span>
        </div>
      </div>
      <div className="verdict-line group">
        <span className="vl-tag">Block verdict</span>
        <span className="vl-body mono">{vd.line}</span>
      </div>
      <div className="twopass-note mono">
        <span className="tp-dot"></span>
        block Δt from the fused timing run · Δ from the instrumented run — two passes, not one
      </div>
    </div>
  );
}

function GroupLatency({ node, cmp }) {
  const sumLat = (cand) => (node.children || []).reduce((a, ch) => a + configLatency(NODE_BY_ID[ch.id] || ch, cand), 0) || configLatency(node, cand);
  const refLat = sumLat(cmp.ref);
  const cands = cmp.cands.map((c, i) => ({ c, i, lat: sumLat(c) }));
  const mx = Math.max(refLat, ...cands.map((x) => x.lat), 0.001);
  return (
    <div className="insp-sec lat-sec">
      <div className="sec-head">
        <span className="ix">03</span><span className="nm">Latency · whole block</span>
        <span className="pill">perf · not the verdict</span>
      </div>
      <div className="cost-grid">
        <div className="cost-row">
          <span className="ck ref">{cmp.ref.ep} · {cmp.ref.precision}</span>
          <span className="cbar"><i style={{ width: (refLat / mx * 100) + "%", background: "var(--tx-1)" }}></i></span>
          <span className="cv mono">{refLat.toFixed(2)} ms</span>
        </div>
        {cands.map(({ c, i, lat }) => (
          <div className="cost-row" key={i}>
            <span className="ck" style={{ color: EP_COLORS[c.ep] }}>{c.ep} · {c.precision}</span>
            <span className="cbar"><i style={{ width: (lat / mx * 100) + "%", background: EP_COLORS[c.ep] }}></i></span>
            <span className="cv mono" style={{ color: EP_COLORS[c.ep] }}>{lat.toFixed(2)} ms</span>
          </div>
        ))}
      </div>
      <div className="twopass-note mono">
        <span className="tp-dot"></span>
        block wall time summed over child nodes, per EP — independent of the tolerance call
      </div>
    </div>
  );
}

/* ============================================================================
   BOTTOM STRIP
   ============================================================================ */
const SC_XMAX = 1.62, SC_YMAX = 0.050;

function DepthChart({ selected, onSelect, ramp, cmp }) {
  const max = cmpMax(cmp);
  const selNode = NODE_BY_ID[selected];
  const effSel = selNode && selNode.parent ? selNode.parent : selected;
  return (
    <div className="depth-chart">
      <div className="depth-axis">
        <span>{max.toFixed(3)}</span>
        <span>{(max / 2).toFixed(3)}</span>
        <span>0</span>
      </div>
      <div className="depth-bars">
        <div className="depth-line" style={{ top: "50%", borderTop: "1px dashed var(--line)" }}></div>
        <div className="depth-line" style={{ top: "2px", borderTop: "1px dashed oklch(0.30 0.008 250 / 0.5)" }}></div>
        {NODES.map((n) => {
          const d = nodeDiff(n, cmp);
          const t = d / max;
          return (
            <div key={n.id}
              className={"dbar" + (n.id === effSel ? " sel" : "")}
              style={{ "--c": heatColor(t, ramp) }}
              onClick={() => onSelect(n.id)}
              title={`${n.name} · Δ ${d.toFixed(4)}`}>
              <div className="bar" style={{ height: `calc(${Math.max(2, t * 100)}% - 16px)` }}></div>
              <span className="xlbl">{n.name.replace("layer", "L").replace("/conv2", "/c2")}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SpeedAccuracy({ selected, onSelect }) {
  const selNode = NODE_BY_ID[selected];
  const effSel = selNode && selNode.parent ? selNode.parent : selected;
  const lineY0 = 100;
  const lineY1 = 100 - (ACCEPT_SLOPE * SC_XMAX) / SC_YMAX * 100;
  const CCLR = { good: "var(--heat-0)", marginal: "var(--heat-2)", reject: "var(--heat-4)" };
  return (
    <div className="scatter">
      <div className="sc-ylab">divergence Δ <span className="ar">↑ worse</span></div>
      <div className="sc-yaxis">
        <span>{SC_YMAX.toFixed(3)}</span><span>{(SC_YMAX / 2).toFixed(3)}</span><span>0</span>
      </div>
      <div className="sc-plot">
        <svg className="sc-grid" viewBox="0 0 100 100" preserveAspectRatio="none">
          <polygon points={`0,0 100,0 100,${lineY1} 0,${lineY0}`} fill="oklch(0.60 0.22 27 / 0.07)"></polygon>
          <polygon points={`0,${lineY0} 100,${lineY1} 100,100 0,100`} fill="oklch(0.78 0.17 150 / 0.06)"></polygon>
          <line x1="0" y1={lineY0} x2="100" y2={lineY1} stroke="var(--tx-2)" strokeWidth="0.6" strokeDasharray="2.4 2"></line>
          <line x1="0" y1="50" x2="100" y2="50" stroke="var(--line)" strokeWidth="0.4"></line>
          <line x1="50" y1="0" x2="50" y2="100" stroke="var(--line)" strokeWidth="0.4"></line>
        </svg>
        <span className="sc-thr-lab mono">acceptance threshold</span>
        <span className="sc-zone tl mono">expensive · low gain</span>
        <span className="sc-zone br mono">cheap win</span>
        {SCATTER.map((p) => {
          const left = (p.x / SC_XMAX) * 100;
          const bottom = (p.y / SC_YMAX) * 100;
          const isSel = p.id === selected || p.id === effSel;
          return (
            <div key={p.id}
              className={"sc-pt " + p.cls + (p.hot ? " hot" : "") + (isSel ? " sel" : "")}
              style={{ left: left + "%", bottom: bottom + "%", "--dc": CCLR[p.cls] }}
              onClick={() => onSelect(p.id)}
              title={`${p.label} · saved ${p.x.toFixed(2)} ms · Δ ${p.y.toFixed(4)}`}>
              <span className="dot"></span>
              <span className={"sc-name mono" + (p.hot ? " hot" : "")}>{p.label}</span>
              {p.cls === "reject" && !(p.id === "layer4") && <span className="sc-tag mono">keep FP32?</span>}
            </div>
          );
        })}
      </div>
      <div className="sc-xaxis mono">
        <span>0</span>
        <span className="sc-xlab">Δt saved by INT8 (ms) <span className="ar">more gain →</span></span>
        <span>{SC_XMAX.toFixed(1)}</span>
      </div>
    </div>
  );
}

/* backend: per-candidate worst divergence vs total latency, side by side */
function CandSummary({ cmp }) {
  const rows = cmp.cands.map((c, i) => {
    let wd = 0; for (const n of NODES) wd = Math.max(wd, candDiff(n, cmp, i));
    const totLat = NODES.reduce((a, n) => {
      if (n.group && n.children) return a + n.children.reduce((b, ch) => b + configLatency(NODE_BY_ID[ch.id] || ch, c), 0);
      return a + configLatency(n, c);
    }, 0);
    return { c, i, wd, totLat };
  });
  const refLat = NODES.reduce((a, n) => {
    if (n.group && n.children) return a + n.children.reduce((b, ch) => b + configLatency(NODE_BY_ID[ch.id] || ch, cmp.ref), 0);
    return a + configLatency(n, cmp.ref);
  }, 0);
  const dmx = Math.max(...rows.map((r) => r.wd), 0.001);
  const lmx = Math.max(refLat, ...rows.map((r) => r.totLat), 0.001);
  return (
    <div className="candsum">
      <div className="candsum-cols">
        <span className="csc up">candidate</span>
        <span className="csc up">worst Δ vs gold</span>
        <span className="csc up">total latency</span>
      </div>
      <div className="candsum-row ref">
        <span className="csr-name mono"><span className="csr-dot" style={{ background: "var(--tx-1)" }}></span>ref · {cfgLabel(cmp.ref)}</span>
        <span className="csr-bar"><span className="csr-zero mono">gold · 0.0000</span></span>
        <span className="csr-bar"><i style={{ width: (refLat / lmx * 100) + "%", background: "var(--tx-1)" }}></i><span className="csr-v mono">{refLat.toFixed(2)} ms</span></span>
      </div>
      {rows.map(({ c, i, wd, totLat }) => (
        <div className="candsum-row" key={i} style={{ "--cc": EP_COLORS[c.ep] }}>
          <span className="csr-name mono"><span className="csr-dot" style={{ background: EP_COLORS[c.ep] }}></span>{CAND_LETTERS[i]} · {cfgLabel(c)}</span>
          <span className="csr-bar"><i style={{ width: (wd / dmx * 100) + "%", background: "var(--heat-4)" }}></i><span className="csr-v mono">{wd.toFixed(4)}</span></span>
          <span className="csr-bar"><i style={{ width: (totLat / lmx * 100) + "%", background: EP_COLORS[c.ep] }}></i><span className="csr-v mono">{totLat.toFixed(2)} ms</span></span>
        </div>
      ))}
      <div className="candsum-note mono"><span className="tp-dot"></span>worst-node divergence and end-to-end latency are independent axes — fastest backend is not the most accurate</div>
    </div>
  );
}

function BottomStrip({ selected, onSelect, ramp, cmp }) {
  const [tab, setTab] = React.useState("depth");
  const secondTab = cmp.mode === "quant" ? "speed" : "cands";
  // keep tab valid when mode flips
  const activeSecond = tab === "depth" ? false : true;
  const curTab = tab === "depth" ? "depth" : secondTab;
  return (
    <div className="bottom" data-screen-label="Bottom strip">
      <div className="panel-head bottom-head">
        <div className="btabs">
          <button className={"btab" + (tab === "depth" ? " active" : "")} onClick={() => setTab("depth")}>
            {cmp.mode === "quant" ? "Error vs. Depth" : "Divergence vs. Depth"}
          </button>
          <button className={"btab" + (activeSecond ? " active" : "")} onClick={() => setTab(secondTab)}>
            {cmp.mode === "quant" ? "Speed ↔ Accuracy" : "Candidates · Δ vs latency"}
          </button>
        </div>
        <span className="meta">
          {curTab === "depth"
            ? (cmp.mode === "quant"
                ? "max-abs error per stage · quantization drift grows with depth"
                : "divergence per stage · spikes at the diverging op, not monotonic")
            : cmp.mode === "quant" ? "fused Δt × instrumented Δ · paired from two runs"
            : "per-candidate worst Δ vs end-to-end latency"}
        </span>
      </div>
      {curTab === "depth" && <DepthChart selected={selected} onSelect={onSelect} ramp={ramp} cmp={cmp} />}
      {curTab === "speed" && <SpeedAccuracy selected={selected} onSelect={onSelect} />}
      {curTab === "cands" && <CandSummary cmp={cmp} />}
    </div>
  );
}

Object.assign(window, { Inspector, GroupInspector, BottomStrip });
