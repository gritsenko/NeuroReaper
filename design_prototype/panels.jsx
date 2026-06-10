/* panels.jsx — composed UI regions */
const { useState: useStateP } = React;

/* candidate identity helpers ------------------------------------------------ */
const CAND_LETTERS = ["A", "B", "C"];
function candColor(cmp, i) {
  if (!cmp || cmp.mode === "quant") return "var(--accent)";
  const c = cmp.cands[i];
  return c ? (EP_COLORS[c.ep] || "var(--accent)") : "var(--accent)";
}
/* which candidate index drives single-candidate coloring/labels at a node */
function focusIdxAt(node, cmp) {
  return cmp.focus === "worst" ? worstCandAt(node, cmp) : (cmp.focus | 0);
}
/* axes that differ between two configs, as short "k: a→b" strings */
function axisDiffs(ref, cand) {
  const out = [];
  if (ref.ep !== cand.ep)               out.push(["backend", ref.ep, cand.ep]);
  if (ref.precision !== cand.precision) out.push(["precision", ref.precision, cand.precision]);
  if (ref.device !== cand.device)       out.push(["device", ref.device, cand.device]);
  return out;
}

/* styled native dropdown ---------------------------------------------------- */
function Drop({ label, value, options, onChange, tone }) {
  return (
    <label className={"drop" + (tone ? " " + tone : "")}>
      <span className="drop-k">{label}</span>
      <span className="drop-sel">
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
        <span className="drop-caret">▾</span>
      </span>
    </label>
  );
}

/* one config = three dropdowns (EP / precision / device).
   Picking an EP snaps the device to where that EP actually executes
   (QNN→NPU, CUDA/TRT/DML→GPU, …) so meaningless combos can't be authored. */
function ConfigEditor({ cfg: c, onChange, dimDevice }) {
  return (
    <div className="cfg-edit">
      <Drop label="Backend / EP" value={c.ep} options={BACKENDS} onChange={(v) => onChange({ ep: v, device: epDevice(v) })} />
      <Drop label="Precision" value={c.precision} options={PRECISIONS} onChange={(v) => onChange({ precision: v })} />
      <Drop label="Device" value={c.device} options={DEVICES} onChange={(v) => onChange({ device: v })} tone={dimDevice ? "dim" : ""} />
    </div>
  );
}

/* ---------------- TOP BAR ---------------- */
const PRESET_OPTS = [["quant", "Quantization"], ["backend", "Backend divergence"]];

function TopBar({ running, onRerun, summary, cmp, onMode }) {
  return (
    <div className="topbar" data-screen-label="Top bar">
      <div className="brand">
        <span className="brand-mark"></span>
        <span className="brand-name">Neuro<b>Reaper</b></span>
        <span className="divider-v" style={{ height: 20, margin: "0 4px" }}></span>
        <span className="model-name mono">resnet18.onnx</span>
      </div>

      <div className="preset-wrap">
        <span className="preset-k">Preset</span>
        <div className="seg preset-seg" role="tablist">
          {PRESET_OPTS.map(([k, lbl]) => (
            <button key={k} className={"seg-btn" + (cmp.mode === k ? " active" : "")}
              onClick={() => onMode(k)}>{lbl}</button>
          ))}
        </div>
        <span className="preset-blurb mono">{PRESETS[cmp.mode].blurb}</span>
      </div>

      <div className="summary">
        <div className="kv">
          <span className="k">{cmp.mode === "quant" ? "Max abs diff" : "Worst Δ across cands"}</span>
          <span className="v"><em>{summary.maxAbs}</em> <span className="at">@ {summary.at}</span></span>
        </div>
        <button className={"btn primary" + (running ? " running" : "")} onClick={onRerun} disabled={running}>
          <span className="glyph">{running ? "↻" : "▷"}</span>
          {running ? "Running…" : "Re-run"}
        </button>
      </div>
    </div>
  );
}

/* ---------------- COMPARE BAR (editable configs) ---------------- */
function CandCard({ cmp, i, onCand, onRemove }) {
  const c = cmp.cands[i];
  const diffs = axisDiffs(cmp.ref, c);
  const col = candColor(cmp, i);
  const canRemove = cmp.mode === "backend" && cmp.cands.length > 1;
  const multi = diffs.length > 1;
  return (
    <div className="cand-card" style={{ "--cc": col }}>
      <div className="cand-head">
        <span className="cand-dot"></span>
        <span className="cand-name">candidate {CAND_LETTERS[i]}</span>
        <span className="cand-cfg mono">{cfgLabel(c)}</span>
        {canRemove && (
          <button className="cand-x" title="remove candidate" onClick={() => onRemove(i)}>✕</button>
        )}
      </div>
      <ConfigEditor cfg={c} onChange={(p) => onCand(i, p)} dimDevice={cmp.mode === "quant"} />
      <div className={"cand-diffs" + (multi ? " warn" : "")}>
        {diffs.length === 0
          ? <span className="cd-none mono">identical to reference</span>
          : diffs.map(([k, a, b]) => (
              <span className="cd-chip mono" key={k}><span className="cd-k">{k}</span>{a}<span className="cd-arr">→</span>{b}</span>
            ))}
        {multi && <span className="cd-warn mono" title="more than one axis differs — divergence can't be attributed to a single cause">⚠ {diffs.length} axes — cause is ambiguous</span>}
        {EP_INFO[c.ep] && EP_INFO[c.ep].legacy && (
          <span className="cd-chip mono legacy" title={EP_INFO[c.ep].note}>⚑ legacy EP — {EP_INFO[c.ep].note}</span>
        )}
      </div>
    </div>
  );
}

function CompareBar({ cmp, onRef, onCand, onAdd, onRemove }) {
  return (
    <div className="comparebar" data-screen-label="Compare bar">
      <div className="cb-ref-side">
        <div className="cb-side-head">
          <span className="ref-dot"></span>
          <span className="cb-side-name">reference</span>
          <span className="cb-gold mono">gold</span>
          <span className="cb-cfg mono">{cfgLabel(cmp.ref)}</span>
        </div>
        <ConfigEditor cfg={cmp.ref} onChange={onRef} dimDevice={cmp.mode === "quant"} />
      </div>

      <div className="cb-arrow"><span>→</span></div>

      <div className="cb-cands">
        {cmp.cands.map((_, i) => (
          <CandCard key={i} cmp={cmp} i={i} onCand={onCand} onRemove={onRemove} />
        ))}
        {cmp.mode === "backend" && cmp.cands.length < 3 && (
          <button className="cand-add" onClick={onAdd}>
            <span className="ca-plus">＋</span>
            <span>add candidate</span>
            <span className="ca-sub mono">vs same reference</span>
          </button>
        )}
      </div>
    </div>
  );
}

/* ---------------- LEFT RAIL ---------------- */
function LeftRail({ cmp }) {
  // build the config list: reference + candidates, each with its top-1 prediction
  const refPred = predOf(cmp.ref);
  const rows = [{ label: cfgLabel(cmp.ref), pred: refPred, color: "var(--tx-1)", isRef: true, tag: "ref" }];
  cmp.cands.forEach((c, i) => rows.push({
    label: cfgLabel(c), pred: predOf(c), color: candColor(cmp, i),
    isRef: false, tag: cmp.mode === "backend" ? CAND_LETTERS[i] : "cand",
  }));
  const worst = rows.slice(1).reduce((m, r) => (r.pred.p < m.pred.p ? r : m), rows[1] || rows[0]);
  const labelPreserved = rows.every((r) => r.pred.cls === refPred.cls);
  const dp = worst.pred.p - refPred.p;

  return (
    <div className="col left" data-screen-label="Input & predictions">
      <div className="panel-head"><span className="title">Input</span><span className="meta">tensor: input</span></div>
      <div className="scroll">
        <div className="rail-sec">
          <div className="input-preview">
            <img className="input-img" src="assets/coffee_mug.png" alt="white coffee mug, center-crop" draggable="false" />
            <svg className="crosshair" viewBox="0 0 100 100" preserveAspectRatio="none">
              <line x1="50" y1="0" x2="50" y2="100" stroke="oklch(0.92 0 0 / 0.16)" strokeWidth="0.4"></line>
              <line x1="0" y1="50" x2="100" y2="50" stroke="oklch(0.92 0 0 / 0.16)" strokeWidth="0.4"></line>
              <rect x="14" y="14" width="72" height="72" fill="none" stroke="oklch(0.92 0 0 / 0.2)" strokeWidth="0.5" strokeDasharray="2 2"></rect>
            </svg>
            <span className="tag">coffee_mug.jpg · center-crop</span>
          </div>
          <div className="shape-line"><span className="k">input:</span> 1×3×224×224 · <span className="k">FP32</span></div>
        </div>

        {/* ---- VERDICT (lead conclusion) ---- */}
        <div className="rail-sec verdict-sec">
          <div className={"verdict" + (labelPreserved ? "" : " bad")}>
            <div className="vd-row">
              <span className="vd-tick">{labelPreserved ? "✓" : "!"}</span>
              <span className="vd-head">{labelPreserved ? "LABEL PRESERVED" : "LABEL FLIPPED"}</span>
            </div>
            <div className="vd-body">
              top-1 {labelPreserved ? "unchanged" : "changed"} — reference calls it <b>"{refPred.cls}"</b>
              {cmp.mode === "backend" ? " across all backends" : " on both precisions"}
            </div>
            <div className="vd-conf">
              <span className="vd-k">top-1 confidence · ref → {cmp.mode === "backend" ? "worst cand" : "candidate"}</span>
              <span className="vd-val mono">{refPred.p.toFixed(3)} <span className="vd-arrow">→</span> {worst.pred.p.toFixed(3)}</span>
              <span className="vd-delta mono">Δp {dp.toFixed(3)}</span>
            </div>
            <div className="vd-foot">{cmp.mode === "backend" ? "backend choice shifted confidence, not the decision" : "quantization shifted confidence, not the decision"}</div>
          </div>
        </div>

        <div className="rail-sec" style={{ borderBottom: "none" }}>
          <div className="sub up">Predictions · top-1 per config</div>
          {rows.map((r, i) => (
            <div className="pred-row" key={i}>
              <span className="bk" style={{ color: r.isRef ? "var(--tx-2)" : r.color }}>{r.tag}</span>
              <span className="cls">"{r.pred.cls}"</span>
              <span className="p">{r.pred.p.toFixed(3)}</span>
              <div className="pred-bar"><i style={{ width: (r.pred.p * 100) + "%", background: r.color }}></i></div>
            </div>
          ))}
          <div className="ru-note" style={{ marginTop: 8 }}>
            {cmp.mode === "backend"
              ? "same input, same weights — only the execution backend differs per row"
              : "FP32 reference vs INT8 candidate on the same backend"}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- GRAPH ---------------- */
const COLORBY = {
  divergence: { title: "Divergence", hint: "max abs ref↔cand drift" },
  latency:    { title: "Latency vs reference", hint: "wall-time recovered vs CPU ref" },
  tradeoff:   { title: "Δ cost per ms saved", hint: "divergence bought per ms saved" },
  ep:         { title: "Execution provider", hint: "which EP each node ran on · ⚠ = CPU fallback" },
};
const COLORBY_OPTS = [["divergence", "DIVERGENCE"], ["latency", "LATENCY"], ["tradeoff", "TRADE-OFF"], ["ep", "EXEC PROVIDER"]];

function epColorAt(node, cmp, idx) {
  const cand = cmp.cands[idx];
  const e = nodeEP(node, cand);
  return { color: EP_COLORS[e.ep] || "var(--tx-1)", fallback: e.fallback, partial: e.partial, ep: e.ep, intended: e.intended };
}

/* recursive raw-ONNX subgraph card */
function Subgraph({ group, level, openPath, onNode, ramp, colorBy, selected, cmp }) {
  return (
    <div className="gchildren docked" data-level={level}>
      <div className="gchildren-note">
        raw ONNX subgraph · Δ is max-abs at each node — peak entered at the
        <b> hot</b> node, not the group output
      </div>
      {group.children.map((ch) => {
        const full = NODE_BY_ID[ch.id] || ch;
        const idx = focusIdxAt(full, cmp);
        const epc = colorBy === "ep" ? epColorAt(full, cmp, idx) : null;
        const cc = epc ? epc.color : heatColor(metricT(full, cmp, colorBy), ramp);
        const d = candDiff(full, cmp, idx);
        const hasKids = !!(ch.children && ch.children.length);
        const isOpenHere = hasKids && openPath[level] === ch.id;
        return (
          <div className="gchild-wrap" key={ch.id}>
            <div
              className={"gchild" + (ch.id === selected ? " selected" : "") + (ch.hot ? " hot" : "") + (hasKids ? " hasgroup" : "") + (isOpenHere ? " open" : "") + (epc && epc.fallback ? " fallback" : "")}
              style={{ "--c": cc }}
              onClick={() => onNode(ch, level)}
            >
              <div className="cheat"></div>
              <div className="cbody">
                <span className="cop">{ch.op}{hasKids ? " · " + (ch.nInner || ch.children.length) + " nodes" : ""}</span>
                <span className="cnm mono">{ch.id}</span>
              </div>
              <div className="cright">
                {epc && <span className={"ep-badge mono" + (epc.fallback ? " fb" : "")}>{epc.fallback ? "⚠ CPU" : epc.ep}</span>}
                {ch.hot && <span className="hotflag">PEAK</span>}
                <span className="cdiff mono">Δ {d.toFixed(4)}</span>
                {hasKids && <span className="ccaret">{isOpenHere ? "▾" : "▸"}</span>}
                <span className="cswatch"></span>
              </div>
            </div>
            {isOpenHere && (
              <Subgraph group={ch} level={level + 1} openPath={openPath} onNode={onNode} ramp={ramp} colorBy={colorBy} selected={selected} cmp={cmp} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function GraphView({ selected, onSelect, ramp, colorBy, onColorBy, cmp, focus, onFocus }) {
  const [tip, setTip] = useStateP(null);
  const [openPath, setOpenPath] = useStateP(() => ["layer3"]);
  const conf = COLORBY[colorBy] || COLORBY.divergence;
  const isEP = colorBy === "ep";
  const multiCand = cmp.mode === "backend" && cmp.cands.length > 1;

  const handleNode = (node, depth) => {
    onSelect(node.id);
    const hasKids = !!(node.children && node.children.length);
    if (!hasKids) return;
    setOpenPath((p) => (p[depth] === node.id ? p.slice(0, depth) : p.slice(0, depth).concat(node.id)));
  };

  const mx = cmpMax(cmp);
  // EP swatches present in the current comparison (for the legend)
  const epsPresent = isEP ? Array.from(new Set(cmp.cands.map((c) => c.ep).concat(["CPU"]))) : [];

  return (
    <div className="col center" data-screen-label="Model graph">
      <div className="panel-head">
        <span className="title">Model Graph · {isEP ? "Execution Provider Map" : "Divergence Heatmap"}</span>
        <span className="meta">resnet18.onnx · folded · click a block to expand</span>
      </div>
      <div className="graph-toolbar">
        <span className="cb-label">Color by</span>
        <div className="seg" role="tablist">
          {COLORBY_OPTS.map(([k, lbl]) => (
            <button key={k} className={"seg-btn" + (colorBy === k ? " active" : "")} onClick={() => onColorBy(k)}>{lbl}</button>
          ))}
        </div>
        {multiCand && (
          <div className="focus-wrap">
            <span className="cb-label">Candidate</span>
            <div className="seg focus-seg">
              <button className={"seg-btn" + (focus === "worst" ? " active" : "")} onClick={() => onFocus("worst")}>WORST</button>
              {cmp.cands.map((c, i) => (
                <button key={i} className={"seg-btn" + (focus === i ? " active" : "")}
                  style={focus === i ? { background: EP_COLORS[c.ep], color: "oklch(0.16 0.02 250)" } : null}
                  onClick={() => onFocus(i)}>{CAND_LETTERS[i]}</button>
              ))}
            </div>
          </div>
        )}
        <span className="cb-hint mono">{conf.hint}</span>
      </div>
      <div className="scroll graph-scroll">
        <div className="graph">
          {NODES.map((n, i) => {
            const idx = focusIdxAt(n, cmp);
            const epc = isEP ? epColorAt(n, cmp, idx) : null;
            const c = epc ? epc.color : heatColor(metricT(n, cmp, colorBy), ramp);
            const d = nodeDiff(n, cmp);
            const isSel = n.id === selected;
            const isGroup = !!n.group;
            const isOpen = isGroup && openPath[0] === n.id;
            const childSel = isGroup && n.children.some((ch) => ch.id === selected);
            return (
              <React.Fragment key={n.id}>
                <div className={"gnode-wrap" + (isOpen ? " open" : "")} data-nodeid={n.id}>
                  <div
                    className={"gnode" + (isGroup ? " group" : "") + (isSel ? " selected" : "") + (childSel ? " childsel" : "") + (isOpen ? " open" : "") + (epc && epc.fallback ? " fallback" : "")}
                    style={{ "--c": c }}
                    onClick={() => handleNode(n, 0)}
                  >
                    <div className="heat"></div>
                    <div className="body">
                      <div className="op">{n.op}</div>
                      <div className="nm">{n.name}</div>
                      {isGroup && <div className="gfold">{isOpen ? "▾ " : "▸ "}{n.nInner} ONNX nodes · click to {isOpen ? "collapse" : "expand"}</div>}
                    </div>
                    <div className="right">
                      {epc && <span className={"ep-badge mono" + (epc.fallback ? " fb" : epc.partial ? " partial" : "")}>{epc.fallback ? "⚠ CPU" : epc.ep}</span>}
                      <div className="diff">{d === 0 ? "0.0000" : "Δ " + d.toFixed(4)}</div>
                      <div className="swatch"></div>
                    </div>
                  </div>

                  {isOpen && (
                    <Subgraph group={n} level={1} openPath={openPath} onNode={handleNode} ramp={ramp} colorBy={colorBy} selected={selected} cmp={cmp} />
                  )}
                </div>

                {i < NODES.length - 1 && (
                  <div className="gedge"
                    onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, shape: n.shape })}
                    onMouseLeave={() => setTip(null)}>
                    <span className="shape-tip">{n.shape}</span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
          {openPath.length > 0 && <div className="graph-tail" aria-hidden="true"></div>}
        </div>
      </div>

      {tip && (
        <div className="gedge-label-float" style={{ left: tip.x + 14, top: tip.y - 14 }}>
          <span className="k">tensor </span>{tip.shape}
        </div>
      )}

      <div className="legend">
        <div className="up">{conf.title}{cmp.focus !== "worst" && multiCand ? " · cand " + CAND_LETTERS[cmp.focus | 0] : multiCand ? " · worst-of-N" : ""}</div>
        {isEP ? (
          <div className="legend-eps">
            {epsPresent.map((e) => (
              <span className="leg-ep mono" key={e}><i style={{ background: EP_COLORS[e] }}></i>{e}</span>
            ))}
            <span className="leg-ep mono fb"><i></i>⚠ fallback</span>
          </div>
        ) : (
          <>
            <div className="ramp" style={{ background: `linear-gradient(90deg, ${heatColor(0, ramp)}, ${heatColor(0.5, ramp)}, ${heatColor(1, ramp)})` }}></div>
            <div className="ticks"><span>identical</span><span></span><span>{mx.toFixed(4)}</span></div>
          </>
        )}
      </div>
    </div>
  );
}

Object.assign(window, { TopBar, CompareBar, LeftRail, GraphView, Drop, ConfigEditor, candColor, CAND_LETTERS, focusIdxAt });
