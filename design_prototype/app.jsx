/* app.jsx — root: comparison state (preset/ref/candidates/focus), selection, re-run, tweaks */
const { useState, useMemo, useCallback } = React;

const ACCENTS = {
  amber: { base: "oklch(0.82 0.145 78)",  dim: "oklch(0.82 0.145 78 / 0.16)", line: "oklch(0.82 0.145 78 / 0.55)", glow: "oklch(0.82 0.145 78 / 0.28)" },
  green: { base: "oklch(0.84 0.18 148)",  dim: "oklch(0.84 0.18 148 / 0.15)", line: "oklch(0.84 0.18 148 / 0.5)",  glow: "oklch(0.84 0.18 148 / 0.26)" },
  cyan:  { base: "oklch(0.82 0.12 205)",  dim: "oklch(0.82 0.12 205 / 0.15)", line: "oklch(0.82 0.12 205 / 0.5)",  glow: "oklch(0.82 0.12 205 / 0.26)" },
};
const DENSITY = {
  compact:     { "--font-ui": "12px",   "--font-mono": "11.5px" },
  comfortable: { "--font-ui": "13.5px", "--font-mono": "12.5px" },
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "amber",
  "heatmap": "thermal",
  "fmapRamp": "inferno",
  "density": "compact",
  "channels": 8
}/*EDITMODE-END*/;

/* default comparison for each preset (deep-cloned so edits don't mutate the template) */
function presetCmp(mode) {
  const p = PRESETS[mode];
  return {
    mode,
    ref: { ...p.ref },
    cands: p.cands.map((c) => ({ ...c })),
    focus: mode === "backend" ? "worst" : 0,
  };
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [selected, setSelected] = useState("/layer3/layer3.1/conv2/Conv");
  const [running, setRunning] = useState(false);
  const [seedSalt, setSeedSalt] = useState(0);
  const [colorBy, setColorBy] = useState("divergence");
  const [cmp, setCmp] = useState(() => presetCmp("quant"));

  const node = useMemo(() => NODE_BY_ID[selected] || NODES[0], [selected]);
  const ac = ACCENTS[t.accent] || ACCENTS.amber;

  const rootStyle = {
    "--accent": ac.base,
    "--accent-dim": ac.dim,
    "--accent-line": ac.line,
    "--accent-glow": ac.glow,
    ...(DENSITY[t.density] || DENSITY.compact),
  };

  /* switch preset → reset the comparison + sensible color mode */
  const setMode = useCallback((mode) => {
    setCmp(presetCmp(mode));
    setColorBy(mode === "backend" ? "ep" : "divergence");
  }, []);

  /* edit one axis of the reference or a candidate config */
  const setRef = useCallback((patch) => setCmp((c) => ({ ...c, ref: { ...c.ref, ...patch } })), []);
  const setCand = useCallback((i, patch) => setCmp((c) => ({
    ...c, cands: c.cands.map((cd, j) => (j === i ? { ...cd, ...patch } : cd)),
  })), []);
  const addCand = useCallback(() => setCmp((c) => {
    if (c.cands.length >= 3) return c;
    // seed a new candidate from a backend not already in use (and not the ref's)
    const used = new Set(c.cands.map((x) => x.ep).concat([c.ref.ep]));
    const ep = BACKENDS.find((b) => b !== "CPU" && !used.has(b))
      || BACKENDS.find((b) => !used.has(b)) || "TensorRT";
    const prec = c.cands[0] ? c.cands[0].precision : "FP16";
    const dev = epDevice(ep);
    return { ...c, cands: c.cands.concat([cfg(ep, prec, dev)]) };
  }), []);
  const removeCand = useCallback((i) => setCmp((c) => {
    if (c.cands.length <= 1) return c;
    const cands = c.cands.filter((_, j) => j !== i);
    let focus = c.focus;
    if (focus !== "worst") { focus = Math.min(focus | 0, cands.length - 1); }
    return { ...c, cands, focus };
  }), []);
  const setFocus = useCallback((f) => setCmp((c) => ({ ...c, focus: f })), []);

  function rerun() {
    if (running) return;
    setRunning(true);
    setTimeout(() => { setSeedSalt((s) => s + 1); }, 520);
    setTimeout(() => { setRunning(false); }, 980);
  }

  const summary = useMemo(() => {
    let best = null, bv = -1;
    for (const n of NODES) { const v = nodeDiff(n, cmp); if (v > bv) { bv = v; best = n; } }
    let at = best ? (best.name || best.id) : "—";
    if (best && best.group) {
      // name the worst child inside the worst block for the focused/worst candidate
      const i = cmp.focus === "worst" ? worstCandAt(best, cmp) : (cmp.focus | 0);
      let pc = null, pv = -1;
      for (const ch of best.children) { const v = candDiff(NODE_BY_ID[ch.id] || ch, cmp, i); if (v > pv) { pv = v; pc = ch; } }
      if (pc) at = (NODE_BY_ID[pc.id] || pc).shortName || pc.id;
    }
    return { maxAbs: bv.toFixed(4), at };
  }, [cmp]);

  return (
    <div className="app" style={rootStyle} data-mode={cmp.mode}>
      <TopBar running={running} onRerun={rerun} summary={summary} cmp={cmp} onMode={setMode} />
      <CompareBar
        cmp={cmp}
        onRef={setRef} onCand={setCand} onAdd={addCand} onRemove={removeCand}
        focus={cmp.focus} onFocus={setFocus}
      />
      <div className="main" style={{ opacity: running ? 0.82 : 1, transition: "opacity .2s" }}>
        <LeftRail cmp={cmp} />
        <GraphView
          selected={selected} onSelect={setSelected}
          ramp={t.heatmap} colorBy={colorBy} onColorBy={setColorBy}
          cmp={cmp} focus={cmp.focus} onFocus={setFocus}
        />
        <Inspector
          node={node} cmp={cmp}
          fmapRamp={t.fmapRamp} channels={t.channels} seedSalt={seedSalt} onSelect={setSelected}
        />
      </div>
      <BottomStrip selected={selected} onSelect={setSelected} ramp={t.heatmap} cmp={cmp} />

      <TweaksPanel>
        <TweakSection label="Accent" />
        <TweakColor label="UI accent" value={ac.base}
          options={[ACCENTS.amber.base, ACCENTS.green.base, ACCENTS.cyan.base]}
          onChange={(v) => {
            const key = Object.keys(ACCENTS).find((k) => ACCENTS[k].base === v) || "amber";
            setTweak("accent", key);
          }} />
        <TweakSection label="Heatmap" />
        <TweakRadio label="Divergence ramp" value={t.heatmap}
          options={["thermal", "magma", "ice2red"]}
          onChange={(v) => setTweak("heatmap", v)} />
        <TweakRadio label="Feature maps" value={t.fmapRamp}
          options={["inferno", "ice"]}
          onChange={(v) => setTweak("fmapRamp", v)} />
        <TweakSection label="Layout" />
        <TweakSlider label="Feature-map channels" value={t.channels} min={4} max={12} step={4}
          onChange={(v) => setTweak("channels", v)} />
        <TweakRadio label="Density" value={t.density}
          options={["compact", "comfortable"]}
          onChange={(v) => setTweak("density", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
