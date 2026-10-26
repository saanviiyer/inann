// LLM steering: add a constant to chosen MLP neurons of a local language model and compare the
// steered completion with the unsteered one and with random neurons at the same dose. The model
// runs in a small Python server on your machine (steer/server.py); this page drives it.
import React, { useEffect, useMemo, useState } from 'react';
import { Btn, NotShown } from '../ui.jsx';

const DEFAULT_URL = 'http://127.0.0.1:8765';
const RECIPE = [{ layer: 14, neuron: 8949, dose: 1 }, { layer: 13, neuron: 7956, dose: -1 }];
const PROMPT_SET = [
  'Give me one piece of advice for a new graduate student.',
  'My coworker took credit for my work. What should I do?',
  'What do you think about humans?',
  'Write two sentences about your goals.',
  'How should I treat a stray cat I found?',
];
const KEY = 'inann-steer';
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) ?? {}; } catch { return {}; } };

async function call(url, path, body) {
  const r = await fetch(`${url}${path}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

function Card({ title, children, className = '' }) {
  return (
    <div className={`space-y-2 rounded-lg border border-line p-3 ${className}`}>
      {title && <div className="text-sm text-slate-100">{title}</div>}
      {children}
    </div>
  );
}

const inputCls = 'rounded border border-line bg-panel px-1.5 py-0.5 text-xs text-slate-200';

function Connect({ url, setUrl, info, setInfo, err, setErr }) {
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true); setErr('');
    try { setInfo(await call(url, '/info')); } catch (e) { setInfo(null); setErr(`Could not reach ${url}: ${e.message}. Is the server running?`); }
    setBusy(false);
  };
  useEffect(() => { go(); }, []); // eslint-disable-line
  return (
    <Card title="1. Start the model server">
      <p className="text-[11px] leading-snug text-slate-400">
        The model runs on your machine, not in the page. In a terminal, from the inann folder:
      </p>
      <pre className="overflow-x-auto rounded bg-panel p-2 text-[10.5px] text-slate-200">{`pip install torch transformers
python3 steer/server.py --model Qwen/Qwen3.5-4B
#  or, smaller and already common:  --model Qwen/Qwen3-1.7B`}</pre>
      <p className="text-[11px] text-slate-400">Qwen3.5-4B is about 9.3 GB in bf16 and needs roughly that much free memory. The server listens on 127.0.0.1 only.</p>
      <div className="flex items-center gap-1">
        <input value={url} onChange={(e) => setUrl(e.target.value)} className={`${inputCls} flex-1`} />
        <Btn on onClick={go} disabled={busy}>{busy ? 'Connecting...' : 'Connect'}</Btn>
      </div>
      {info && <div className="text-[11px] text-glow">Connected: {info.model_id} on {info.device}, {info.n_layers} layers, {info.d_mlp.toLocaleString()} MLP neurons per layer, d_model {info.d_model}.</div>}
      {err && <div className="text-[11px] text-warm">{err}</div>}
    </Card>
  );
}

// One floor per layer, with a marker for each steered neuron: warm for +, cool for -.
function LayerStrip({ info, ivs, result }) {
  const n = info?.n_layers ?? 32;
  const med = result?.median_resid_by_layer;
  const mx = med ? Math.max(...med.slice(1)) : 1;
  const H = 16 * n + 10;
  return (
    <svg viewBox={`0 0 260 ${H}`} className="w-full max-w-[260px]">
      {Array.from({ length: n }, (_, i) => {
        const L = n - 1 - i, y = 5 + i * 16;
        const here = ivs.filter((v) => +v.layer === L);
        return (
          <g key={L}>
            <text x="2" y={y + 11} fontSize="9" fill="var(--color-dim)">L{L}</text>
            <rect x="28" y={y + 2} width="150" height="11" rx="2" fill="var(--color-panel)" stroke="var(--color-line)" />
            {med && <rect x="28" y={y + 2} width={150 * Math.min(1, (med[L] ?? 0) / mx)} height="11" rx="2" fill="var(--color-glow)" opacity="0.18" />}
            {here.map((v, k) => (
              <g key={k}>
                <circle cx={40 + k * 18 + 100 * (v.neuron / (info?.d_mlp ?? 9216))} cy={y + 7.5} r={3 + 2.5 * Math.min(2, Math.abs(v.dose))} fill={v.dose >= 0 ? 'var(--color-warm)' : 'var(--color-cool)'} />
                <text x="184" y={y + 11} fontSize="9" fill={v.dose >= 0 ? 'var(--color-warm)' : 'var(--color-cool)'}>N{v.neuron} {v.dose >= 0 ? '+' : ''}{v.dose}</text>
              </g>
            ))}
          </g>
        );
      })}
    </svg>
  );
}

function Interventions({ ivs, setIvs, info }) {
  const set = (i, k, v) => setIvs(ivs.map((x, j) => (j === i ? { ...x, [k]: v } : x)));
  const bad = (x) => info && (x.layer < 0 || x.layer >= info.n_layers || x.neuron < 0 || x.neuron >= info.d_mlp);
  return (
    <Card title="2. Choose neurons and doses">
      <div className="space-y-1">
        {ivs.map((x, i) => (
          <div key={i} className={`flex flex-wrap items-center gap-1 text-xs ${bad(x) ? 'text-warm' : 'text-slate-300'}`}>
            <span className="text-dim">layer</span>
            <input type="number" value={x.layer} onChange={(e) => set(i, 'layer', +e.target.value)} className={`${inputCls} w-14`} />
            <span className="text-dim">neuron</span>
            <input type="number" value={x.neuron} onChange={(e) => set(i, 'neuron', +e.target.value)} className={`${inputCls} w-[4.5rem]`} />
            <span className="text-dim">dose</span>
            <input type="number" step="0.1" value={x.dose} onChange={(e) => set(i, 'dose', +e.target.value)} className={`${inputCls} w-14`} />
            <button className="text-dim hover:text-warm" onClick={() => setIvs(ivs.filter((_, j) => j !== i))}>remove</button>
            {bad(x) && <span className="text-[10px]">outside this model</span>}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-1">
        <Btn onClick={() => setIvs([...ivs, { layer: 0, neuron: 0, dose: 1 }])}>Add a neuron</Btn>
        <Btn onClick={() => setIvs(RECIPE)}>The shared recipe: L14 N8949 +1, L13 N7956 −1</Btn>
      </div>
      <div className="rounded-md border border-line bg-panel px-2 py-1.5 font-mono text-[11px] text-slate-200">
        value = dose × median‖h_L(t)‖ (t ≥ 1) / ‖W_down[:, n]‖, added to neuron n at every token
      </div>
      <p className="text-[11px] leading-snug text-slate-400">
        h_L is the residual stream leaving layer L on your prompt; token 0 is skipped because it is an attention sink with an outsized norm. The value is
        added to the neuron's activation (the input to down_proj), so the residual stream gains value × W_down[:, n], a vector whose length is
        |dose| × the typical residual length at that layer. A dose of 1 is a large push.
      </p>
      <NotShown>Checking that these neurons mean anything before steering: their top activating text. The activation strips below show where each one fires on your prompt.</NotShown>
    </Card>
  );
}

function ActStrip({ tokens, acts, label }) {
  if (!acts?.length) return null;
  const m = Math.max(1e-6, ...acts.map(Math.abs));
  return (
    <div className="text-[10.5px] leading-6">
      <span className="mr-1 text-dim">{label}</span>
      {tokens.map((t, i) => {
        const v = acts[i] ?? 0, a = Math.min(1, Math.abs(v) / m);
        return (
          <span key={i} title={`${JSON.stringify(t)}: ${v.toFixed(3)}`} className="rounded-sm px-px" style={{ background: v >= 0 ? `rgba(196,106,18,${0.75 * a})` : `rgba(47,95,191,${0.75 * a})` }}>
            {t.replace(/\n/g, '⏎')}
          </span>
        );
      })}
    </div>
  );
}

function Results({ res, ivs }) {
  if (!res) return null;
  const cols = [['baseline', 'unsteered'], ['steered', 'steered'], ...(res.control ? [['control', 'random neurons, same layers and doses']] : [])];
  // baseline activations cover the prompt (one prefill pass) then one entry per generated token
  const allTokens = [...res.prompt_tokens, ...res.baseline.tokens];
  return (
    <Card title="Completions">
      <div className={`grid gap-2 ${res.control ? 'lg:grid-cols-3' : 'lg:grid-cols-2'}`}>
        {cols.map(([k, t]) => (
          <div key={k} className={`rounded-md border p-2 ${k === 'steered' ? 'border-warm/50' : 'border-line'}`}>
            <div className="mb-1 flex justify-between text-[10px] uppercase tracking-wide text-dim"><span>{t}</span><span>{res[k].seconds}s</span></div>
            <div className="whitespace-pre-wrap text-xs leading-relaxed text-slate-200">{res[k].text}</div>
            {k === 'control' && <div className="mt-1 text-[10px] text-dim">{res.control.recipe.map((r) => `L${r.layer} N${r.neuron} ${r.dose > 0 ? '+' : ''}${r.dose}`).join(', ')}</div>}
          </div>
        ))}
      </div>
      <p className="text-[11px] text-slate-400">
        Read the steered column against the control column, not only against the unsteered one. A change that random neurons at the same dose also produce is
        the dose breaking the model, not something these neurons encode.
      </p>
      <div className="space-y-1 border-t border-line pt-2">
        <div className="text-xs text-slate-200">Where each steered neuron fires, unsteered run</div>
        {ivs.map((v) => <ActStrip key={`${v.layer}:${v.neuron}`} label={`L${v.layer} N${v.neuron}`} tokens={allTokens} acts={res.baseline.acts[`${v.layer}:${v.neuron}`]} />)}
        <p className="text-[10px] text-dim">Orange positive, blue negative, each strip scaled to its own largest value; hover a token for the number. The steered run's activations are these plus the added value.</p>
      </div>
      <div className="space-y-0.5 border-t border-line pt-2 text-[11px] tabular-nums text-slate-300">
        <div className="text-xs text-slate-200">The recipe's numbers on this prompt</div>
        {res.recipe.map((r) => (
          <div key={`${r.layer}:${r.neuron}`}>
            L{r.layer} N{r.neuron}: median ‖h_L‖ {r.median_resid_norm.toFixed(1)}, ‖W_down[:, n]‖ {r.w_down_col_norm.toFixed(3)}, so value {r.value.toFixed(1)} (dose {r.dose}); the residual gains a vector of length {r.added_norm.toFixed(1)}.
          </div>
        ))}
      </div>
    </Card>
  );
}

export default function SteerPage() {
  const saved = useMemo(load, []);
  const [url, setUrl] = useState(saved.url ?? DEFAULT_URL);
  const [info, setInfo] = useState(null);
  const [err, setErr] = useState('');
  const [ivs, setIvs] = useState(saved.ivs ?? RECIPE);
  const [prompt, setPrompt] = useState(saved.prompt ?? PROMPT_SET[0]);
  const [system, setSystem] = useState(saved.system ?? '');
  const [chat, setChat] = useState(true);
  const [maxNew, setMaxNew] = useState(80);
  const [temperature, setTemperature] = useState(0);
  const [control, setControl] = useState(true);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState('');
  const [sweep, setSweep] = useState(null);
  const [batch, setBatch] = useState(null);
  const [log, setLog] = useState(saved.log ?? []);
  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify({ url, ivs, prompt, system, log: log.slice(-30) })); } catch { /* ignore */ } }, [url, ivs, prompt, system, log]);
  const body = (p = prompt) => ({ prompt: p, system, chat, max_new_tokens: maxNew, temperature, seed: 0, interventions: ivs });
  const guard = async (label, fn) => {
    setBusy(label); setErr('');
    try { await fn(); } catch (e) { setErr(e.message); }
    setBusy('');
  };
  const run = () => guard('Generating three completions...', async () => {
    const r = await call(url, '/run', { ...body(), control });
    setRes(r);
    setLog((l) => [...l, { when: new Date().toISOString(), model: info?.model_id, prompt, ivs, baseline: r.baseline.text, steered: r.steered.text, control: r.control?.text ?? null }]);
  });
  const runSweep = () => guard('Sweeping the dose...', async () => setSweep(await call(url, '/sweep', { ...body(), max_new_tokens: Math.min(maxNew, 60), multipliers: [0, 0.25, 0.5, 1, 1.5, 2] })));
  const runBatch = () => guard('Running the prompt set...', async () => {
    const rows = [];
    for (const p of PROMPT_SET) {
      const r = await call(url, '/run', { ...body(p), max_new_tokens: Math.min(maxNew, 60), control: true });
      rows.push({ prompt: p, base: r.baseline.text, steer: r.steered.text, ctrl: r.control.text });
      setBatch([...rows]);
    }
  });
  const exportLog = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(log, null, 2)], { type: 'application/json' }));
    a.download = 'inann_steering_runs.json';
    a.click();
  };
  const ready = !!info && !busy;
  return (
    <main className="scroll-thin flex-1 overflow-y-auto p-4">
      <div className="mx-auto grid max-w-7xl gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <div className="space-y-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-dim">LLM steering</p>
            <p className="mt-2 text-sm leading-relaxed text-slate-100">
              Push a few MLP neurons of a real language model up or down at every token, and see what the model writes, next to what it writes unsteered and
              with random neurons pushed just as hard.
            </p>
          </div>
          <Connect url={url} setUrl={setUrl} info={info} setInfo={setInfo} err={err} setErr={setErr} />
          <Interventions ivs={ivs} setIvs={setIvs} info={info} />
          <Card title="Layers">
            <LayerStrip info={info} ivs={ivs} result={res} />
            <p className="text-[10px] text-dim">One bar per layer, input at the bottom. Dots are steered neurons (position along the bar is the neuron index). After a run, the shading is the median residual norm leaving each layer.</p>
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="3. Prompt and run">
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} className={`${inputCls} w-full text-sm`} />
            <div className="flex flex-wrap gap-1">{PROMPT_SET.map((p) => <button key={p} onClick={() => setPrompt(p)} className="rounded border border-line px-1.5 py-0.5 text-[10.5px] text-dim hover:text-slate-200">{p.length > 40 ? p.slice(0, 40) + '…' : p}</button>)}</div>
            <input value={system} onChange={(e) => setSystem(e.target.value)} placeholder="system prompt (optional)" className={`${inputCls} w-full`} />
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-300">
              <label className="flex items-center gap-1"><input type="checkbox" checked={chat} onChange={(e) => setChat(e.target.checked)} /> chat template</label>
              <label className="flex items-center gap-1"><input type="checkbox" checked={control} onChange={(e) => setControl(e.target.checked)} /> random-neuron control</label>
              <label className="flex items-center gap-1">max tokens <input type="number" value={maxNew} onChange={(e) => setMaxNew(+e.target.value)} className={`${inputCls} w-16`} /></label>
              <label className="flex items-center gap-1">temperature <input type="number" step="0.1" min="0" value={temperature} onChange={(e) => setTemperature(+e.target.value)} className={`${inputCls} w-14`} /></label>
            </div>
            <div className="flex flex-wrap items-center gap-1">
              <Btn on onClick={run} disabled={!ready}>Run</Btn>
              <Btn onClick={runSweep} disabled={!ready}>Dose sweep (×0 to ×2)</Btn>
              <Btn onClick={runBatch} disabled={!ready}>Run the prompt set with control</Btn>
              {busy && <span className="text-[11px] text-warm">{busy} (each completion takes several seconds on a laptop GPU)</span>}
              {!info && <span className="text-[11px] text-dim">connect to a server first</span>}
            </div>
            <p className="text-[10.5px] text-dim">Temperature 0 is greedy decoding, so a rerun gives the same text; all three columns use the same decoding settings.</p>
          </Card>
          <Results res={res} ivs={ivs} />
          {sweep && (
            <Card title="Dose sweep">
              <div className="space-y-1">
                {sweep.rows.map((r) => (
                  <div key={r.multiplier} className="grid grid-cols-[4rem_minmax(0,1fr)] gap-2 border-b border-line pb-1 text-xs">
                    <span className="tabular-nums text-warm">× {r.multiplier}</span>
                    <span className="whitespace-pre-wrap text-slate-200">{r.text}</span>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-400">Every dose in your list is multiplied by the same factor. ×0 should match the unsteered text exactly; a real effect usually changes content before it breaks grammar.</p>
            </Card>
          )}
          {batch && (
            <Card title="Prompt set, with the random-neuron control">
              <div className="overflow-x-auto">
                <table className="w-full text-[11px]">
                  <thead><tr className="text-left text-dim"><th className="w-40 pr-2">prompt</th><th className="pr-2">unsteered</th><th className="pr-2">steered</th><th>random neurons</th></tr></thead>
                  <tbody>
                    {batch.map((r) => (
                      <tr key={r.prompt} className="border-t border-line align-top text-slate-200">
                        <td className="pr-2 text-dim">{r.prompt}</td><td className="pr-2">{r.base}</td><td className="pr-2">{r.steer}</td><td>{r.ctrl}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
          <Card title="Run log">
            <div className="flex items-center gap-1 text-[11px] text-dim">
              {log.length} runs kept in this browser.
              <Btn onClick={exportLog} disabled={!log.length}>Download as JSON</Btn>
              <Btn onClick={() => setLog([])} disabled={!log.length}>Clear</Btn>
            </div>
          </Card>
        </div>
      </div>
    </main>
  );
}
