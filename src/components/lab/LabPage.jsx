// The Lab: load any ONNX model and walk through it. Rooms for its layers, a room per unit with
// its dataset statistics, live clamping, a sparse autoencoder trained in the browser and docked
// beside any layer, and the latent space of every layer.
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import LabScene from './LabScene.jsx';
import LatentCloud from '../LatentCloud.jsx';
import { Thumb, VectorStrip, Hist, Curve, Field, selectCls } from './LabParts.jsx';
import { Btn, Term, Cite, NotShown, Boundary } from '../ui.jsx';
import { LabModel, layoutOf, softmax } from '../../lib/lab/runtime.js';
import {
  inputKind, NORMS, normFromMeta, toModel, drawingToPixels, imageFileToPixels, datasetFromJson, datasetFromImages,
  parseNpy, datasetFromNpy, modelInputOf, displayOf, textToIds, idsToText,
} from '../../lib/lab/data.js';
import { collect, channelActs, topItems, histogramOf, labelPurity } from '../../lib/lab/store.js';
import { itemVectors, sampleRows, featuresNow, saeTransform, neuronTransform } from '../../lib/lab/saeLab.js';
import { toJSON, fromJSON } from '../../lib/lab/saeCore.js';
import { linearSource, incomingWeights, preFlattenTensor } from '../../lib/lab/graph.js';
import { PixelGrid } from '../ui.jsx';
import { describeUnit, searchUnits } from '../../lib/lab/summary.js';
import { ScoreBadge } from '../FeatureRoom.jsx';
import { occlusionMap, readClassProb, readChannelMax } from '../../lib/lab/occlusion.js';
import { unitEdges } from '../../lib/lab/edges.js';
import { pcaOf, knnAccOf, separationOf } from '../../lib/latent.js';

const BASE = import.meta.env.BASE_URL;
export const EXAMPLES = [
  { file: 'mnist_cnn', label: 'MNIST CNN' },
  { file: 'mnist_mlp', label: 'MNIST MLP' },
  { file: 'optdigits_cnn', label: 'Tour 8x8 CNN' },
  { file: 'paren_transformer', label: 'Paren transformer' },
  { file: 'cifar_cnn', label: 'CIFAR-10 CNN (colour)' },
];

const unitOf = (L, i) => (L.kind === 'spatial' ? { c: Math.floor(i / L.P), p: i % L.P } : L.kind === 'seq' ? { c: i % L.C, p: Math.floor(i / L.C) } : { c: i, p: 0 });
const posText = (L, p) => (L.kind === 'spatial' ? `row ${Math.floor(p / L.W)}, col ${p % L.W}` : L.kind === 'seq' ? `token ${p}` : '');
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };

// where in the input a spatial unit at position p sits (approximate: scales the grid onto the image)
function boxFor(L, p, kind) {
  if (L.kind !== 'spatial' || kind?.kind !== 'image') return null;
  const y = Math.floor(p / L.W), x = p % L.W, sy = kind.H / L.H, sx = kind.W / L.W;
  return { x: x * sx, y: y * sy, w: Math.max(1, sx), h: Math.max(1, sy) };
}

function labelRooms(rooms) {
  const seen = {};
  return rooms.map((r) => {
    seen[r.op] = (seen[r.op] ?? 0) + 1;
    return { ...r, label: r.isInputRoom ? 'Input' : r.isOutput ? 'Output' : `${r.op} ${seen[r.op]}` };
  });
}

export default function LabPage() {
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [model, setModel] = useState(null);
  const [meta, setMeta] = useState({});
  const [kind, setKind] = useState(null);
  const [norm, setNorm] = useState('unit');
  const [prep, setPrep] = useState('mnist');
  const [pixels, setPixels] = useState(null); // display form of the current input
  const [ds, setDs] = useState(null);
  const [item, setItemRaw] = useState(0);
  const [source, setSource] = useState('item'); // 'item' (a dataset item) or 'custom' (drawn, uploaded, random)
  const setItem = useCallback((v) => { setSource('item'); setItemRaw(v); }, []);
  const setCustom = useCallback((p) => { setSource('custom'); setPixels(p); }, []);
  const [store, setStore] = useState(null);
  const [progress, setProgress] = useState(null);
  const [maxItems, setMaxItems] = useState(1000);
  const [roomNames, setRoomNames] = useState([]);
  const [outs, setOuts] = useState(null);
  const [baseOuts, setBaseOuts] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  const [steer, setSteer] = useState(null);
  const [sae, setSae] = useState(null); // {room, sae, metrics, stats, iv}
  const [view, setView] = useState('space');
  const [frameNonce, setFrameNonce] = useState(0);
  const [focusRoom, setFocusRoom] = useState(null);
  const [expl, setExpl] = useState(null); // explanations from scripts/explain_lab.py
  const [pinned, setPinned] = useState(null);
  const [outKind, setOutKind] = useState('classes'); // how to read the first output: class scores or raw values
  const [modelSrc, setModelSrc] = useState(null); // {example} or {url, metaUrl, dataUrl}, saved in sessions
  const [pendingSession, setPendingSession] = useState(null); // {outs, pixels, label} of a pinned input, for difference mode
  const explFor = useCallback((room, unit, isFeat) => {
    if (!expl || expl.tensor !== room || expl.sae !== !!isFeat) return null;
    return expl.byUnit.get(unit) ?? null;
  }, [expl]);
  const runId = useRef(0);
  const rightPanel = useRef(null);

  // ---------- loading ----------
  const lastBytes = useRef(null);
  const loadModel = useCallback(async (bytes, m = {}, fileName = 'model.onnx', dataset = null) => {
    lastBytes.current = { bytes, m, fileName, dataset };
    setError(''); setStatus(`Loading ${fileName}...`);
    passId.current++; setProgress(null);
    setStore(null); setSae(null); setSteer(null); setSelected(null); setOuts(null); setPinned(null);
    try {
      const lm = await LabModel.load(bytes, m, fileName);
      const k = inputKind(lm.inputDims, m, lm.inputType);
      setModel(lm); setMeta(m); setKind(k);
      setNorm(normFromMeta(m));
      setPrep(m.input?.preprocess ?? (k.kind === 'image' && k.H === 28 ? 'mnist' : 'fit'));
      setRoomNames(lm.rooms.map((r) => r.name));
      const perItem = lm.rooms.reduce((a, r) => a + (r.layout?.size ?? 0), 0);
      setMaxItems(lm.summary.params > 5e6 || perItem > 2e5 ? 100 : 1000);
      // one output value is a regression unless the meta file says otherwise
      const outSize = (lm.shapes[lm.outputName] ?? []).reduce((a, b) => a * b, 1);
      setOutKind(m.output?.kind ?? (outSize === 1 ? 'values' : 'classes'));
      setPixels(new Float32Array(k.kind === 'image' ? k.C * k.H * k.W : lm.inputSize));
      if (k.kind === 'tokens' && !dataset) setSource('custom');
      setDs(dataset);
      setSource(dataset ? 'item' : 'custom');
      setItemRaw(0);
      setFrameNonce((n) => n + 1);
      setStatus('');
    } catch (e) {
      setError(`Could not load this model: ${e.message ?? e}`);
      setStatus('');
    }
  }, []);

  const loadExample = useCallback(async (file) => {
    setStatus(`Fetching ${file}...`);
    const [bytes, m] = await Promise.all([
      fetch(`${BASE}examples/${file}.onnx`).then((r) => r.arrayBuffer()),
      fetch(`${BASE}examples/${file}.meta.json`).then((r) => r.json()),
    ]);
    const dsJson = m.dataset ? await fetch(`${BASE}examples/${m.dataset}`).then((r) => r.json()) : null;
    await loadModel(bytes, m, `${file}.onnx`, dsJson ? datasetFromJson(dsJson, m.dataset.replace('.json', '')) : null);
    setModelSrc({ example: file });
  }, [loadModel]);

  const loadUrl = useCallback(async (url, metaUrl, dataUrl) => {
    try {
      setStatus(`Fetching ${url}...`);
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const bytes = await res.arrayBuffer();
      const m = metaUrl ? await fetch(metaUrl).then((r) => r.json()) : {};
      const dsUrl = dataUrl ?? (m.dataset ? new URL(m.dataset, new URL(metaUrl ?? url, location.href)).href : null);
      const dsJson = dsUrl ? await fetch(dsUrl).then((r) => r.json()) : null;
      await loadModel(bytes, m, url.split('/').pop().split('?')[0], dsJson ? datasetFromJson(dsJson, dsUrl.split('/').pop()) : null);
      setModelSrc({ url, metaUrl, dataUrl });
    } catch (e) { setError(`Could not fetch ${url}: ${e.message ?? e}. The server must allow cross-origin requests.`); setStatus(''); }
  }, [loadModel]);

  useEffect(() => {
    const q = new URLSearchParams(location.search);
    if (q.get('model')) loadUrl(q.get('model'), q.get('meta'), q.get('data'));
    else loadExample('mnist_cnn');
  }, [loadExample, loadUrl]);

  const onFiles = async (files) => {
    const list = [...files];
    const onnx = list.find((f) => f.name.endsWith('.onnx'));
    const metaFile = list.find((f) => f.name.endsWith('.json'));
    let m = {};
    if (metaFile) {
      try { m = JSON.parse(await metaFile.text()); } catch { setError('The .json next to the model is not valid JSON; ignoring it.'); }
    }
    if (!onnx && m.kind === 'inann-lab-explanations') {
      setExpl({ ...m, byUnit: new Map(m.units.map((u) => [u.unit, u])) });
      return;
    }
    if (!onnx && m.kind === 'inann-lab-session') { openSession(m); return; }
    if (onnx) { await loadModel(await onnx.arrayBuffer(), m, onnx.name, null); setModelSrc({ file: onnx.name }); }
    else if (metaFile && !onnx) setError('Drop the .onnx file too (the .json only describes it).');
  };

  // ---------- the current input ----------
  const useItem = source === 'item' && ds && model && kind;
  const itemIdx = ds ? Math.min(item, ds.n - 1) : 0;
  const shownPixels = useMemo(() => (useItem ? displayOf(ds, itemIdx, kind) : pixels), [useItem, ds, itemIdx, kind, pixels]);
  const xFinal = useMemo(() => {
    if (!model || !kind) return null;
    if (useItem) return modelInputOf(ds, itemIdx, kind, norm);
    if (!pixels) return null;
    return kind.kind === 'image' ? toModel(pixels, kind, norm) : Float32Array.from(pixels);
  }, [model, kind, useItem, ds, itemIdx, norm, pixels]);

  // run it (and the steered version when a clamp is set)
  useEffect(() => {
    if (!model || !xFinal) return;
    const id = ++runId.current;
    (async () => {
      try {
        const base = await model.run([xFinal]);
        let out = base;
        if (steer) {
          const r = model.describe(steer.room);
          const tf = steer.sae && sae ? saeTransform(sae.sae, r.layout, 'steer', steer.feature, steer.value) : neuronTransform(r.layout, steer.c, steer.p, steer.value);
          out = await model.runWith(xFinal, steer.room, tf);
        }
        if (id === runId.current) { setBaseOuts(base); setOuts(out); }
      } catch (e) { setError(`Inference failed: ${e.message ?? e}`); }
    })();
  }, [model, xFinal, steer, sae]);

  // ---------- dataset analysis ----------
  const rooms = useMemo(() => (model ? roomNames.map((n) => model.describe(n)).filter((r) => r.dims) : []), [model, roomNames]);
  const passId = useRef(0);
  const analyze = useCallback(async () => {
    if (!model || !ds) return;
    const id = ++passId.current; // a newer pass, model or dataset makes this one stale
    setProgress(0);
    try {
      const st = await collect(model, ds, kind, norm, rooms, { max: maxItems, onProgress: (f) => id === passId.current && setProgress(f) });
      if (id !== passId.current) return;
      setSae(null); setSteer(null); // an SAE's statistics index the store it was built on
      setStore(st);
    } catch (e) { if (id === passId.current) setError(`Dataset pass failed: ${e.message ?? e}`); }
    if (id === passId.current) setProgress(null);
  }, [model, ds, kind, norm, rooms, maxItems]);
  // whenever the store is cleared (new model, dataset, rooms, session), start a fresh pass
  useEffect(() => { if (model && ds && !store) analyze(); }, [model, ds, store, rooms]); // eslint-disable-line

  const onDatasetFiles = async (files, invert) => {
    const list = [...files];
    try {
      if (list.some((f) => f.name.endsWith('.npy'))) {
        const xs = list.find((f) => /x|data|input/i.test(f.name) && f.name.endsWith('.npy')) ?? list.find((f) => f.name.endsWith('.npy'));
        const ys = list.find((f) => f !== xs && f.name.endsWith('.npy'));
        const X = parseNpy(await xs.arrayBuffer());
        const Y = ys ? parseNpy(await ys.arrayBuffer()) : null;
        if (X.data.length / X.shape[0] !== model.inputSize) throw new Error(`each item has ${X.data.length / X.shape[0]} values but the model takes ${model.inputSize}`);
        setDs(datasetFromNpy(X, Y, kind, norm !== 'unit'));
      } else if (list.length === 1 && list[0].name.endsWith('.json')) {
        setDs(datasetFromJson(JSON.parse(await list[0].text()), list[0].name));
      } else {
        setStatus('Reading images...');
        setDs(await datasetFromImages(list, kind, invert));
        setStatus('');
      }
      setStore(null); setSae(null); setItemRaw(0); setSource('item');
    } catch (e) { setError(`Could not read that dataset: ${e.message ?? e}`); }
  };

  // ---------- 3D values ----------
  const inputRoom = useMemo(() => {
    if (!model || !kind) return null;
    const dims = kind.kind === 'image' ? [kind.C, kind.H, kind.W] : kind.kind === 'tokens' ? [kind.T] : model.inputDims;
    return { name: '__input', op: 'Input', dims, layout: layoutOf(dims), isInputRoom: true, isTokens: kind.kind === 'tokens' };
  }, [model, kind]);
  const sceneRooms = useMemo(() => {
    if (!inputRoom) return [];
    const r = { ...inputRoom };
    if (r.isTokens && xFinal) r.podLabels = Array.from(xFinal, (v) => (kind.vocab ? kind.vocab[v] ?? '?' : String(v)));
    return labelRooms([r, ...rooms]);
  }, [inputRoom, rooms, xFinal, kind]);
  const values = useMemo(() => {
    const v = {};
    if (!outs) return v;
    if (pinned) {
      // difference mode: this input minus the pinned one, per unit
      for (const r of rooms) {
        const a = outs[r.name]?.data, b = pinned.outs[r.name]?.data;
        if (a && b && a.length === b.length) v[r.name] = Float32Array.from(a, (x, i) => x - b[i]);
      }
      return v;
    }
    for (const r of rooms) if (outs[r.name]) v[r.name] = r.isOutput && r.layout.size <= 32 && outKind === 'classes' ? Float32Array.from(maybeSoftmax(outs[r.name].data)) : outs[r.name].data;
    if (inputRoom && kind?.kind === 'tokens' && xFinal) v.__input = Float32Array.from(xFinal, () => 0.6);
    else if (inputRoom && kind?.kind !== 'image' && xFinal) v.__input = xFinal;
    return v;
  }, [outs, rooms, inputRoom, kind, xFinal, pinned, outKind]);
  const scales = useMemo(() => {
    const s = {};
    for (const r of rooms) {
      const st = store?.stats[r.name];
      if (pinned) {
        let m = 0;
        for (const x of values[r.name] ?? []) m = Math.max(m, Math.abs(x));
        s[r.name] = m || 1; s[`${r.name}:signed`] = true;
        continue;
      }
      if (r.isOutput && r.layout.size <= 32 && outKind === 'classes') { s[r.name] = 1; continue; }
      if (st) { s[r.name] = st.scale; s[`${r.name}:signed`] = st.signed; }
      else if (values[r.name]) { let m = 0; for (const v of values[r.name]) m = Math.max(m, Math.abs(v)); s[r.name] = m || 1; s[`${r.name}:signed`] = values[r.name].some((v) => v < 0); }
    }
    if (values.__input && kind?.kind === 'tokens') s.__input = 1;
    else if (values.__input) { let m = 0; for (const v of values.__input) m = Math.max(m, Math.abs(v)); s.__input = m || 1; s['__input:signed'] = values.__input.some((v) => v < 0); }
    return s;
  }, [rooms, store, values, pinned, outKind]);

  const out = outs && model ? outs[model.outputName]?.data : null;
  const probs = out && out.length <= 1000 && outKind === 'classes' ? maybeSoftmax(out) : null;
  const pred = probs ? probs.indexOf(Math.max(...probs)) : null;
  const classes = meta.classes ?? ds?.classes ?? null;

  const saeNow = useMemo(() => {
    if (!sae || !outs?.[sae.room]) return null;
    const L = model.describe(sae.room).layout;
    const f = featuresNow(sae.sae, outs[sae.room].data, L);
    const { D, Wd } = sae.sae;
    const links = (fi) => Array.from({ length: D }, (_, c) => ({ c, w: Wd[fi * D + c] })).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, Math.min(12, D));
    return { room: sae.room, F: sae.sae.F, k: sae.sae.k, featNow: f.max, argNow: f.arg, scale: sae.featScale, links };
  }, [sae, outs, model]);

  const onSelect = useCallback((s) => { if (s.room !== '__input') setSelected(s); }, []);
  // incoming edges of the selected unit on the current input (none for SAE features)
  const edges = useMemo(() => {
    if (!selected || selected.sae || !outs || !model) return null;
    try {
      const r = model.describe(selected.room);
      return unitEdges(model, roomNames, outs, selected.room, selected.index, r.layout, xFinal);
    } catch { return null; }
  }, [selected, outs, model, roomNames, xFinal]);
  // a clamp on an SAE feature belongs to that SAE; retraining or removing it releases the clamp
  useEffect(() => { setSteer((st) => (st?.sae ? null : st)); }, [sae?.sae]);
  const summaries = useMemo(() => {
    if (!store) return [];
    const out = [];
    const labelled = labelRooms([{ isInputRoom: true, op: 'Input' }, ...rooms]).slice(1);
    labelled.forEach((r) => {
      const st = store.stats[r.name];
      if (!st || r.isOutput) return;
      for (let c = 0; c < st.layout.C && c < 4096; c++) {
        const acts = channelActs(store, r.name, c);
        const sm = describeUnit(acts, (i) => st.chArg[i * st.layout.C + c], st.layout, store.labels, classes);
        out.push({ key: `${r.name}:${c}`, room: r.name, roomLabel: r.label, index: st.layout.idx(c, 0), c, summary: sm, className: sm.purity ? classes?.[sm.purity.label] ?? sm.purity.label : null, expl: explFor(r.name, c, false) });
      }
    });
    if (sae?.stats) {
      const L = model.describe(sae.room).layout, F = sae.sae.F;
      for (let f = 0; f < F; f++) {
        const acts = Float32Array.from({ length: sae.N }, (_, i) => sae.stats.featMax[i * F + f]);
        if (!acts.some((v) => v > 0)) continue;
        const sm = describeUnit(acts, (i) => sae.stats.featArg[i * F + f], L, store.labels, classes);
        out.push({ key: `sae:${f}`, room: sae.room, roomLabel: 'SAE', index: f, c: f, sae: true, summary: sm, className: sm.purity ? classes?.[sm.purity.label] ?? sm.purity.label : null, expl: explFor(sae.room, f, true) });
      }
    }
    return out;
  }, [store, rooms, sae, classes, model, explFor]);
  useEffect(() => { if (sae) setFrameNonce((n) => n + 1); }, [sae?.room, sae?.sae]); // eslint-disable-line
  useEffect(() => { if (selected) rightPanel.current?.scrollTo({ top: 0, behavior: 'smooth' }); }, [selected]);
  const onHover = useCallback((h) => setHover(h), []);

  // ---------- sessions: save what you set up, reopen it later ----------
  const saveSession = () => {
    const sess = {
      kind: 'inann-lab-session', version: 1, savedAt: new Date().toISOString(),
      model: { fileName: model.fileName, params: model.summary.params, src: modelSrc }, meta,
      settings: { norm, prep, outKind, maxItems, roomNames },
      sae: sae ? { room: sae.room, metrics: sae.metrics, json: toJSON(sae.sae) } : null,
      explanations: expl ? { ...expl, byUnit: undefined } : null,
    };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(sess)], { type: 'application/json' }));
    a.download = `inann_session_${model.fileName.replace(/\.onnx$/, '')}.json`;
    a.click();
  };
  const openSession = async (sess) => {
    const src = sess.model?.src;
    if (src?.example) await loadExample(src.example);
    else if (src?.url) await loadUrl(src.url, src.metaUrl, src.dataUrl);
    else if (!model || model.fileName !== sess.model.fileName) {
      setError(`This session is for ${sess.model.fileName}. Drop that model first, then the session again.`);
      return;
    }
    setPendingSession(sess);
  };
  useEffect(() => {
    // apply settings once the model is in; the SAE waits for the dataset pass
    if (!pendingSession || !model || model.fileName !== pendingSession.model.fileName) return;
    const st = pendingSession.settings;
    if (!pendingSession.appliedSettings) {
      setNorm(st.norm); setPrep(st.prep); setOutKind(st.outKind); setMaxItems(st.maxItems);
      setRoomNames(st.roomNames.filter((n) => model.shapes[n]));
      if (pendingSession.explanations) setExpl({ ...pendingSession.explanations, byUnit: new Map(pendingSession.explanations.units.map((u) => [u.unit, u])) });
      setStore(null);
      setPendingSession({ ...pendingSession, appliedSettings: true });
      return;
    }
    if (pendingSession.sae && store && store.stats[pendingSession.sae.room]?.full) {
      const s = fromJSON(pendingSession.sae.json);
      const iv = itemVectors(store, pendingSession.sae.room);
      const w = new Worker(new URL('../../lib/lab/sae.worker.js', import.meta.url), { type: 'module' });
      w.onmessage = (e) => {
        if (e.data.type !== 'done') return;
        const a = Array.from(e.data.stats.featMax).filter((v) => v > 0).sort((x, y) => x - y);
        const alive = new Set(Array.from(e.data.stats.featMax.keys()).filter((i) => e.data.stats.featMax[i] > 0).map((i) => i % s.F)).size;
        setSae({ room: pendingSession.sae.room, sae: s, metrics: pendingSession.sae.metrics, stats: e.data.stats, N: store.N, featScale: a[Math.floor(a.length * 0.99)] || 1, alive });
        w.terminate();
      };
      w.postMessage({ mode: 'stats', sae: s, items: { items: iv.items, N: iv.N, P: iv.P } });
      setPendingSession(null);
    } else if (!pendingSession.sae && store) setPendingSession(null);
  }, [pendingSession, model, store]); // eslint-disable-line

  // keyboard: arrows step through dataset items, P pins or unpins, Esc closes the room
  useEffect(() => {
    const key = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'ArrowRight' && ds) setItem((i) => Math.min(ds.n - 1, i + 1));
      else if (e.key === 'ArrowLeft' && ds) setItem((i) => Math.max(0, i - 1));
      else if ((e.key === 'p' || e.key === 'P') && outs) setPinned((pn) => (pn ? null : { outs, pixels: shownPixels, x: xFinal }));
      else if (e.key === 'Escape') setSelected(null);
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [ds, outs, shownPixels, xFinal, setItem]);

  // drop a model anywhere on the page
  const [dragAll, setDragAll] = useState(false);
  useEffect(() => {
    const over = (e) => { if ([...(e.dataTransfer?.items ?? [])].some((i) => i.kind === 'file')) { e.preventDefault(); setDragAll(true); } };
    const leave = (e) => { if (!e.relatedTarget) setDragAll(false); };
    const drop = (e) => {
      const files = [...(e.dataTransfer?.files ?? [])];
      setDragAll(false);
      if (!files.length) return;
      e.preventDefault();
      if (files.some((f) => f.name.endsWith('.onnx')) || (files.length === 1 && /explanations|session/i.test(files[0].name))) onFiles(files);
      else onDatasetFiles(files, false);
    };
    window.addEventListener('dragover', over);
    window.addEventListener('dragleave', leave);
    window.addEventListener('drop', drop);
    return () => { window.removeEventListener('dragover', over); window.removeEventListener('dragleave', leave); window.removeEventListener('drop', drop); };
  }); // re-bind each render so the handlers see the current model

  // ---------- render ----------
  return (
    <main className="relative flex min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
      {dragAll && (
        <div className="pointer-events-none absolute inset-0 z-50 flex items-center justify-center border-2 border-dashed border-glow bg-black/60 text-sm text-glow">
          Drop an .onnx model (with an optional .meta.json), or images / .npy / .json as a dataset
        </div>
      )}
      <aside className="scroll-thin order-2 w-full shrink-0 space-y-3 border-line p-4 lg:order-1 lg:w-80 lg:overflow-y-auto lg:border-r">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-dim">Lab · your model</p>
          <p className="mt-2 text-sm leading-relaxed text-slate-100">Load any ONNX model and walk through its layers, then train a sparse autoencoder on one of them, right here in the browser.</p>
          <details className="mt-2 rounded-md border border-line px-2 py-1 text-xs text-slate-300">
            <summary className="cursor-pointer text-dim hover:text-slate-200">How to read the Lab</summary>
            <ol className="mt-1 list-decimal space-y-1 pl-4 leading-snug">
              <li>Each floor of the tower is one layer. Brightness is how strongly each unit fires on the current input.</li>
              <li>Click a unit, search for one, or pick it by number. Its room shows what it fires on across the dataset, its weights, and a clamp.</li>
              <li>"What does it control?" clamps the unit on many items and counts how the answer moves. That is evidence of cause, not just correlation.</li>
              <li>Train a sparse autoencoder on a layer to split it into features that are often easier to name. They dock beside the layer.</li>
              <li>The latent view places every dataset item by how a layer represents it.</li>
              <li>Keys: ← → step through dataset items, P pins the input for comparison, Esc closes a room.</li>
            </ol>
          </details>
        </div>
        <ModelCard model={model} meta={meta} onFiles={onFiles} onExample={loadExample} onUrl={loadUrl} status={status} error={error} clearError={() => setError('')} onSave={saveSession} pending={!!pendingSession}
          onDims={(dims) => { const L = lastBytes.current; if (L) loadModel(L.bytes, { ...L.m, input: { ...(L.m.input ?? {}), dims } }, L.fileName, L.dataset); }} />
        {model && kind && (
          <InputCard
            model={model} kind={kind} norm={norm} setNorm={setNorm} prep={prep} setPrep={setPrep} pixels={shownPixels} setPixels={setCustom}
            ds={ds} item={item} setItem={setItem} probs={probs} classes={classes} pred={pred} xFinal={xFinal} source={source}
            outKind={outKind} setOutKind={setOutKind} rawOut={out} metaClasses={!!meta.classes}
          />
        )}
        {model && kind && (
          <DataCard ds={ds} store={store} progress={progress} analyze={analyze} onFiles={onDatasetFiles} kind={kind} maxItems={maxItems} setMaxItems={setMaxItems} />
        )}
        {model && <ExplCard expl={expl} setExpl={setExpl} model={model} sae={sae} />}
        {model && (
          <RoomsCard model={model} roomNames={roomNames} setRoomNames={(n) => { setRoomNames(n); setStore(null); setSae(null); setSteer(null); }} />
        )}
      </aside>

      <section className="relative order-1 h-[62vh] min-w-0 shrink-0 overflow-hidden lg:order-2 lg:h-auto lg:flex-1 lg:shrink">
        <div className="absolute left-3 top-3 z-20 flex overflow-hidden rounded-md border border-line text-xs">
          {[['space', '3D space'], ['gallery', 'Gallery'], ['report', 'Report'], ['latent', 'Latent space']].map(([k, t]) => (
            <button key={k} onClick={() => setView(k)} className={`px-2.5 py-1 ${view === k ? 'bg-glow/15 text-glow' : 'bg-panel text-dim hover:text-slate-200'}`}>{t}</button>
          ))}
        </div>
        {view === 'space' && sceneRooms.length > 0 && (
          <Boundary>
            <LabScene
              rooms={sceneRooms} values={values} scales={scales} inputPixels={kind?.kind === 'image' ? shownPixels : null} inputChannels={kind?.C}
              classes={classes} pred={pred} selected={selected} onSelect={onSelect} onHover={onHover} sae={saeNow} frameNonce={frameNonce} focusRoom={focusRoom} edges={edges}
            />
          </Boundary>
        )}
        {view === 'gallery' && (
          <LabGallery store={store} rooms={sceneRooms.slice(1)} sae={sae} ds={ds} kind={kind} classes={classes} explFor={explFor}
            onPick={(sel) => setSelected(sel)} selected={selected} />
        )}
        {view === 'report' && <LabReport store={store} rooms={sceneRooms.slice(1)} summaries={summaries} sae={sae} onPick={(e) => setSelected({ room: e.room, index: e.index, sae: e.sae })} />}
        {view === 'latent' && <LabLatent model={model} store={store} rooms={rooms} sae={sae} outs={outs} ds={ds} setItem={setItem} classes={classes} />}
        <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/50 px-2 py-1 text-[11px] text-slate-300">
          {hover && outs ? <HoverText hover={hover} model={model} outs={outs} saeNow={saeNow} /> : view === 'space' ? 'Drag to orbit, scroll to zoom, click any unit.' : view === 'report' ? 'Every number here is counted from the dataset pass.' : view === 'gallery' ? 'Click a card to open that unit\'s room.' : 'Each mark is one dataset item, placed by PCA of the chosen layer.'}
        </div>
        {summaries.length > 0 && <LabSearch entries={summaries} classes={classes} onPick={(e) => { setView('space'); setSelected({ room: e.room, index: e.index, sae: e.sae }); setFocusRoom({ i: sceneRooms.findIndex((r) => r.name === e.room), sae: e.sae, nonce: Math.random() }); }} />}
        {view === 'space' && sceneRooms.length > 3 && (
          <div className="absolute right-3 top-14 z-20 hidden max-h-[70%] space-y-px overflow-y-auto scroll-thin rounded-md border border-line bg-panel/80 p-1 text-[10px] md:block">
            <div className="px-1 pb-0.5 text-dim">fly to</div>
            {[...sceneRooms.map((r, i) => ({ r, i }))].reverse().map(({ r, i }) => (
              <button key={r.name} onClick={() => setFocusRoom({ i, nonce: Math.random() })} className="block w-full rounded px-1.5 py-0.5 text-left text-slate-300 hover:bg-glow/10 hover:text-glow">
                {r.label}
              </button>
            ))}
            {saeNow && (
              <button onClick={() => setFocusRoom({ i: sceneRooms.findIndex((r) => r.name === saeNow.room), sae: true, nonce: Math.random() })} className="block w-full rounded px-1.5 py-0.5 text-left text-warm hover:bg-warm/10">
                SAE features
              </button>
            )}
            <button onClick={() => { setFocusRoom(null); setFrameNonce((n) => n + 1); }} className="block w-full rounded px-1.5 py-0.5 text-left text-dim hover:text-slate-200">whole model</button>
          </div>
        )}
        {view === 'space' && outs && (
          <button
            className="absolute bottom-12 right-3 z-20 rounded-md border border-line bg-panel/90 px-2 py-1 text-[11px] text-slate-300 hover:border-glow/40"
            onClick={(e) => {
              const cv = e.currentTarget.parentElement.querySelector('canvas');
              const a = document.createElement('a');
              a.href = cv.toDataURL('image/png');
              a.download = `inann_${(model?.fileName ?? 'model').replace(/\.onnx$/, '')}.png`;
              a.click();
            }}
          >
            Save image
          </button>
        )}
        {view === 'space' && outs && (
          <div className="absolute bottom-12 left-3 z-20 flex items-center gap-2 rounded-md border border-line bg-panel/90 px-2 py-1 text-[11px] text-slate-300">
            {pinned ? (
              <>
                <span>
                  Showing <b className="text-warm">this input minus the pinned one</b>; orange is higher now, blue lower. Each room is scaled to its own largest change.
                </span>
                {kind?.kind === 'image' && <Thumb pixels={pinned.pixels} kind={kind} size={28} />}
                <Btn onClick={() => setPinned(null)}>unpin</Btn>
                {outKind === 'classes' && <PatchPanel model={model} rooms={sceneRooms.slice(1)} pinned={pinned} xNow={xFinal} classes={classes} />}
              </>
            ) : (
              <Btn onClick={() => setPinned({ outs, pixels: shownPixels, x: xFinal })} title="freeze this input, then change the input to compare">Pin this input to compare</Btn>
            )}
          </div>
        )}
        {steer && (
          <div className="absolute right-3 top-3 z-20 flex items-center gap-2 rounded-md border border-warm/50 bg-black/70 px-2 py-1 text-xs text-warm">
            Clamped {steer.sae ? `SAE feature ${steer.feature}` : `channel ${steer.c}${steer.p != null ? ' at one position' : ''}`} = {steer.value.toFixed(2)}
            <button className="text-slate-200 hover:text-white" onClick={() => setSteer(null)}>release</button>
          </div>
        )}
      </section>

      <aside ref={rightPanel} className="scroll-thin order-3 w-full shrink-0 space-y-3 border-line p-4 lg:w-[23rem] lg:overflow-y-auto lg:border-l">
        <Boundary key={selected ? `${selected.room}:${selected.index}:${!!selected.sae}` : 'none'}>
          <UnitCard
            model={model} sel={selected} outs={outs} baseOuts={baseOuts} rooms={sceneRooms} onPick={onSelect} norm={norm} explFor={explFor} pixelsNow={shownPixels} rawNow={useItem && !!ds?.raw} outKind={outKind} pinnedNow={pinned} xNow={xFinal} edges={edges} store={store} ds={ds} kind={kind} sae={sae} saeNow={saeNow}
            steer={steer} setSteer={setSteer} classes={classes} onClose={() => setSelected(null)} setItem={setItem}
            onPickFeature={(f) => setSelected({ room: sae.room, index: f, sae: true })}
          />
        </Boundary>
        {model && <Boundary><SaeCard model={model} rooms={sceneRooms.slice(1)} store={store} sae={sae} setSae={setSae} ds={ds} kind={kind} norm={norm} onShow={() => { setView('space'); }} saeNow={saeNow} onPickFeature={(f) => setSelected({ room: sae.room, index: f, sae: true })} outKind={outKind} /></Boundary>}
      </aside>
    </main>
  );
}

function maybeSoftmax(v) {
  const a = Array.from(v);
  const sum = a.reduce((s, t) => s + t, 0);
  const probsAlready = a.every((t) => t >= 0 && t <= 1) && Math.abs(sum - 1) < 1e-3;
  return probsAlready ? a : softmax(a);
}

function HoverText({ hover, model, outs, saeNow }) {
  if (hover.sae) return <>SAE feature {hover.index}: <span className="text-glow">{saeNow?.featNow[hover.index]?.toFixed(3)}</span></>;
  const r = model.describe(hover.room);
  if (!r?.dims) return null;
  const { c, p } = unitOf(r.layout, hover.index);
  const v = outs[hover.room]?.data[hover.index];
  return <>{r.op} · channel {c}{r.layout.P > 1 ? ` · ${posText(r.layout, p)}` : ''}: <span className="text-glow">{v?.toFixed(3)}</span></>;
}

// ---------- model ----------
function ModelCard({ model, meta, onFiles, onExample, onUrl, status, error, clearError, onSave, pending, onDims }) {
  const [drag, setDrag] = useState(false);
  const [url, setUrl] = useState('');
  const s = model?.summary;
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer.files); }}
        className={`rounded-md border border-dashed p-3 text-center text-xs ${drag ? 'border-glow bg-glow/10 text-glow' : 'border-line text-slate-300'}`}
      >
        Drop an <b>.onnx</b> file here (plus an optional <b>.meta.json</b>)
        <label className="mt-2 block cursor-pointer text-glow/90 hover:text-glow">
          or choose files
          <input type="file" accept=".onnx,.json" multiple className="hidden" onChange={(e) => onFiles(e.target.files)} />
        </label>
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[10px] text-dim">examples:</span>
        {EXAMPLES.map((ex) => <Btn key={ex.file} onClick={() => onExample(ex.file)}>{ex.label}</Btn>)}
      </div>
      <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); if (url.trim()) onUrl(url.trim()); }}>
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="or paste a URL to an .onnx file" className="min-w-0 flex-1 rounded border border-line bg-panel px-2 py-1 text-[11px] text-slate-100 placeholder:text-dim" />
        <Btn type="submit">Load</Btn>
      </form>
      {status && <div className="text-xs text-glow">{status}</div>}
      {error && (
        <div className="rounded border border-red-900 p-2 text-xs text-red-300">
          {error} <button className="ml-1 text-slate-300 underline" onClick={clearError}>dismiss</button>
        </div>
      )}
      {model && (
        <div className="space-y-1 text-xs text-slate-300">
          <div className="font-semibold text-slate-100">{meta.name ?? model.fileName}</div>
          {meta.description && <div className="text-dim">{meta.description}</div>}
          <div>
            {s.nodes} ops, {s.params.toLocaleString()} parameters · input {model.input.name} [{model.inputDims.join('×')}]{model.batchable ? ', batchable' : ''}
          </div>
          <div className="text-[10px] text-dim">
            {Object.entries(s.opCounts).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}×${v}`).join('  ')}
          </div>
          {model.symbolic?.some(Boolean) && <DimsEditor model={model} onDims={onDims} />}
          {model.extraInputs?.length > 0 && (
            <div className="text-[10px] text-dim">
              Other inputs ({model.extraInputs.map((i) => i.name).join(', ')}) are filled automatically: masks with ones, position ids with 0, 1, 2, ..., the rest with zeros.
            </div>
          )}
        </div>
      )}
      {model && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Btn onClick={onSave}>Save session</Btn>
          <label className="cursor-pointer rounded-md border border-line px-2.5 py-1 text-xs text-slate-200 hover:border-glow/40">
            Open session<input type="file" accept=".json" className="hidden" onChange={(e) => e.target.files[0] && onFiles(e.target.files)} />
          </label>
          {pending && <span className="text-[10px] text-glow">restoring session...</span>}
        </div>
      )}
      <p className="text-[10px] leading-snug text-dim">
        Everything runs on your machine with ONNX Runtime Web; the file never leaves the browser. From PyTorch: <code className="text-slate-300">torch.onnx.export(model, x, "m.onnx")</code>.
      </p>
    </div>
  );
}

// ---------- input ----------
function DrawBox({ onDrawn }) {
  const cv = useRef(null), drawing = useRef(false), last = useRef(null);
  useEffect(() => {
    const ctx = cv.current.getContext('2d');
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 18;
  }, []);
  const pos = (e) => { const r = cv.current.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 200, ((e.clientY - r.top) / r.height) * 200]; };
  return (
    <div>
      <canvas
        ref={cv} width={200} height={200}
        className="h-[110px] w-[110px] touch-none cursor-crosshair rounded border border-line bg-[#070a12]"
        onPointerDown={(e) => { drawing.current = true; last.current = pos(e); e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={(e) => {
          if (!drawing.current) return;
          const p = pos(e), ctx = cv.current.getContext('2d');
          ctx.beginPath(); ctx.moveTo(...last.current); ctx.lineTo(...p); ctx.stroke(); last.current = p;
          onDrawn(cv.current);
        }}
        onPointerUp={() => { drawing.current = false; onDrawn(cv.current); }}
      />
      <button className="mt-1 text-[10px] text-dim hover:text-slate-200" onClick={() => { cv.current.getContext('2d').clearRect(0, 0, 200, 200); onDrawn(cv.current); }}>clear drawing</button>
    </div>
  );
}

function InputCard({ model, kind, norm, setNorm, prep, setPrep, pixels, setPixels, ds, item, setItem, probs, classes, pred, xFinal, source, outKind, setOutKind, rawOut, metaClasses }) {
  const rawItem = source === 'item' && !!ds?.raw;
  const [invert, setInvert] = useState(false);
  const top = probs ? probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5) : [];
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Input</div>
      {kind.kind === 'tokens' ? (
        <TokenInput kind={kind} ids={pixels} setIds={setPixels} />
      ) : kind.kind === 'image' ? (
        <>
          <div className="flex items-start gap-3">
            <DrawBox onDrawn={(c) => setPixels(drawingToPixels(c, kind, prep))} />
            <div className="space-y-1">
              <Thumb pixels={pixels} kind={kind} size={78} />
              <div className="text-[10px] leading-tight text-dim">what the model sees, {kind.H}×{kind.W}{kind.C === 3 ? ' RGB' : ''}</div>
            </div>
          </div>
          <Field label="drawing fit">
            <select value={prep} onChange={(e) => setPrep(e.target.value)} className={selectCls}>
              <option value="mnist">MNIST (20px box, centre of mass)</option>
              <option value="fit">fit with margin</option>
              <option value="stretch">stretch</option>
              {kind.H === 8 && <option value="optdigits">optdigits 8x8</option>}
            </select>
          </Field>
          <label className="block text-xs text-dim">
            or upload a picture{' '}
            <input type="file" accept="image/*" className="w-40 text-[10px]" onChange={async (e) => e.target.files[0] && setPixels(await imageFileToPixels(e.target.files[0], kind, invert))} />
          </label>
          <label className="flex items-center gap-1 text-[10px] text-dim">
            <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)} /> invert uploads (dark ink on light paper)
          </label>
        </>
      ) : (
        <div className="space-y-1">
          <VectorStrip values={xFinal} size={240} />
          <div className="flex gap-1">
            <Btn onClick={() => { let s = 1; setPixels(Float32Array.from({ length: model.inputSize }, () => { s = (s * 16807) % 2147483647; return (s / 2147483647) * 2 - 1; })); }}>random</Btn>
            <Btn onClick={() => setPixels(new Float32Array(model.inputSize))}>zeros</Btn>
          </div>
          <div className="text-[10px] text-dim">A {model.inputDims.join('×')} vector input; use a dataset below to feed real items.</div>
        </div>
      )}
      {ds && (
        <div className="flex items-center gap-1 text-xs">
          <span className="text-dim">dataset item</span>
          <Btn onClick={() => setItem((i) => Math.max(0, i - 1))}>‹</Btn>
          <span className="w-16 text-center text-slate-200">{item + 1} / {ds.n}</span>
          <Btn onClick={() => setItem((i) => Math.min(ds.n - 1, i + 1))}>›</Btn>
          <Btn onClick={() => setItem(Math.floor(Math.random() * ds.n))}>random</Btn>
          {ds.labels && <span className="ml-1 text-dim">label {ds.classes?.[ds.labels[item]] ?? ds.labels[item]}</span>}
          {source === 'custom' && <span className="ml-1 text-warm">showing your input</span>}
        </div>
      )}
      {kind.kind === 'image' && (
        <Field label="normalize pixels as">
          <select value={norm} onChange={(e) => setNorm(e.target.value)} className={selectCls}>
            {Object.entries(NORMS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
          </select>
        </Field>
      )}
      <Field label="read the output as">
        <select value={outKind} onChange={(e) => setOutKind(e.target.value)} className={selectCls}>
          <option value="classes">class scores (softmax){metaClasses ? '' : ', assumed'}</option>
          <option value="values">raw values</option>
        </select>
      </Field>
      {outKind === 'values' && rawOut && (
        <div className="text-[11px] text-slate-300">
          First output, {rawOut.length} value{rawOut.length === 1 ? '' : 's'}: {Array.from(rawOut).slice(0, 8).map((v) => v.toFixed(3)).join(', ')}{rawOut.length > 8 ? ', ...' : ''}
        </div>
      )}
      {probs && probs.length <= 1000 && (
        <div className="space-y-0.5 pt-1 text-xs">
          <div className="flex items-baseline justify-between">
            <span className="text-dim">Model says</span>
            <span className="text-xl font-semibold text-warm">{classes?.[pred] ?? pred}</span>
          </div>
          {top.map(([p, i]) => (
            <div key={i} className="flex items-center gap-2">
              <span className="w-10 truncate text-dim">{classes?.[i] ?? i}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded bg-line"><div className={`h-full ${i === pred ? 'bg-warm' : 'bg-glow/60'}`} style={{ width: `${p * 100}%` }} /></div>
              <span className="w-9 text-right text-dim">{(p * 100).toFixed(0)}%</span>
            </div>
          ))}
          <div className="text-[10px] text-dim">top 5 of {probs.length} (softmax of the first output if it is not already probabilities)</div>
          <OcclusionView model={model} kind={kind} norm={norm} pixels={pixels} label={`the answer "${classes?.[pred] ?? pred}"`}
            target={readClassProb(model.outputName, pred, maybeSoftmax)} deps={[pixels, pred]} raw={rawItem} />
        </div>
      )}
    </div>
  );
}

// ---------- dataset ----------
function DataCard({ ds, store, progress, analyze, onFiles, kind, maxItems, setMaxItems }) {
  const [invert, setInvert] = useState(false);
  return (
    <div className="space-y-2 rounded-lg border border-line p-3 text-xs">
      <div className="text-sm text-slate-100">Dataset</div>
      {ds ? (
        <div className="text-slate-300">
          {ds.name}: {ds.n} items{ds.labels ? `, ${ds.classes?.length ?? '?'} labels` : ', unlabeled'}. <span className="text-dim">{ds.source}</span>
        </div>
      ) : (
        <div className="text-dim">No dataset yet. Rooms, top examples, the SAE and the latent view all need one.</div>
      )}
      <label className="block text-dim">
        add one: images (labels from folder or "label_name.png"), .npy (x and optional y), or a dataset .json
        <input type="file" multiple accept="image/*,.npy,.json" className="mt-1 block w-full text-[10px]" onChange={(e) => onFiles(e.target.files, invert)} />
      </label>
      <label className="block cursor-pointer text-dim">
        or a whole folder
        <input type="file" multiple webkitdirectory="" className="mt-1 block w-full text-[10px]" onChange={(e) => onFiles(e.target.files, invert)} />
      </label>
      {kind.kind === 'image' && (
        <label className="flex items-center gap-1 text-[10px] text-dim">
          <input type="checkbox" checked={invert} onChange={(e) => setInvert(e.target.checked)} /> invert images (dark ink on light paper)
        </label>
      )}
      {ds && (
        <Field label="items to analyse">
          <select value={maxItems} onChange={(e) => setMaxItems(+e.target.value)} className={selectCls}>
            {[50, 100, 250, 500, 1000, 2000, 5000].map((n) => <option key={n} value={n}>{n}{n >= ds.n ? ' (all)' : ''}</option>)}
          </select>
        </Field>
      )}
      {ds && (
        <div className="flex items-center gap-2">
          <Btn onClick={analyze} disabled={progress != null}>{store ? 'Re-run dataset' : 'Run dataset'}</Btn>
          {progress != null && (
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-line"><div className="h-full bg-glow" style={{ width: `${progress * 100}%` }} /></div>
          )}
          {store && progress == null && <span className="text-dim">{store.N} items analysed{store.labels ? `, accuracy ${(100 * store.preds.filter((p, i) => p === store.labels[i]).length / store.N).toFixed(1)}%` : ''}</span>}
        </div>
      )}
    </div>
  );
}

// ---------- which tensors are rooms ----------
function RoomsCard({ model, roomNames, setRoomNames }) {
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState(new Set(roomNames));
  useEffect(() => setPick(new Set(roomNames)), [roomNames]);
  return (
    <div className="space-y-2 rounded-lg border border-line p-3 text-xs">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-100">Rooms</span>
        <Btn onClick={() => setOpen((o) => !o)}>{open ? 'done' : `choose (${roomNames.length} of ${model.allRooms.length})`}</Btn>
      </div>
      {!open && <div className="text-dim">Each chosen tensor is one room. The defaults are activation outputs and the final output.</div>}
      {open && (
        <>
          <div className="max-h-64 space-y-0.5 overflow-y-auto scroll-thin">
            {model.allRooms.map((r) => (
              <label key={r.name} className="flex items-center gap-2 text-[11px]">
                <input type="checkbox" checked={pick.has(r.name)} onChange={(e) => { const s = new Set(pick); e.target.checked ? s.add(r.name) : s.delete(r.name); setPick(s); }} />
                <span className="w-24 truncate text-slate-200">{r.op}</span>
                <span className="text-dim">{r.dims.join('×')}</span>
                <span className="truncate text-[10px] text-dim">{r.name}</span>
              </label>
            ))}
          </div>
          <Btn on onClick={() => { setRoomNames(model.allRooms.map((r) => r.name).filter((n) => pick.has(n))); setOpen(false); }}>Apply (re-runs the dataset)</Btn>
        </>
      )}
    </div>
  );
}

// ---------- one unit's room ----------
function UnitCard(props) {
  if (!props.model || !props.sel) return <UnitPicker {...props} />;
  return <UnitBody {...props} />;
}

// Open a room without hunting for a tiny sphere: pick a layer and a channel number.
function UnitPicker({ model, rooms, onPick }) {
  const [room, setRoom] = useState('');
  const [c, setC] = useState(0);
  const list = (rooms ?? []).filter((r) => !r.isInputRoom);
  const r = list.find((x) => x.name === room) ?? list[0];
  return (
    <div className="space-y-2 rounded-lg border border-line p-3 text-sm text-slate-300">
      <div>Click any unit in the space to open its room, or pick one here. Rooms fill in once a dataset has run.</div>
      {r && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <select value={r.name} onChange={(e) => { setRoom(e.target.value); setC(0); }} className={selectCls}>
            {list.map((x) => <option key={x.name} value={x.name}>{x.label}</option>)}
          </select>
          <span className="text-dim">channel</span>
          <input type="number" min={0} max={r.layout.C - 1} value={c} onChange={(e) => setC(Math.max(0, Math.min(r.layout.C - 1, +e.target.value)))} className={`${selectCls} w-16`} />
          <span className="text-dim">of {r.layout.C}</span>
          <Btn onClick={() => onPick({ room: r.name, index: r.layout.idx(c, 0) })}>Open</Btn>
        </div>
      )}
    </div>
  );
}

function UnitBody({ model, sel, outs, baseOuts, store, ds, kind, norm, sae, saeNow, steer, setSteer, classes, onClose, setItem, onPickFeature, explFor, pixelsNow, rawNow, outKind, pinnedNow, xNow, edges }) {
  const r = model.describe(sel.room);
  const L = r.layout;
  const isFeat = !!sel.sae;
  const { c, p } = isFeat ? { c: sel.index, p: saeNow?.argNow?.[sel.index] ?? 0 } : unitOf(L, sel.index);
  const valueNow = isFeat ? saeNow?.featNow[sel.index] ?? 0 : outs?.[sel.room]?.data[sel.index] ?? 0;
  const chNow = isFeat ? valueNow : (() => { let m = -Infinity; for (let q = 0; q < L.P; q++) m = Math.max(m, outs?.[sel.room]?.data[L.idx(c, q)] ?? -Infinity); return m; })();

  // dataset statistics for this channel or feature
  const acts = useMemo(() => {
    if (isFeat) return sae?.stats ? Float32Array.from({ length: sae.N }, (_, i) => sae.stats.featMax[i * sae.sae.F + c]) : null;
    return store?.stats[sel.room] ? channelActs(store, sel.room, c) : null;
  }, [isFeat, sae, store, sel.room, c]);
  const args = (i) => (isFeat ? sae.stats.featArg[i * sae.sae.F + c] : store.stats[sel.room].chArg[i * L.C + c]);
  const top = acts ? topItems(acts, 12) : [];
  const hist = acts ? histogramOf(acts) : null;
  const labels = store?.labels;
  const pur = acts && labels ? labelPurity(acts, labels, 12) : null;
  const max = acts ? Math.max(...acts) : Math.abs(chNow) || 1;
  const lo = acts ? Math.min(0, Math.min(...acts)) : 0;
  const summary = acts ? describeUnit(acts, args, L, labels, classes).text : null;
  const explanation = explFor?.(sel.room, c, isFeat);

  const steering = steer && steer.room === sel.room && (isFeat ? steer.sae && steer.feature === c : !steer.sae && steer.c === c);
  const [scope, setScope] = useState('channel');
  const setValue = (v) => setSteer(isFeat ? { room: sel.room, sae: true, feature: c, value: v } : { room: sel.room, c, p: scope === 'unit' ? p : null, value: v });
  const bOut = baseOuts?.[model.outputName]?.data, sOut = outs?.[model.outputName]?.data;
  const argmax = (a) => (a ? Array.from(a).indexOf(Math.max(...a)) : null);

  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-dim">room</div>
          <div className="text-sm font-semibold text-slate-100">{isFeat ? `SAE feature ${c}` : `${r.op}, channel ${c}`}</div>
          <div className="text-[10px] text-dim">
            {isFeat ? `of the SAE on ${r.op} (${r.dims.join('×')})` : `${r.name} · ${r.dims.join('×')}${L.P > 1 ? ` · selected at ${posText(L, p)}` : ''}`}
          </div>
        </div>
        <Btn onClick={onClose}>Close</Btn>
      </div>
      {explanation && (
        <div className="space-y-1">
          <p className="text-sm leading-snug text-slate-100">"{explanation.explanation}"</p>
          <div className="flex flex-wrap items-center gap-1.5">
            <ScoreBadge ex={explanation} />
            <span className="text-[10px] text-dim">written by {explanation.model ?? 'Claude'} from 12 examples, scored by simulation on 20 others<Cite id="bills2023" /></span>
          </div>
        </div>
      )}
      {summary && <div className="rounded bg-glow/5 px-2 py-1 text-[11px] text-slate-200">{summary}<span className="ml-1 text-dim">(measured)</span></div>}
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-semibold text-glow">{chNow.toFixed(2)}</span>
        <span className="text-xs text-dim">{L.P > 1 ? 'max over positions for this input' : 'for this input'}{acts ? `, ${max ? Math.round((chNow / max) * 100) : 0}% of its dataset max` : ''}</span>
      </div>
      {!isFeat && L.P > 1 && <div className="text-[11px] text-dim">This unit alone: {valueNow.toFixed(3)}</div>}

      {!acts && <NotShown>Dataset statistics. Add a dataset and run it to fill this room.</NotShown>}
      {hist && (
        <div>
          <div className="mb-1 text-xs text-slate-200">Over {acts.length} dataset items</div>
          <Hist hist={hist} current={chNow} />
          <div className="text-[10px] text-dim">active on {Array.from(acts).filter((v) => v > 0).length} of {acts.length}; orange line is this input</div>
        </div>
      )}
      {top.length > 0 && ds && (
        <div>
          <div className="mb-1 text-xs text-slate-200">Top 12 items {L.P > 1 && <span className="text-dim">({kind.kind === 'tokens' ? 'highlight' : 'box'}: where it fired)</span>}</div>
          {kind.kind === 'tokens' ? (
            <div className="space-y-0.5">
              {top.map((i) => {
                const a = args(i);
                const tokPos = L.kind === 'seq' ? a : L.kind === 'spatial' ? Math.floor(a / L.W) : null;
                return (
                  <button key={i} className="flex w-full items-center gap-2 text-left" onClick={() => setItem(store.dsIndex[i])} title="load this item">
                    <TokenThumb ids={ds.item(store.dsIndex[i])} kind={kind} pos={tokPos} />
                    <span className="text-[9px] text-dim">{labels ? `${classes?.[labels[i]] ?? labels[i]} · ` : ''}{acts[i].toFixed(2)}</span>
                  </button>
                );
              })}
              {L.kind === 'spatial' && <div className="text-[10px] text-dim">For a [heads, query, key] map the highlight is the query token.</div>}
            </div>
          ) : (
          <div className="grid grid-cols-6 gap-1">
            {top.map((i) => (
              <button key={i} className="text-center" onClick={() => setItem(store.dsIndex[i])} title="load this item">
                <Thumb pixels={kind.kind === 'image' ? displayOf(ds, store.dsIndex[i], kind) : null} kind={kind} size={44} box={boxFor(L, args(i), kind)} />
                {kind.kind !== 'image' && <div className="h-6 rounded bg-line" />}
                <div className="text-[9px] text-dim">{labels ? `${classes?.[labels[i]] ?? labels[i]} · ` : ''}{acts[i].toFixed(1)}</div>
              </button>
            ))}
          </div>
          )}
          {pur && <div className="mt-1 text-[11px] text-slate-300">{pur.count} of its top 12 are labeled {classes?.[pur.label] ?? pur.label}.</div>}
          {kind.kind === 'image' && <AverageInput ds={ds} kind={kind} top={top.map((i) => store.dsIndex[i])} acts={(j) => acts[top[top.map((i) => store.dsIndex[i]).indexOf(j)]]} />}
          {L.P > 1 && kind.kind === 'image' && <div className="text-[10px] text-dim">Boxes scale the layer's grid onto the image; they show roughly where, not the exact receptive field.</div>}
        </div>
      )}

      {!isFeat && kind.kind === 'image' && (
        <OcclusionView model={model} kind={kind} norm={norm} pixels={pixelsNow} label="this channel"
          target={readChannelMax(sel.room, L, c)} deps={[pixelsNow, sel.room, c]} raw={rawNow} />
      )}
      {!isFeat && pinnedNow && <ChannelPatch model={model} room={sel.room} L={L} c={c} pinned={pinnedNow} xNow={xNow} classes={classes} />}
      {!isFeat && edges?.edges?.length > 0 && (
        <div className="rounded-md border border-line p-2 text-[11px] text-slate-300">
          <div className="text-xs text-slate-200">What feeds it on this input</div>
          <div className="mt-0.5">
            {edges.edges.slice(0, 6).map((e) => (
              <span key={e.k} className={`mr-2 ${e.v >= 0 ? 'text-warm' : 'text-cool'}`}>
                {edges.perChannel ? 'ch' : '#'}{e.k} {e.v >= 0 ? '+' : ''}{e.v.toFixed(2)}
              </span>
            ))}
          </div>
          <div className="mt-0.5 text-[10px] text-dim">
            Weight times the activation it multiplies, {edges.perChannel ? 'summed per input channel' : 'per input'}; the glowing edges in the space show the same{edges.exact ? '' : ' (approximate: a strided conv, so weight sum times channel mean)'}. Biases not included.
          </div>
        </div>
      )}
      {!isFeat && <Weights model={model} room={sel.room} c={c} />}
      {!isFeat && kind.kind === 'tokens' && r.op === 'Softmax' && L.kind === 'spatial' && L.H === kind.T && (
        <div className="rounded-md border border-line p-2 text-[11px] text-slate-300">
          Attention: head {c}, query token {Math.floor(p / L.W)} attends to key token {p % L.W} with weight {valueNow.toFixed(3)}.
          The arcs over the input row show this query's weights to every key.
        </div>
      )}
      {!isFeat && sae && sae.room === sel.room && <WrittenBy sae={sae} c={c} onPick={(f) => onPickFeature?.(f)} />}
      <div className="space-y-1.5 rounded-md border border-line p-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-200"><Term k="clamp">Clamp</Term> {isFeat ? 'this feature' : scope === 'channel' ? 'this channel' : 'this unit'}</span>
          <Btn on={steering} onClick={() => (steering ? setSteer(null) : setValue(+max.toFixed(3)))}>{steering ? 'Clamped' : 'Clamp it'}</Btn>
        </div>
        {!isFeat && L.P > 1 && (
          <div className="flex gap-1">
            <Btn on={scope === 'channel'} onClick={() => setScope('channel')}>every position</Btn>
            <Btn on={scope === 'unit'} onClick={() => setScope('unit')}>only {posText(L, p)}</Btn>
          </div>
        )}
        <input
          type="range" min={+lo.toFixed(3)} max={+(1.5 * max || 1).toFixed(3)} step={(1.5 * max || 1) / 200}
          value={steering ? steer.value : max} onChange={(e) => setValue(Number(e.target.value))} className="w-full accent-[#ffb547]"
        />
        <div className="flex justify-between text-[10px] text-dim"><span>{lo.toFixed(1)}</span><span>dataset max {max.toFixed(2)}</span><span>{(1.5 * max).toFixed(1)}</span></div>
        {steering && bOut && sOut && (
          <div className="text-xs text-slate-300">
            Answer without the clamp: <span className="text-glow">{classes?.[argmax(bOut)] ?? argmax(bOut)}</span>, with it:{' '}
            <span className={argmax(bOut) !== argmax(sOut) ? 'text-warm' : 'text-glow'}>{classes?.[argmax(sOut)] ?? argmax(sOut)}</span>.
            {isFeat && ' The SAE error term is kept, so only this feature changes.'}
          </div>
        )}
        <EffectTest model={model} ds={ds} kind={kind} norm={norm} store={store} classes={classes} room={sel.room}
          makeTransform={(v) => (isFeat ? saeTransform(sae.sae, L, 'steer', c, v) : neuronTransform(L, c, scope === 'unit' ? p : null, v))}
          value={steering ? steer.value : max} outKind={outKind} />
      </div>
    </div>
  );
}

// ---------- SAE ----------
function SaeCard({ model, rooms, store, sae, setSae, ds, kind, norm, onShow, saeNow, onPickFeature, outKind: outKindRef }) {
  const candidates = rooms.filter((r) => !r.isOutput && r.layout.C >= 4 && store?.stats[r.name]?.full);
  const [room, setRoom] = useState('');
  const [mult, setMult] = useState(8);
  const [k, setK] = useState(8);
  const [steps, setSteps] = useState(1500);
  const [train, setTrain] = useState(null); // {curve, dead, step, phase}
  const worker = useRef(null);
  const candKey = candidates.map((c) => c.name).join('|');
  useEffect(() => {
    // default to the deepest hidden layer; reset whenever the model or its rooms change
    if (candidates.length && !candidates.some((c) => c.name === room)) setRoom(candidates[candidates.length - 1].name);
  }, [candKey]); // eslint-disable-line
  const r = rooms.find((x) => x.name === room);
  const D = r?.layout.C ?? 0;
  useEffect(() => { if (D) setK(Math.max(1, Math.min(16, Math.round(D / 8)))); }, [D]);

  const start = () => {
    const iv = itemVectors(store, room);
    if (!iv) return;
    const X = sampleRows(iv, 20000);
    worker.current?.terminate();
    const w = new Worker(new URL('../../lib/lab/sae.worker.js', import.meta.url), { type: 'module' });
    worker.current = w;
    const curve = [];
    setTrain({ curve: [], step: 0, dead: 0, phase: 'training' });
    w.onmessage = async (e) => {
      const m = e.data;
      if (m.type === 'progress') {
        if (m.fvu != null) curve.push([m.step, m.fvu]);
        setTrain({ curve: [...curve], step: m.step, dead: m.dead, phase: 'training' });
      } else if (m.type === 'stats') setTrain((t) => ({ ...t, phase: `scoring features ${(m.frac * 100).toFixed(0)}%` }));
      else if (m.type === 'done') {
        const s = m.sae;
        const featScale = (() => { const a = Array.from(m.stats.featMax).filter((v) => v > 0).sort((x, y) => x - y); return a[Math.floor(a.length * 0.99)] || 1; })();
        const alive = new Set();
        for (let i = 0; i < m.stats.featMax.length; i++) if (m.stats.featMax[i] > 0) alive.add(i % s.F);
        setSae({ room, sae: s, metrics: m.metrics, stats: m.stats, N: store.N, featScale, alive: alive.size });
        setTrain((t) => ({ ...t, phase: 'measuring accuracy with the SAE in the loop' }));
        // accuracy when every vector of this layer is replaced by its reconstruction
        const n = Math.min(200, store.N);
        let agree = 0, correct = 0, correctBase = 0, absChange = 0, absBase = 0;
        const tf = saeTransform(s, r.layout, 'recon');
        for (let k = 0; k < n; k++) {
          const i = Math.floor((k * store.N) / n); // evenly spaced: datasets are often sorted by class
          const xi = modelInputOf(ds, store.dsIndex[i], kind, norm);
          const o = await model.runWith(xi, room, tf);
          const lg = o[model.outputName].data;
          const pr = Array.from(lg).indexOf(Math.max(...lg));
          if (outKindRef === 'values') {
            const base = (await model.run([xi]))[model.outputName].data;
            lg.forEach((v, j) => { absChange += Math.abs(v - base[j]); absBase += Math.abs(base[j]); });
          }
          if (pr === store.preds[i]) agree++;
          if (store.labels) { if (pr === store.labels[i]) correct++; if (store.preds[i] === store.labels[i]) correctBase++; }
        }
        setSae((x) => ({ ...x, loop: { n, agree: agree / n, acc: store.labels ? correct / n : null, accBase: store.labels ? correctBase / n : null, rel: outKindRef === 'values' && absBase ? absChange / absBase : null } }));
        setTrain((t) => ({ ...t, phase: 'done' }));
        w.terminate();
        onShow();
      }
    };
    w.postMessage({ X, D, F: D * mult, k, steps, lr: 2e-3, batch: 128, seed: 3, items: { items: iv.items, N: iv.N, P: iv.P } });
  };

  // quick SAEs at several k, to see the sparsity / reconstruction trade-off before a full run
  const [sweepRes, setSweepRes] = useState(null);
  const sweep = async () => {
    const iv = itemVectors(store, room);
    if (!iv) return;
    const X = sampleRows(iv, 8000);
    const ks = [1, 2, 4, 8, 16, 32].filter((kk) => kk <= D);
    const points = [];
    setSweepRes({ running: true, points: [], F: D * mult });
    for (const kk of ks) {
      const res = await new Promise((resolve) => {
        const w = new Worker(new URL('../../lib/lab/sae.worker.js', import.meta.url), { type: 'module' });
        let dead = 0;
        w.onmessage = (e) => {
          if (e.data.type === 'progress') dead = e.data.dead;
          if (e.data.type === 'done') { w.terminate(); resolve({ k: kk, ve: e.data.metrics.varianceExplained, alive: D * mult - dead }); }
        };
        w.postMessage({ X, D, F: D * mult, k: kk, steps: 600, lr: 2e-3, batch: 128, seed: 3 });
      });
      points.push(res);
      setSweepRes({ running: true, points: [...points], F: D * mult });
    }
    setSweepRes({ running: false, points, F: D * mult });
  };

  const purityCompare = useMemo(() => {
    if (!sae?.stats || !store?.labels) return null;
    const F = sae.sae.F, N = sae.N;
    const pf = [], pn = [];
    for (let f = 0; f < F; f++) {
      const a = Float32Array.from({ length: N }, (_, i) => sae.stats.featMax[i * F + f]);
      const p = labelPurity(a, store.labels, 12);
      if (p) pf.push(p.value);
    }
    const L = model.describe(sae.room).layout;
    for (let c = 0; c < L.C; c++) { const p = labelPurity(channelActs(store, sae.room, c), store.labels, 12); if (p) pn.push(p.value); }
    return { pf, pn };
  }, [sae, store, model]);

  const download = () => {
    const blob = new Blob([JSON.stringify(toJSON(sae.sae, { model: model.fileName, tensor: sae.room, metrics: sae.metrics }))], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sae_${sae.room.replace(/[^a-z0-9]+/gi, '_')}.json`;
    a.click();
  };
  const importSae = async (file) => {
    const j = JSON.parse(await file.text());
    const target = candidates.find((x) => x.name === j.tensor) ?? (r && candidates.includes(r) ? r : null);
    if (!target) { alert(`This SAE was trained on ${j.tensor ?? 'another layer'}, which is not a room with stored activations here. Add that room, or analyse fewer items so it fits.`); return; }
    if (target.layout.C !== j.D) { alert(`This SAE expects ${j.D} channels; ${target.label ?? target.name} has ${target.layout.C}.`); return; }
    const s = fromJSON(j);
    const iv = itemVectors(store, target.name);
    const { featureStats } = await import('../../lib/lab/saeLab.js');
    const stats = featureStats(s, iv);
    const a = Array.from(stats.featMax).filter((v) => v > 0).sort((x, y) => x - y);
    setSae({ room: target.name, sae: s, metrics: j.metrics ?? null, stats, N: store.N, featScale: a[Math.floor(a.length * 0.99)] || 1, alive: new Set(Array.from(stats.featMax.keys()).filter((i) => stats.featMax[i] > 0).map((i) => i % s.F)).size });
    onShow();
  };

  return (
    <div className="space-y-2 rounded-lg border border-line p-3 text-xs">
      <div className="text-sm text-slate-100">Add a sparse autoencoder</div>
      <p className="leading-snug text-slate-400">
        Pick a layer and train a <Term k="sparse autoencoder" /> on it here, in a background thread. Its features dock beside the layer in the space, each with its own room.
        <Cite id="bricken2023" /><Cite id="gao2024" />
      </p>
      {!store && <NotShown>An SAE until a dataset has run; it learns from the layer's activations on that data.</NotShown>}
      {store && candidates.length === 0 && <NotShown>A layer small enough to train on here. Full activations are kept for at most 40 million values across all rooms, smallest rooms first; analyse fewer items or choose fewer rooms to free space.</NotShown>}
      {store && candidates.length > 0 && (
        <>
          <Field label="layer">
            <select value={room} onChange={(e) => setRoom(e.target.value)} className={`${selectCls} max-w-44`}>
              {candidates.map((c) => <option key={c.name} value={c.name}>{c.label ?? c.op} · {c.dims.join('×')}</option>)}
            </select>
          </Field>
          <div className="text-[10px] text-dim">
            {r?.layout.kind === 'spatial' ? `One ${D}-dim vector per pixel of the ${r.layout.H}×${r.layout.W} map, ` : r?.layout.kind === 'seq' ? `One ${D}-dim vector per token, ` : `One ${D}-dim vector per item, `}
            {store.N * (r?.layout.P ?? 1)} vectors in all (up to 20,000 sampled for training).
          </div>
          <Field label="dictionary size">
            <select value={mult} onChange={(e) => setMult(+e.target.value)} className={selectCls}>
              {[2, 4, 8, 16].map((m) => <option key={m} value={m}>{m}× = {D * m} features</option>)}
            </select>
          </Field>
          <Field label="active per vector (k)">
            <input type="number" min={1} max={64} value={k} onChange={(e) => setK(Math.max(1, +e.target.value))} className={`${selectCls} w-16`} />
          </Field>
          <Field label="training steps">
            <select value={steps} onChange={(e) => setSteps(+e.target.value)} className={selectCls}>
              {[500, 1500, 3000, 6000].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
          <div className="flex gap-1.5">
            <Btn on onClick={start} disabled={train?.phase === 'training'}>{sae ? 'Train again' : 'Train SAE'}</Btn>
            {train?.phase === 'training' && <Btn onClick={() => { worker.current?.terminate(); setTrain(null); }}>Stop</Btn>}
            <Btn onClick={sweep} disabled={!!sweepRes?.running || train?.phase === 'training'}>Sweep k</Btn>
            <label className="cursor-pointer rounded-md border border-line px-2.5 py-1 text-slate-200 hover:border-glow/40">
              Load SAE .json<input type="file" accept=".json" className="hidden" onChange={(e) => e.target.files[0] && importSae(e.target.files[0])} />
            </label>
          </div>
        </>
      )}
      {sweepRes && <SweepChart res={sweepRes} onPick={(kk) => setK(kk)} />}
      {train && (
        <div className="space-y-1">
          <Curve points={train.curve} label={`held-out unexplained variance by step · step ${train.step} of ${steps} · ${train.dead} features idle · ${train.phase}`} />
        </div>
      )}
      {sae?.metrics && (
        <div className="space-y-1 rounded-md border border-line p-2 text-slate-300">
          <div>
            Explains <span className="text-glow">{(100 * sae.metrics.varianceExplained).toFixed(1)}%</span> of held-out variance ({sae.metrics.nTest} vectors) with {sae.sae.k} of {sae.sae.F} features on;{' '}
            {sae.alive} features fire on at least one item.
          </div>
          {sae.loop && sae.loop.rel != null && (
            <div>With the SAE's reconstruction in place of the layer, the outputs move by {(100 * sae.loop.rel).toFixed(1)}% of their size on average over {sae.loop.n} items.</div>
          )}
          {sae.loop && sae.loop.rel == null && (
            <div>
              With the SAE's reconstruction in place of the layer, the model's answer is unchanged on {(100 * sae.loop.agree).toFixed(0)}% of {sae.loop.n} items
              {sae.loop.acc != null && ` (accuracy ${(100 * sae.loop.acc).toFixed(1)}% vs ${(100 * sae.loop.accBase).toFixed(1)}%)`}.
            </div>
          )}
          {purityCompare && (
            <div className="text-[11px]">
              Share of top 12 items with one label: channels median {median(purityCompare.pn).toFixed(2)} (n = {purityCompare.pn.length}), features median {median(purityCompare.pf).toFixed(2)} (n = {purityCompare.pf.length}).
              <span className="text-dim"> Units active on fewer than 12 items are left out; a label is a rough stand-in for one meaning.</span>
            </div>
          )}
          {saeNow && (
            <div>
              <div className="text-[11px] text-slate-200">Features on for this input (click to open):</div>
              <div className="mt-1 flex flex-wrap gap-1">
                {Array.from(saeNow.featNow.keys()).filter((f) => saeNow.featNow[f] > 0).sort((a, b) => saeNow.featNow[b] - saeNow.featNow[a]).slice(0, 16).map((f) => (
                  <button key={f} onClick={() => onPickFeature(f)} className="rounded border border-warm/40 px-1.5 py-px text-[10px] text-warm hover:bg-warm/10">
                    #{f} · {saeNow.featNow[f].toFixed(1)}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="flex gap-1.5 pt-1">
            <Btn onClick={download}>Download SAE .json</Btn>
            <Btn onClick={() => setSae(null)}>Remove</Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------- latent space ----------
function LabLatent({ model, store, rooms, sae, outs, ds, setItem, classes }) {
  const options = useMemo(() => {
    const o = rooms.filter((r) => store?.stats[r.name]?.full).map((r) => ({ key: r.name, label: `${r.op} ${r.dims.join('×')}` }));
    if (sae?.stats) o.push({ key: '__sae', label: `SAE features (${sae.sae.F})` });
    return o;
  }, [rooms, store, sae]);
  const [key, setKey] = useState('');
  const [highlight, setHighlight] = useState(null);
  const [measures, setMeasures] = useState({});
  useEffect(() => { if (!options.find((o) => o.key === key) && options.length) setKey(options[Math.max(0, options.length - 2)].key); }, [options]); // eslint-disable-line
  const X = useMemo(() => {
    if (!store || !key) return null;
    if (key === '__sae') return Array.from({ length: sae.N }, (_, i) => sae.stats.featMax.subarray(i * sae.sae.F, (i + 1) * sae.sae.F));
    const st = store.stats[key];
    return Array.from({ length: store.N }, (_, i) => st.full.subarray(i * st.layout.size, (i + 1) * st.layout.size));
  }, [store, key, sae]);
  const P = useMemo(() => (X ? pcaOf(X) : null), [X]);
  const me = useMemo(() => {
    if (!outs || !key) return null;
    if (key === '__sae') { const L = model.describe(sae.room).layout; return featuresNow(sae.sae, outs[sae.room].data, L).max; }
    return outs[key]?.data ?? null;
  }, [outs, key, sae, model]);
  if (!store) return <div className="flex h-full items-center justify-center p-6 text-sm text-dim">Run a dataset first; the latent view places every item by its activations.</div>;
  const measure = () => {
    if (!store.labels) return;
    setMeasures((m) => ({ ...m, [key]: { knn: knnAccOf(X, store.labels), sep: separationOf(X, store.labels) } }));
  };
  return (
    <div className="relative h-full w-full">
      {P && <LatentCloud P={P} labels={store.labels} classes={classes} me={me} highlight={highlight} onPick={(i) => setItem(store.dsIndex[i])} />}
      <div className="absolute right-3 top-3 z-20 w-72 space-y-2 rounded-lg border border-line bg-panel/90 p-3 text-xs">
        <Field label="layer">
          <select value={key} onChange={(e) => setKey(e.target.value)} className={`${selectCls} max-w-44`}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </Field>
        {P && <div className="text-slate-300">These 3 axes keep {(100 * P.explained.reduce((a, b) => a + b, 0)).toFixed(0)}% of the spread in {P.D} dimensions.<Cite id="jolliffe2016" /></div>}
        {store.labels && (
          <Field label="highlight">
            <select value={highlight ?? ''} onChange={(e) => setHighlight(e.target.value === '' ? null : +e.target.value)} className={selectCls}>
              <option value="">none</option>
              {(classes ?? []).map((c, i) => <option key={i} value={i}>{c}</option>)}
            </select>
          </Field>
        )}
        {store.labels && (
          <div>
            {measures[key] ? (
              <div className="text-slate-300">
                Full space: 5-nearest-neighbour accuracy <span className="text-glow">{(100 * measures[key].knn).toFixed(1)}%</span>, spacing ratio <span className="text-glow">{measures[key].sep.toFixed(2)}</span>
                <Cite id="alain2016" />
              </div>
            ) : (
              <Btn onClick={measure}>Measure how this layer groups labels</Btn>
            )}
          </div>
        )}
        <div className="text-[10px] text-dim">Click a mark to load that item. The 3D view hides the other {P ? P.D - 3 : 0} dimensions; the measures use all of them.<Cite id="wattenberg2016" /></div>
      </div>
    </div>
  );
}

// The activation-weighted mean of a unit's top items: a forward-only stand-in for feature
// visualization, which needs gradients that ONNX Runtime Web does not provide.
function AverageInput({ ds, kind, top, acts }) {
  const avg = useMemo(() => {
    if (!top.length) return null;
    const out = new Float32Array(kind.C * kind.H * kind.W);
    let w = 0;
    for (const i of top) {
      const a = Math.max(0, typeof acts === 'function' ? acts(i) : acts[i]);
      const px = displayOf(ds, i, kind);
      for (let j = 0; j < out.length; j++) out[j] += a * px[j];
      w += a;
    }
    let mx = 0;
    for (let j = 0; j < out.length; j++) { out[j] /= w || 1; mx = Math.max(mx, out[j]); }
    for (let j = 0; j < out.length; j++) out[j] /= mx || 1;
    return out;
  }, [ds, kind, top, acts]);
  if (!avg) return null;
  return (
    <div className="mt-2 flex items-center gap-2">
      <Thumb pixels={avg} kind={kind} size={56} />
      <div className="text-[10px] leading-snug text-dim">
        Average of its top 12, weighted by activation. Gradient-based feature visualization is not shown here: ONNX Runtime Web gives no gradients.
      </div>
    </div>
  );
}

// Which SAE features write most to channel c of the layer: decoder weight times how much the
// feature is used across the dataset (Bricken et al. 2023 read neurons through features this way).
function WrittenBy({ sae, c, onPick }) {
  const list = useMemo(() => {
    const { F, D, Wd } = sae.sae;
    const use = new Float64Array(F);
    for (let i = 0; i < sae.stats.featMax.length; i++) use[i % F] += sae.stats.featMax[i];
    return Array.from({ length: F }, (_, f) => [f, Wd[f * D + c] * use[f]]).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [sae, c]);
  if (!list.length) return null;
  const top = list[0][1];
  return (
    <div className="rounded-md border border-line p-2 text-xs">
      <div className="text-slate-200">Features that write to this channel most</div>
      <div className="mt-1 space-y-0.5">
        {list.map(([f, v]) => (
          <button key={f} onClick={() => onPick(f)} className="flex w-full items-center gap-2 text-left text-[11px] hover:text-warm">
            <span className="w-10 text-warm">#{f}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-line"><div className="h-full bg-warm/70" style={{ width: `${(v / top) * 100}%` }} /></div>
          </button>
        ))}
      </div>
      <div className="mt-1 text-[10px] text-dim">decoder weight on this channel times the feature's total activation over the dataset</div>
    </div>
  );
}

function TokenInput({ kind, ids, setIds }) {
  const [text, setText] = useState('');
  const [info, setInfo] = useState(null);
  const shown = ids ? idsToText(ids, kind) : '';
  const apply = (t) => { setText(t); const r = textToIds(t, kind); setIds(r.ids); setInfo(r); };
  return (
    <div className="space-y-1.5">
      <input
        value={text}
        onChange={(e) => apply(e.target.value)}
        placeholder={kind.vocab ? `type using: ${kind.vocab.join(' ')}` : 'token ids, separated by spaces'}
        className="w-full rounded border border-line bg-panel px-2 py-1 font-mono text-xs text-slate-100 placeholder:text-dim"
      />
      <div className="font-mono text-sm tracking-wider text-glow">{shown}</div>
      <div className="text-[10px] text-dim">
        {kind.T} tokens. {kind.vocab ? `Characters outside the vocabulary are dropped; short input is padded with "${kind.vocab[0]}".` : 'No vocabulary was given, so type ids directly.'}
        {info?.dropped ? ` ${info.dropped} extra tokens were cut.` : ''}
      </div>
      <Btn onClick={() => { const n = kind.vocab?.length ?? 2; apply(Array.from({ length: kind.T }, () => (kind.vocab ? kind.vocab[Math.floor(Math.random() * n)] : Math.floor(Math.random() * n))).join(kind.vocab ? '' : ' ')); }}>random</Btn>
    </div>
  );
}

// A token item as text, with the position where a unit fired highlighted.
function TokenThumb({ ids, kind, pos }) {
  return (
    <div className="rounded bg-[#070a12] px-1 py-0.5 font-mono text-[10px] leading-tight tracking-tight text-slate-300">
      {Array.from(ids).map((v, j) => (
        <span key={j} className={j === pos ? 'rounded-sm bg-warm/80 text-black' : ''}>{kind.vocab ? kind.vocab[v] ?? '?' : `${v} `}</span>
      ))}
    </div>
  );
}

// Incoming weights of a channel, read from the model file: a conv kernel per input channel, or
// one weight per input for a dense layer. Orange is positive, blue negative.
function Weights({ model, room, c }) {
  const { w, shape } = useMemo(() => {
    const src = linearSource(model.model, room);
    const w = incomingWeights(src, c);
    let shape = null;
    if (w?.kind === 'row' && src.input) {
      // lay a dense layer's weights out like the tensor it read, before flattening
      const t = preFlattenTensor(model.model, src.input);
      const dims = t === model.input.name ? model.inputDims : model.shapes[t];
      if (dims && dims.reduce((a, b) => a * b, 1) === w.data.length) {
        if (dims.length === 3) shape = { C: dims[0], H: dims[1], W: dims[2] };
        else if (dims.length === 1 && t === model.input.name) { const sq = Math.round(Math.sqrt(dims[0])); if (sq * sq === dims[0]) shape = { C: 1, H: sq, W: sq }; }
      }
    }
    return { w, shape };
  }, [model, room, c]);
  if (!w) return null;
  let m = 0;
  for (const v of w.data) m = Math.max(m, Math.abs(v));
  const norm = (a) => Array.from(a, (v) => v / (m || 1));
  const tiles = (count, h, wd, get, note) => (
    <>
      <div className="mt-1 flex flex-wrap gap-1">
        {Array.from({ length: Math.min(count, 48) }, (_, k) => (
          <PixelGrid key={k} values={norm(get(k))} w={wd} h={h} size={Math.min(56, Math.max(14, wd * (count === 1 ? 4 : 7)))} map="diverging" />
        ))}
      </div>
      <div className="mt-1 text-[10px] text-dim">{note}; orange positive, blue negative</div>
    </>
  );
  return (
    <div className="rounded-md border border-line p-2 text-xs">
      <div className="text-slate-200">Incoming weights</div>
      {w.kind === 'kernel'
        ? tiles(w.cin, w.kh, w.kw, (k) => w.data.subarray(k * w.kh * w.kw, (k + 1) * w.kh * w.kw), `one ${w.kh}×${w.kw} kernel per input channel (${w.cin}${w.cin > 48 ? ', first 48 shown' : ''})`)
        : shape
          ? tiles(shape.C, shape.H, shape.W, (k) => w.data.subarray(k * shape.H * shape.W, (k + 1) * shape.H * shape.W),
              shape.C === 1 ? `${w.data.length} weights laid out like the ${shape.H}×${shape.W} input they read` : `${w.data.length} weights laid out like the ${shape.C}×${shape.H}×${shape.W} map they read, one tile per channel`)
          : w.data.length <= 256 ? (
            <>
              <div className="mt-1 flex h-10 items-center gap-px">
                {Array.from(w.data).map((v, i) => (
                  <div key={i} className="flex h-full flex-1 flex-col justify-center" title={`input ${i}: ${v.toFixed(3)}`}>
                    <div style={{ height: `${(Math.abs(v) / (m || 1)) * 50}%`, transform: v < 0 ? 'translateY(50%)' : 'translateY(-50%)' }} className={v >= 0 ? 'bg-warm' : 'bg-cool'} />
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-dim">{w.data.length} incoming weights; orange positive, blue negative</div>
            </>
          ) : tiles(1, Math.ceil(w.data.length / Math.ceil(Math.sqrt(w.data.length))), Math.ceil(Math.sqrt(w.data.length)), () => {
              const side = Math.ceil(Math.sqrt(w.data.length));
              const out = new Float32Array(side * Math.ceil(w.data.length / side));
              out.set(w.data);
              return out;
            }, `${w.data.length} incoming weights in reading order`)}
    </div>
  );
}

function LabSearch({ entries, classes, onPick }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const res = useMemo(() => searchUnits(q, entries, classes), [q, entries, classes]);
  return (
    <div className="absolute left-3 right-3 top-12 z-30 md:right-auto md:w-72">
      <input
        value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => { if (e.key === 'Enter' && res[0]) { onPick(res[0]); setOpen(false); } }}
        placeholder='find units: a class ("5"), "upper left", "token 3", "rare"'
        className="w-full rounded-md border border-line bg-panel px-2.5 py-1 text-xs text-slate-100 placeholder:text-dim focus:border-glow/50 focus:outline-none"
      />
      {open && q.trim() && (
        <div className="mt-1 max-h-80 overflow-y-auto rounded-md border border-line bg-panel shadow-xl">
          {res.length === 0 && <div className="px-3 py-2 text-xs text-dim">No measured summary matches.</div>}
          {res.map((e) => (
            <button key={e.key} onMouseDown={() => onPick(e)} className="block w-full border-b border-line/60 px-3 py-1.5 text-left hover:bg-glow/10">
              <div className="text-xs text-slate-100">{e.sae ? `SAE feature ${e.index}` : `${e.roomLabel}, channel ${e.c}`}</div>
              {e.expl && <div className="text-[10px] italic text-slate-200">"{e.expl.explanation}" <span className="not-italic text-dim">r {e.expl.score.toFixed(2)}</span></div>}
              <div className="text-[10px] text-slate-400">{e.summary.text}</div>
            </button>
          ))}
          <div className="px-3 py-1 text-[10px] text-dim">Matches measured summaries (labels, frequency, position), plus written explanations if loaded.</div>
        </div>
      )}
    </div>
  );
}

// Clamp this unit on many dataset items and count how the answer moves: the Lab's version of the
// Tour's measured steering presets. Selectivity = share of the main source class sent to the main
// target, minus the share of all other items sent there (a clamp that moves everything is not
// evidence about one class).
function EffectTest({ model, ds, kind, norm, store, classes, room, makeTransform, value, outKind }) {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  if (!ds || !store) return null;
  const run = async () => {
    setBusy(true);
    const n = Math.min(100, store.N);
    const tf = makeTransform(value);
    const moves = {};
    let changed = 0, absChange = 0, absBase = 0;
    const perClass = {};
    // evenly spaced over the analysed items, since datasets are often sorted by class
    for (let k = 0; k < n; k++) {
      const i = Math.floor((k * store.N) / n);
      const o = await model.runWith(modelInputOf(ds, store.dsIndex[i], kind, norm), room, tf);
      const lg = o[model.outputName].data;
      if (outKind === 'values') {
        const base = (await model.run([modelInputOf(ds, store.dsIndex[i], kind, norm)]))[model.outputName].data;
        lg.forEach((v, j) => { absChange += Math.abs(v - base[j]); absBase += Math.abs(base[j]); });
        continue;
      }
      const p = Array.from(lg).indexOf(Math.max(...lg));
      const b = store.preds[i];
      perClass[b] = (perClass[b] ?? 0) + 1;
      if (p !== b) { changed++; const k = `${b}>${p}`; moves[k] = (moves[k] ?? 0) + 1; }
    }
    const top = Object.entries(moves).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, cnt]) => {
      const [from, to] = k.split('>').map(Number);
      const othersTo = Object.entries(moves).filter(([kk]) => kk.endsWith(`>${to}`) && !kk.startsWith(`${from}>`)).reduce((a, [, v]) => a + v, 0);
      const othersN = n - (perClass[from] ?? 0) - (perClass[to] ?? 0);
      return { from, to, cnt, of: perClass[from] ?? 0, othersTo, othersN, sel: cnt / (perClass[from] || 1) - othersTo / (othersN || 1) };
    });
    setRes({ n, changed, top, value, rel: absBase ? absChange / absBase : null });
    setBusy(false);
  };
  const name = (k) => classes?.[k] ?? k;
  return (
    <div className="mt-1 border-t border-line pt-1.5 text-[11px]">
      <Btn onClick={run} disabled={busy}>{busy ? 'Testing...' : `What does it control? Test at ${value.toFixed(2)} on ${Math.min(100, store.N)} items`}</Btn>
      {res && (
        <div className="mt-1 space-y-0.5 text-slate-300">
          {outKind === 'values' ? (
            <div>At {res.value.toFixed(2)}, the outputs moved by <span className="text-warm">{(100 * (res.rel ?? 0)).toFixed(1)}%</span> of their size on average, over {res.n} items.</div>
          ) : (
          <div>At {res.value.toFixed(2)}, the answer changed on <span className="text-warm">{res.changed}</span> of {res.n} items.</div>
          )}
          {res.top.map((t) => (
            <div key={`${t.from}>${t.to}`}>
              {t.cnt} of {t.of} predicted {name(t.from)} became {name(t.to)}; {t.othersTo} of {t.othersN} others also became {name(t.to)}
              <span className={t.sel > 0.5 ? ' text-glow' : ' text-dim'}> (selectivity {t.sel.toFixed(2)})</span>
            </div>
          ))}
          {res.changed === 0 && outKind !== 'values' && <div className="text-dim">Nothing moved at this value; try a larger clamp.</div>}
          <div className="text-[10px] text-dim">Selectivity near 1 means the clamp moves one kind of item; near 0 means it pushes everything the same way.<Cite id="olah2020" /></div>
        </div>
      )}
    </div>
  );
}

function ExplCard({ expl, setExpl, model, sae }) {
  const load = async (f) => {
    try {
      const j = JSON.parse(await f.text());
      if (j.kind !== 'inann-lab-explanations') throw new Error('not an explanations file from scripts/explain_lab.py');
      setExpl({ ...j, byUnit: new Map(j.units.map((u) => [u.unit, u])) });
    } catch (e) { alert(e.message); }
  };
  const tensor = sae?.room ?? model.rooms[model.rooms.length - 2]?.name ?? model.rooms[0]?.name;
  return (
    <div className="space-y-1.5 rounded-lg border border-line p-3 text-xs">
      <div className="text-sm text-slate-100">Written explanations</div>
      {expl ? (
        <div className="text-slate-300">
          {expl.units.length} {expl.sae ? 'SAE features' : 'channels'} of {expl.tensor} explained by {expl.explainer_model}; median simulation score{' '}
          {median(expl.units.map((u) => u.score)).toFixed(2)}. They appear in each room and in the unit search.
          <button className="ml-1 text-dim underline" onClick={() => setExpl(null)}>remove</button>
        </div>
      ) : (
        <>
          <p className="leading-snug text-slate-400">
            One sentence per unit, written by Claude from its top examples and scored by how well the sentence alone predicts the unit on other items.<Cite id="bills2023" /> Generate them offline with your own API key, then load the file here.
          </p>
          <pre className="overflow-x-auto rounded bg-black/40 p-1.5 text-[10px] text-slate-300">{`python3 scripts/explain_lab.py \\
  --model ${model.fileName} --data dataset.json \\
  ${sae ? '--sae sae.json' : `--tensor "${tensor}"`}`}</pre>
          <label className="inline-block cursor-pointer rounded-md border border-line px-2.5 py-1 text-slate-200 hover:border-glow/40">
            Load explanations .json<input type="file" accept=".json" className="hidden" onChange={(e) => e.target.files[0] && load(e.target.files[0])} />
          </label>
        </>
      )}
    </div>
  );
}

// "Where does it look?" for image models: the occlusion map of a target over the current input.
function OcclusionView({ model, kind, norm, pixels, target, label, deps, raw }) {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => setRes(null), deps); // eslint-disable-line
  if (kind?.kind !== 'image' || !pixels) return null;
  if (raw) return <div className="text-[10px] text-dim">Occlusion is not shown for this item: the dataset holds pre-normalized inputs, so its display picture is not what the model reads.</div>;
  const run = async () => {
    setBusy(true);
    try { setRes(await occlusionMap(model, pixels, kind, norm, target)); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-1 text-[11px]">
      <Btn onClick={run} disabled={busy}>{busy ? 'Occluding...' : `Where does ${label} look?`}</Btn>
      {res && (
        <div className="flex items-start gap-2">
          <div className="relative">
            <Thumb pixels={pixels} kind={kind} size={84} />
            <div className="pointer-events-none absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${kind.W}, 1fr)` }}>
              {Array.from(res.heat, (v, i) => <div key={i} style={{ background: `rgba(255,181,71,${0.8 * Math.max(0, v) / (res.max || 1)})` }} />)}
            </div>
          </div>
          <div className="leading-snug text-dim">
            Orange marks regions whose covering lowers {label} most (largest drop {res.max.toFixed(3)} from {res.base.toFixed(3)}). {res.runs} covered copies, {res.patch}px patches.<Cite id="zeiler2014" />
          </div>
        </div>
      )}
    </div>
  );
}

// Every unit of a layer at once, in the spirit of OpenAI Microscope: each card shows what the
// unit fires on (the activation-weighted average of its top items, or its top token string) and
// its measured summary. Click a card to open its room.
function LabGallery({ store, rooms, sae, ds, kind, classes, explFor, onPick, selected }) {
  const options = useMemo(() => {
    const o = rooms.filter((r) => store?.stats[r.name] && !r.isOutput).map((r) => ({ key: r.name, label: `${r.label} · ${r.dims.join('×')}`, n: store.stats[r.name].layout.C }));
    if (sae?.stats) o.push({ key: '__sae', label: `SAE features (${sae.sae.F})`, n: sae.sae.F });
    return o;
  }, [rooms, store, sae]);
  const [key, setKey] = useState('');
  const [sort, setSort] = useState('index');
  const [limit, setLimit] = useState(96);
  useEffect(() => { if (!options.find((o) => o.key === key) && options.length) setKey(options[0].key); }, [options]); // eslint-disable-line
  const cards = useMemo(() => {
    if (!store || !key || !ds) return [];
    const isSae = key === '__sae';
    const room = isSae ? sae.room : key;
    const st = store.stats[room], L = st.layout;
    const n = isSae ? sae.sae.F : L.C;
    const out = [];
    for (let c = 0; c < n; c++) {
      const acts = isSae ? Float32Array.from({ length: sae.N }, (_, i) => sae.stats.featMax[i * sae.sae.F + c]) : channelActs(store, room, c);
      const on = Array.from(acts).filter((v) => v > 0).length;
      if (isSae && !on) continue;
      const args = isSae ? (i) => sae.stats.featArg[i * sae.sae.F + c] : (i) => st.chArg[i * L.C + c];
      const sm = describeUnit(acts, args, L, store.labels, classes);
      out.push({ c, acts, args, sm, on, isSae, room, L, purity: sm.purity?.value ?? 0 });
    }
    if (sort === 'freq') out.sort((a, b) => b.on - a.on);
    if (sort === 'purity') out.sort((a, b) => b.purity - a.purity);
    return out;
  }, [store, key, sae, ds, classes, sort]);
  if (!store) return <div className="flex h-full items-center justify-center p-6 text-sm text-dim">Run a dataset first; the gallery shows what every unit fires on.</div>;
  const thumbFor = (card) => {
    const top = topItems(card.acts, 8).filter((i) => card.acts[i] > 0);
    if (!top.length) return <div className="h-14 w-14 rounded bg-line" />;
    if (kind.kind === 'image') {
      const out = new Float32Array(kind.C * kind.H * kind.W);
      let w = 0;
      for (const i of top) {
        const px = displayOf(ds, store.dsIndex[i], kind), a = card.acts[i];
        for (let j = 0; j < out.length; j++) out[j] += a * px[j];
        w += a;
      }
      let mx = 0;
      for (let j = 0; j < out.length; j++) { out[j] /= w || 1; mx = Math.max(mx, out[j]); }
      for (let j = 0; j < out.length; j++) out[j] /= mx || 1;
      return <Thumb pixels={out} kind={kind} size={56} />;
    }
    if (kind.kind === 'tokens') {
      const i = top[0], a = card.args(i);
      const pos = card.L.kind === 'seq' ? a : card.L.kind === 'spatial' ? Math.floor(a / card.L.W) : null;
      return <TokenThumb ids={ds.item(store.dsIndex[i])} kind={kind} pos={pos} />;
    }
    return <div className="text-[10px] text-dim">top label {store.labels ? classes?.[store.labels[top[0]]] ?? store.labels[top[0]] : 'n/a'}</div>;
  };
  return (
    <div className="h-full overflow-y-auto scroll-thin p-3 pt-24">
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <select value={key} onChange={(e) => { setKey(e.target.value); setLimit(96); }} className={selectCls}>
          {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
        </select>
        <span className="text-dim">sort</span>
        {[['index', 'by number'], ['freq', 'most active'], ['purity', 'most single-label']].map(([k, t]) => <Btn key={k} on={sort === k} onClick={() => setSort(k)}>{t}</Btn>)}
        <span className="text-[10px] text-dim">{cards.length} units. Pictures are the activation-weighted average of each unit's top 8 items.<Cite id="schubert2020" /></span>
      </div>
      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))' }}>
        {cards.slice(0, limit).map((card) => {
          const ex = explFor(card.room, card.c, card.isSae);
          const on = selected && selected.room === card.room && !!selected.sae === card.isSae && (card.isSae ? selected.index === card.c : unitOf(card.L, selected.index).c === card.c);
          return (
            <button key={card.c} onClick={() => onPick({ room: card.room, index: card.isSae ? card.c : card.L.idx(card.c, 0), sae: card.isSae })}
              className={`space-y-1 rounded-md border p-2 text-left ${on ? 'border-warm/70 bg-warm/5' : 'border-line hover:border-glow/40'}`}>
              <div className="flex items-start gap-2">
                {thumbFor(card)}
                <div className="min-w-0 text-[10px] leading-tight">
                  <div className="text-xs text-slate-100">{card.isSae ? 'feature' : 'channel'} {card.c}</div>
                  {ex && <div className="mt-0.5 italic text-slate-200">"{ex.explanation}"</div>}
                </div>
              </div>
              <div className="text-[10px] leading-tight text-dim">{card.sm.text}</div>
            </button>
          );
        })}
      </div>
      {cards.length > limit && <div className="mt-3 text-center"><Btn onClick={() => setLimit((l) => l + 96)}>show more ({cards.length - limit} left)</Btn></div>}
    </div>
  );
}

// Models with symbolic sizes (e.g. a sequence length) need a concrete value to run.
function DimsEditor({ model, onDims }) {
  const [dims, setDims] = useState(model.inputDims);
  return (
    <div className="flex flex-wrap items-center gap-1 text-[11px]">
      <span className="text-dim">input size</span>
      {dims.map((d, i) => model.symbolic[i]
        ? <input key={i} type="number" min={1} value={d} onChange={(e) => setDims(dims.map((x, j) => (j === i ? Math.max(1, +e.target.value) : x)))} className={`${selectCls} w-14`} />
        : <span key={i} className="text-slate-300">{d}</span>)}
      <Btn onClick={() => onDims(dims)}>apply</Btn>
      <span className="text-[10px] text-dim">(editable sizes are symbolic in the file)</span>
    </div>
  );
}

// The model at a glance, per layer: how many units never fire, how single-label the units are,
// and the most label-specific ones. All counted from the dataset pass.
function LabReport({ store, rooms, summaries, sae, onPick }) {
  if (!store) return <div className="flex h-full items-center justify-center p-6 text-sm text-dim">Run a dataset first; the report is counted from it.</div>;
  const groups = [...rooms.filter((r) => store.stats[r.name] && !r.isOutput).map((r) => ({ key: r.name, label: `${r.label} · ${r.dims.join('×')}`, items: summaries.filter((e) => e.room === r.name && !e.sae) })),
    ...(sae ? [{ key: 'sae', label: `SAE on ${sae.room} (${sae.sae.F} features)`, items: summaries.filter((e) => e.sae), sae: true }] : [])];
  const acc = store.labels ? store.preds.filter((p, i) => p === store.labels[i]).length / store.N : null;
  return (
    <div className="h-full overflow-y-auto scroll-thin p-4 pt-24 text-xs">
      <div className="mb-3 text-sm text-slate-100">
        Model report over {store.N} items{acc != null ? `, accuracy ${(100 * acc).toFixed(1)}%` : ''}
      </div>
      <div className="space-y-3">
        {groups.map((g) => {
          const n = g.items.length;
          const dead = g.sae ? (sae.sae.F - n) : g.items.filter((e) => e.summary.freq === 0).length;
          const always = g.items.filter((e) => e.summary.freq >= 0.99).length;
          const pur = g.items.map((e) => e.summary.purity?.value).filter((v) => v != null);
          const top = g.items.filter((e) => e.summary.purity).sort((a, b) => b.summary.purity.value - a.summary.purity.value).slice(0, 5);
          const total = g.sae ? sae.sae.F : n;
          return (
            <div key={g.key} className="rounded-lg border border-line p-3">
              <div className="mb-1 text-slate-100">{g.label}</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-slate-300 md:grid-cols-4">
                <div>never fire: <span className={dead ? 'text-warm' : 'text-glow'}>{dead} of {total}</span></div>
                <div>fire on 99%+ of items: {always}</div>
                <div>median top-12 label purity: {pur.length ? median(pur).toFixed(2) : 'n/a'} {pur.length ? <span className="text-dim">(n = {pur.length})</span> : null}</div>
                <div>fully one label: {pur.filter((v) => v === 1).length}</div>
              </div>
              {top.length > 0 && (
                <div className="mt-2 space-y-0.5">
                  <div className="text-[10px] uppercase tracking-wide text-dim">most label-specific</div>
                  {top.map((e) => (
                    <button key={e.key} onClick={() => onPick(e)} className="block w-full truncate text-left text-[11px] text-slate-300 hover:text-glow">
                      {e.sae ? `feature ${e.c}` : `channel ${e.c}`}: {e.summary.text}{e.expl ? `  "${e.expl.explanation}"` : ''}
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[10px] text-dim">
        Label purity is the share of a unit's 12 strongest items that carry its most common label; it needs labels and at least 12 active items. It is a rough stand-in for a unit meaning one thing.
      </p>
    </div>
  );
}

// Activation patching: run the current input, but copy one room's activations from the pinned
// input. If the answer follows the pinned input, that room carries what decides it (Meng et al.
// 2022; see Heimersheim and Nanda 2024 for how to read patching results).
function PatchPanel({ model, rooms, pinned, xNow, classes }) {
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const argmax = (a) => Array.from(a).indexOf(Math.max(...a));
  const name = (k) => classes?.[k] ?? k;
  const run = async () => {
    setBusy(true);
    try { await runInner(); } catch (e) { setRes({ error: String(e?.message ?? e) }); } finally { setBusy(false); }
  };
  const runInner = async () => {
    const pinnedPred = argmax(pinned.outs[model.outputName].data);
    const base = await model.run([xNow]);
    const nowPred = argmax(base[model.outputName].data);
    const rows = [];
    for (const r of rooms) {
      if (r.isOutput || !pinned.outs[r.name]) continue;
      const src = pinned.outs[r.name].data;
      const o = await model.runWith(xNow, r.name, () => Float32Array.from(src));
      const p = maybeSoftmax(o[model.outputName].data);
      rows.push({ label: r.label, pred: p.indexOf(Math.max(...p)), pPinned: p[pinnedPred] });
    }
    setRes({ pinnedPred, nowPred, rows });
  };
  return (
    <div className="relative">
      <Btn onClick={run} disabled={busy}>{busy ? 'Patching...' : 'Patch layers from the pinned input'}</Btn>
      {res?.error && <div className="absolute bottom-8 left-0 w-72 rounded border border-red-900 bg-panel p-2 text-[11px] text-red-300">Patching failed: {res.error}</div>}
      {res && !res.error && (
        <div className="absolute bottom-8 left-0 w-80 rounded-md border border-line bg-panel p-2 text-[11px] text-slate-300 shadow-xl">
          <div className="mb-1">
            Current answer <span className="text-glow">{name(res.nowPred)}</span>, pinned answer <span className="text-warm">{name(res.pinnedPred)}</span>. Copying one layer from the pinned input gives:
          </div>
          {res.rows.map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span className="w-28 truncate text-dim">{r.label}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded bg-line"><div className="h-full bg-warm" style={{ width: `${r.pPinned * 100}%` }} /></div>
              <span className={`w-16 text-right ${r.pred === res.pinnedPred ? 'text-warm' : 'text-slate-300'}`}>{name(r.pred)} {(r.pPinned * 100).toFixed(0)}%</span>
            </div>
          ))}
          <div className="mt-1 text-[10px] text-dim">
            Bars: probability of the pinned answer after patching that layer. Without skip connections every later layer is computed from the patched one, so a whole-layer patch hands over the answer by construction; with skip connections it need not. To find which units carry the answer, patch single channels from their rooms.<Cite id="meng2022" /><Cite id="heimersheim2024" />
          </div>
          <button className="mt-1 text-[10px] text-dim underline" onClick={() => setRes(null)}>close</button>
        </div>
      )}
    </div>
  );
}

// Patch only this channel (every position) from the pinned input into the current one.
function ChannelPatch({ model, room, L, c, pinned, xNow, classes }) {
  const [res, setRes] = useState(null);
  const argmax = (a) => Array.from(a).indexOf(Math.max(...a));
  const name = (k) => classes?.[k] ?? k;
  const run = async () => {
    try { await runInner(); } catch (e) { setRes({ error: String(e?.message ?? e) }); }
  };
  const runInner = async () => {
    const src = pinned.outs[room]?.data;
    if (!src) return;
    const base = maybeSoftmax((await model.run([xNow]))[model.outputName].data);
    const o = await model.runWith(xNow, room, (d) => { const out = Float32Array.from(d); for (let p = 0; p < L.P; p++) out[L.idx(c, p)] = src[L.idx(c, p)]; return out; });
    const p = maybeSoftmax(o[model.outputName].data);
    const k = argmax(pinned.outs[model.outputName].data);
    setRes({ before: base[k], after: p[k], pred: argmax(p), k });
  };
  return (
    <div className="rounded-md border border-line p-2 text-[11px] text-slate-300">
      <Btn onClick={run}>Patch this channel from the pinned input</Btn>
      {res?.error && <div className="mt-1 text-red-300">Patching failed: {res.error}</div>}
      {res && !res.error && (
        <div className="mt-1">
          Probability of the pinned answer {name(res.k)}: {(100 * res.before).toFixed(1)}% before, {(100 * res.after).toFixed(1)}% with this channel copied over; the answer is now {name(res.pred)}.
          <Cite id="heimersheim2024" />
        </div>
      )}
    </div>
  );
}

function SweepChart({ res, onPick }) {
  return (
    <div className="space-y-1 rounded-md border border-line p-2">
      <div className="text-slate-200">k sweep ({res.F} features, 600 quick steps each{res.running ? ', running' : ''})</div>
      {res.points.map((p) => (
        <button key={p.k} onClick={() => onPick(p.k)} className="flex w-full items-center gap-2 text-left text-[11px] hover:text-glow">
          <span className="w-10 text-dim">k = {p.k}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded bg-line"><div className="h-full rounded bg-[#3987e5]" style={{ width: `${Math.max(0, p.ve) * 100}%` }} /></div>
          <span className="w-24 text-right text-slate-300">{(100 * p.ve).toFixed(1)}% · {p.alive} live</span>
        </button>
      ))}
      <div className="text-[10px] text-dim">Held-out variance explained by k. Larger k reconstructs better but each feature carries less on its own (Gao et al. 2024). Click a row to use that k.</div>
    </div>
  );
}
