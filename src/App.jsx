import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Scene from './components/Scene.jsx';
import Prologue from './components/Prologue.jsx';
import DrawPad from './components/DrawPad.jsx';
import Inspector from './components/Inspector.jsx';
import { FeatureVis, Patches, Neighborhood } from './components/SeePanel.jsx';
import { Adversarial, Occlusion, occlude } from './components/BreakPanel.jsx';
import { Term, Cite, NotShown, Btn, Kbd, Sources, Boundary } from './components/ui.jsx';
import { net, META, DIGITS, LABELS, describe } from './lib/model.js';
import { forward, receptiveField, strongestPath } from './lib/net.js';
import { topPatches } from './lib/analysis.js';

const CHAPTERS = [
  { n: 1, name: 'Why this exists' },
  { n: 2, name: 'Inside the network' },
  { n: 3, name: 'What neurons see' },
  { n: 4, name: 'Break it' },
];

const INTRO = {
  2: <>This is a real trained network laid out as rooms, with the input at the bottom and the answer at the top; draw a digit and watch the light climb.</>,
  3: <>Pick a neuron and ask it what it responds to.</>,
  4: <>Now change the input on purpose and watch the flow reroute.</>,
};

const TAKEAWAY = {
  2: (
    <>
      Each neuron only adds up what the layer below sends it, so the answer is built in steps you can watch. The brightest path is the chain of largest
      contributions for this input, but many dimmer neurons contribute too.
    </>
  ),
  3: (
    <>
      Feature visualization shows what could excite a neuron, and the top real patches show what does; you need both. Neurons rarely mean one clean thing,
      and that is still an open research problem.
    </>
  ),
  4: (
    <>
      Small input changes can redirect the whole internal flow. Occlusion shows which parts of the input a decision actually depends on.
    </>
  ),
};

const firstSeven = DIGITS[LABELS.indexOf(7)];

export default function App() {
  const [chapter, setChapter] = useState(1);
  const [x, setXRaw] = useState(() => Float32Array.from(firstSeven));
  const [drawKey, setDrawKey] = useState(0);
  const [occ, setOcc] = useState({ on: false, size: 2, pos: 0 });
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  const [xray, setXray] = useState(true);
  const [follow, setFollow] = useState(false);
  const [showPath, setShowPath] = useState(false);
  const [pausedUI, setPausedUI] = useState(false);
  const [fv, setFv] = useState(null);
  const [sources, setSources] = useState(false);
  const paused = useRef(false);
  const [entered, setEntered] = useState(false); // mount the 3D space the first time it is needed, then keep it
  useEffect(() => { if (chapter > 1) setEntered(true); }, [chapter]);

  const setX = useCallback((v, src) => {
    setXRaw(Float32Array.from(v));
    if (src === 'draw' || src === 'sample') setDrawKey((k) => k + 1);
  }, []);

  const xe = useMemo(() => (chapter === 4 ? occlude(x, occ) : x), [x, occ, chapter]);
  const f = useMemo(() => forward(net, xe), [xe]);
  const stamp = f; // a new forward pass starts a new wave
  const path = useMemo(() => strongestPath(net, f), [f]);
  const rfMask = useMemo(() => (selected ? receptiveField(selected.layer, selected.index) : null), [selected]);
  const patches = useMemo(() => {
    if (chapter !== 3 || !selected || !['conv1', 'conv2', 'dense'].includes(selected.layer)) return null;
    return topPatches(selected.layer, selected.index, 5).map((t) => ({ values: t.patch, w: t.w, h: t.h, caption: `digit ${t.label}` }));
  }, [chapter, selected]);

  const onSelect = useCallback((s) => setSelected(s), []);
  const onHover = useCallback((h) => setHover(h), []);
  const onFvFrame = useCallback((v) => setFv(v), []);

  useEffect(() => { if (chapter !== 3) setFv(null); if (chapter === 1) setFollow(false); }, [chapter]);

  useEffect(() => {
    const key = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        paused.current = !paused.current;
        setPausedUI(paused.current);
      } else if (e.key === 'f' || e.key === 'F') {
        if (chapter > 1) setFollow((v) => !v);
      }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [chapter]);

  const hoverInfo = hover ? { ...describe(hover.layer, hover.index), a: f.acts[hover.layer][hover.index], layer: hover.layer } : null;

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-2">
        <div className="text-sm font-semibold tracking-[0.3em] text-glow">INANN</div>
        <nav className="flex flex-wrap gap-1">
          {CHAPTERS.map((c) => (
            <button
              key={c.n}
              onClick={() => setChapter(c.n)}
              className={`rounded px-2 py-1 text-xs ${chapter === c.n ? 'bg-glow/15 text-glow' : 'text-dim hover:text-slate-200'}`}
            >
              {c.n}. {c.name}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-[11px] text-dim">
          <span><Kbd>space</Kbd> {pausedUI ? 'resume' : 'pause'}</span>
          {chapter > 1 && <span><Kbd>F</Kbd> follow the signal</span>}
          <button onClick={() => setSources(true)} className="text-glow/80 hover:text-glow">Sources</button>
        </div>
      </header>

      {chapter === 1 && (
        <main className="flex-1 overflow-y-auto">
          <Prologue paused={paused} onNext={() => setChapter(2)} />
        </main>
      )}

      <main className={`${chapter === 1 ? 'hidden' : 'flex'} min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden`}>
        {/* left: intro, input, controls, takeaway */}
        <aside className="scroll-thin order-2 w-full shrink-0 space-y-4 border-line p-4 lg:overflow-y-auto lg:order-1 lg:w-80 lg:shrink-0 lg:border-r">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-dim">Chapter {chapter} · {CHAPTERS[chapter - 1]?.name}</p>
            <p className="mt-2 text-sm leading-relaxed text-slate-100">{INTRO[chapter]}</p>
          </div>
          <DrawPad x={x} onChange={setX} />
          <div className="rounded-lg border border-line p-3 text-xs">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="text-dim">Model says</span>
              <span className="text-2xl font-semibold text-warm">{f.pred}</span>
            </div>
            <div className="space-y-0.5">
              {Array.from(f.probs).map((p, d) => (
                <div key={d} className="flex items-center gap-2">
                  <span className="w-3 text-dim">{d}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded bg-line">
                    <div className={`h-full ${d === f.pred ? 'bg-warm' : 'bg-glow/60'}`} style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="w-9 text-right text-dim">{(p * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Btn on={xray} onClick={() => setXray((v) => !v)} title="spread the layers apart, or stack them tightly">X-ray {xray ? 'on' : 'off'}</Btn>
            <Btn on={follow} onClick={() => setFollow((v) => !v)}>Follow the signal</Btn>
            <Btn on={showPath} onClick={() => setShowPath((v) => !v)}>Show strongest path</Btn>
            <Btn on={pausedUI} onClick={() => { paused.current = !paused.current; setPausedUI(paused.current); }}>{pausedUI ? 'Paused' : 'Pause'}</Btn>
          </div>
          {chapter === 2 && (
            <div className="space-y-2 text-xs leading-relaxed text-slate-400">
              <p>
                Rooms from bottom to top: the 8x8 input, Conv 1 (6 <Term k="channel">channels</Term> of 8x8), Conv 2 (12 channels of 4x4), a{' '}
                <Term k="dense layer" /> of 32, and 10 output pods. Brightness is <Term k="activation" />. Nothing here learns; this is{' '}
                <Term k="inference" /> only.
              </p>
              <p>
                Reading a network one neuron at a time follows the Circuits thread<Cite id="olah2020" />. The walk-through layout borrows from 3D node-link views
                of small digit networks<Cite id="harley2015" /> and from the direct manipulation of TensorFlow Playground<Cite id="smilkov2017" />.
              </p>
              <p>
                The model is real: {META.n_train} training digits, {(META.test_acc * 100).toFixed(1)}% correct on {META.n_test} held-out ones. The digits are the
                8x8 UCI set<Cite id="alpaydin1998" />, not MNIST.
              </p>
              <NotShown>Max pooling has no room of its own. Edges from a pooled layer are drawn to the cell that won the pool.</NotShown>
              <NotShown>Biases. They are in the math, not in the picture.</NotShown>
            </div>
          )}
          <div className="rounded-lg border border-glow/20 bg-glow/5 p-3 text-xs leading-relaxed text-slate-200">{TAKEAWAY[chapter]}</div>
          {chapter === 4 && (
            <div className="rounded-lg border border-line p-3 text-xs text-slate-400">
              <span className="text-[10px] uppercase tracking-[0.2em] text-dim">Coming next</span>
              <p className="mt-1 text-slate-300">Transformers, where each word uses attention to choose which other words to read from.<Cite id="vaswani2017" /></p>
            </div>
          )}
        </aside>

        {/* center: the space */}
        <section className="relative order-1 h-[60vh] min-w-0 shrink-0 overflow-hidden lg:order-2 lg:h-auto lg:flex-1 lg:shrink">
          {entered && <Scene
            active={chapter > 1}
            f={f}
            stamp={stamp}
            selected={selected}
            onSelect={onSelect}
            onHover={onHover}
            xray={xray}
            follow={follow}
            paused={paused}
            path={path}
            rfMask={rfMask}
            fv={chapter === 3 ? fv : null}
            patches={patches}
            showPath={showPath}
          />}
          <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/50 px-2 py-1 text-[11px] text-slate-300">
            {hoverInfo ? (
              <>
                {hoverInfo.title}: <span className="text-glow">{hoverInfo.layer === 'out' ? `${(hoverInfo.a * 100).toFixed(1)}%` : hoverInfo.a.toFixed(3)}</span>
              </>
            ) : follow ? (
              <>Following the strongest contributions from input to answer. <Kbd>F</Kbd> to stop.</>
            ) : (
              <>Drag to orbit, scroll to zoom, W A S D Q E to fly, click a neuron.</>
            )}
          </div>
          {pausedUI && <div className="pointer-events-none absolute right-3 top-3 rounded bg-black/60 px-2 py-1 text-xs text-warm">paused</div>}
        </section>

        {/* right: inspector and chapter tools */}
        <aside className="scroll-thin order-3 w-full shrink-0 space-y-3 border-line p-4 lg:overflow-y-auto lg:w-[22rem] lg:shrink-0 lg:border-l">
          <Boundary key={`bi-${selected?.layer}-${selected?.index}`}><Inspector sel={selected} f={f} rfMask={rfMask} onClear={() => setSelected(null)} /></Boundary>
          {chapter === 3 && (
            <Boundary>
              <FeatureVis sel={selected} f={f} paused={paused} onFrame={onFvFrame} />
              <Patches sel={selected} />
              <Neighborhood sel={selected} onSelect={onSelect} />
            </Boundary>
          )}
          {chapter === 4 && (
            <Boundary>
              <Adversarial key={drawKey} x={x} f={forward(net, x)} setX={setX} paused={paused} />
              <Occlusion x={x} occ={occ} setOcc={setOcc} />
            </Boundary>
          )}
        </aside>
      </main>
      {sources && <Sources onClose={() => setSources(false)} />}
    </div>
  );
}
