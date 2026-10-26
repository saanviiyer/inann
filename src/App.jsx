import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Scene from './components/Scene.jsx';
import Prologue from './components/Prologue.jsx';
import DrawPad from './components/DrawPad.jsx';
import Inspector from './components/Inspector.jsx';
import FeatureRoom from './components/FeatureRoom.jsx';
import SearchBar from './components/SearchBar.jsx';
import SteerPanel from './components/SteerPanel.jsx';
import SaePanel from './components/SaePanel.jsx';
import { LatentScene, LatentPanel } from './components/Latent.jsx';
import { ProbeScene, ProbePanel } from './components/Probes.jsx';
import LabPage from './components/lab/LabPage.jsx';
import SteerPage from './components/steer/SteerPage.jsx';
import CoursePage from './components/course/CoursePage.jsx';
import { FeatureVis, Patches, Neighborhood } from './components/SeePanel.jsx';
import { Adversarial, Occlusion, occlude } from './components/BreakPanel.jsx';
import { Term, Cite, NotShown, Btn, Kbd, Sources, Boundary, About } from './components/ui.jsx';
import { useTheme, setTheme } from './lib/theme.js';
import { net, sae, META, DIGITS, LABELS, describe, roomOf, roomName, bestUnitInRoom } from './lib/model.js';
import { topExamples } from './lib/rooms.js';
import { forward, receptiveField, strongestPath } from './lib/net.js';
import { topPatches } from './lib/analysis.js';

const CHAPTERS = [
  { n: 1, name: 'Why this exists' },
  { n: 2, name: 'Inside the network' },
  { n: 3, name: 'What neurons see' },
  { n: 4, name: 'Break it' },
  { n: 5, name: 'Feature rooms' },
  { n: 6, name: 'Steer it' },
  { n: 7, name: 'Neurons vs features' },
  { n: 8, name: 'Latent space' },
  { n: 9, name: 'Probes' },
];

const INTRO = {
  2: <>This is a real trained network laid out as rooms, with the input at the bottom and the answer at the top; draw a digit and watch the light climb.</>,
  3: <>Pick a neuron and ask it what it responds to.</>,
  4: <>Now change the input on purpose and watch the flow reroute.</>,
  5: <>Every unit has a room that shows what it responds to, how often, and a box where you can test it yourself.</>,
  6: <>Grab a unit, fix its value, and watch the answer change.</>,
  7: <>Swap the 32 dense neurons for 256 sparse features and compare how single-minded each one is.</>,
  8: <>Every digit becomes a point in each layer's space; see where yours lands and who its neighbours are.</>,
  9: <>Hang a thermometer on every layer: train a straight-line reader on each one and see how much of the digit it can read.</>,
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
  5: (
    <>
      A unit's top examples and its histogram together say what it does and how often it does it. A written explanation is only as good as its simulation score, so read the badge before you trust the sentence.
    </>
  ),
  6: (
    <>
      Fixing a unit and watching the output move shows that the unit causes the change, which watching activations alone cannot. A good demo moves one kind of digit and not every digit, so read the counts under each preset.
    </>
  ),
  7: (
    <>
      Sparse features give up a little accuracy for units that are usually easier to name. Whether a given feature is clean is a measurement shown in its room, not a promise.
    </>
  ),
  8: (
    <>
      A layer's space is where the network does its work, and distances there show which inputs it treats as alike. A 3D picture of that space always hides most of it, so trust the full-space numbers over the picture.
    </>
  ),
  9: (
    <>
      A probe shows what a layer makes readable, not what the network uses. Only the controls give the number meaning, and only an intervention, like steering along the probed direction, tests use.
    </>
  ),
};

const firstSeven = DIGITS[LABELS.indexOf(7)];

export default function App() {
  const [chapter, setChapter] = useState(1);
  const areaFromHash = () => {
    if (typeof location === 'undefined') return 'tour';
    if (location.hash === '#lab') return 'lab';
    if (location.hash === '#steer') return 'steer';
    if (location.hash.startsWith('#course')) return 'course';
    return 'tour';
  };
  const [area, setArea] = useState(areaFromHash);
  useEffect(() => {
    // leave a #course/N deep link untouched; only set the bare hash when switching areas
    try {
      if (area === 'course' && location.hash.startsWith('#course')) return;
      history.replaceState(null, '', `${location.pathname}${location.search}${area === 'lab' ? '#lab' : area === 'steer' ? '#steer' : area === 'course' ? '#course' : ''}`);
    } catch { /* ignore */ }
  }, [area]);
  useEffect(() => {
    const onHash = () => setArea(areaFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
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
  const [about, setAbout] = useState(false);
  const theme = useTheme();
  const paused = useRef(false);
  const [mode, setModeRaw] = useState('neurons');
  const [clamp, setClamp] = useState(null);
  const [flyTo, setFlyTo] = useState(null);
  const [latentLayer, setLatentLayer] = useState('dense');
  const [latentAxes, setLatentAxes] = useState(0);
  const [probeLayer, setProbeLayer] = useState('dense');
  const [highlight, setHighlight] = useState(null);
  const [trail, setTrail] = useState([]);
  const [entered, setEntered] = useState(false); // mount the 3D space the first time it is needed, then keep it
  useEffect(() => { if (chapter > 1) setEntered(true); }, [chapter]);

  const setX = useCallback((v, src) => {
    setXRaw(Float32Array.from(v));
    if (src === 'draw' || src === 'sample') setDrawKey((k) => k + 1);
  }, []);

  const xe = useMemo(() => (chapter === 4 ? occlude(x, occ) : x), [x, occ, chapter]);
  const fBase = useMemo(() => forward(net, xe, { sae }), [xe]);
  const f = useMemo(() => (clamp ? forward(net, xe, { sae, clamp }) : fBase), [xe, clamp, fBase]);
  const stamp = f; // a new forward pass starts a new wave
  // in the rooms chapter, the current runs from the input to the selected unit ("test this neuron")
  const pulseTo = chapter === 5 && selected && selected.layer !== 'input' ? selected : null;
  const path = useMemo(() => strongestPath(net, f, pulseTo, mode === 'features' ? sae : null), [f, pulseTo, mode]);
  const ring = useMemo(() => {
    if (chapter < 5 || chapter > 7 || !selected || selected.layer === 'input' || selected.layer === 'out') return null;
    return topExamples(roomOf(selected.layer, selected.index), 12).map((t) => ({ values: t.img, caption: `${t.label} · ${t.act.toFixed(1)}` }));
  }, [chapter, selected]);
  const rfMask = useMemo(() => (selected ? receptiveField(selected.layer, selected.index) : null), [selected]);
  const patches = useMemo(() => {
    if (chapter !== 3 || !selected || !['conv1', 'conv2', 'dense'].includes(selected.layer)) return null;
    return topPatches(selected.layer, selected.index, 5).map((t) => ({ values: t.patch, w: t.w, h: t.h, caption: `digit ${t.label}` }));
  }, [chapter, selected]);

  const onSelect = useCallback((s) => setSelected(s), []);
  const setMode = useCallback((m) => {
    setModeRaw(m);
    setSelected((s) => (s && ((m === 'features' && s.layer === 'dense') || (m === 'neurons' && s.layer === 'sae')) ? null : s));
  }, []);
  // open a room from search, a preset or the SAE panel: switch mode if needed, select, and fly there
  const openRoom = useCallback((room, acts) => {
    if (room.layer === 'sae') setModeRaw('features');
    if (room.layer === 'dense') setModeRaw('neurons');
    const unit = room.channel ? bestUnitInRoom(acts, room) : room.index;
    const s = { layer: room.layer, index: unit };
    setSelected(s);
    setFlyTo({ ...s, nonce: Math.random() });
  }, []);
  const onSearchPick = useCallback((r) => {
    if (chapter < 2 || chapter >= 8) setChapter(5);
    openRoom(r.room, f.acts);
  }, [chapter, f, openRoom]);
  const onPreset = useCallback((p, c, i) => {
    const xi = i >= 0 ? DIGITS[i] : x;
    if (i >= 0) setX(DIGITS[i], 'preset');
    setClamp(c);
    openRoom({ layer: p.layer, index: p.index, channel: p.scope === 'channel' }, forward(net, xi, { sae }).acts);
  }, [x, setX, openRoom]);
  const onHover = useCallback((h) => setHover(h), []);
  const onFvFrame = useCallback((v) => setFv(v), []);

  useEffect(() => { if (chapter !== 3) setFv(null); if (chapter === 1 || chapter >= 8) setFollow(false); }, [chapter]);

  useEffect(() => {
    const key = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      if (e.code === 'Space') {
        e.preventDefault();
        paused.current = !paused.current;
        setPausedUI(paused.current);
      } else if (e.key === 'f' || e.key === 'F') {
        if (chapter > 1 && chapter < 8) setFollow((v) => !v);
      }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [chapter]);

  const hoverInfo = hover ? { ...describe(hover.layer, hover.index), a: f.acts[hover.layer][hover.index], layer: hover.layer } : null;

  return (
    <div className="flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line px-4 py-2">
        <button onClick={() => setAbout(true)} className="flex items-baseline gap-2 text-left" title="About this project">
          <span className="text-sm font-semibold tracking-[0.3em] text-glow">INANN</span>
          <span className="hidden text-[11px] text-dim sm:inline">in a neural network!</span>
        </button>
        <div className="flex overflow-hidden rounded-md border border-line text-xs">
          {[['tour', 'Tour'], ['course', 'Course'], ['lab', 'Lab: your model'], ['steer', 'LLM steering']].map(([k, t]) => (
            <button key={k} onClick={() => setArea(k)} className={`px-2.5 py-1 ${area === k ? 'bg-glow/15 text-glow' : 'text-dim hover:text-slate-200'}`}>{t}</button>
          ))}
        </div>
        {area === 'tour' && <nav className="flex flex-wrap gap-1">
          {CHAPTERS.map((c) => (
            <button
              key={c.n}
              onClick={() => setChapter(c.n)}
              className={`rounded px-2 py-1 text-xs ${chapter === c.n ? 'bg-glow/15 text-glow' : 'text-dim hover:text-slate-200'}`}
            >
              {c.n}. {c.name}
            </button>
          ))}
        </nav>}
        <div className="ml-auto flex items-center gap-3 text-[11px] text-dim">
          <span><Kbd>space</Kbd> {pausedUI ? 'resume' : 'pause'}</span>
          {chapter > 1 && chapter < 8 && <span><Kbd>F</Kbd> follow the signal</span>}
          <button onClick={() => setAbout(true)} className="text-glow/80 hover:text-glow">About</button>
          <button onClick={() => setSources(true)} className="text-glow/80 hover:text-glow">Sources</button>
          <button
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            className="rounded-md border border-line px-2 py-0.5 text-slate-300 hover:border-glow/50"
            title="switch between the light and dark themes"
          >
            {theme === 'light' ? 'Dark mode' : 'Light mode'}
          </button>
        </div>
        {area === 'tour' && chapter > 1 && (
          <div className="flex w-full flex-wrap items-center gap-2">
            <SearchBar onPick={onSearchPick} />
            <div className="flex overflow-hidden rounded-md border border-line text-xs" title="show raw dense neurons, or the sparse autoencoder's features in their place">
              {['neurons', 'features'].map((m) => (
                <button key={m} onClick={() => setMode(m)} className={`px-2.5 py-1 ${mode === m ? 'bg-glow/15 text-glow' : 'text-dim hover:text-slate-200'}`}>
                  {m === 'neurons' ? 'Neurons' : 'Features'}
                </button>
              ))}
            </div>
            {clamp && (
              <span className="flex items-center gap-1.5 rounded-md border border-warm/50 bg-warm/10 px-2 py-1 text-xs text-warm">
                Clamped: {roomName({ layer: clamp.layer, index: clamp.index, channel: clamp.scope === 'channel' })} = {clamp.value.toFixed(2)}
                <button onClick={() => setClamp(null)} className="text-slate-200 hover:text-white" title="release the clamp">release</button>
              </span>
            )}
          </div>
        )}
      </header>

      {area === 'lab' && <LabPage />}
      {area === 'steer' && <SteerPage />}
      {area === 'course' && <CoursePage onGoTour={() => setArea('tour')} />}
      {area === 'tour' && chapter === 1 && (
        <main className="flex-1 overflow-y-auto">
          <Prologue paused={paused} onNext={() => setChapter(2)} />
        </main>
      )}

      <main className={`${area !== 'tour' || chapter === 1 ? 'hidden' : 'flex'} min-h-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden`}>
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
          {chapter < 8 && <div className="flex flex-wrap gap-1.5">
            <Btn on={xray} onClick={() => setXray((v) => !v)} title="spread the layers apart, or stack them tightly">X-ray {xray ? 'on' : 'off'}</Btn>
            <Btn on={follow} onClick={() => setFollow((v) => !v)}>Follow the signal</Btn>
            <Btn on={showPath} onClick={() => setShowPath((v) => !v)}>Show strongest path</Btn>
            <Btn on={pausedUI} onClick={() => { paused.current = !paused.current; setPausedUI(paused.current); }}>{pausedUI ? 'Paused' : 'Pause'}</Btn>
          </div>}
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
          {chapter === 9 && (
            <div className="rounded-lg border border-line p-3 text-xs text-slate-400">
              <span className="text-[10px] uppercase tracking-[0.2em] text-dim">Coming next</span>
              <p className="mt-1 text-slate-300">Transformers, where each word uses attention to choose which other words to read from.<Cite id="vaswani2017" /></p>
            </div>
          )}
          {chapter === 9 && (
            <button onClick={() => setArea('lab')} className="w-full rounded-lg border border-glow/40 bg-glow/5 p-3 text-left text-xs text-slate-200 hover:bg-glow/10">
              <span className="text-[10px] uppercase tracking-[0.2em] text-glow">Your turn</span>
              <p className="mt-1">Open the Lab to load your own ONNX model, walk through its layers, train a sparse autoencoder on any of them and probe every layer, all in your browser.</p>
            </button>
          )}
        </aside>

        {/* center: the space */}
        <section className="relative order-1 h-[60vh] min-w-0 shrink-0 overflow-hidden lg:order-2 lg:h-auto lg:flex-1 lg:shrink">
          {chapter === 8 && (
            <div className="absolute inset-0 z-10">
              <LatentScene layer={latentLayer} f={f} highlight={highlight} trail={trail} onPick={(i) => setX(DIGITS[i], 'sample')} axes={latentAxes} />
            </div>
          )}
          {chapter === 9 && (
            <div className="absolute inset-0 z-10">
              <ProbeScene layer={probeLayer} setLayer={setProbeLayer} />
            </div>
          )}
          <div className={`absolute inset-0 ${chapter >= 8 ? 'invisible' : ''}`}>
          {entered && <Scene
            active={area === 'tour' && chapter > 1 && chapter < 8}
            mode={mode}
            flyTo={flyTo}
            ring={ring}
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
            showPath={showPath || !!pulseTo}
          />}
          </div>
          <div className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/50 px-2 py-1 text-[11px] text-slate-300">
            {hoverInfo ? (
              <>
                {hoverInfo.title}: <span className="text-glow">{hoverInfo.layer === 'out' ? `${(hoverInfo.a * 100).toFixed(1)}%` : hoverInfo.a.toFixed(3)}</span>
              </>
            ) : chapter === 8 ? (
              <>Drag to orbit, scroll to zoom, click a numeral to load that digit.</>
            ) : chapter === 9 ? (
              <>Each floor is a layer; its tube fills to the probe's held-out accuracy. Click a floor to steer it.</>
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
          {chapter < 8 && (
            <Boundary key={`bi-${selected?.layer}-${selected?.index}`}>
              <FeatureRoom sel={selected} f={f} fBase={fBase} setX={setX} clamp={clamp} setClamp={setClamp} rfMask={rfMask} onClear={() => setSelected(null)} />
            </Boundary>
          )}
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
          {chapter === 6 && <Boundary><SteerPanel onRun={onPreset} clamp={clamp} /></Boundary>}
          {chapter === 7 && <Boundary><SaePanel onOpen={(room) => openRoom(room, f.acts)} /></Boundary>}
          {chapter === 8 && (
            <Boundary>
              <LatentPanel layer={latentLayer} setLayer={(l) => { setLatentLayer(l); setLatentAxes(0); }} f={f} highlight={highlight} setHighlight={setHighlight} setX={setX} setTrail={setTrail} axes={latentAxes} setAxes={setLatentAxes} />
            </Boundary>
          )}
          {chapter === 9 && (
            <Boundary>
              <ProbePanel layer={probeLayer} setLayer={setProbeLayer} x={xe} />
            </Boundary>
          )}
        </aside>
      </main>
      <footer className="border-t border-line px-4 py-1.5 text-[10px] text-dim">
        Feature rooms, test-this-neuron, explanation search and steering follow Neuronpedia<Cite id="lin2023" />; explanations follow Bills et al.<Cite id="bills2023" />. {area === 'steer' ? 'The LLM steering page talks to a model server on your own machine; nothing leaves it.' : 'Everything runs in your browser from precomputed files.'}
      </footer>
      {sources && <Sources onClose={() => setSources(false)} />}
      {about && <About onClose={() => setAbout(false)} />}
    </div>
  );
}
