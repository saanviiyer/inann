import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import { LATENT_LAYERS, pca, neighbours, knnAccuracy, separation } from '../lib/latent.js';
import { net, sae, DIGITS, LABELS } from '../lib/model.js';
import { forward } from '../lib/net.js';

const R = 5; // world radius of the cloud
const NEUTRAL = new THREE.Color('#c3c2b7');
const DIM = new THREE.Color('#4a4f5c');
const HIGHLIGHT = new THREE.Color('#3987e5');

// one sprite texture per numeral, so each point is labelled by its own digit (identity is never color alone)
const numeralTex = Array.from({ length: 10 }, (_, d) => {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 52px Inter, Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(d), 32, 36);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
});

const byDigit = Array.from({ length: 10 }, (_, d) => LABELS.map((y, i) => (y === d ? i : -1)).filter((i) => i >= 0));

function DigitCloud({ d, target, highlight, onPick }) {
  const idx = byDigit[d];
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(idx.length * 3), 3));
    return g;
  }, [idx]);
  const mat = useRef();
  useFrame(() => {
    // glide toward the new layer's positions
    const a = geo.attributes.position.array;
    let moving = false;
    idx.forEach((i, k) => {
      for (let c = 0; c < 3; c++) {
        const t = target[i][c], v = a[k * 3 + c];
        const nv = v + (t - v) * 0.08;
        if (Math.abs(t - nv) > 1e-3) moving = true;
        a[k * 3 + c] = nv;
      }
    });
    if (moving) geo.attributes.position.needsUpdate = true;
    const want = highlight == null ? NEUTRAL : highlight === d ? HIGHLIGHT : DIM;
    mat.current.color.lerp(want, 0.15);
  });
  return (
    <points geometry={geo} onClick={(e) => { e.stopPropagation(); onPick(idx[e.index]); }}>
      <pointsMaterial ref={mat} map={numeralTex[d]} size={0.42} sizeAttenuation transparent alphaTest={0.3} depthWrite={false} />
    </points>
  );
}

function You({ p, trail }) {
  const ref = useRef();
  const line = useMemo(() => new THREE.BufferGeometry(), []);
  useEffect(() => {
    line.setFromPoints(trail.map((q) => new THREE.Vector3(...q)));
  }, [trail, line]);
  useFrame(({ clock }) => {
    ref.current.position.lerp(new THREE.Vector3(...p), 0.2);
    ref.current.scale.setScalar(1 + 0.15 * Math.sin(clock.elapsedTime * 4));
  });
  return (
    <group>
      <mesh ref={ref}>
        <sphereGeometry args={[0.18, 20, 14]} />
        <meshBasicMaterial color="#ffb547" toneMapped={false} />
        <Html center position={[0, 0.42, 0]} style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded bg-black/60 px-1 text-[10px] text-warm">your input</div>
        </Html>
      </mesh>
      {trail.length > 1 && (
        <line geometry={line}>
          <lineBasicMaterial color="#ffd27a" transparent opacity={0.8} toneMapped={false} />
        </line>
      )}
    </group>
  );
}

function Axes() {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-R, 0, 0, R, 0, 0, 0, -R, 0, 0, R, 0, 0, 0, -R, 0, 0, R]), 3));
    return g;
  }, []);
  return (
    <group>
      <lineSegments geometry={geo}>
        <lineBasicMaterial color="#1c2436" />
      </lineSegments>
      {[['PC1', [R + 0.4, 0, 0]], ['PC2', [0, R + 0.4, 0]], ['PC3', [0, 0, R + 0.4]]].map(([t, p]) => (
        <Html key={t} position={p} center style={{ pointerEvents: 'none' }}>
          <div className="text-[10px] text-dim">{t}</div>
        </Html>
      ))}
    </group>
  );
}

export function LatentScene({ layer, f, highlight, trail, onPick }) {
  const P = useMemo(() => pca(layer), [layer]);
  const target = useMemo(() => P.pts.map((q) => q.map((v) => (v / P.scale) * R)), [P]);
  const me = useMemo(() => P.project(f.acts[layer]).map((v) => (v / P.scale) * R), [P, f, layer]);
  const trailW = useMemo(() => trail.map((acts) => P.project(acts[layer]).map((v) => (v / P.scale) * R)), [trail, P, layer]);
  return (
    <Canvas camera={{ position: [8, 6, 10], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={['#05070d']} />
      <Axes />
      {Array.from({ length: 10 }, (_, d) => <DigitCloud key={d} d={d} target={target} highlight={highlight} onPick={onPick} />)}
      <You p={me} trail={trailW} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </Canvas>
  );
}

export function LatentPanel({ layer, setLayer, f, highlight, setHighlight, setX, setTrail }) {
  const P = useMemo(() => pca(layer), [layer]);
  const sep = useMemo(() => LATENT_LAYERS.map((l) => ({ ...l, acc: knnAccuracy(l.key), ratio: separation(l.key) })), []);
  const maxRatio = Math.max(...sep.map((l) => l.ratio));
  const sepOf = (k) => sep.find((l) => l.key === k);
  const nb = useMemo(() => neighbours(layer, f.acts[layer], 5), [layer, f]);
  const firstOf = (d) => LABELS.indexOf(d);
  const [a, setA] = useState(1);
  const [b, setB] = useState(7);
  const [t, setT] = useState(0);
  const blend = (A, B, tt) => DIGITS[firstOf(A)].map((v, i) => (1 - tt) * v + tt * DIGITS[firstOf(B)][i]);
  const runPath = (A, B, tt) => {
    setX(blend(A, B, tt), 'latent');
    // the path through every layer at 21 steps, drawn as a trail in whichever layer is shown
    setTrail(Array.from({ length: 21 }, (_, k) => forward(net, blend(A, B, k / 20), { sae }).acts));
  };
  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Latent space</div>
      <p className="text-xs leading-snug text-slate-400">
        Each numeral is one held-out digit, placed by how a layer represents it, squeezed to 3D with <Term k="PCA" />.<Cite id="jolliffe2016" /> Switch layers and watch the same 450 digits move.
      </p>
      <div className="flex flex-wrap gap-1">
        {LATENT_LAYERS.map((l) => <Btn key={l.key} on={layer === l.key} onClick={() => setLayer(l.key)}>{l.label}</Btn>)}
      </div>
      <div className="text-xs text-slate-300">
        This view keeps <span className="text-glow">{(100 * P.explained.reduce((x, y) => x + y, 0)).toFixed(0)}%</span> of the spread in {P.D} dimensions
        (PC1 {(100 * P.explained[0]).toFixed(0)}%, PC2 {(100 * P.explained[1]).toFixed(0)}%, PC3 {(100 * P.explained[2]).toFixed(0)}%).
      </div>

      <div>
        <div className="mb-1 text-xs text-slate-200">Two ways to ask how each layer groups same digits<Cite id="alain2016" /></div>
        <div className="flex text-[10px] text-dim">
          <span className="w-28" />
          <span className="flex-1"><Term k="nearest neighbours">5-nearest-neighbour</Term> accuracy</span>
          <span className="w-24 text-right"><Term k="separation ratio">spacing ratio</Term></span>
        </div>
        {sep.map((l) => (
          <button key={l.key} onClick={() => setLayer(l.key)} className="flex w-full items-center gap-2 text-left text-[11px]">
            <span className={`w-28 ${layer === l.key ? 'text-glow' : 'text-dim'}`}>{l.label}</span>
            <span className="w-10 text-slate-300">{(l.acc * 100).toFixed(1)}%</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-line">
              <div className="h-full rounded bg-[#3987e5]" style={{ width: `${(l.ratio / maxRatio) * 100}%` }} />
            </div>
            <span className="w-10 text-right text-slate-300">{l.ratio.toFixed(2)}</span>
          </button>
        ))}
        <div className="mt-1 text-[10px] text-dim">Both use the full space, not this 3D view. Neighbour accuracy is leave-one-out.</div>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span className="text-slate-200">Highlight</span>
        <select value={highlight ?? ''} onChange={(e) => setHighlight(e.target.value === '' ? null : Number(e.target.value))} className="rounded border border-line bg-panel px-1 py-0.5 text-slate-200">
          <option value="">none</option>
          {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>digit {d}</option>)}
        </select>
        <span className="text-[10px] text-dim">click any numeral to load that digit</span>
      </div>

      <div>
        <div className="mb-1 text-xs text-slate-200">Your input's nearest held-out digits in this layer</div>
        <div className="flex gap-1.5">
          {nb.map((n) => (
            <div key={n.i} className="text-center">
              <PixelGrid values={DIGITS[n.i]} size={38} />
              <div className="text-[9px] text-dim">{n.label} · {n.d.toFixed(1)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-1.5 rounded-md border border-line p-2">
        <div className="text-xs text-slate-200">Walk between two digits</div>
        <div className="flex items-center gap-1.5 text-xs">
          <select value={a} onChange={(e) => { setA(+e.target.value); runPath(+e.target.value, b, t); }} className="rounded border border-line bg-panel px-1 text-slate-200">
            {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
          </select>
          <input type="range" min={0} max={1} step={0.05} value={t} onChange={(e) => { setT(+e.target.value); runPath(a, b, +e.target.value); }} className="flex-1 accent-[#ffb547]" />
          <select value={b} onChange={(e) => { setB(+e.target.value); runPath(a, +e.target.value, t); }} className="rounded border border-line bg-panel px-1 text-slate-200">
            {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div className="text-[11px] text-slate-400">
          The input fades pixel by pixel from one held-out {a} to one held-out {b}. Its trail shows where the layer puts each in-between image; the model reads it as{' '}
          <span className="text-warm">{f.pred}</span> right now.
        </div>
      </div>

      <p className="text-xs leading-snug text-slate-200">
        These digits are easy, so raw pixels already put most digits next to a same-digit neighbour ({(sepOf('input').acc * 100).toFixed(1)}%), and depth barely moves that number.
        What depth changes is the spacing: different digits sit {sepOf('input').ratio.toFixed(1)} times farther apart than same digits spread in pixels, and {sepOf('out').ratio.toFixed(1)} times at the output.
      </p>
      <NotShown>
        The other {P.D - 3} dimensions. Points that look close here can be far apart in the full space; the neighbour list uses the full space. Nonlinear maps like t-SNE are left out because they can invent clusters and distances.
        <Cite id="wattenberg2016" />
      </NotShown>
    </div>
  );
}
