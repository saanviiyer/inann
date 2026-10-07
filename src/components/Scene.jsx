import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { LAYERS, ORDER_NEURONS, ORDER_FEATURES, incoming, resolveSource } from '../lib/net.js';
import { net, sae, SCALE, layerSize } from '../lib/model.js';
import { featureLogitEffect } from '../lib/sae.js';
import { cmap } from './ui.jsx';

// ---------- layout: every neuron gets a fixed spot inside its layer's room ----------
// SAE features take the dense room's slot in the stack when feature mode is on
const SLOT = { input: 0, conv1: 1, conv2: 2, dense: 3, sae: 3, out: 4 };
const BASE = {}; // layer -> Float32Array(n*3) local positions
const SIZE = { input: 0.27, conv1: 0.17, conv2: 0.2, dense: 0.12, sae: 0.075, out: 0.3 };
const FLOOR = { input: [3.2, 3.2], conv1: [7.2, 4.6], conv2: [5.8, 4.4], dense: [5.4, 5.4], sae: [7, 7], out: [8.6, 1.4] };
(function buildLayout() {
  const put = (key, n, fn) => {
    const a = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const [x, y, z] = fn(i); a[i * 3] = x; a[i * 3 + 1] = y; a[i * 3 + 2] = z; }
    BASE[key] = a;
  };
  put('input', 64, (i) => [((i % 8) - 3.5) * 0.34, 0, (Math.floor(i / 8) - 3.5) * 0.34]);
  put('conv1', 384, (i) => {
    const c = Math.floor(i / 64), p = i % 64;
    return [((c % 3) - 1) * 2.25 + ((p % 8) - 3.5) * 0.22, 0, (Math.floor(c / 3) - 0.5) * 2.25 + (Math.floor(p / 8) - 3.5) * 0.22];
  });
  put('conv2', 192, (i) => {
    const c = Math.floor(i / 16), p = i % 16;
    return [((c % 4) - 1.5) * 1.35 + ((p % 4) - 1.5) * 0.26, 0, (Math.floor(c / 4) - 1) * 1.35 + (Math.floor(p / 4) - 1.5) * 0.26];
  });
  put('dense', 32, (i) => {
    const r = 2.3 * Math.sqrt((i + 0.5) / 32), t = i * 2.39996;
    return [r * Math.cos(t), 0.28 * Math.sin(i * 1.7), r * Math.sin(t)];
  });
  put('sae', 256, (i) => {
    const r = 3.3 * Math.sqrt((i + 0.5) / 256), t = i * 2.39996;
    return [r * Math.cos(t), 0.2 * Math.sin(i * 1.3), r * Math.sin(t)];
  });
  put('out', 10, (d) => [(d - 4.5) * 0.8, 0, 0]);
})();

export const layerY = (key, gap) => SLOT[key] * gap;
const gapFor = (xray) => (xray ? 3.1 : 0.75);
export function worldPos(key, i, gap, v = new THREE.Vector3()) {
  const b = BASE[key];
  return v.set(b[i * 3], b[i * 3 + 1] + layerY(key, gap), b[i * 3 + 2]);
}

const BASE_COL = new THREE.Color('#121a2b');
const GLOW_COL = new THREE.Color('#8af0ff');
const HOT_COL = new THREE.Color('#ffffff');
const RF_COL = new THREE.Color('#ffb547');
const WIN_COL = new THREE.Color('#ffd27a');

// ---------- one layer: an instanced mesh whose brightness is the activation ----------
function Layer({ k, target, live, rfMask, pred, onSelect, onHover, gapRef, paused, stamp }) {
  const n = layerSize(k);
  const mesh = useRef();
  const group = useRef();
  const disp = useMemo(() => new Float32Array(n), [n]);
  const from = useMemo(() => new Float32Array(n), [n]);
  const anim = useRef({ t: 0, start: 0, active: true });
  const tmp = useMemo(() => new THREE.Object3D(), []);
  const col = useMemo(() => new THREE.Color(), []);
  const geom = useMemo(() => {
    const s = SIZE[k];
    return k === 'dense' || k === 'out' ? new THREE.SphereGeometry(s, 20, 14) : new THREE.BoxGeometry(s, s * 0.35, s);
  }, [k]);

  useEffect(() => {
    group.current.position.y = layerY(k, gapRef.current);
    for (let i = 0; i < n; i++) {
      tmp.position.set(BASE[k][i * 3], BASE[k][i * 3 + 1], BASE[k][i * 3 + 2]);
      tmp.updateMatrix();
      mesh.current.setMatrixAt(i, tmp.matrix);
    }
    mesh.current.instanceMatrix.needsUpdate = true;
  }, [k, n, tmp]);

  // a new input starts a wave: each layer lights up a beat after the one below it
  useEffect(() => {
    from.set(disp);
    const a = anim.current;
    a.start = a.t + SLOT[k] * 0.11;
    a.active = true;
  }, [stamp, k, from, disp]);
  useEffect(() => { anim.current.active = true; }, [rfMask, pred, live]);

  useFrame((_, dt) => {
    const a = anim.current;
    if (!paused.current) a.t += Math.min(dt, 0.05);
    group.current.position.y += (layerY(k, gapRef.current) - group.current.position.y) * 0.18;
    if (!a.active) return;
    let k01 = Math.min(1, Math.max(0, (a.t - a.start) / 0.25));
    k01 = k01 * k01 * (3 - 2 * k01);
    const sc = SCALE[k];
    for (let i = 0; i < n; i++) {
      disp[i] = from[i] + (target[i] - from[i]) * k01;
      const v = Math.min(1, Math.max(0, disp[i] / sc));
      col.copy(BASE_COL).lerp(GLOW_COL, Math.min(1, v * 1.2));
      if (v > 0.85) col.lerp(HOT_COL, (v - 0.85) * 3);
      if (k === 'out' && i === pred && k01 >= 1) col.lerp(WIN_COL, 0.55);
      if (rfMask && k === 'input' && rfMask[i]) col.lerp(RF_COL, 0.45);
      mesh.current.setColorAt(i, col);
    }
    mesh.current.instanceColor.needsUpdate = true;
    if (k01 >= 1 && a.t > a.start) a.active = false;
  });

  const label = LAYERS.find((l) => l.key === k).label;
  return (
    <group ref={group}>
      <instancedMesh
        ref={mesh}
        args={[geom, undefined, n]}
        onClick={(e) => { e.stopPropagation(); onSelect({ layer: k, index: e.instanceId }); }}
        onPointerMove={(e) => { e.stopPropagation(); onHover({ layer: k, index: e.instanceId }); }}
        onPointerOut={() => onHover(null)}
      >
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      {/* floor of the room */}
      <mesh rotation-x={-Math.PI / 2} position-y={-0.06}>
        <planeGeometry args={FLOOR[k]} />
        <meshBasicMaterial color="#0d1424" transparent opacity={0.35} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Html position={[k === 'conv1' ? -4.4 : k === 'out' ? -4.9 : k === 'sae' ? -4.6 : -3.9, 0, 0]} center style={{ pointerEvents: 'none' }}>
        <div className="whitespace-nowrap text-[11px] uppercase tracking-[0.18em] text-dim">{label}</div>
      </Html>
      {k === 'out' &&
        Array.from({ length: 10 }, (_, d) => (
          <Html key={d} position={[BASE.out[d * 3], 0.55, BASE.out[d * 3 + 2]]} center style={{ pointerEvents: 'none' }}>
            <div className={`text-sm font-semibold ${d === pred ? 'text-warm' : 'text-slate-400'}`}>{d}</div>
          </Html>
        ))}
    </group>
  );
}

// ---------- incoming weights of the selected neuron, as glowing lines ----------
// In feature mode an SAE feature shows its outgoing effect on each digit's score
// (W_out . decoder direction), since the dense neurons it reads from are not drawn.
function Edges({ selected, f, gapRef, mode }) {
  const geo = useMemo(() => new THREE.BufferGeometry(), []);
  const data = useMemo(() => {
    if (!selected || selected.layer === 'input') return null;
    let inc;
    if (selected.layer === 'sae') {
      const e = featureLogitEffect(net, sae, selected.index);
      inc = Array.from(e, (w, d) => ({ layer: 'out', index: d, w }));
    } else if (selected.layer === 'out' && mode === 'features') {
      inc = [];
      for (let fi = 0; fi < sae.F; fi++) inc.push({ layer: 'sae', index: fi, w: featureLogitEffect(net, sae, fi)[selected.index] });
      inc = inc.sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 24);
    } else {
      inc = incoming(net, selected.layer, selected.index).map((s) => ({ layer: s.layer, index: resolveSource(f, s), w: s.w }));
    }
    const m = Math.max(...inc.map((s) => Math.abs(s.w))) || 1;
    return inc.map((s) => ({ ...s, w: s.w / m }));
  }, [selected, f, mode]);
  const v1 = useMemo(() => new THREE.Vector3(), []);
  const v2 = useMemo(() => new THREE.Vector3(), []);
  const ys = useRef({});

  useEffect(() => {
    if (!data) return;
    const pos = new Float32Array(data.length * 6), c = new Float32Array(data.length * 6);
    data.forEach((e, j) => {
      const [r, g, b] = cmap.diverging(e.w);
      const a = 0.45 + 1.1 * Math.abs(e.w);
      c.set([(r / 255) * a * 0.4, (g / 255) * a * 0.4, (b / 255) * a * 0.4, (r / 255) * a, (g / 255) * a, (b / 255) * a], j * 6);
    });
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
  }, [data, geo]);

  useFrame(() => {
    if (!data) return;
    const g = gapRef.current;
    // ease line endpoints with the layers while X-ray animates
    const cur = (ys.current.g = (ys.current.g ?? g) + (g - (ys.current.g ?? g)) * 0.18);
    const pos = geo.attributes.position.array;
    worldPos(selected.layer, selected.index, cur, v2);
    data.forEach((e, j) => {
      worldPos(e.layer, e.index, cur, v1);
      pos.set([v1.x, v1.y, v1.z, v2.x, v2.y, v2.z], j * 6);
    });
    geo.attributes.position.needsUpdate = true;
  });
  if (!data) return null;
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

function SelectionMarker({ selected, gapRef }) {
  const ref = useRef();
  const v = useMemo(() => new THREE.Vector3(), []);
  const g = useRef(gapRef.current);
  useFrame(({ clock }) => {
    if (!selected) return;
    g.current += (gapRef.current - g.current) * 0.18;
    worldPos(selected.layer, selected.index, g.current, v);
    ref.current.position.copy(v);
    const s = 1 + 0.15 * Math.sin(clock.elapsedTime * 4);
    ref.current.scale.setScalar(s * SIZE[selected.layer] * 2.4);
  });
  if (!selected) return null;
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 16, 12]} />
      <meshBasicMaterial color="#ffb547" wireframe transparent opacity={0.7} toneMapped={false} />
    </mesh>
  );
}

// ---------- the strongest path, and a light that runs along it ----------
function SignalPath({ path, gapRef, show }) {
  const line = useMemo(() => new THREE.BufferGeometry(), []);
  const dots = useRef([]);
  const g = useRef(gapRef.current);
  const pts = useMemo(() => path.map(() => new THREE.Vector3()), [path]);
  useEffect(() => {
    line.setAttribute('position', new THREE.BufferAttribute(new Float32Array(path.length * 3), 3));
  }, [path, line]);
  useFrame(({ clock }) => {
    if (!show || path.length < 2) return;
    g.current += (gapRef.current - g.current) * 0.18;
    path.forEach((p, i) => worldPos(p.layer, p.index, g.current, pts[i]));
    if (!line.attributes.position || line.attributes.position.count !== pts.length)
      line.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.length * 3), 3));
    const arr = line.attributes.position.array;
    pts.forEach((p, i) => arr.set([p.x, p.y, p.z], i * 3));
    line.attributes.position.needsUpdate = true;
    const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
    dots.current.forEach((m, j) => {
      if (!m) return;
      const u = (clock.elapsedTime * 0.22 + j / 4) % 1;
      m.position.copy(curve.getPoint(u));
    });
  });
  if (!show) return null;
  return (
    <group>
      <line geometry={line}>
        <lineBasicMaterial color="#ffd27a" transparent opacity={0.85} toneMapped={false} />
      </line>
      {[0, 1, 2, 3].map((j) => (
        <mesh key={j} ref={(m) => (dots.current[j] = m)}>
          <sphereGeometry args={[0.09, 12, 8]} />
          <meshBasicMaterial color="#fff2c4" toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}

// ---------- camera: orbit, WASD fly, and follow-the-signal ----------
function CameraRig({ follow, path, gapRef, paused, controls, xrayKey, flyTo }) {
  const { camera } = useThree();
  const keys = useRef({});
  const u = useRef(0);
  const look = useMemo(() => new THREE.Vector3(), []);
  const pts = useMemo(() => path.map(() => new THREE.Vector3()), [path]);

  useEffect(() => {
    const down = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      keys.current[e.key.toLowerCase()] = true;
    };
    const up = (e) => (keys.current[e.key.toLowerCase()] = false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, []);
  const size = useThree((s) => s.size);
  // frame the whole stack on load and whenever the view is resized or X-ray changes
  const frame = () => {
    if (!controls.current) return;
    const g = gapRef.current, h = 4 * g + 2, w = 8;
    const t = Math.tan(((camera.fov / 2) * Math.PI) / 180);
    const d = 1.15 * Math.max(h / 2 / t, w / 2 / t / camera.aspect);
    const dir = new THREE.Vector3(0.55, 0.32, 0.77).normalize();
    controls.current.target.set(0, 2 * g, 0);
    camera.position.copy(controls.current.target).addScaledVector(dir, d);
    controls.current.update();
  };
  const framedFor = useRef('');
  // fly to a unit picked by search: ease the orbit target onto it and the camera to a close view
  const fly = useRef(null);
  useEffect(() => {
    if (!flyTo || !controls.current) return;
    const p = worldPos(flyTo.layer, flyTo.index, gapRef.current);
    const dir = camera.position.clone().sub(controls.current.target).normalize();
    fly.current = { t: 0, fromCam: camera.position.clone(), fromTgt: controls.current.target.clone(), toTgt: p, toCam: p.clone().addScaledVector(dir, 10) };
  }, [flyTo]); // eslint-disable-line
  useEffect(() => { u.current = 0; }, [follow]);
  useEffect(() => {
    if (!follow && controls.current) { controls.current.target.copy(look); controls.current.update(); }
  }, [follow, controls, look]);

  useFrame((_, dt) => {
    dt = Math.min(dt, 0.05);
    const fk = `${size.width}x${size.height}:${xrayKey}`;
    if (framedFor.current !== fk && controls.current && !follow) { framedFor.current = fk; frame(); }
    if (fly.current && controls.current && !follow) {
      const F = fly.current;
      F.t = Math.min(1, F.t + dt / 1.4);
      const e = F.t < 0.5 ? 2 * F.t * F.t : 1 - (-2 * F.t + 2) ** 2 / 2;
      camera.position.lerpVectors(F.fromCam, F.toCam, e);
      controls.current.target.lerpVectors(F.fromTgt, F.toTgt, e);
      controls.current.update();
      if (F.t >= 1) fly.current = null;
      look.copy(controls.current.target);
      return;
    }
    if (follow && path.length >= 2) {
      if (!paused.current) u.current = Math.min(1.25, u.current + dt / 9);
      path.forEach((p, i) => worldPos(p.layer, p.index, gapRef.current, pts[i]));
      const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
      const t = Math.min(1, u.current);
      const here = curve.getPoint(t);
      const ahead = curve.getPoint(Math.min(1, t + 0.06));
      // ride beside the current, pulling back near the top so the output ring fits in view
      const back = 1 + 1.4 * Math.max(0, (t - 0.7) / 0.3);
      const want = here.clone().add(new THREE.Vector3(2.6 * back, 1.4 * back, 3.4 * back));
      camera.position.lerp(want, 0.08);
      look.lerp(t >= 1 ? here : ahead, 0.1);
      camera.lookAt(look);
      if (u.current >= 1.25) u.current = 0; // hold at the output, then go again
      return;
    }
    const k = keys.current;
    const sp = (k.shift ? 9 : 4) * dt;
    const fwd = new THREE.Vector3();
    camera.getWorldDirection(fwd);
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    const mv = new THREE.Vector3();
    if (k.w) mv.add(fwd); if (k.s) mv.sub(fwd);
    if (k.d) mv.add(right); if (k.a) mv.sub(right);
    if (k.e) mv.y += 1; if (k.q) mv.y -= 1;
    if (mv.lengthSq() > 0 && controls.current) {
      mv.normalize().multiplyScalar(sp);
      camera.position.add(mv);
      controls.current.target.add(mv);
    }
    if (controls.current) look.copy(controls.current.target);
  });
  return null;
}

function Stars() {
  const geo = useMemo(() => {
    let s = 3;
    const r = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
    const p = new Float32Array(1500 * 3);
    for (let i = 0; i < 1500; i++) {
      const v = new THREE.Vector3(r(), r(), r()).normalize().multiplyScalar(40 + 30 * Math.abs(r()));
      p.set([v.x, v.y, v.z], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    return g;
  }, []);
  return (
    <points geometry={geo}>
      <pointsMaterial size={0.08} color="#5d6b8a" sizeAttenuation />
    </points>
  );
}

// a small image floating in space, always facing the camera
function FloatImage({ values, w = 8, h = 8, map = 'gray', position, size = 1.2, caption }) {
  const tex = useMemo(() => {
    const t = new THREE.DataTexture(new Uint8Array(w * h * 4), w, h, THREE.RGBAFormat);
    t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter; t.flipY = false;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [w, h]);
  useEffect(() => {
    const d = tex.image.data;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        // DataTexture row 0 is at the bottom, so flip rows to keep the digit upright
        const [r, g, b] = cmap[map](values[y * w + x]);
        const j = ((h - 1 - y) * w + x) * 4;
        d[j] = r; d[j + 1] = g; d[j + 2] = b; d[j + 3] = 255;
      }
    tex.needsUpdate = true;
  }, [values, tex, w, h, map]);
  return (
    <group position={position}>
      <mesh>
        <planeGeometry args={[size * (w / Math.max(w, h)) + 0.08, size * (h / Math.max(w, h)) + 0.08]} />
        <meshBasicMaterial color="#ffb547" toneMapped={false} />
      </mesh>
      <mesh position-z={0.01}>
        <planeGeometry args={[size * (w / Math.max(w, h)), size * (h / Math.max(w, h))]} />
        <meshBasicMaterial map={tex} toneMapped={false} />
      </mesh>
      {caption && (
        <Html position={[0, -size / 2 - 0.18, 0]} center style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded bg-black/60 px-1 text-[10px] text-slate-200">{caption}</div>
        </Html>
      )}
    </group>
  );
}

function Floating({ selected, gapRef, fv, patches, ring }) {
  const grp = useRef();
  const v = useMemo(() => new THREE.Vector3(), []);
  const g = useRef(gapRef.current);
  useFrame(() => {
    if (!selected) return;
    g.current += (gapRef.current - g.current) * 0.18;
    worldPos(selected.layer, selected.index, g.current, v);
    grp.current.position.copy(v);
  });
  if (!selected) return null;
  return (
    <group ref={grp}>
      {/* one camera-facing card, so the row never overlaps itself from an angle */}
      {/* a room's top examples, as cards in a ring around the unit */}
      {ring?.map((c, i) => {
        const t = (i / ring.length) * Math.PI * 2;
        return (
          <Billboard key={i} position={[1.9 * Math.cos(t), 0.55, 1.9 * Math.sin(t)]}>
            <FloatImage values={c.values} position={[0, 0, 0]} size={0.6} caption={c.caption} />
          </Billboard>
        );
      })}
      <Billboard>
        {fv && <FloatImage values={fv.values} map="gray" position={[patches ? 3.3 : 1.6, 0.5, 0]} size={1.3} caption={fv.caption} />}
        {patches?.map((p, i) => (
          <FloatImage key={i} values={p.values} w={p.w} h={p.h} position={[(i - 2) * 1.05, 1.5, 0]} size={0.85} caption={p.caption} />
        ))}
      </Billboard>
    </group>
  );
}

function Scene(props) {
  const { f, stamp, selected, onSelect, onHover, xray, follow, paused, path, rfMask, fv, patches, ring, showPath, active, mode, flyTo } = props;
  const gapRef = useRef(gapFor(xray));
  gapRef.current = gapFor(xray);
  const controls = useRef();
  const order = mode === 'features' ? ORDER_FEATURES : ORDER_NEURONS;
  // a path may pass through a room that is not drawn in this mode (the dense neurons behind an SAE feature)
  const shownPath = useMemo(() => path.filter((p) => order.includes(p.layer)), [path, order]);
  const sel = selected && order.includes(selected.layer) ? selected : null;
  return (
    <Canvas
      frameloop={active ? 'always' : 'demand'}
      camera={{ position: [9, 9, 13], fov: 42, near: 0.1, far: 300 }}
      dpr={[1, 2]}
      gl={{ antialias: true, powerPreference: 'high-performance' }}
    >
      <color attach="background" args={['#05070d']} />
      <fog attach="fog" args={['#05070d', 28, 70]} />
      <Stars />
      {order.map((k) => (
        <Layer
          key={k}
          k={k}
          target={f.acts[k]}
          live={f}
          stamp={stamp}
          rfMask={rfMask}
          pred={f.pred}
          onSelect={onSelect}
          onHover={onHover}
          gapRef={gapRef}
          paused={paused}
        />
      ))}
      <Edges selected={sel} f={f} gapRef={gapRef} mode={mode} />
      <SelectionMarker selected={sel} gapRef={gapRef} />
      <SignalPath path={shownPath} gapRef={gapRef} show={follow || showPath} />
      <Floating selected={sel} gapRef={gapRef} fv={fv} patches={patches} ring={ring} />
      <OrbitControls ref={controls} makeDefault enabled={!follow} enableDamping dampingFactor={0.1} />
      <CameraRig follow={follow} path={shownPath} gapRef={gapRef} paused={paused} controls={controls} xrayKey={xray} flyTo={flyTo} />
    </Canvas>
  );
}

export default React.memo(Scene);
