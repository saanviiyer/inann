// The Lab's 3D space for any uploaded model. Each chosen tensor is a room stacked above the last.
// Big tensors are drawn as one texture atlas per room (one tile per channel), so a 64x56x56 map
// costs one draw call; small vectors are instanced spheres; a small output is a row of pods.
// A trained SAE docks beside its layer as a disk of feature spheres joined by a beam.
import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { cmap } from '../ui.jsx';

const GAP = 3.2;
const ROOM_W = 6.5;

// how a room is drawn
export function roomGeometry(room, isFirst) {
  const L = room.layout;
  if (room.isTokens && L.size <= 64) return { type: 'pods', n: L.size, spacing: Math.min(0.9, 9 / L.size), width: Math.min(0.9, 9 / L.size) * L.size };
  if (room.isOutput && L.size <= 32) return { type: 'pods', n: L.size, spacing: Math.min(0.9, 8 / L.size), width: Math.min(0.9, 8 / L.size) * L.size };
  if (L.kind === 'vector' && L.size <= 1024) {
    const r = Math.min(3.4, Math.max(1.2, 0.11 * Math.sqrt(L.size) * 2.2));
    return { type: 'spheres', n: L.size, r, size: Math.min(0.14, (r / Math.sqrt(L.size)) * 0.7), width: 2 * r };
  }
  // atlas
  let tiles, tH, tW, idx, inv;
  if (L.kind === 'spatial') {
    // the input picture is one colour tile, not one tile per colour channel
    tiles = room.isInputRoom ? 1 : L.C; tH = L.H; tW = L.W;
    idx = (t, y, x) => t * L.H * L.W + y * L.W + x;
    inv = (i) => { const p = i % (L.H * L.W); return [Math.floor(i / (L.H * L.W)), Math.floor(p / L.W), p % L.W]; };
  } else if (L.kind === 'seq') {
    tiles = 1; tH = L.H; tW = L.C;
    idx = (t, y, x) => y * L.C + x;
    inv = (i) => [0, Math.floor(i / L.C), i % L.C];
  } else {
    const w = Math.ceil(Math.sqrt(L.size)); tiles = 1; tH = Math.ceil(L.size / w); tW = w;
    idx = (t, y, x) => (y * w + x < L.size ? y * w + x : -1);
    inv = (i) => [0, Math.floor(i / w), i % w];
  }
  const cols = Math.ceil(Math.sqrt(tiles * (tH / tW)));
  const rows = Math.ceil(tiles / cols);
  const pad = tiles > 1 ? 1 : 0;
  const texW = cols * (tW + pad), texH = rows * (tH + pad);
  const px = Math.min(0.32, ROOM_W / Math.max(texW, texH));
  return { type: 'atlas', tiles, tH, tW, cols, rows, pad, texW, texH, px, idx, inv, width: texW * px, depth: texH * px, isInput: isFirst };
}

// world position of unit `i` in a room (for markers)
export function unitPosition(geo, i, y) {
  if (geo.type === 'pods') return [(i - (geo.n - 1) / 2) * geo.spacing, y, 0];
  if (geo.type === 'spheres') return spherePos(geo, i, y);
  const [t, yy, x] = geo.inv(i);
  const ax = (t % geo.cols) * (geo.tW + geo.pad) + x, ay = Math.floor(t / geo.cols) * (geo.tH + geo.pad) + yy;
  return [(ax + 0.5) * geo.px - geo.width / 2, y + 0.05, (ay + 0.5) * geo.px - geo.depth / 2];
}
function spherePos(geo, i, y) {
  const r = geo.r * Math.sqrt((i + 0.5) / geo.n), t = i * 2.39996;
  return [r * Math.cos(t), y + 0.12 * Math.sin(i * 1.3), r * Math.sin(t)];
}

const BASE = new THREE.Color('#121a2b'), GLOW = new THREE.Color('#8af0ff'), HOT = new THREE.Color('#ffffff'), WIN = new THREE.Color('#ffd27a');
function colorFor(v, scale, signed, out = new THREE.Color()) {
  if (signed) {
    const [r, g, b] = cmap.diverging(v / scale);
    return out.setRGB(r / 255, g / 255, b / 255);
  }
  const t = Math.min(1, Math.max(0, v / scale));
  out.copy(BASE).lerp(GLOW, Math.min(1, t * 1.2));
  if (t > 0.85) out.lerp(HOT, (t - 0.85) * 3);
  return out;
}

function AtlasRoom({ geo, data, scale, signed, y, onPick, onHover, inputPixels, inputChannels }) {
  const tex = useMemo(() => {
    const t = new THREE.DataTexture(new Uint8Array(geo.texW * geo.texH * 4), geo.texW, geo.texH, THREE.RGBAFormat);
    t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter; t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }, [geo]);
  useEffect(() => {
    const d = tex.image.data;
    d.fill(8);
    const col = new THREE.Color();
    for (let t = 0; t < geo.tiles; t++) {
      const ox = (t % geo.cols) * (geo.tW + geo.pad), oy = Math.floor(t / geo.cols) * (geo.tH + geo.pad);
      for (let yy = 0; yy < geo.tH; yy++)
        for (let x = 0; x < geo.tW; x++) {
          const i = geo.idx(t, yy, x);
          if (i < 0) continue;
          let r, g, b;
          if (inputPixels) {
            // the input room shows the picture itself (RGB when it has 3 channels)
            const p = yy * geo.tW + x, HW = geo.tH * geo.tW;
            const v = (c) => Math.min(1, Math.max(0, inputPixels[c * HW + p] ?? 0));
            [r, g, b] = inputChannels === 3 ? [v(0), v(1), v(2)] : [v(0), v(0), v(0)];
          } else {
            colorFor(data ? data[i] : 0, scale, signed, col);
            [r, g, b] = [col.r, col.g, col.b];
          }
          const j = ((geo.texH - 1 - (oy + yy)) * geo.texW + ox + x) * 4;
          d[j] = r * 255; d[j + 1] = g * 255; d[j + 2] = b * 255; d[j + 3] = 255;
        }
    }
    tex.needsUpdate = true;
  }, [tex, geo, data, scale, signed, inputPixels, inputChannels]);
  const pick = (e, fn) => {
    e.stopPropagation();
    const ax = Math.floor(e.uv.x * geo.texW), ay = geo.texH - 1 - Math.floor(e.uv.y * geo.texH);
    const col = Math.floor(ax / (geo.tW + geo.pad)), row = Math.floor(ay / (geo.tH + geo.pad));
    const x = ax % (geo.tW + geo.pad), yy = ay % (geo.tH + geo.pad);
    if (x >= geo.tW || yy >= geo.tH) return fn(null);
    const t = row * geo.cols + col;
    if (t >= geo.tiles) return fn(null);
    const i = geo.idx(t, yy, x);
    fn(i >= 0 ? i : null);
  };
  return (
    <mesh position={[0, y, 0]} rotation-x={-Math.PI / 2} onClick={(e) => pick(e, (i) => i != null && onPick(i))} onPointerMove={(e) => pick(e, onHover)} onPointerOut={() => onHover(null)}>
      <planeGeometry args={[geo.width, geo.depth]} />
      <meshBasicMaterial map={tex} toneMapped={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

function SphereRoom({ geo, data, scale, signed, y, onPick, onHover, x0 = 0, pods, classes, pred, podLabels }) {
  const mesh = useRef();
  const g = useMemo(() => new THREE.SphereGeometry(pods ? 0.28 : geo.size, 16, 12), [geo, pods]);
  useEffect(() => {
    const o = new THREE.Object3D();
    for (let i = 0; i < geo.n; i++) {
      const p = pods ? unitPosition(geo, i, 0) : spherePos(geo, i, 0);
      o.position.set(p[0] + x0, p[1], p[2]); o.updateMatrix();
      mesh.current.setMatrixAt(i, o.matrix);
    }
    mesh.current.instanceMatrix.needsUpdate = true;
  }, [geo, pods, x0]);
  useEffect(() => {
    const c = new THREE.Color();
    for (let i = 0; i < geo.n; i++) {
      colorFor(data ? data[i] : 0, scale, signed, c);
      if (pods && i === pred) c.lerp(WIN, 0.55);
      mesh.current.setColorAt(i, c);
    }
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true;
  }, [geo, data, scale, signed, pods, pred]);
  return (
    <group position={[0, y, 0]}>
      <instancedMesh
        ref={mesh}
        args={[g, undefined, geo.n]}
        onClick={(e) => { e.stopPropagation(); onPick(e.instanceId); }}
        onPointerMove={(e) => { e.stopPropagation(); onHover(e.instanceId); }}
        onPointerOut={() => onHover(null)}
      >
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
      {pods && (podLabels || classes) && geo.n <= 64 &&
        Array.from({ length: geo.n }, (_, i) => (
          <Html key={i} position={[unitPosition(geo, i, 0)[0], 0.55, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`font-semibold ${podLabels ? 'font-mono text-sm text-glow' : i === pred ? 'text-xs text-warm' : 'text-xs text-slate-400'}`}>
              {String((podLabels ?? classes)[i] ?? i).slice(0, 6)}
            </div>
          </Html>
        ))}
    </group>
  );
}

function Marker({ pos, size = 0.25 }) {
  const ref = useRef();
  useFrame(({ clock }) => {
    if (!ref.current) return;
    ref.current.position.set(...pos);
    ref.current.scale.setScalar(size * (1 + 0.15 * Math.sin(clock.elapsedTime * 4)));
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 16, 12]} />
      <meshBasicMaterial color="#ffb547" wireframe transparent opacity={0.8} toneMapped={false} />
    </mesh>
  );
}

// Frame the whole tower, or (focus) one room. Tall towers open on their first few rooms.
// The traced answer: a warm line through the chosen unit of each room, with a light running along it.
function TracePath({ points }) {
  const line = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setFromPoints(points.map((p) => new THREE.Vector3(...p)));
    return g;
  }, [points]);
  const dot = useRef();
  const curve = useMemo(() => new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)), false, 'centripetal'), [points]);
  useFrame(({ clock }) => { if (dot.current) dot.current.position.copy(curve.getPoint((clock.elapsedTime * 0.25) % 1)); });
  return (
    <group>
      <line geometry={line}>
        <lineBasicMaterial color="#ffd27a" toneMapped={false} />
      </line>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.09, 12, 8]} />
          <meshBasicMaterial color="#ffb547" toneMapped={false} />
        </mesh>
      ))}
      <mesh ref={dot}>
        <sphereGeometry args={[0.12, 12, 8]} />
        <meshBasicMaterial color="#fff2c4" toneMapped={false} />
      </mesh>
    </group>
  );
}

// Attention arcs: for a selected cell of a [heads, query, key] attention map, arcs from every key
// token to the query token over the input row, thickness and brightness set by the weight.
function AttentionArcs({ podPos, query, weights }) {
  const geo = useMemo(() => {
    const pos = [], col = [];
    const m = Math.max(...weights, 1e-9);
    const q = podPos(query);
    weights.forEach((w, k) => {
      if (w < 0.02 * m) return;
      const a = podPos(k);
      const h = 0.6 + 0.12 * Math.abs(k - query);
      const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(a[0], a[1] + 0.3, a[2]), new THREE.Vector3((a[0] + q[0]) / 2, a[1] + 0.3 + h, a[2]), new THREE.Vector3(q[0], q[1] + 0.3, q[2]));
      const pts = curve.getPoints(16);
      const t = w / m;
      for (let i = 0; i < pts.length - 1; i++) {
        pos.push(pts[i].x, pts[i].y, pts[i].z, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z);
        col.push(t, 0.71 * t, 0.28 * t, t, 0.71 * t, 0.28 * t);
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return g;
  }, [podPos, query, weights]);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

// lines from an SAE feature to the channels its decoder writes most; orange adds, blue subtracts
function Beams({ from, targets }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = [], col = [];
    const m = Math.max(...targets.map((t) => Math.abs(t.w)), 1e-9);
    targets.forEach(({ pos: p, w }) => {
      pos.push(...from, ...p);
      const a = 0.3 + 0.7 * Math.abs(w) / m;
      const [r, gg, b] = w >= 0 ? [1, 0.71, 0.28] : [0.36, 0.55, 1];
      col.push(r * a * 0.4, gg * a * 0.4, b * a * 0.4, r * a, gg * a, b * a);
    });
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    return g;
  }, [from, targets]);
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial vertexColors transparent blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
    </lineSegments>
  );
}

function Frame({ height, nonce, cx = 0, width = ROOM_W, focus }) {
  const { camera, controls, size } = useThree();
  const fly = useRef(null);
  useEffect(() => {
    if (!controls) return;
    const t = Math.tan(((camera.fov / 2) * Math.PI) / 180);
    let ty, span, w = width, x = cx;
    if (focus) { ty = focus.y; span = GAP * 1.6; w = Math.max(3, focus.width); x = focus.x ?? 0; }
    else if (height > GAP * 6) { ty = GAP * 1.5; span = GAP * 4; }
    else { ty = height / 2; span = height; }
    const d = 1.1 * Math.max(span / 2 / t, w / t / camera.aspect, 4);
    const toTgt = new THREE.Vector3(x, ty, 0);
    const toCam = toTgt.clone().addScaledVector(new THREE.Vector3(0.55, 0.32, 0.77).normalize(), d);
    fly.current = { t: 0, fromTgt: controls.target.clone(), toTgt, fromCam: camera.position.clone(), toCam, instant: !focus };
  }, [height, nonce, controls, size.width, size.height, focus]); // eslint-disable-line
  useFrame((_, dt) => {
    const F = fly.current;
    if (!F || !controls) return;
    F.t = F.instant ? 1 : Math.min(1, F.t + dt / 1.1);
    const e = F.t < 0.5 ? 2 * F.t * F.t : 1 - (-2 * F.t + 2) ** 2 / 2;
    controls.target.lerpVectors(F.fromTgt, F.toTgt, e);
    camera.position.lerpVectors(F.fromCam, F.toCam, e);
    controls.update();
    if (F.t >= 1) fly.current = null;
  });
  return null;
}

// rooms: [{name, op, dims, layout, isOutput, label}]; values: {name: Float32Array without batch}
// sae: {room, F, featNow: Float32Array(F), scale} or null
export default function LabScene({ rooms, values, scales, inputPixels, inputChannels, classes, pred, selected, onSelect, onHover, sae, frameNonce, focusRoom, edges, trace }) {
  const geos = useMemo(() => rooms.map((r, i) => roomGeometry(r, i === 0)), [rooms]);
  const ys = rooms.map((_, i) => i * GAP);
  const height = (rooms.length - 1) * GAP;
  const saeIdx = sae ? rooms.findIndex((r) => r.name === sae.room) : -1;
  const saeGeo = useMemo(() => (sae ? { type: 'spheres', n: sae.F, r: Math.min(3.6, Math.max(1.6, 0.1 * Math.sqrt(sae.F) * 2)), size: Math.min(0.12, 2.2 / Math.sqrt(sae.F) / 2.5) } : null), [sae?.F]); // eslint-disable-line
  const saeX = saeIdx >= 0 ? (geos[saeIdx].width ?? ROOM_W) / 2 + 2.2 + saeGeo.r : 0;

  // rebuilt only when a new fly-to is requested, so re-renders (hover, new values) do not restart it
  const focus = useMemo(
    () => (focusRoom != null && rooms[focusRoom.i] ? { y: ys[focusRoom.i], width: geos[focusRoom.i].width ?? ROOM_W, nonce: focusRoom.nonce, x: focusRoom.sae && saeIdx >= 0 ? saeX : 0 } : null),
    [focusRoom?.nonce], // eslint-disable-line
  );
  let marker = null;
  if (selected) {
    if (selected.sae && saeIdx >= 0) {
      const p = spherePos(saeGeo, selected.index, ys[saeIdx]);
      marker = { pos: [p[0] + saeX, p[1], p[2]], size: 0.22 };
    } else {
      const k = rooms.findIndex((r) => r.name === selected.room);
      if (k >= 0) marker = { pos: unitPosition(geos[k], selected.index, ys[k]), size: geos[k].type === 'atlas' ? Math.max(0.12, geos[k].px * 1.2) : 0.3 };
    }
  }

  return (
    <Canvas camera={{ position: [9, 9, 13], fov: 42, near: 0.1, far: 500 }} dpr={[1, 2]} gl={{ preserveDrawingBuffer: true }}>
      <color attach="background" args={['#05070d']} />
      {rooms.map((r, i) => {
        const g = geos[i];
        const common = { geo: g, data: values[r.name], scale: scales[r.name] ?? 1, signed: !!scales[`${r.name}:signed`], y: ys[i] };
        const pick = (idx) => onSelect({ room: r.name, index: idx });
        const hov = (idx) => onHover(idx == null ? null : { room: r.name, index: idx });
        return (
          <group key={r.name}>
            {g.type === 'atlas' && <AtlasRoom {...common} onPick={pick} onHover={hov} inputPixels={i === 0 ? inputPixels : null} inputChannels={inputChannels} />}
            {g.type === 'spheres' && <SphereRoom {...common} onPick={pick} onHover={hov} />}
            {g.type === 'pods' && <SphereRoom {...common} onPick={pick} onHover={hov} pods classes={classes} pred={r.isTokens ? -1 : pred} podLabels={r.podLabels} />}
            <Html position={[-(Math.max(g.width ?? 0, 2 * (g.r ?? 0)) / 2) - 1.2, ys[i], 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
              <div className="whitespace-nowrap text-right text-[10px] leading-tight text-dim">
                <div className="uppercase tracking-[0.14em] text-slate-300">{r.label}</div>
                <div>{r.dims.join('×')}</div>
              </div>
            </Html>
          </group>
        );
      })}
      {sae && saeIdx >= 0 && (
        <group>
          {/* the beam joining a layer to its dictionary */}
          <mesh position={[(geos[saeIdx].width ?? ROOM_W) / 2 + 1.1, ys[saeIdx], 0]} rotation-z={Math.PI / 2}>
            <cylinderGeometry args={[0.04, 0.04, 2.2, 8]} />
            <meshBasicMaterial color="#ffb547" transparent opacity={0.6} toneMapped={false} />
          </mesh>
          <SphereRoom
            geo={saeGeo} data={sae.featNow} scale={sae.scale || 1} signed={false} y={ys[saeIdx]} x0={saeX}
            onPick={(idx) => onSelect({ room: sae.room, index: idx, sae: true })}
            onHover={(idx) => onHover(idx == null ? null : { room: sae.room, index: idx, sae: true })}
          />
          <Html position={[saeX, ys[saeIdx] + 0.9, -saeGeo.r - 0.4]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="whitespace-nowrap rounded bg-black/50 px-1.5 text-[10px] text-warm">SAE · {sae.F} features · top {sae.k}</div>
          </Html>
        </group>
      )}
      {marker && <Marker {...marker} />}
      {trace && trace.length > 1 && (
        <TracePath points={trace.map((t) => { const k = rooms.findIndex((r) => r.name === t.room); return k < 0 ? null : unitPosition(geos[k], rooms[k].isInputRoom && rooms[k].layout.kind === 'spatial' ? t.index % rooms[k].layout.P : t.index, ys[k]); }).filter(Boolean)} />
      )}
      {marker && edges && (() => {
        // incoming edges: from the feeding units (or channel tiles) of the room below to the selection
        const k = rooms.findIndex((r) => r.name === edges.fromRoom);
        if (k < 0) return null;
        const g = geos[k], L = rooms[k].layout;
        const pos = (e) => {
          if (edges.perChannel && L.kind === 'spatial') {
            if (g.type !== 'atlas') return unitPosition(g, e.k, ys[k]);
            // centre of channel tile e.k (the input picture is a single tile)
            const t = rooms[k].isInputRoom ? 0 : e.k;
            const ax = (t % g.cols) * (g.tW + g.pad) + g.tW / 2, ay = Math.floor(t / g.cols) * (g.tH + g.pad) + g.tH / 2;
            return [ax * g.px - g.width / 2, ys[k] + 0.05, ay * g.px - g.depth / 2];
          }
          return unitPosition(g, Math.min(e.k, L.size - 1), ys[k]);
        };
        return <Beams from={marker.pos} targets={edges.edges.map((e) => ({ pos: pos(e), w: e.v }))} />;
      })()}
      {(() => {
        // attention arcs when a cell of a [heads, T, T] map is selected and the input is a token row
        if (!selected || selected.sae || !rooms[0]?.isTokens) return null;
        const k = rooms.findIndex((r) => r.name === selected.room);
        const r = rooms[k];
        const T = rooms[0].layout.size;
        if (!r || r.layout.kind !== 'spatial' || r.layout.H !== T || r.layout.W !== T || !values[r.name]) return null;
        const head = Math.floor(selected.index / (T * T)), q = Math.floor((selected.index % (T * T)) / T);
        const row = Array.from(values[r.name].subarray(head * T * T + q * T, head * T * T + (q + 1) * T));
        return <AttentionArcs podPos={(i) => unitPosition(geos[0], i, ys[0])} query={q} weights={row} />;
      })()}
      {selected?.sae && sae?.links && saeIdx >= 0 && (
        <Beams
          from={(() => { const p = spherePos(saeGeo, selected.index, ys[saeIdx]); return [p[0] + saeX, p[1], p[2]]; })()}
          targets={sae.links(selected.index).map(({ c, w }) => ({ pos: unitPosition(geos[saeIdx], rooms[saeIdx].layout.idx(c, Math.floor(rooms[saeIdx].layout.P / 2)), ys[saeIdx]), w }))}
        />
      )}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <Frame
        focus={focus}
        height={height} nonce={frameNonce} cx={saeIdx >= 0 ? (saeX + saeGeo.r) / 2 : 0} width={saeIdx >= 0 ? (ROOM_W + saeX + saeGeo.r) / 1.6 : ROOM_W} />
    </Canvas>
  );
}
