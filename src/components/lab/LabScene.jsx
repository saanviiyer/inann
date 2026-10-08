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
  if (room.isOutput && L.size <= 32) return { type: 'pods', n: L.size, spacing: Math.min(0.9, 8 / L.size), width: Math.min(0.9, 8 / L.size) * L.size };
  if (L.kind === 'vector' && L.size <= 1024) {
    const r = Math.min(3.4, Math.max(1.2, 0.11 * Math.sqrt(L.size) * 2.2));
    return { type: 'spheres', n: L.size, r, size: Math.min(0.14, (r / Math.sqrt(L.size)) * 0.7), width: 2 * r };
  }
  // atlas
  let tiles, tH, tW, idx, inv;
  if (L.kind === 'spatial') {
    tiles = L.C; tH = L.H; tW = L.W;
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

function SphereRoom({ geo, data, scale, signed, y, onPick, onHover, x0 = 0, pods, classes, pred }) {
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
      {pods && classes && geo.n <= 20 &&
        Array.from({ length: geo.n }, (_, i) => (
          <Html key={i} position={[unitPosition(geo, i, 0)[0], 0.55, 0]} center style={{ pointerEvents: 'none' }}>
            <div className={`text-xs font-semibold ${i === pred ? 'text-warm' : 'text-slate-400'}`}>{String(classes[i] ?? i).slice(0, 6)}</div>
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

function Frame({ height, nonce }) {
  const { camera, controls, size } = useThree();
  useEffect(() => {
    if (!controls) return;
    const t = Math.tan(((camera.fov / 2) * Math.PI) / 180);
    const d = 1.1 * Math.max(height / 2 / t, ROOM_W / t / camera.aspect);
    controls.target.set(0, height / 2, 0);
    camera.position.copy(controls.target).addScaledVector(new THREE.Vector3(0.55, 0.32, 0.77).normalize(), d);
    controls.update();
  }, [height, nonce, controls, size.width, size.height]); // eslint-disable-line
  return null;
}

// rooms: [{name, op, dims, layout, isOutput, label}]; values: {name: Float32Array without batch}
// sae: {room, F, featNow: Float32Array(F), scale} or null
export default function LabScene({ rooms, values, scales, inputPixels, inputChannels, classes, pred, selected, onSelect, onHover, sae, frameNonce }) {
  const geos = useMemo(() => rooms.map((r, i) => roomGeometry(r, i === 0)), [rooms]);
  const ys = rooms.map((_, i) => i * GAP);
  const height = (rooms.length - 1) * GAP;
  const saeIdx = sae ? rooms.findIndex((r) => r.name === sae.room) : -1;
  const saeGeo = useMemo(() => (sae ? { type: 'spheres', n: sae.F, r: Math.min(3.6, Math.max(1.6, 0.1 * Math.sqrt(sae.F) * 2)), size: Math.min(0.12, 2.2 / Math.sqrt(sae.F) / 2.5) } : null), [sae?.F]); // eslint-disable-line
  const saeX = saeIdx >= 0 ? (geos[saeIdx].width ?? ROOM_W) / 2 + 2.2 + saeGeo.r : 0;

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
    <Canvas camera={{ position: [9, 9, 13], fov: 42, near: 0.1, far: 500 }} dpr={[1, 2]}>
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
            {g.type === 'pods' && <SphereRoom {...common} onPick={pick} onHover={hov} pods classes={classes} pred={pred} />}
            <Html position={[-(Math.max(g.width ?? 0, 2 * (g.r ?? 0)) / 2) - 1.2, ys[i], 0]} center style={{ pointerEvents: 'none' }}>
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
          <Html position={[saeX, ys[saeIdx] + 0.9, -saeGeo.r - 0.4]} center style={{ pointerEvents: 'none' }}>
            <div className="whitespace-nowrap rounded bg-black/50 px-1.5 text-[10px] text-warm">SAE · {sae.F} features · top {sae.k}</div>
          </Html>
        </group>
      )}
      {marker && <Marker {...marker} />}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <Frame height={height} nonce={frameNonce} />
    </Canvas>
  );
}
