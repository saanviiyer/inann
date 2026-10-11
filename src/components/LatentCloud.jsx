// A 3D PCA cloud of items, each drawn as its class name (identity never rides on color alone).
// Shared by the Tour's latent chapter and the Lab. Points glide when the projection changes.
import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';

const R = 5;
import { PAL, useTheme, sceneHex } from '../lib/theme.js';
// numerals: neutral ink, dimmed when another class is highlighted (theme-aware, shared)
const NEUTRAL = PAL.neutral, DIM = PAL.dimPoint, HIGHLIGHT = PAL.highlight;

const texCache = new Map();
function labelTex(text) {
  if (!texCache.has(text)) {
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    if (text === null) {
      ctx.beginPath(); ctx.arc(32, 32, 14, 0, 7); ctx.fill();
    } else {
      const t = String(text).slice(0, 3);
      ctx.font = `bold ${t.length > 2 ? 30 : t.length > 1 ? 40 : 52}px Inter, Helvetica, Arial, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(t, 32, 36);
    }
    const tx = new THREE.CanvasTexture(c);
    tx.colorSpace = THREE.SRGBColorSpace;
    texCache.set(text, tx);
  }
  return texCache.get(text);
}

function Group({ idx, target, text, on, highlightActive, onPick }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(idx.length * 3), 3));
    return g;
  }, [idx]);
  const mat = useRef();
  useFrame(() => {
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
    mat.current.color.lerp(!highlightActive ? NEUTRAL : on ? HIGHLIGHT : DIM, 0.15);
  });
  return (
    <points geometry={geo} onClick={(e) => { e.stopPropagation(); onPick?.(idx[e.index]); }}>
      <pointsMaterial ref={mat} map={labelTex(text)} size={text === null ? 0.22 : 0.42} sizeAttenuation transparent alphaTest={0.3} depthWrite={false} />
    </points>
  );
}

function You({ p, trail }) {
  const ref = useRef();
  const line = useMemo(() => new THREE.BufferGeometry(), []);
  useEffect(() => { line.setFromPoints(trail.map((q) => new THREE.Vector3(...q))); }, [trail, line]);
  useFrame(({ clock }) => {
    ref.current.position.lerp(new THREE.Vector3(...p), 0.2);
    ref.current.scale.setScalar(1 + 0.15 * Math.sin(clock.elapsedTime * 4));
  });
  return (
    <group>
      <mesh ref={ref}>
        <sphereGeometry args={[0.18, 20, 14]} />
        <meshBasicMaterial color={sceneHex().accent} toneMapped={false} />
        <Html center position={[0, 0.42, 0]} zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
          <div className="whitespace-nowrap rounded bg-black/60 px-1 text-[10px] text-warm">your input</div>
        </Html>
      </mesh>
      {trail.length > 1 && (
        <line geometry={line}>
          <lineBasicMaterial color={sceneHex().accentSoft} transparent opacity={0.8} toneMapped={false} />
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
        <lineBasicMaterial color={sceneHex().axis} />
      </lineSegments>
      {[['PC1', [R + 0.4, 0, 0]], ['PC2', [0, R + 0.4, 0]], ['PC3', [0, 0, R + 0.4]]].map(([t, p]) => (
        <Html key={t} position={p} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
          <div className="text-[10px] text-dim">{t}</div>
        </Html>
      ))}
    </group>
  );
}

// P: result of pcaOf. labels: int per item or null; classes: names per label.
// me: the current input's vector in the same space (or null); trail: list of vectors.
export default function LatentCloud({ P, labels, classes, me, trail = [], highlight, onPick }) {
  const theme = useTheme();
  const target = useMemo(() => P.pts.map((q) => q.map((v) => (v / P.scale) * R)), [P]);
  const groups = useMemo(() => {
    if (!labels) return [{ key: 'all', text: null, idx: P.pts.map((_, i) => i) }];
    const K = classes?.length ?? Math.max(...labels) + 1;
    return Array.from({ length: K }, (_, c) => ({ key: c, text: K > 40 ? null : classes?.[c] ?? String(c), idx: labels.map((y, i) => (y === c ? i : -1)).filter((i) => i >= 0) })).filter((g) => g.idx.length);
  }, [labels, classes, P]);
  const meW = useMemo(() => (me ? P.project(me).map((v) => (v / P.scale) * R) : null), [P, me]);
  const trailW = useMemo(() => trail.map((v) => P.project(v).map((t) => (t / P.scale) * R)), [trail, P]);
  return (
    <Canvas camera={{ position: [8, 6, 10], fov: 45 }} dpr={[1, 2]}>
      <color attach="background" args={[sceneHex(theme).bg]} />
      <Axes />
      {groups.map((g) => (
        <Group key={g.key} idx={g.idx} target={target} text={g.text} on={highlight === g.key} highlightActive={highlight != null} onPick={onPick} />
      ))}
      {meW && <You p={meW} trail={trailW} />}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </Canvas>
  );
}
