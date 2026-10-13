// "A thermometer on every layer" (Alain & Bengio 2016): one floor per layer, stacked from the
// input up, each with a glass tube filled to its probe's held-out accuracy. A cool tick marks the
// same probe on a network with shuffled weights, a gray stub the shuffled-label control, and a
// dashed ring the majority-class floor. Shared by the Tour and the Lab.
import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { PAL, useTheme, sceneHex } from '../lib/theme.js';

const GAP = 1.15, TUBE = 0.85;

function Fill({ value, color, radius, height }) {
  const ref = useRef();
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const h = Math.max(0.001, value) * height;
    m.scale.y += (h - m.scale.y) * 0.12;
    m.position.y = m.scale.y / 2;
  });
  return (
    <mesh ref={ref} scale={[1, 0.001, 1]}>
      <cylinderGeometry args={[radius, radius, 1, 24]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function Tick({ y, color, width }) {
  return (
    <mesh position={[0, y, 0]}>
      <boxGeometry args={[width, width * 0.055, width * 0.055]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

// One thermometer: a glass tube filled to `test`, a gray stub for the shuffled-label control,
// a cool tick for the shuffled-weight network and a short tick for the majority-class floor.
export function Thermometer({ test, ctrlTest, rand, chance, height = TUBE, r = 0.12, busy }) {
  const t = sceneHex();
  return (
    <group>
      <mesh position={[0, height / 2, 0]}>
        <cylinderGeometry args={[r, r, height, 24, 1, true]} />
        <meshBasicMaterial color={PAL.axis} transparent opacity={0.55} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <mesh position={[0, -r * 0.2, 0]}>
        <sphereGeometry args={[r * 1.35, 24, 12]} />
        <meshBasicMaterial color={test == null ? PAL.axis : PAL.glow} />
      </mesh>
      {test != null && <Fill value={test} color={PAL.glow} radius={r * 0.75} height={height} />}
      {ctrlTest != null && (
        <group position={[r * 2.4, 0, 0]}>
          <Fill value={ctrlTest} color={PAL.dimPoint} radius={r * 0.42} height={height} />
        </group>
      )}
      {rand != null && <Tick y={rand * height} color={PAL.cool} width={r * 2.8} />}
      {chance != null && <Tick y={chance * height} color={PAL.neutral} width={r * 1.8} />}
      <Html position={[r * 3.5, test != null ? test * height : r, 0]} zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
        <div className="whitespace-nowrap text-[12px] font-semibold" style={{ color: t.glowText }}>
          {test == null ? (busy ? '...' : '') : `${(100 * test).toFixed(1)}%`}
        </div>
      </Html>
    </group>
  );
}

function Floor({ row, i, n, on, onSelect, chance }) {
  const y = (i - (n - 1) / 2) * GAP;
  const t = sceneHex();
  return (
    <group position={[0, y, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-0.9, 0, 0]} onClick={(e) => { e.stopPropagation(); onSelect?.(row.key); }}
        onPointerOver={() => (document.body.style.cursor = 'pointer')} onPointerOut={() => (document.body.style.cursor = '')}>
        <planeGeometry args={[2.6, 1.3]} />
        <meshBasicMaterial color={on ? PAL.accentSoft : PAL.floor} transparent opacity={on ? 0.35 : 0.8} side={THREE.DoubleSide} />
      </mesh>
      <Html position={[-2.15, 0.02, 0]} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
        <div className="whitespace-nowrap text-[11px]" style={{ color: on ? t.accent : t.neutral }}>{row.label}{row.D ? <span style={{ opacity: 0.6 }}> · {row.D}</span> : null}</div>
      </Html>
      <group position={[0.75, 0, 0]}>
        <Thermometer test={row.test} ctrlTest={row.ctrlTest} rand={row.rand} chance={chance} busy={row.busy} />
      </group>
    </group>
  );
}

// shift moves the view target right, so the tower sits left of an overlaid panel
export default function ProbeTower({ rows, chance, selected, onSelect, shift = 0 }) {
  const theme = useTheme();
  const n = rows.length;
  const h = n * GAP;
  return (
    <Canvas camera={{ position: [1.6 + shift, 1.4, Math.max(6, h * 1.75)], fov: 42 }} dpr={[1, 2]}>
      <color attach="background" args={[sceneHex(theme).bg]} />
      {rows.map((r, i) => <Floor key={r.key} row={r} i={i} n={n} on={selected === r.key} onSelect={onSelect} chance={chance} />)}
      <OrbitControls enablePan={false} minDistance={2.5} maxDistance={18} target={[-0.5 + shift, 0.2, 0]} />
    </Canvas>
  );
}

// The legend shared by both pages, in DOM so it reads at any zoom.
export function ProbeLegend({ hasRand }) {
  const t = sceneHex();
  const sw = (c, w = 10, h = 10, r = 2) => <span className="inline-block align-middle" style={{ width: w, height: h, borderRadius: r, background: c }} />;
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-dim">
      <span>{sw(t.glow)} probe, held-out items</span>
      <span>{sw(t.dimPoint, 6)} same probe, shuffled labels</span>
      {hasRand && <span>{sw(t.cool, 14, 3, 0)} same probe, shuffled-weight network</span>}
      <span>{sw(t.neutral, 10, 3, 0)} majority-class floor</span>
    </div>
  );
}
