// Light (default) and dark themes. CSS colors switch through [data-theme] tokens in index.css;
// three.js colors live in one shared palette that is updated in place, so every scene follows.
import { useSyncExternalStore } from 'react';
import * as THREE from 'three';

const KEY = 'inann-theme';
const SCENE = {
  dark: {
    bg: '#05070d', floor: '#0d1424', stars: '#5d6b8a', base: '#121a2b', glow: '#8af0ff', hot: '#ffffff',
    accent: '#ffb547', accentSoft: '#ffd27a', spark: '#fff2c4', axis: '#1c2436',
    neutral: '#c3c2b7', dimPoint: '#4a4f5c', highlight: '#3987e5', ink: '#e6ebf5', glowText: '#7ee7ff', cool: '#5b8cff',
  },
  light: {
    bg: '#f4eee2', floor: '#e6dccb', stars: '#d8cdb8', base: '#e3dac8', glow: '#1f86b0', hot: '#0b3550',
    accent: '#c56a12', accentSoft: '#d98a2b', spark: '#8a4b0f', axis: '#d3c8b3',
    neutral: '#4b4234', dimPoint: '#d2c8b5', highlight: '#2a78d6', ink: '#2b241b', glowText: '#0f6e8c', cool: '#2f5fbf',
  },
};

function initial() {
  try { const t = localStorage.getItem(KEY); if (t === 'light' || t === 'dark') return t; } catch { /* storage unavailable */ }
  return 'light';
}
let current = initial();
const subs = new Set();

// shared THREE.Color objects, mutated on theme change
export const PAL = Object.fromEntries(Object.keys(SCENE.dark).map((k) => [k, new THREE.Color()]));
function apply(t) {
  if (typeof document !== 'undefined') document.documentElement.dataset.theme = t;
  for (const [k, v] of Object.entries(SCENE[t])) PAL[k].set(v);
}
apply(current);

export function setTheme(t) {
  current = t;
  apply(t);
  try { localStorage.setItem(KEY, t); } catch { /* ignore */ }
  subs.forEach((f) => f());
}
export const getTheme = () => current;
export function useTheme() {
  return useSyncExternalStore((cb) => { subs.add(cb); return () => subs.delete(cb); }, () => current);
}
// hex strings for JSX props and 2D canvases
export const sceneHex = (t = current) => SCENE[t];
// glowing lines: additive light on dark, plain lines on light (additive vanishes on beige)
export const lineBlending = (t = current) => (t === 'light' ? THREE.NormalBlending : THREE.AdditiveBlending);
