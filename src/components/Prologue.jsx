import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Term, Cite, Btn, Kbd, AboutBlurb } from './ui.jsx';
import { sceneHex } from '../lib/theme.js';

// A seeded, linearly separable 2D dataset.
function makeData(seed = 7) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const pts = [];
  while (pts.length < 40) {
    const x = rnd(), y = rnd();
    const m = 0.8 * x - 0.6 * y + 0.15;
    if (Math.abs(m) < 0.2) continue;
    pts.push({ x, y, label: m > 0 ? 1 : -1 });
  }
  return pts;
}

const STEP_MS = 330; // converges, then confirms with a clean pass, in about 30 seconds
const LR = 0.5;

export default function Prologue({ paused, onNext }) {
  const data = useMemo(() => makeData(), []);
  const canvas = useRef(null);
  const st = useRef(null);
  const [info, setInfo] = useState({ step: 0, updates: 0, wrong: 0, done: false });
  const [run, setRun] = useState(0);

  useEffect(() => {
    st.current = { w: [-0.35, 0.55], b: 0, i: 0, step: 0, updates: 0, last: null, cleanRun: 0, t: 0, acc: 0 };
  }, [run]);

  useEffect(() => {
    let raf, prev = performance.now();
    const cv = canvas.current, ctx = cv.getContext('2d');
    const W = cv.width, H = cv.height, R = 1.25;
    const X = (x) => ((x + R) / (2 * R)) * W;
    const Y = (y) => H - ((y + R) / (2 * R)) * H;
    const countWrong = (s) => data.filter((p) => Math.sign(s.w[0] * p.x + s.w[1] * p.y + s.b) !== p.label).length;

    const tick = (now) => {
      const s = st.current;
      const dt = now - prev; prev = now;
      if (!paused.current) {
        s.acc += dt;
        while (s.acc > STEP_MS) {
          s.acc -= STEP_MS;
          const p = data[s.i % data.length];
          const wrong = Math.sign(s.w[0] * p.x + s.w[1] * p.y + s.b) !== p.label;
          s.last = { p, wrong, at: now };
          if (wrong) {
            s.w = [s.w[0] + LR * p.label * p.x, s.w[1] + LR * p.label * p.y];
            s.b += LR * p.label;
            s.updates++; s.cleanRun = 0;
          } else s.cleanRun++;
          s.i++; s.step++;
          setInfo({ step: s.step, updates: s.updates, wrong: countWrong(s), done: s.cleanRun >= data.length });
        }
      }
      // draw
      ctx.clearRect(0, 0, W, H);
      const img = ctx.createImageData(W / 4, H / 4);
      for (let j = 0; j < H / 4; j++)
        for (let i = 0; i < W / 4; i++) {
          const x = ((i * 4) / W) * 2 * R - R, y = R - ((j * 4) / H) * 2 * R;
          const v = s.w[0] * x + s.w[1] * y + s.b;
          const k = (j * (W / 4) + i) * 4;
          if (v > 0) { img.data[k] = 255; img.data[k + 1] = 181; img.data[k + 2] = 71; }
          else { img.data[k] = 91; img.data[k + 1] = 140; img.data[k + 2] = 255; }
          img.data[k + 3] = 26;
        }
      const off = document.createElement('canvas');
      off.width = W / 4; off.height = H / 4;
      off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, W, H);

      // boundary: w0 x + w1 y + b = 0
      ctx.strokeStyle = sceneHex().ink; ctx.lineWidth = 2;
      ctx.beginPath();
      if (Math.abs(s.w[1]) > 1e-6) {
        ctx.moveTo(X(-R), Y((-s.b - s.w[0] * -R) / s.w[1]));
        ctx.lineTo(X(R), Y((-s.b - s.w[0] * R) / s.w[1]));
      } else {
        ctx.moveTo(X(-s.b / s.w[0]), 0); ctx.lineTo(X(-s.b / s.w[0]), H);
      }
      ctx.stroke();

      for (const p of data) {
        ctx.fillStyle = p.label > 0 ? sceneHex().accent : sceneHex().cool;
        ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 5, 0, 7); ctx.fill();
      }
      if (s.last) {
        const age = (now - s.last.at) / STEP_MS;
        ctx.strokeStyle = s.last.wrong ? `rgba(255,90,90,${1 - age * 0.6})` : `rgba(126,231,255,${1 - age * 0.8})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(X(s.last.p.x), Y(s.last.p.y), 11, 0, 7); ctx.stroke();
      }
      // weight vector arrow from the origin
      const n = Math.hypot(s.w[0], s.w[1]) || 1, L = 0.75;
      const ex = (s.w[0] / n) * L, ey = (s.w[1] / n) * L;
      ctx.strokeStyle = sceneHex().glowText; ctx.fillStyle = sceneHex().glowText; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
      const a = Math.atan2(-(ey), ex);
      ctx.beginPath();
      ctx.moveTo(X(ex), Y(ey));
      ctx.lineTo(X(ex) - 12 * Math.cos(a - 0.4), Y(ey) - 12 * Math.sin(a - 0.4));
      ctx.lineTo(X(ex) - 12 * Math.cos(a + 0.4), Y(ey) - 12 * Math.sin(a + 0.4));
      ctx.fill();
      ctx.font = '12px Inter, system-ui'; ctx.fillText('w', X(ex) + 6, Y(ey) - 6);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [data, paused, run]);

  return (
    <div className="mx-auto flex min-h-full max-w-5xl flex-col justify-start gap-6 px-4 py-6 lg:justify-center">
    <AboutBlurb />
    <div className="flex flex-col items-center gap-6 lg:flex-row lg:gap-10">
      <canvas ref={canvas} width={440} height={440} className="aspect-square w-full max-w-[440px] rounded-lg border border-line bg-[var(--pad)]" />
      <div className="max-w-md space-y-4 text-sm leading-relaxed text-slate-300">
        <p className="text-xs uppercase tracking-[0.2em] text-dim">Chapter 1 · Why this exists</p>
        <p className="text-base text-slate-100">
          This is a <Term k="perceptron" />, the smallest neural network, learning to separate orange dots from blue ones.
          <Cite id="rosenblatt1958" />
        </p>
        <p>
          Every update is just nudging a line: when it gets a dot wrong (red ring), it tilts and shifts its{' '}
          <Term k="decision boundary" /> a little toward that dot. The arrow is its <Term k="weight vector" />.
        </p>
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="rounded border border-line p-2"><div className="text-lg text-slate-100">{info.step}</div>checks</div>
          <div className="rounded border border-line p-2"><div className="text-lg text-slate-100">{info.updates}</div>nudges</div>
          <div className="rounded border border-line p-2"><div className="text-lg text-slate-100">{info.wrong}</div>dots wrong now</div>
        </div>
        <p className="h-5 text-xs text-glow">{info.done ? 'A full pass with no mistakes, so the line stops moving.' : ''}</p>
        <div className="flex items-center gap-2">
          <Btn onClick={() => setRun((r) => r + 1)}>Restart</Btn>
          <span className="text-xs text-dim"><Kbd>space</Kbd> pauses</span>
        </div>
        <p className="pt-2 text-slate-100">Now let's go inside a bigger one.</p>
        <Btn on onClick={onNext} className="px-4 py-2 text-sm">Enter the network</Btn>
      </div>
    </div>
    </div>
  );
}
