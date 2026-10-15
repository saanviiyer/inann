// Lecture 3: the linear model I. PLA thrashes on non-separable data; the pocket keeps the best line.
import React, { useEffect, useRef, useState } from 'react';
import { Plane, Slides, Lead, Controls, Btn, Stat, boundaryLine, sign, rng } from '../common.jsx';

function makeData(seed, flip) {
  const r = rng(seed);
  const pts = [];
  while (pts.length < 40) {
    const x = (r() * 2 - 1) * 0.95, y = (r() * 2 - 1) * 0.95;
    let label = 0.8 * x - 0.6 * y + 0.1 > 0 ? 1 : -1;
    if (r() < flip) label = -label; // noise makes it non-separable
    pts.push({ x, y, label });
  }
  return pts;
}
const errOf = (w, pts) => pts.filter((p) => sign(w[0] * p.x + w[1] * p.y + w[2]) !== p.label).length;

export default function Lesson03() {
  const [seed, setSeed] = useState(5);
  const [pts, setPts] = useState(() => makeData(5, 0.15));
  const [w, setW] = useState([0.2, -0.1, 0]); // live PLA weights
  const [pocket, setPocket] = useState([0.2, -0.1, 0]); // best weights seen
  const [running, setRunning] = useState(false);
  const [n, setN] = useState(0);
  const idx = useRef(0);
  const timer = useRef(null);

  const step = () => {
    setW((ww) => {
      let i = idx.current % pts.length, guard = 0;
      while (sign(ww[0] * pts[i].x + ww[1] * pts[i].y + ww[2]) === pts[i].label && guard < pts.length) { i = (i + 1) % pts.length; guard++; }
      idx.current = i + 1;
      const p = pts[i];
      const nw = [ww[0] + p.label * p.x, ww[1] + p.label * p.y, ww[2] + p.label];
      setPocket((pk) => (errOf(nw, pts) < errOf(pk, pts) ? nw : pk));
      setN((c) => c + 1);
      return nw;
    });
  };

  useEffect(() => {
    if (!running) { clearInterval(timer.current); return; }
    timer.current = setInterval(step, 160);
    return () => clearInterval(timer.current);
  }, [running, pts]);

  const reset = (d = pts) => { setRunning(false); idx.current = 0; setN(0); setW([0.2, -0.1, 0]); setPocket([0.2, -0.1, 0]); };

  return (
    <div className="space-y-4">
      <Lead>
        <p>real data is rarely perfectly separable. when no line can get everything right, plain pla never stops: every pass it finds a mistake and jumps, so its line keeps lurching around.</p>
        <p>the pocket trick is simple. keep running pla, but hold on to the best line you have seen so far (fewest mistakes) and ignore the thrashing. the glowing line is live pla; the steady amber line is what the pocket is holding.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane
          points={pts}
          classify={(x, y) => sign(pocket[0] * x + pocket[1] * y + pocket[2])}
          lines={[
            { ...boundaryLine(w), color: 'var(--color-glow)', width: 1.5, dash: '4 4' },
            { ...boundaryLine(pocket), color: 'var(--color-warm)', width: 2.5 },
          ]}
          interactive={false}
        />
        <div className="flex-1 space-y-3">
          <Controls>
            <Btn on={running} onClick={() => setRunning((v) => !v)}>{running ? 'pause' : 'run'}</Btn>
            <Btn onClick={() => { setRunning(false); step(); }}>step</Btn>
            <Btn onClick={() => reset()}>reset</Btn>
            <Btn onClick={() => { const d = makeData(seed + 1, 0.15); setSeed((s) => s + 1); setPts(d); idx.current = 0; setN(0); setW([0.2, -0.1, 0]); setPocket([0.2, -0.1, 0]); }}>new data</Btn>
          </Controls>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="updates" value={n} />
            <Stat label="pla mistakes" value={`${errOf(w, pts)}/${pts.length}`} tone="glow" />
            <Stat label="pocket mistakes" value={`${errOf(pocket, pts)}/${pts.length}`} tone="warm" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            watch the pla count bounce while the pocket count only ever drops or holds. the shaded regions show the pocket line's decision, the one you would actually ship.
          </p>
        </div>
      </div>
      <Slides n={3} />
    </div>
  );
}
