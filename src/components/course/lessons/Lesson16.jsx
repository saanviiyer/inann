// Lecture 16: radial basis functions. Place Gaussian centers; the fit is a sum of bumps.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Slider, Stat, Controls, Btn, lstsq, sign, rng } from '../common.jsx';

function makeData() {
  const r = rng(7);
  return Array.from({ length: 70 }, () => {
    const x = (r() * 2 - 1) * 0.95, y = (r() * 2 - 1) * 0.95;
    let label = Math.sin(3 * x) * 0.5 > y ? 1 : -1; // a wavy boundary
    if (r() < 0.04) label = -label;
    return { x, y, label };
  });
}

export default function Lesson16() {
  const data = useMemo(makeData, []);
  const [centers, setCenters] = useState([]);
  const [gamma, setGamma] = useState(6);

  const phi = (x, y) => [...centers.map((c) => Math.exp(-gamma * ((x - c.x) ** 2 + (y - c.y) ** 2))), 1];
  const w = useMemo(() => {
    if (!centers.length) return null;
    return lstsq(data.map((p) => phi(p.x, p.y)), data.map((p) => p.label), 1e-6);
  }, [data, centers, gamma]);
  const h = (x, y) => {
    if (!w) return 0;
    return sign(phi(x, y).reduce((s, z, i) => s + z * w[i], 0));
  };
  const eIn = w ? data.filter((p) => h(p.x, p.y) !== p.label).length / data.length : null;

  const autoPlace = () => {
    const r = rng(Date.now() % 9999);
    setCenters(Array.from({ length: 8 }, () => data[Math.floor(r() * data.length)]).map((p) => ({ x: p.x, y: p.y })));
  };

  return (
    <div className="space-y-4">
      <Lead>
        <p>a radial basis function network builds its answer from bumps. each center drops a gaussian hill; near the center its feature is large, far away it fades. the fit is a weighted sum of those hills, and the weights are solved so the sum matches the labels.</p>
        <p>click to drop centers on the plane (or auto-place a handful), then watch the boundary form. few centers give a smooth, stiff fit; many give a flexible one that can chase the wavy boundary, and eventually the noise.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane
          points={data}
          classify={h}
          marks={centers}
          onBg={(x, y) => setCenters((c) => [...c, { x, y }])}
          grid={28}
        />
        <div className="flex-1 space-y-3">
          <Slider label="gamma (bump width)" value={gamma} min={1} max={30} step={1} onChange={setGamma} />
          <Controls>
            <Btn onClick={autoPlace}>auto-place 8 centers</Btn>
            <Btn onClick={() => setCenters([])}>clear centers</Btn>
          </Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="centers" value={centers.length} tone="glow" />
            <Stat label="E_in" value={eIn == null ? '-' : eIn.toFixed(2)} tone="warm" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            with no centers there is nothing to fit. add a few and the boundary snaps toward the data. this is the same idea as the rbf kernel from the last lecture, but here you choose the centers instead of using every point.
          </p>
        </div>
      </div>
      <Slides n={16} />
    </div>
  );
}
