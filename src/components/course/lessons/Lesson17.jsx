// Lecture 17: three learning principles. Occam's razor, sampling bias, data snooping.
import React, { useMemo, useState } from 'react';
import { useTheme, sceneHex } from '../../../lib/theme.js';
import { Slides, Lead, Slider, Stat, Controls, Btn, LinePlot, lstsq, gauss, sign, rng } from '../common.jsx';

const vander = (x, d) => Array.from({ length: d + 1 }, (_, k) => x ** k);
const PLOT = Array.from({ length: 61 }, (_, i) => -1 + (2 * i) / 60);

function Occam() {
  const [complex, setComplex] = useState(false);
  const target = (x) => 0.7 * x + 0.1;
  const data = useMemo(() => { const r = rng(3); return Array.from({ length: 6 }, () => { const x = r() * 2 - 1; return { x, y: target(x) + 0.15 * gauss(r) }; }); }, []);
  const d = complex ? 5 : 1;
  const w = useMemo(() => lstsq(data.map((p) => vander(p.x, d)), data.map((p) => p.y), 1e-9), [data, d]);
  const fit = (x) => vander(x, d).reduce((s, z, i) => s + z * w[i], 0);
  const eIn = data.reduce((s, p) => s + (fit(p.x) - p.y) ** 2, 0) / data.length;
  const eOut = PLOT.reduce((s, x) => s + (fit(x) - target(x)) ** 2, 0) / PLOT.length;
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-300">the simplest fit that explains the data is the one to trust. with only 6 noisy points, a degree-5 curve nails them (tiny E_in) but swings wildly between them (large E_out). the straight line is less impressive in-sample and far better out.</p>
      <LinePlot series={[{ label: 'target', color: 'var(--color-warm)', pts: PLOT.map((x) => [x, target(x)]) }, { label: `degree ${d}`, color: 'var(--color-glow)', pts: PLOT.map((x) => [x, fit(x)]) }]} xlabel="x" ylabel="y" xDomain={[-1, 1]} yDomain={[-1.5, 1.5]} height={180} />
      <Controls><Btn on={!complex} onClick={() => setComplex(false)}>simple (line)</Btn><Btn on={complex} onClick={() => setComplex(true)}>complex (degree 5)</Btn></Controls>
      <div className="grid grid-cols-2 gap-2"><Stat label="E_in" value={eIn.toFixed(3)} tone="glow" /><Stat label="E_out" value={eOut.toFixed(3)} tone="warm" /></div>
    </div>
  );
}

function SamplingBias() {
  const [bias, setBias] = useState(0.6); // chance a +1 case even makes it into the sample
  const trueRate = 0.5; // the population is half +1
  // a biased sample keeps +1 cases with prob `bias` and -1 cases with prob (1-bias)
  const est = useMemo(() => {
    const r = rng(21); let pos = 0, n = 0;
    for (let i = 0; i < 4000; i++) {
      const isPos = r() < trueRate;
      if (r() < (isPos ? bias : 1 - bias)) { n++; if (isPos) pos++; }
    }
    return n ? pos / n : 0;
  }, [bias]);
  const hex = sceneHex(useTheme());
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-300">if the data is gathered in a way that favours one outcome, the sample lies no matter how big it gets. the population here is exactly half +1, but the sampler lets +1 cases through more often.</p>
      <Slider label="how strongly the sample favours +1" value={bias} min={0.1} max={0.9} step={0.01} onChange={setBias} fmt={(v) => v.toFixed(2)} />
      <div className="space-y-1">
        {[['population (truth)', trueRate, hex.cool], ['biased sample estimate', est, hex.accent]].map(([lab, v, c]) => (
          <div key={lab} className="flex items-center gap-2 text-xs">
            <span className="w-44 text-dim">{lab}</span>
            <div className="h-3 flex-1 rounded bg-line"><div className="h-full rounded" style={{ width: `${v * 100}%`, background: c }} /></div>
            <span className="w-10 text-right text-slate-200">{(v * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
      <p className="text-xs text-slate-400">the test set must be drawn the same way the model will be used, or the number it reports is about a world that does not exist.</p>
    </div>
  );
}

function Snooping() {
  const [tried, setTried] = useState(1);
  // search many random linear rules on a dataset with NO signal (random labels); keep the best in-sample
  const { inAcc, outAcc } = useMemo(() => {
    const r = rng(5);
    const train = Array.from({ length: 20 }, () => ({ x: r() * 2 - 1, y: r() * 2 - 1, label: r() < 0.5 ? 1 : -1 }));
    const test = Array.from({ length: 400 }, () => ({ x: r() * 2 - 1, y: r() * 2 - 1, label: r() < 0.5 ? 1 : -1 }));
    const rr = rng(1000);
    let best = -1, bestW = [0, 0, 0];
    for (let t = 0; t < tried; t++) {
      const w = [rr() * 2 - 1, rr() * 2 - 1, rr() * 2 - 1];
      const acc = train.filter((p) => sign(w[0] * p.x + w[1] * p.y + w[2]) === p.label).length / train.length;
      if (acc > best) { best = acc; bestW = w; }
    }
    const outAcc = test.filter((p) => sign(bestW[0] * p.x + bestW[1] * p.y + bestW[2]) === p.label).length / test.length;
    return { inAcc: best, outAcc };
  }, [tried]);
  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-300">the labels here are pure coin flips, so no rule can really predict them. but if you try enough rules and keep the one that scores best on your sample, it will look great in-sample and do no better than chance on fresh data. that gap is snooping.</p>
      <Slider label="rules tried before picking the best" value={tried} min={1} max={2000} step={1} onChange={setTried} />
      <div className="grid grid-cols-2 gap-2"><Stat label="best in-sample accuracy" value={`${(inAcc * 100).toFixed(0)}%`} tone="glow" /><Stat label="its fresh-data accuracy" value={`${(outAcc * 100).toFixed(0)}%`} tone="warm" /></div>
      <p className="text-xs text-slate-400">every time you reuse the test set to make a choice, you have snooped a little. the more you look, the better the in-sample number and the emptier it is.</p>
    </div>
  );
}

const TABS = { occam: { name: "occam's razor", C: Occam }, bias: { name: 'sampling bias', C: SamplingBias }, snoop: { name: 'data snooping', C: Snooping } };

export default function Lesson17() {
  const [tab, setTab] = useState('occam');
  const Body = TABS[tab].C;
  return (
    <div className="space-y-4">
      <Lead>
        <p>three warnings that sit on top of all the theory. they are less about algorithms and more about not fooling yourself. pick one.</p>
      </Lead>
      <Controls>
        {Object.entries(TABS).map(([k, v]) => (<Btn key={k} on={tab === k} onClick={() => setTab(k)}>{v.name}</Btn>))}
      </Controls>
      <Body />
      <Slides n={17} />
    </div>
  );
}
