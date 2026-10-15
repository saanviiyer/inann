// Lecture 10: neural networks. A short frame, then a hand-off to the inann Tour that teaches them in 3D.
import React from 'react';
import { useTheme, sceneHex } from '../../../lib/theme.js';
import { Slides, Lead, Btn } from '../common.jsx';

export default function Lesson10({ ctx }) {
  const hex = sceneHex(useTheme());
  const layers = [3, 5, 5, 2];
  const W = 340, H = 180, colX = (i) => 40 + (i * (W - 80)) / (layers.length - 1);
  const nodeY = (c, n) => (H * (c + 1)) / (n + 1);
  const nodes = layers.flatMap((n, li) => Array.from({ length: n }, (_, c) => ({ x: colX(li), y: nodeY(c, n), li })));

  return (
    <div className="space-y-4">
      <Lead>
        <p>stack perceptrons in layers and let each layer feed the next, and you get a neural network. one perceptron draws one line; a hidden layer of them can carve the plane into regions, and the next layer combines those regions into any shape you like.</p>
        <p>training is still the same idea as lecture 1, a nudge per mistake, but now the nudge flows backward through every layer at once. that is backpropagation, and it is far easier to watch than to read.</p>
      </Lead>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-line bg-[var(--pad)]" role="img">
        {nodes.map((a, i) => nodes.filter((b) => b.li === a.li + 1).map((b, j) => (
          <line key={`${i}-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={hex.axis} strokeWidth="1" />
        )))}
        {nodes.map((n, i) => (
          <circle key={i} cx={n.x} cy={n.y} r={7} fill={n.li === 0 ? hex.cool : n.li === layers.length - 1 ? hex.accent : hex.glowText} />
        ))}
      </svg>
      <div className="rounded-lg border border-glow/30 bg-glow/5 p-3">
        <p className="text-sm text-slate-200">inann already renders a real trained network as rooms you can fly through, pick neurons, and watch the signal climb. that is the best way to meet this lecture.</p>
        <div className="mt-2">
          <Btn on onClick={() => ctx?.onGoTour?.()}>open the tour and go inside one</Btn>
        </div>
      </div>
      <Slides n={10} />
    </div>
  );
}
