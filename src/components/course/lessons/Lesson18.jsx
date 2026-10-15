// Lecture 18: epilogue. A map of the course, and a way back into any lesson.
import React from 'react';
import { Slides, Lead, Btn, TELECOURSE } from '../common.jsx';

const MAP = [
  { group: 'what learning is', items: [[1, 'the problem'], [2, 'is it feasible'], [3, 'the linear model'], [4, 'error and noise']] },
  { group: 'why it can work (the theory)', items: [[5, 'dichotomies'], [6, 'generalization'], [7, 'vc dimension'], [8, 'bias-variance']] },
  { group: 'the models', items: [[9, 'nonlinear transforms'], [10, 'neural networks'], [14, 'svm'], [15, 'kernels'], [16, 'rbf']] },
  { group: 'making it generalize', items: [[11, 'overfitting'], [12, 'regularization'], [13, 'validation']] },
  { group: 'staying honest', items: [[17, 'three principles']] },
];

export default function Lesson18({ ctx }) {
  return (
    <div className="space-y-4">
      <Lead>
        <p>the course splits into three questions. can we learn at all? that is the theory, lectures 5 to 8, and the answer is yes when the hypothesis set has a break point. what do we learn with? the models, from the linear model through neural networks, svms, kernels and rbfs. and how do we keep it honest? overfitting, regularization, validation, and the three principles.</p>
        <p>everything here grew from the single perceptron nudge in lecture 1. click any stop to revisit it.</p>
      </Lead>
      <div className="space-y-3">
        {MAP.map((m) => (
          <div key={m.group}>
            <p className="mb-1 text-xs uppercase tracking-[0.2em] text-dim">{m.group}</p>
            <div className="flex flex-wrap gap-2">
              {m.items.map(([n, label]) => (
                <Btn key={n} onClick={() => ctx?.onGoLesson?.(n)}>{n}. {label}</Btn>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="text-sm text-slate-300">
        beyond here is the bias-variance of your own projects, and the discipline of not snooping. the lectures, slides and homework all live at{' '}
        <a className="text-glow/90 hover:text-glow" href={TELECOURSE} target="_blank" rel="noreferrer">work.caltech.edu/telecourse</a>.
      </p>
      <Slides n={18} />
    </div>
  );
}
