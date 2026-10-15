// Course mode: interactive lessons, one per lecture of Caltech CS156 "Learning From Data".
// An index of all lectures plus a selected-lesson view, with prev/next and a quick-jump nav.
import React, { useEffect, useMemo, useState } from 'react';
import { Cite, Boundary } from '../ui.jsx';
import { LESSONS } from './lessons/index.jsx';
import { Btn, TELECOURSE } from './common.jsx';

const lessonFromHash = () => {
  const m = (typeof location !== 'undefined' ? location.hash : '').match(/^#course\/(\d+)/);
  const n = m ? Number(m[1]) : NaN;
  return LESSONS.some((l) => l.n === n) ? n : 1;
};

function IndexList({ current, onPick }) {
  return (
    <ol className="space-y-1">
      {LESSONS.map((l) => (
        <li key={l.n}>
          <button
            onClick={() => onPick(l.n)}
            className={`w-full rounded-md px-2 py-1.5 text-left text-xs ${l.n === current ? 'bg-glow/15 text-glow' : 'text-slate-300 hover:bg-line/60'}`}
          >
            <span className="font-medium">{l.n}. {l.title}</span>
            <span className="block text-[11px] leading-snug text-dim">{l.blurb}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}

export default function CoursePage({ onGoTour }) {
  const [current, setCurrent] = useState(lessonFromHash);
  useEffect(() => {
    const onHash = () => setCurrent(lessonFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  const pick = (n) => {
    setCurrent(n);
    try { history.replaceState(null, '', `${location.pathname}${location.search}#course/${n}`); } catch { /* ignore */ }
    const main = document.getElementById('course-main');
    if (main) main.scrollTop = 0;
  };

  const idx = LESSONS.findIndex((l) => l.n === current);
  const lesson = LESSONS[idx];
  const prev = LESSONS[idx - 1];
  const next = LESSONS[idx + 1];
  const ctx = useMemo(() => ({ onGoTour, onGoLesson: pick }), [onGoTour]);
  const Body = lesson.C;

  return (
    <main className="flex min-h-0 flex-1 flex-col overflow-hidden lg:flex-row">
      {/* index: all lectures, title and one-line description */}
      <aside className="scroll-thin hidden w-72 shrink-0 overflow-y-auto border-r border-line p-3 lg:block">
        <p className="mb-2 px-2 text-[10px] uppercase tracking-[0.2em] text-dim">lessons</p>
        <IndexList current={current} onPick={pick} />
      </aside>

      <section id="course-main" className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-4 px-4 py-5">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-dim">
              course · learning from data (cs156)<Cite id="abumostafa2012" />
            </p>
            <h1 className="mt-1 text-xl font-semibold text-slate-100">lecture {lesson.n}. {lesson.title}</h1>
            <p className="mt-1 text-sm text-slate-400">{lesson.blurb}</p>
          </div>

          {/* quick-jump nav, keyboard accessible, works on phones */}
          <div className="flex flex-wrap items-center gap-2">
            <Btn onClick={() => prev && pick(prev.n)} disabled={!prev}>prev</Btn>
            <label className="flex items-center gap-1 text-xs text-dim">
              <span className="sr-only">jump to lecture</span>
              <select
                value={current}
                onChange={(e) => pick(Number(e.target.value))}
                className="rounded-md border border-line bg-panel px-2 py-1 text-xs text-slate-200"
              >
                {LESSONS.map((l) => (<option key={l.n} value={l.n}>{l.n}. {l.title}</option>))}
              </select>
            </label>
            <Btn onClick={() => next && pick(next.n)} disabled={!next}>next</Btn>
            <a className="ml-auto text-xs text-dim hover:text-glow" href={TELECOURSE} target="_blank" rel="noreferrer">course home</a>
          </div>

          {/* full index on phones, where the sidebar is hidden */}
          <details className="rounded-lg border border-line p-2 lg:hidden">
            <summary className="cursor-pointer px-1 text-xs text-glow">all 18 lessons</summary>
            <div className="mt-2"><IndexList current={current} onPick={pick} /></div>
          </details>

          <Boundary key={lesson.n}>
            <Body ctx={ctx} />
          </Boundary>
        </div>
      </section>
    </main>
  );
}
