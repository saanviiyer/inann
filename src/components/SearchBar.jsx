import React, { useMemo, useState } from 'react';
import { search } from '../lib/search.js';
import { roomName, EXPLAINED } from '../lib/model.js';
import { scoreBand } from '../lib/rooms.js';

export default function SearchBar({ onPick }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const results = useMemo(() => (q.trim().length > 1 ? search(q) : []), [q]);
  const pick = (r) => { onPick(r); setOpen(false); };
  return (
    <div className="relative w-full max-w-sm">
      <input
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => { if (e.key === 'Enter' && results[0]) pick(results[0]); if (e.key === 'Escape') setOpen(false); }}
        placeholder='search by concept, e.g. "horizontal line" or "the 7 detector"'
        className="w-full rounded-md border border-line bg-panel px-3 py-1.5 text-xs text-slate-100 placeholder:text-dim focus:border-glow/50 focus:outline-none"
      />
      {open && q.trim().length > 1 && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 max-h-96 overflow-y-auto rounded-md border border-line bg-panel shadow-xl">
          {!EXPLAINED && (
            <div className="border-b border-line px-3 py-2 text-[11px] text-dim">
              Explanations not generated yet, so only measured digit matches can be found.
            </div>
          )}
          {results.length === 0 && <div className="px-3 py-2 text-xs text-dim">No unit matches that.</div>}
          {results.map((r, i) => {
            const b = r.kind === 'text' ? scoreBand(r.sim) : null;
            return (
              <button key={i} onMouseDown={() => pick(r)} className="block w-full border-b border-line/60 px-3 py-2 text-left hover:bg-glow/10">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-slate-100">{roomName(r.room)}</span>
                  {b ? <span className={`rounded border px-1 text-[9px] ${b.cls}`}>{b.label}, r {r.sim.toFixed(2)}</span> : <span className="rounded border border-line px-1 text-[9px] text-slate-300">measured</span>}
                </div>
                <div className="mt-0.5 text-[11px] leading-snug text-slate-400">{r.why}</div>
              </button>
            );
          })}
          <div className="px-3 py-1.5 text-[10px] text-dim">Enter flies to the top result. Matching is by keywords over the written explanations, not embeddings.</div>
        </div>
      )}
    </div>
  );
}
