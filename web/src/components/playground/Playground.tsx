"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { RaceIndexEntry, PlaygroundGridEntry, PlaygroundInput } from "@/lib/types";
import { GRID_2026 } from "@/lib/grid2026";
import { teamColor, onTeamColor } from "@/lib/teams";
import { pct, fixed } from "@/lib/format";
import Select, { type Option } from "@/components/ui/Select";
import PressButton from "@/components/ui/PressButton";
import DriverAvatar from "@/components/ui/DriverAvatar";
import SectionHead from "@/components/ui/SectionHead";
import { Caret, WeatherGlyph } from "@/components/ui/icons";
import { useScorer } from "./useScorer";

// map RaceIndexEntry -> circuit selector options (value = GP race_name)
function circuitOptions(races: RaceIndexEntry[]): Option[] {
  const seen = new Set<string>();
  const opts: Option[] = [];
  for (const r of races) {
    if (seen.has(r.race_name)) continue;
    seen.add(r.race_name);
    opts.push({ value: r.race_name, label: r.race_name });
  }
  return opts;
}

function reassignStarts(grid: PlaygroundGridEntry[]): PlaygroundGridEntry[] {
  return grid.map((g, i) => ({ ...g, start: i + 1 }));
}

export default function Playground({ races }: { races: RaceIndexEntry[] }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(false);
  const { status, progress, rows, error, run } = useScorer(active);

  const cOptions = useMemo(() => circuitOptions(races), [races]);
  const [circuitName, setCircuitName] = useState(cOptions[0]?.value ?? "British Grand Prix");
  const [rain, setRain] = useState(false);
  const [temp, setTemp] = useState(35);
  const [humidity, setHumidity] = useState(45);
  const [grid, setGrid] = useState<PlaygroundGridEntry[]>(GRID_2026);

  // Activate (and lazy-load model) when scrolled into view.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActive(true)),
      { rootMargin: "200px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Debounced re-score on any input change (once model ready / scoring).
  useEffect(() => {
    if (!active) return;
    const input: PlaygroundInput = {
      grid,
      year: 2026,
      circuitName,
      rainfall: rain ? 1 : 0,
      avg_track_temp: temp,
      min_humidity: humidity,
      nSims: 2000,
    };
    const id = setTimeout(() => run(input), 180);
    return () => clearTimeout(id);
  }, [active, grid, circuitName, rain, temp, humidity, run]);

  function move(i: number, dir: -1 | 1) {
    setGrid((g) => {
      const j = i + dir;
      if (j < 0 || j >= g.length) return g;
      const next = [...g];
      [next[i], next[j]] = [next[j], next[i]];
      return reassignStarts(next);
    });
  }

  const maxWin = Math.max(...(rows ?? []).map((r) => r.p_win), 0.0001);
  const loadingModel = status === "loading";

  return (
    <section
      id="playground"
      ref={rootRef}
      className="section-anchor relative mx-auto w-full max-w-[1180px] px-5 py-24 md:px-8 md:py-32"
    >
      <SectionHead
        sector={5}
        channel="SANDBOX.LIVE"
        title="What-if"
        lede="The full LightGBM model runs in a Web Worker on your machine — no server. Reorder the grid, change the weather, switch circuits; win and podium odds re-simulate live via Plackett-Luce Monte-Carlo."
      />

      {/* model loading state */}
      <AnimatePresence>
        {loadingModel && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-6 rounded-lg border border-rosso/30 bg-rosso/5 p-4"
          >
            <div className="flex items-center justify-between font-mono text-xs text-chalk">
              <span>Loading model into browser…</span>
              <span className="t-data text-rosso">{pct(progress, 0)}</span>
            </div>
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">
              <motion.div
                className="h-full bg-rosso"
                animate={{ width: `${Math.max(4, progress * 100)}%` }}
                transition={{ ease: "linear" }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <div className="mb-6 rounded-lg border border-rosso bg-rosso/10 p-4 font-mono text-xs text-rosso">
          Worker error: {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* ── Controls + grid editor ── */}
        <div className="space-y-5">
          <div className="border border-line bg-surface/40 p-5">
            <div className="mb-4">
              <label className="mb-2 block font-mono text-[10px] uppercase tracking-widest text-muted">Circuit</label>
              <Select value={circuitName} options={cOptions} onChange={setCircuitName} ariaLabel="Circuit" />
            </div>
            <div className="mb-4 flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-widest text-muted">Conditions</span>
              <PressButton
                active={rain}
                onClick={() => setRain((r) => !r)}
                className="inline-flex items-center gap-1.5"
              >
                <WeatherGlyph wet={rain} size={13} />
                {rain ? "Wet" : "Dry"}
              </PressButton>
            </div>
            <Slider label="Track temp" value={temp} min={10} max={55} unit="°C" onChange={setTemp} />
            <Slider label="Min humidity" value={humidity} min={10} max={100} unit="%" onChange={setHumidity} />
            <div className="mt-4 flex gap-2">
              <PressButton onClick={() => setGrid(GRID_2026)}>Reset grid</PressButton>
              <PressButton
                onClick={() =>
                  setGrid((g) => reassignStarts([...g].sort(() => Math.random() - 0.5)))
                }
              >
                Shuffle quali
              </PressButton>
            </div>
          </div>

          <div className="border border-line bg-surface/40 p-3">
            <div className="mb-2 px-2 font-mono text-[10px] uppercase tracking-widest text-muted">
              Starting grid — reorder to change grid slots
            </div>
            <div className="max-h-[420px] space-y-1 overflow-y-auto pr-1">
              {grid.map((g, i) => (
                <div key={g.driver} className="flex items-center gap-2 rounded-md bg-white/[0.02] px-2 py-1.5">
                  <span className="t-data w-6 text-center text-sm text-muted">{g.start}</span>
                  <DriverAvatar driver={g.driver} team={g.team} size={24} />
                  <span className="font-text text-[14px] font-semibold tracking-tight text-chalk">{g.driver}</span>
                  <span className="ml-auto flex items-center gap-1.5">
                    <span className="h-2.5 w-0.5" style={{ background: teamColor(g.team) }} />
                    <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-faint">
                      {g.team}
                    </span>
                  </span>
                  <div className="flex flex-col text-muted">
                    <button onClick={() => move(i, -1)} className="px-1 py-0.5 transition-colors hover:text-chalk" aria-label="Move up">
                      <Caret dir="up" />
                    </button>
                    <button onClick={() => move(i, 1)} className="px-1 py-0.5 transition-colors hover:text-chalk" aria-label="Move down">
                      <Caret dir="down" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Live results ── */}
        <div className="border border-line bg-surface/40 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="t-display text-[1.4rem] text-chalk">Simulated result</h3>
            <span className="font-mono text-[10px] uppercase tracking-widest text-muted">
              {status === "scoring" ? "simulating…" : status === "ready" && rows ? "2000 sims" : status}
            </span>
          </div>

          {!rows && !loadingModel && (
            <div className="grid h-64 place-items-center font-mono text-xs text-muted">
              {active ? "warming up…" : "scroll to activate"}
            </div>
          )}

          <div className="space-y-1">
            <AnimatePresence>
              {rows?.map((r) => {
                const color = teamColor(r.team);
                return (
                  <motion.div
                    key={r.driver}
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ layout: { type: "spring", stiffness: 400, damping: 34 }, opacity: { duration: 0.2 } }}
                    className="flex items-center gap-2.5 rounded-md bg-white/[0.02] px-2.5 py-2"
                  >
                    <span
                      className="t-data grid h-6 w-6 place-items-center rounded text-xs font-bold"
                      style={{
                        background: r.pred_finish <= 3 ? ["#FFD24A", "#C9D1D9", "#CD7F32"][r.pred_finish - 1] : "transparent",
                        color: r.pred_finish <= 3 ? "#0a0a0a" : onTeamColor(r.team) === "#0a0a0a" ? "#f4f4f5" : "#f4f4f5",
                      }}
                    >
                      {r.pred_finish}
                    </span>
                    <DriverAvatar driver={r.driver} team={r.team} size={26} />
                    <span className="w-24 font-text text-[14px] font-semibold tracking-tight text-chalk">{r.driver}</span>
                    <div className="relative h-5 flex-1 overflow-hidden bg-white/[0.04]">
                      <motion.div
                        className="absolute inset-y-0 left-0"
                        style={{ background: color, boxShadow: `0 0 10px -3px ${color}` }}
                        animate={{ width: `${(r.p_win / maxWin) * 100}%` }}
                        transition={{ type: "spring", stiffness: 120, damping: 22 }}
                      />
                    </div>
                    <span className="t-data w-12 text-right text-xs text-chalk">{pct(r.p_win, 1)}</span>
                    <span className="t-data hidden w-14 text-right text-[10px] text-muted sm:inline">
                      dnf {pct(r.p_dnf, 0)}
                    </span>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>

          {rows && (
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 border-t border-line pt-3 font-mono text-[10px] text-muted">
              <span>P1 <span className="text-chalk">{rows[0]?.driver}</span> · {pct(rows[0]?.p_win ?? 0, 1)}</span>
              <span>expected pos {fixed(rows[0]?.expected_pos ?? 0, 2)}</span>
              <span className="text-rosso">τ = 0.5 · Plackett-Luce</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  const t = (value - min) / (max - min);
  return (
    <div className="mb-3">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted">{label}</span>
        <span className="t-data text-sm text-chalk">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full outline-none"
        style={{ background: `linear-gradient(90deg, #DC0000 ${t * 100}%, #26262a ${t * 100}%)` }}
      />
    </div>
  );
}
