"use client";

import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { RaceArtifact } from "@/lib/types";
import { teamColor } from "@/lib/teams";
import { fixed } from "@/lib/format";
import Section from "./ui/Section";
import Select, { type Option } from "./ui/Select";

const NEG = "#6b7280";
const POS = "#DC0000";

export default function Shap({ race }: { race: RaceArtifact }) {
  const drivers = useMemo(
    () => race.predictions.map((p) => p.driver).filter((d) => race.shap.drivers[d]),
    [race],
  );
  const [driver, setDriver] = useState(drivers[0] ?? "");

  const teamOf = useMemo(() => {
    const m = new Map<string, string>();
    race.predictions.forEach((p) => m.set(p.driver, p.team));
    return m;
  }, [race]);

  const options: Option[] = drivers.map((d) => ({ value: d, label: d, hint: teamOf.get(d) }));

  const base = race.shap.base_value;
  const feats = useMemo(() => {
    const list = [...(race.shap.drivers[driver] ?? [])];
    list.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    return list;
  }, [race, driver]);

  // Cumulative walk from base value.
  const steps = useMemo(() => {
    let cum = base;
    const out = feats.map((f) => {
      const start = cum;
      cum += f.shap;
      return { ...f, start, end: cum };
    });
    return { out, final: cum };
  }, [feats, base]);

  // Geometry
  const W = 720;
  const PADL = 230;
  const PADR = 70;
  const rowH = 40;
  const H = (feats.length + 1) * rowH + 30;
  const plotW = W - PADL - PADR;

  const allX = [base, steps.final, ...steps.out.flatMap((s) => [s.start, s.end])];
  const min = Math.min(...allX);
  const max = Math.max(...allX);
  const span = max - min || 1;
  const x = (v: number) => PADL + ((v - min) / span) * plotW;

  const accent = teamColor(teamOf.get(driver) ?? "");

  return (
    <Section
      id="shap"
      sector={3}
      channel="EXPLAIN.SHAP"
      title="Why it thinks that"
      lede="A SHAP waterfall decomposes each driver's ranker score: start at the model's base value, then add every feature's contribution. Red pushes the driver up the order; grey pulls them down."
    >
      <div className="mb-6 flex flex-wrap items-center gap-5">
        <div className="w-64">
          <Select value={driver} options={options} onChange={setDriver} ariaLabel="Select a driver" />
        </div>
        <div className="t-data text-xs text-muted">
          base <span className="text-chalk">{fixed(base, 2)}</span>
          <span className="mx-2 text-faint">→</span> score{" "}
          <span style={{ color: accent }}>{fixed(steps.final, 2)}</span>
        </div>
      </div>

      <div className="overflow-x-auto border border-line bg-surface/40 p-4">
        <AnimatePresence mode="wait">
          <motion.svg
            key={driver}
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            className="min-w-[640px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            {/* base gridline */}
            <line x1={x(base)} y1={18} x2={x(base)} y2={H - 12} stroke="#3a3a40" strokeDasharray="4 4" />
            <text x={x(base)} y={12} fill="#8a8a92" fontSize={10} fontFamily="monospace" textAnchor="middle">
              base {fixed(base, 2)}
            </text>

            {steps.out.map((s, i) => {
              const y = 24 + i * rowH;
              const positive = s.shap >= 0;
              const x0 = x(Math.min(s.start, s.end));
              const barW = Math.max(2, Math.abs(x(s.end) - x(s.start)));
              const color = positive ? POS : NEG;
              return (
                <g key={s.feature}>
                  {/* connector to previous cumulative */}
                  <line
                    x1={x(s.start)}
                    y1={y - rowH + 26}
                    x2={x(s.start)}
                    y2={y + 8}
                    stroke="#2a2a2e"
                    strokeWidth={1}
                  />
                  {/* feature label + value */}
                  <text x={PADL - 12} y={y + 15} fill="#c9c9cf" fontSize={12.5} textAnchor="end" fontFamily="var(--font-text)">
                    {s.feature}
                  </text>
                  <text x={PADL - 12} y={y + 29} fill="#6f6f77" fontSize={10} textAnchor="end" fontFamily="monospace">
                    = {fixed(s.value, 2)}
                  </text>
                  {/* bar */}
                  <motion.rect
                    x={x0}
                    y={y}
                    height={20}
                    rx={3}
                    fill={color}
                    initial={{ width: 0, opacity: 0 }}
                    animate={{ width: barW, opacity: 1 }}
                    transition={{ delay: 0.05 + i * 0.06, type: "spring", stiffness: 140, damping: 20 }}
                  />
                  {/* value label */}
                  <motion.text
                    x={positive ? x0 + barW + 6 : x0 - 6}
                    y={y + 14}
                    fill={color}
                    fontSize={11}
                    fontFamily="monospace"
                    textAnchor={positive ? "start" : "end"}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.2 + i * 0.06 }}
                  >
                    {s.shap >= 0 ? "+" : ""}
                    {fixed(s.shap, 2)}
                  </motion.text>
                </g>
              );
            })}

            {/* final score marker */}
            <line x1={x(steps.final)} y1={18} x2={x(steps.final)} y2={H - 12} stroke={accent} strokeWidth={1.5} />
            <text x={x(steps.final)} y={H - 2} fill={accent} fontSize={10} fontFamily="monospace" textAnchor="middle">
              score {fixed(steps.final, 2)}
            </text>
          </motion.svg>
        </AnimatePresence>
      </div>
      <p className="t-label mt-4 text-faint">
        Top {feats.length} features by |contribution| · {race.race_name}
      </p>
    </Section>
  );
}
