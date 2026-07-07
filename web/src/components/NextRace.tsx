"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import type { RaceArtifact, Prediction } from "@/lib/types";
import { teamColor } from "@/lib/teams";
import { pct } from "@/lib/format";
import { EASE_OUT_EXPO } from "@/lib/motion";
import Section from "./ui/Section";
import DriverAvatar from "./ui/DriverAvatar";
import { WeatherGlyph } from "./ui/icons";

type Metric = "p_win" | "p_podium" | "p_points";
const METRICS: { key: Metric; label: string }[] = [
  { key: "p_win", label: "Win" },
  { key: "p_podium", label: "Podium" },
  { key: "p_points", label: "Points" },
];

// Telemetry track with tick marks — a sector bar, not a rounded progress pill.
function Track({
  value,
  max,
  color,
  height = "h-2.5",
}: {
  value: number;
  max: number;
  color: string;
  height?: string;
}) {
  const width = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className={`relative ${height} w-full overflow-hidden bg-white/[0.04]`}>
      {/* quarter ticks */}
      {[25, 50, 75].map((t) => (
        <span
          key={t}
          className="absolute inset-y-0 w-px bg-white/[0.06]"
          style={{ left: `${t}%` }}
        />
      ))}
      <motion.div
        className="absolute inset-y-0 left-0"
        style={{ background: color, boxShadow: `0 0 12px -2px ${color}` }}
        initial={{ width: 0 }}
        whileInView={{ width: `${width}%` }}
        viewport={{ once: true, amount: 0.5 }}
        transition={{ duration: 0.9, ease: EASE_OUT_EXPO }}
      />
      {/* bright leading edge */}
      <motion.div
        className="absolute inset-y-0 w-[2px] bg-white"
        initial={{ left: 0, opacity: 0 }}
        whileInView={{ left: `calc(${width}% - 2px)`, opacity: 0.8 }}
        viewport={{ once: true, amount: 0.5 }}
        transition={{ duration: 0.9, ease: EASE_OUT_EXPO }}
      />
    </div>
  );
}

function FieldRow({
  pred,
  metric,
  max,
}: {
  pred: Prediction;
  metric: Metric;
  max: number;
}) {
  const color = teamColor(pred.team);
  const value = pred[metric];
  const podium = pred.pred_finish <= 3;
  return (
    <div className="group grid grid-cols-[2rem_1fr] items-center gap-4 border-t border-line/70 py-3 transition-colors hover:bg-white/[0.015] md:grid-cols-[2.5rem_13rem_1fr_4rem]">
      <div className="t-data text-right text-[15px] tabular-nums text-faint group-hover:text-muted">
        {String(pred.pred_finish).padStart(2, "0")}
      </div>
      <div className="col-start-2 flex items-center gap-2.5 md:col-auto">
        <DriverAvatar driver={pred.driver} team={pred.team} size={26} />
        <div className="min-w-0">
          <div className="font-text text-[15px] font-semibold leading-none tracking-tight text-chalk">
            {pred.driver}
          </div>
          <div className="mt-1 flex items-center gap-1.5">
            <span className="h-2 w-0.5" style={{ background: color }} />
            <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-faint">
              {pred.team}
            </span>
          </div>
        </div>
      </div>
      <div className="col-span-2 md:col-auto">
        <Track value={value} max={max} color={podium ? color : `${color}bb`} />
      </div>
      <div className="col-span-2 text-right md:col-auto">
        <span className="t-data text-[14px] tabular-nums text-chalk">
          {pct(value, 1)}
        </span>
      </div>
    </div>
  );
}

function Heatmap({ preds }: { preds: Prediction[] }) {
  const nPos = preds[0]?.position_probs?.length ?? 20;
  const positions = Array.from({ length: nPos }, (_, i) => i + 1);
  return (
    <div className="overflow-x-auto">
      <div className="min-w-[760px]">
        <div className="flex pb-1.5">
          <div className="w-28 shrink-0" />
          {positions.map((p) => (
            <div
              key={p}
              className={`t-data flex-1 text-center text-[9px] ${
                p <= 3 ? "text-rosso" : "text-faint"
              }`}
            >
              {p}
            </div>
          ))}
        </div>
        {preds.map((pred, ri) => (
          <div
            key={pred.driver}
            className="group flex items-center border-t border-line/50"
          >
            <div className="flex w-28 shrink-0 items-center gap-2 py-1">
              <span className="t-data w-5 text-right text-[11px] text-faint">
                {String(pred.pred_finish).padStart(2, "0")}
              </span>
              <span className="font-text text-[13px] font-medium text-muted group-hover:text-chalk">
                {pred.driver}
              </span>
            </div>
            {pred.position_probs.slice(0, nPos).map((prob, i) => {
              const alpha = Math.min(1, Math.pow(prob, 0.6) * 1.15);
              return (
                <div key={i} className="flex-1 px-px py-px">
                  <motion.div
                    className="h-3.5 w-full"
                    style={{
                      background: `rgba(220,0,0,${alpha})`,
                      outline:
                        i + 1 <= 3
                          ? "1px solid rgba(220,0,0,0.14)"
                          : "1px solid rgba(255,255,255,0.02)",
                    }}
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: ri * 0.01 + i * 0.004 }}
                    title={`${pred.driver} — P${i + 1}: ${pct(prob, 1)}`}
                  />
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function NextRace({ race }: { race: RaceArtifact }) {
  const [metric, setMetric] = useState<Metric>("p_win");
  const preds = race.predictions;
  const leader = preds[0];
  const rest = preds.slice(1);
  const max = Math.max(...preds.map((p) => p[metric]), 0.0001);
  const wet = !!race.weather.rainfall;
  const metricLabel = METRICS.find((m) => m.key === metric)!.label;

  return (
    <Section
      id="next-race"
      sector={1}
      channel="NEXT.RACE"
      title="The odds"
      lede="Per-driver probabilities aggregated over thousands of Monte-Carlo race simulations, team-coded and ranked by predicted finish."
      aside={
        <div className="flex items-center gap-2 border border-line px-3 py-2">
          <span className={wet ? "text-[#64C4FF]" : "text-[#f4b740]"}>
            <WeatherGlyph wet={wet} />
          </span>
          <span className="t-data text-[11px] text-muted">
            {wet ? "Wet" : "Dry"} · {race.weather.avg_track_temp}°C track
          </span>
        </div>
      }
    >
      {/* Metric toggle — a segmented control, not three loose buttons. */}
      <div className="mb-8 flex items-center justify-between">
        <div className="inline-flex rounded-[3px] border border-line p-0.5">
          {METRICS.map((m) => {
            const active = metric === m.key;
            return (
              <button
                key={m.key}
                onClick={() => setMetric(m.key)}
                className={`relative rounded-[2px] px-4 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] transition-colors ${
                  active ? "text-white" : "text-faint hover:text-muted"
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="metric-active"
                    className="absolute inset-0 rounded-[2px] bg-rosso"
                    transition={{ type: "spring", stiffness: 420, damping: 34 }}
                  />
                )}
                <span className="relative">{m.label}</span>
              </button>
            );
          })}
        </div>
        <span className="t-label hidden text-faint sm:block">
          P({metricLabel.toLowerCase()}) · {race.predictions.length} drivers
        </span>
      </div>

      {/* Leader — dominant, not one of twenty identical bars. */}
      {leader && (
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.6 }}
          transition={{ duration: 0.7, ease: EASE_OUT_EXPO }}
          className="relative mb-2 grid grid-cols-1 items-end gap-6 border-t-2 border-rosso pt-6 md:grid-cols-[1fr_auto]"
        >
          <div className="flex items-center gap-4">
            <span className="t-display text-[clamp(3rem,7vw,5rem)] leading-[0.8] text-rosso">
              01
            </span>
            <DriverAvatar driver={leader.driver} team={leader.team} size={54} />
            <div>
              <div className="font-text text-[clamp(1.6rem,3vw,2.2rem)] font-bold leading-none tracking-tight text-chalk">
                {leader.driver}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span
                  className="h-3 w-1"
                  style={{ background: teamColor(leader.team) }}
                />
                <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted">
                  {leader.team}
                </span>
              </div>
            </div>
          </div>
          <div className="md:text-right">
            <div className="t-display text-[clamp(2.6rem,6vw,4rem)] leading-[0.8] text-chalk">
              {pct(leader[metric], 1)}
            </div>
            <div className="t-label mt-1 text-faint">P({metricLabel.toLowerCase()})</div>
          </div>
          <div className="md:col-span-2">
            <Track
              value={leader[metric]}
              max={max}
              color={teamColor(leader.team)}
              height="h-4"
            />
          </div>
        </motion.div>
      )}

      {/* The chasing field. */}
      <div>
        {rest.map((p) => (
          <FieldRow key={p.driver} pred={p} metric={metric} max={max} />
        ))}
      </div>

      {/* Heatmap */}
      <div className="mt-20">
        <div className="mb-5 flex items-baseline justify-between border-b border-line pb-3">
          <h3 className="t-display text-[clamp(1.4rem,3vw,2.1rem)] text-chalk">
            Position probability
          </h3>
          <span className="t-label text-faint">driver × finishing slot</span>
        </div>
        <Heatmap preds={preds} />
        <p className="t-label mt-4 text-faint">
          Cell intensity = P(driver finishes in that slot). Columns 1–3 mark the podium.
        </p>
      </div>
    </Section>
  );
}
