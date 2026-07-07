"use client";

import { useMemo } from "react";
import {
  ComposedChart,
  Area,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { TrackRecordEntry } from "@/lib/types";
import { fixed, pct, formatDate } from "@/lib/format";
import Section from "./ui/Section";
import Reveal from "./ui/Reveal";

interface Row extends TrackRecordEntry {
  i: number;
}

const ROSSO = "#DC0000";
const STEEL = "#3b3b43";

// Bespoke dark tooltip — no default recharts chrome.
function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: Row }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="min-w-[190px] border border-lineBright bg-ink/95 p-3 shadow-card backdrop-blur-md">
      <div className="font-text text-[14px] font-bold tracking-tight text-chalk">
        {d.race_name}
      </div>
      <div className="t-label mt-0.5 text-faint">{formatDate(d.race_date)}</div>
      <div className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1.5">
        <TipStat label="Spearman" value={fixed(d.spearman, 3)} accent />
        <TipStat label="NDCG@3" value={fixed(d.ndcg3, 3)} />
        <TipStat label="Podium" value={`${d.podium_hits}/3`} />
        <TipStat label="Winner" value={d.winner_correct ? "HIT" : "MISS"} />
      </div>
    </div>
  );
}

function TipStat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="t-label text-[9px] !tracking-[0.16em] text-faint">{label}</div>
      <div className={`t-data text-[13px] ${accent ? "text-rosso" : "text-chalk"}`}>
        {value}
      </div>
    </div>
  );
}

// Custom red trace dot — filled on winner-correct races, hollow otherwise.
function TraceDot(props: { cx?: number; cy?: number; payload?: Row }) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null || !payload) return <g />;
  return payload.winner_correct ? (
    <circle cx={cx} cy={cy} r={3.4} fill={ROSSO} stroke="#0a0a0a" strokeWidth={1.5} />
  ) : (
    <circle cx={cx} cy={cy} r={2.6} fill="#0a0a0a" stroke={ROSSO} strokeWidth={1.4} />
  );
}

export default function TrackRecord({ data }: { data: TrackRecordEntry[] }) {
  const rows: Row[] = useMemo(() => data.map((e, i) => ({ ...e, i })), [data]);

  const stats = useMemo(() => {
    const n = data.length || 1;
    const mean = (f: (e: TrackRecordEntry) => number) =>
      data.reduce((a, e) => a + f(e), 0) / n;
    return {
      spearman: mean((e) => e.spearman),
      ndcg3: mean((e) => e.ndcg3),
      winner: data.filter((e) => e.winner_correct).length / n,
      podium: mean((e) => e.podium_hits),
      n: data.length,
    };
  }, [data]);

  const tickInterval = Math.max(0, Math.ceil(rows.length / 8) - 1);

  return (
    <Section
      id="track-record"
      sector={4}
      channel="BACKTEST.WALKFWD"
      title="Track record"
      lede={`Walk-forward evaluation across ${stats.n} Grands Prix — retrained before each race on prior data only, then graded on the real result.`}
    >
      {/* Asymmetric stat strip: one hero metric, three supporting readouts. */}
      <Reveal className="mb-12 grid items-end gap-8 border-b border-line pb-10 md:grid-cols-[1.1fr_1.5fr]">
        <div>
          <div className="t-label text-faint">Mean Spearman correlation</div>
          <div className="t-display mt-2 text-[clamp(4rem,10vw,7rem)] leading-[0.8] text-chalk">
            {fixed(stats.spearman, 3)}
          </div>
          <div className="t-data mt-2 text-xs text-rosso">
            rank correlation, predicted vs actual order
          </div>
        </div>
        <div className="grid grid-cols-3 divide-x divide-line">
          <SupportStat label="NDCG@3" value={fixed(stats.ndcg3, 3)} sub="podium quality" />
          <SupportStat label="Winner acc." value={pct(stats.winner, 0)} sub="top pick correct" />
          <SupportStat label="Avg podium" value={fixed(stats.podium, 2)} sub="hits of 3 / race" />
        </div>
      </Reveal>

      <Reveal>
        <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-1">
          <span className="flex items-center gap-2 t-label !tracking-[0.16em] text-muted">
            <span className="h-[2px] w-5 bg-rosso" /> Spearman (0–1)
          </span>
          <span className="flex items-center gap-2 t-label !tracking-[0.16em] text-faint">
            <span className="h-3 w-2.5 bg-[#3b3b43]" /> Podium hits (0–3)
          </span>
          <span className="flex items-center gap-2 t-label !tracking-[0.16em] text-faint">
            <span className="h-2 w-2 rounded-full bg-rosso" /> Winner called
          </span>
        </div>

        <div className="h-[360px] w-full border border-line bg-surface/30 p-2 pr-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 16, right: 16, bottom: 8, left: 0 }}>
              <defs>
                <linearGradient id="spearFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={ROSSO} stopOpacity={0.32} />
                  <stop offset="100%" stopColor={ROSSO} stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid stroke="#1a1a1f" vertical={false} />

              <XAxis
                dataKey="i"
                interval={tickInterval}
                tick={{ fill: "#5a5a62", fontSize: 10, fontFamily: "var(--font-mono)" }}
                tickFormatter={(v: number) => `R${(rows[v]?.i ?? v) + 1}`}
                axisLine={{ stroke: "#232329" }}
                tickLine={false}
                dy={4}
              />
              {/* Spearman axis (0–1) */}
              <YAxis
                yAxisId="rho"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fill: "#5a5a62", fontSize: 10, fontFamily: "var(--font-mono)" }}
                axisLine={false}
                tickLine={false}
                width={44}
              />
              {/* Hidden podium axis (0–3) so bars share the frame */}
              <YAxis yAxisId="pod" domain={[0, 3]} hide />

              <Tooltip
                cursor={{ stroke: "#33333b", strokeWidth: 1 }}
                content={<ChartTooltip />}
              />

              <ReferenceLine
                yAxisId="rho"
                y={stats.spearman}
                stroke={ROSSO}
                strokeDasharray="3 4"
                strokeOpacity={0.55}
                label={{
                  value: `mean ${fixed(stats.spearman, 2)}`,
                  position: "insideTopRight",
                  fill: "#8a8a92",
                  fontSize: 10,
                  fontFamily: "var(--font-mono)",
                }}
              />

              <Bar
                yAxisId="pod"
                dataKey="podium_hits"
                fill={STEEL}
                barSize={6}
                radius={[1, 1, 0, 0]}
                isAnimationActive
                animationDuration={900}
              />
              <Area
                yAxisId="rho"
                dataKey="spearman"
                stroke="none"
                fill="url(#spearFill)"
                isAnimationActive
                animationDuration={1100}
              />
              <Line
                yAxisId="rho"
                dataKey="spearman"
                stroke={ROSSO}
                strokeWidth={2.2}
                dot={<TraceDot />}
                isAnimationActive
                animationDuration={1100}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <p className="t-label mt-4 text-faint">
          Each point is one race, in season order. The trace holds well above chance ({fixed(stats.spearman, 2)} mean) across the full run.
        </p>
      </Reveal>
    </Section>
  );
}

function SupportStat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="px-5 first:pl-0">
      <div className="t-label text-faint">{label}</div>
      <div className="t-data mt-2 text-[clamp(1.6rem,3vw,2.4rem)] font-semibold leading-none text-chalk">
        {value}
      </div>
      <div className="t-label mt-2 !tracking-[0.14em] text-faint">{sub}</div>
    </div>
  );
}
