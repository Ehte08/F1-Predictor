"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { RaceArtifact, RaceIndexEntry, ActualResult } from "@/lib/types";
import { fetchRace } from "@/lib/data";
import { teamColor } from "@/lib/teams";
import { deltaColor, podiumColor, ordinal, fixed } from "@/lib/format";
import Section from "./ui/Section";
import Select, { type Option } from "./ui/Select";
import DriverAvatar from "./ui/DriverAvatar";

function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase();
  if (s === "finished") return null;
  const label = s === "dnf" ? "DNF" : s === "dsq" ? "DSQ" : status.toUpperCase();
  return (
    <span className="rounded-[2px] bg-rosso/15 px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider text-rosso ring-1 ring-rosso/40">
      {label}
    </span>
  );
}

export default function Archive({ races }: { races: RaceIndexEntry[] }) {
  const withActual = useMemo(() => races.filter((r) => r.has_actual), [races]);
  // Newest race first in the dropdown
  const options: Option[] = [...races].reverse().map((r) => ({
    value: r.slug,
    label: r.race_name,
    hint: `${r.year} · R${r.round}`,
  }));

  const [slug, setSlug] = useState(
    () => (withActual.length ? withActual[withActual.length - 1].slug : races[0]?.slug) ?? "",
  );
  const [race, setRace] = useState<RaceArtifact | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchRace(slug)
      .then((r) => {
        if (!cancelled) setRace(r);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [slug]);

  const actualMap = useMemo(() => {
    const m = new Map<string, ActualResult>();
    race?.actual?.forEach((a) => m.set(a.driver, a));
    return m;
  }, [race]);

  return (
    <Section
      id="archive"
      sector={2}
      channel="ARCHIVE"
      title="Every race, scored"
      lede="Pick any Grand Prix from 2025 through the 2026 British GP. Where results exist we grade the model against reality — Δ is the gap between predicted and actual finish."
    >
      <div className="mb-8 max-w-md">
        <Select value={slug} options={options} onChange={setSlug} ariaLabel="Select a race" />
      </div>

      <div className="border-t border-line">
        <AnimatePresence mode="wait">
          <motion.div
            key={slug}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.26, ease: [0.16, 1, 0.3, 1] }}
          >
            {race && (
              <>
                <div className="flex flex-wrap items-end justify-between gap-4 py-6">
                  <div>
                    <div className="t-display text-[clamp(1.8rem,4vw,2.8rem)] text-chalk">
                      {race.race_name}
                    </div>
                    <div className="t-data mt-1 text-xs text-faint">{race.circuit}</div>
                  </div>
                  {race.metrics && (
                    <div className="flex flex-wrap gap-x-8 gap-y-2">
                      <Metric label="Spearman" value={fixed(race.metrics.spearman, 3)} />
                      <Metric label="NDCG@3" value={fixed(race.metrics.ndcg3, 3)} />
                      <Metric label="Podium" value={`${race.metrics.podium_hits}/3`} />
                      <Metric
                        label="Winner"
                        value={race.metrics.winner_correct ? "HIT" : "MISS"}
                        good={race.metrics.winner_correct}
                      />
                    </div>
                  )}
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full min-w-[560px] border-collapse text-sm">
                    <thead>
                      <tr className="t-label border-y border-line text-faint [&>th]:px-2 [&>th]:py-2.5 [&>th]:font-normal">
                        <th className="text-left">Pred</th>
                        <th className="text-left">Driver</th>
                        <th className="text-left">Team</th>
                        <th className="text-right">Start</th>
                        <th className="text-right">Actual</th>
                        <th className="text-right">Δ</th>
                      </tr>
                    </thead>
                    <tbody>
                      {race.predictions.map((p) => {
                        const act = actualMap.get(p.driver);
                        const pod = podiumColor(p.pred_finish);
                        const delta =
                          act && act.status.toLowerCase() === "finished"
                            ? act.finish - p.pred_finish
                            : null;
                        const dColor = delta !== null ? deltaColor(Math.abs(delta)) : "#3a3a40";
                        return (
                          <tr
                            key={p.driver}
                            className="border-b border-line/50 transition-colors hover:bg-white/[0.015]"
                          >
                            <td className="px-2 py-2.5">
                              <span
                                className="t-data inline-grid h-6 w-6 place-items-center rounded-[2px] text-xs font-bold"
                                style={{
                                  background: pod ? pod : "transparent",
                                  color: pod ? "#0a0a0a" : "#8a8a92",
                                  border: pod ? "none" : "1px solid #232329",
                                }}
                              >
                                {p.pred_finish}
                              </span>
                            </td>
                            <td className="px-2 py-2.5">
                              <div className="flex items-center gap-2.5">
                                <DriverAvatar driver={p.driver} team={p.team} size={24} />
                                <span className="font-text text-[14px] font-semibold tracking-tight text-chalk">
                                  {p.driver}
                                </span>
                                {act && <StatusBadge status={act.status} />}
                              </div>
                            </td>
                            <td className="px-2 py-2.5">
                              <span className="flex items-center gap-1.5">
                                <span
                                  className="h-2.5 w-0.5"
                                  style={{ background: teamColor(p.team) }}
                                />
                                <span className="font-mono text-[9.5px] uppercase tracking-[0.12em] text-faint">
                                  {p.team}
                                </span>
                              </span>
                            </td>
                            <td className="t-data px-2 py-2.5 text-right text-muted">{p.start}</td>
                            <td className="t-data px-2 py-2.5 text-right text-chalk">
                              {act
                                ? act.status.toLowerCase() === "finished"
                                  ? ordinal(act.finish)
                                  : "—"
                                : "—"}
                            </td>
                            <td className="px-2 py-2.5 text-right">
                              {delta !== null ? (
                                <span
                                  className="t-data rounded-[2px] px-2 py-0.5 text-xs font-semibold"
                                  style={{ background: `${dColor}1f`, color: dColor }}
                                >
                                  {delta > 0 ? "+" : ""}
                                  {delta}
                                </span>
                              ) : (
                                <span className="text-faint">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {!race.actual && (
                  <p className="t-label mt-4 text-faint">
                    No official result recorded for this race — predictions only.
                  </p>
                )}
              </>
            )}
            {loading && !race && (
              <div className="grid h-40 place-items-center">
                <div className="h-7 w-7 animate-spin rounded-full border-2 border-line border-t-rosso" />
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </Section>
  );
}

function Metric({ label, value, good }: { label: string; value: string; good?: boolean }) {
  return (
    <div className="text-right">
      <div className="t-label text-faint">{label}</div>
      <div
        className={`t-data mt-1 text-[17px] font-semibold ${
          good === undefined ? "text-chalk" : good ? "text-[#2ec16b]" : "text-rosso"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
