"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import type { RaceArtifact, SiteIndex } from "@/lib/types";
import { fetchIndex, fetchRace } from "@/lib/data";
import ScrollProgress from "@/components/ScrollProgress";
import Hero from "@/components/hero/Hero";
import NextRace from "@/components/NextRace";
import Archive from "@/components/Archive";
import Shap from "@/components/Shap";
import TrackRecord from "@/components/TrackRecord";

// Code-split the playground (worker glue + model loader) out of the main bundle.
const Playground = dynamic(() => import("@/components/playground/Playground"), {
  ssr: false,
});

export default function Home() {
  const [index, setIndex] = useState<SiteIndex | null>(null);
  const [featured, setFeatured] = useState<RaceArtifact | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const idx = await fetchIndex();
        setIndex(idx);
        const slug =
          idx.next_race?.slug ?? idx.races[idx.races.length - 1]?.slug;
        if (slug) setFeatured(await fetchRace(slug));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  if (err) {
    return (
      <main className="grid min-h-[100dvh] place-items-center px-6 text-center">
        <div>
          <div className="t-display text-4xl text-rosso">Data unavailable</div>
          <p className="t-data mt-3 text-sm text-muted">{err}</p>
        </div>
      </main>
    );
  }

  if (!index || !featured) {
    return (
      <main className="grid min-h-[100dvh] place-items-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-5"
        >
          <div className="h-9 w-9 animate-spin rounded-full border-2 border-line border-t-rosso" />
          <span className="t-label text-muted">Warming up the grid</span>
        </motion.div>
      </main>
    );
  }

  return (
    <>
      <ScrollProgress />
      <main className="relative">
        <Hero
          race={featured}
          modelVersion={index.model_version}
          trainedThrough={featured.race_date}
        />
        <NextRace race={featured} />
        <Archive races={index.races} />
        <Shap race={featured} />
        <TrackRecord data={index.track_record} />
        <Playground races={index.races} />

        <footer className="border-t border-line">
          <div className="mx-auto grid max-w-[1180px] gap-6 px-5 py-14 md:grid-cols-[1fr_auto] md:items-end md:px-8">
            <div>
              <div className="font-text text-lg font-bold tracking-tight text-chalk">
                Scuderia<span className="text-rosso">Predict</span>
              </div>
              <p className="t-label mt-2 max-w-md !tracking-[0.14em] text-faint">
                Learning-to-rank finishing-order model · not affiliated with Formula 1 or Scuderia Ferrari
              </p>
            </div>
            <p className="t-data text-[11px] leading-relaxed text-faint md:text-right">
              LightGBM · Plackett-Luce · model v{index.model_version}
              <br />
              updated {new Date(index.updated_at).toLocaleDateString("en-GB")}
            </p>
          </div>
        </footer>
      </main>
    </>
  );
}
