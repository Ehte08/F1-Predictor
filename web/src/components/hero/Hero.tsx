"use client";

import dynamic from "next/dynamic";
import { useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { RaceArtifact } from "@/lib/types";
import { formatDate } from "@/lib/format";
import { EASE_OUT_EXPO } from "@/lib/motion";
import Countdown from "./Countdown";

// Code-split the 3D bundle (three/r3f/drei) out of the initial load.
const Podium3D = dynamic(() => import("./Podium3D"), {
  ssr: false,
  loading: () => (
    <div className="grid h-full w-full place-items-center">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-rosso" />
    </div>
  ),
});

export default function Hero({
  race,
  modelVersion,
  trainedThrough,
}: {
  race: RaceArtifact;
  modelVersion: string;
  trainedThrough: string;
}) {
  const top3 = race.predictions.slice(0, 3);
  const rootRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    gsap.registerPlugin(ScrollTrigger);
    const ctx = gsap.context(() => {
      // Copy leaves faster than the podium — a parallax depth cue on exit.
      gsap.to(contentRef.current, {
        yPercent: -22,
        opacity: 0,
        ease: "none",
        scrollTrigger: {
          trigger: rootRef.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });
      gsap.to(canvasWrapRef.current, {
        yPercent: 10,
        ease: "none",
        scrollTrigger: {
          trigger: rootRef.current,
          start: "top top",
          end: "bottom top",
          scrub: true,
        },
      });
    }, rootRef);
    return () => ctx.revert();
  }, []);

  return (
    <section
      id="hero"
      ref={rootRef}
      className="vignette grain relative min-h-[100dvh] w-full overflow-hidden"
    >
      <div className="carbon-weave absolute inset-0 opacity-30" />
      <div className="telemetry-grid absolute inset-0 opacity-70" />

      {/* 3D podium — constrained to the RIGHT half on desktop so floating driver
          cards can never collide with the left-aligned headline block. */}
      <div
        ref={canvasWrapRef}
        className="absolute inset-x-0 bottom-0 top-[46%] md:inset-y-0 md:left-[44%] md:right-0 md:top-[6%]"
      >
        <Podium3D top3={top3} />
      </div>

      {/* Left scrim keeps the headline legible over the 3D scene. */}
      <div
        className="pointer-events-none absolute inset-0 md:w-[62%]"
        style={{
          background:
            "linear-gradient(100deg, rgba(10,10,10,0.96) 30%, rgba(10,10,10,0.6) 62%, transparent 88%)",
        }}
      />

      <div
        ref={contentRef}
        className="relative z-10 mx-auto flex min-h-[100dvh] max-w-[1180px] flex-col justify-between px-5 pb-12 pt-28 md:px-8 md:pt-32"
      >
        <div className="max-w-2xl">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            className="mb-5 flex items-center gap-3"
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rosso opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-rosso" />
            </span>
            <span className="t-label !text-rosso">Predicted podium</span>
            <span className="h-px w-10 bg-line" />
            <span className="t-data text-[11px] text-faint">
              R{String(race.round).padStart(2, "0")} · {race.year}
            </span>
          </motion.div>

          {/* Headline: word-by-word rise. Anton, tight. */}
          <h1 className="t-display text-[clamp(3.2rem,9vw,7rem)] text-chalk">
            {race.race_name.split(" ").map((word, i) => (
              <span key={i} className="block overflow-hidden">
                <motion.span
                  className="inline-block"
                  initial={{ y: "112%" }}
                  animate={{ y: "0%" }}
                  transition={{
                    duration: 0.9,
                    delay: 0.1 + i * 0.09,
                    ease: EASE_OUT_EXPO,
                  }}
                >
                  {word}
                </motion.span>
              </span>
            ))}
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.7, ease: EASE_OUT_EXPO }}
            className="mt-6 max-w-md text-[15px] leading-relaxed text-muted"
          >
            A LightGBM learning-to-rank model, resolved into a full finishing order
            by Plackett-Luce Monte-Carlo simulation. {race.circuit}.
          </motion.p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.7, ease: EASE_OUT_EXPO }}
          className="flex flex-col gap-6"
        >
          <Countdown date={race.race_date} />
          <div className="flex flex-col gap-2 border-t border-line pt-4 md:flex-row md:items-center md:gap-0">
            <MetaCell k="Race date" v={formatDate(race.race_date)} />
            <span className="hidden h-8 w-px bg-line md:mx-6 md:block" />
            <MetaCell k="Trained through" v={formatDate(trainedThrough)} />
            <span className="hidden h-8 w-px bg-line md:mx-6 md:block" />
            <MetaCell k="Model" v={`v${modelVersion}`} accent />
          </div>
        </motion.div>
      </div>

      <div className="pointer-events-none absolute bottom-6 right-6 z-10 hidden items-center gap-3 md:flex">
        <span className="t-label text-faint">Scroll</span>
        <span className="relative block h-10 w-px overflow-hidden bg-line">
          <motion.span
            className="absolute inset-x-0 top-0 h-4 bg-rosso"
            animate={{ y: [-16, 40] }}
            transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          />
        </span>
      </div>
    </section>
  );
}

function MetaCell({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="t-label text-faint">{k}</span>
      <span className={`t-data text-[13px] ${accent ? "text-rosso" : "text-chalk"}`}>
        {v}
      </span>
    </div>
  );
}
