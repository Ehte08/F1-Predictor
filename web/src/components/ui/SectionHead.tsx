"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

/**
 * Editorial section header. The site is framed as one timed lap through six
 * sectors (S1..S6) — the index is a real sequence, not a per-section eyebrow.
 *
 * Motion: the top rule draws left→right scrubbed to scroll (a sector-timing
 * sweep); the title + channel resolve on enter with a strong ease-out. Both
 * are GSAP-choreographed and reduced-motion aware.
 */
export default function SectionHead({
  sector,
  channel,
  title,
  lede,
  aside,
}: {
  sector: number;
  channel: string;
  title: React.ReactNode;
  lede?: React.ReactNode;
  aside?: React.ReactNode;
}) {
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    gsap.registerPlugin(ScrollTrigger);
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".sh-rule",
        { scaleX: 0 },
        {
          scaleX: 1,
          ease: "none",
          scrollTrigger: {
            trigger: root.current,
            start: "top 92%",
            end: "top 58%",
            scrub: 0.6,
          },
        },
      );
      gsap.from(".sh-rise", {
        yPercent: 108,
        duration: 0.9,
        ease: "expo.out",
        stagger: 0.08,
        scrollTrigger: { trigger: root.current, start: "top 80%" },
      });
      gsap.from([".sh-meta", ".sh-lede"], {
        opacity: 0,
        y: 12,
        duration: 0.7,
        ease: "power3.out",
        stagger: 0.06,
        delay: 0.1,
        scrollTrigger: { trigger: root.current, start: "top 80%" },
      });
    }, root);
    return () => ctx.revert();
  }, []);

  return (
    <header ref={root} className="mb-10 md:mb-14">
      <div className="sh-rule rule mb-5 origin-left" />
      <div className="grid grid-cols-1 gap-x-10 gap-y-4 md:grid-cols-12">
        <div className="sh-meta flex items-center gap-3 md:col-span-3 md:flex-col md:items-start md:gap-2">
          <span className="t-data text-[13px] leading-none text-rosso">
            S<span className="text-chalk">{sector}</span>
            <span className="text-faint">/6</span>
          </span>
          <span className="t-label text-faint">{channel}</span>
        </div>

        <div className="md:col-span-6">
          <h2 className="t-display overflow-hidden text-[clamp(2.4rem,6vw,4.6rem)] text-chalk">
            <span className="sh-rise inline-block">{title}</span>
          </h2>
        </div>

        {(lede || aside) && (
          <div className="sh-lede md:col-span-3 md:pt-2">
            {lede && (
              <p className="max-w-[42ch] text-[13.5px] leading-relaxed text-muted">
                {lede}
              </p>
            )}
            {aside && <div className="mt-3">{aside}</div>}
          </div>
        )}
      </div>
    </header>
  );
}
