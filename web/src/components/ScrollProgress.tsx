"use client";

import { motion, useScroll, useSpring } from "framer-motion";
import { useEffect, useState } from "react";
import { spring } from "@/lib/motion";

const LINKS = [
  { id: "hero", label: "Podium" },
  { id: "next-race", label: "Next Race" },
  { id: "archive", label: "Archive" },
  { id: "shap", label: "Why" },
  { id: "track-record", label: "Record" },
  { id: "playground", label: "What-If" },
];

export default function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });
  const [solid, setSolid] = useState(false);
  const [activeId, setActiveId] = useState("hero");

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Scroll-spy: mark the section nearest the top of the viewport as active.
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) setActiveId(visible[0].target.id);
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: [0, 0.25, 0.5] },
    );
    LINKS.forEach((l) => {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  function go(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <header
      className={`fixed inset-x-0 top-0 z-40 transition-colors duration-300 ${
        solid ? "border-b border-line bg-ink/75 backdrop-blur-xl" : "border-b border-transparent bg-transparent"
      }`}
    >
      <div className="mx-auto flex max-w-[1180px] items-center justify-between px-5 py-3.5 md:px-8">
        <button onClick={() => go("hero")} className="group flex items-center gap-2.5">
          <span className="grid h-6 w-6 place-items-center rounded-[3px] bg-rosso text-[11px] font-black text-white shadow-glow">
            SP
          </span>
          <span className="font-text text-[15px] font-bold tracking-tight text-chalk">
            Scuderia<span className="text-rosso">Predict</span>
          </span>
        </button>
        <nav className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => {
            const active = activeId === l.id;
            return (
              <motion.button
                key={l.id}
                whileTap={{ scale: 0.95 }}
                transition={spring}
                onClick={() => go(l.id)}
                aria-current={active ? "true" : undefined}
                className={`relative rounded-[3px] px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.12em] transition-colors ${
                  active ? "text-chalk" : "text-faint hover:text-muted"
                }`}
              >
                {l.label}
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-x-2 -bottom-0.5 h-px bg-rosso"
                    transition={spring}
                  />
                )}
              </motion.button>
            );
          })}
        </nav>
      </div>
      <motion.div
        style={{ scaleX, transformOrigin: "0%" }}
        className="h-px bg-gradient-to-r from-rosso via-rosso to-transparent"
      />
    </header>
  );
}
