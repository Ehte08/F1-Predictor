"use client";

import { useEffect, useState } from "react";

function diff(target: number) {
  const now = Date.now();
  const d = Math.max(0, target - now);
  return {
    past: target < now,
    days: Math.floor(d / 86400000),
    hours: Math.floor((d % 86400000) / 3600000),
    mins: Math.floor((d % 3600000) / 60000),
    secs: Math.floor((d % 60000) / 1000),
  };
}

function Unit({ v, label, live }: { v: number; label: string; live?: boolean }) {
  return (
    <div className="flex flex-col">
      <span
        className={`t-data text-[clamp(2rem,4vw,2.75rem)] font-semibold leading-none tracking-tight ${
          live ? "text-rosso" : "text-chalk"
        }`}
      >
        {String(v).padStart(2, "0")}
      </span>
      <span className="t-label mt-2 text-faint">{label}</span>
    </div>
  );
}

export default function Countdown({ date }: { date: string }) {
  const target = new Date(date + "T13:00:00Z").getTime();
  const [t, setT] = useState(() => diff(target));

  useEffect(() => {
    const id = setInterval(() => setT(diff(target)), 1000);
    return () => clearInterval(id);
  }, [target]);

  if (t.past) {
    return (
      <div className="inline-flex items-center gap-3">
        <span className="h-2 w-2 rounded-full bg-rosso shadow-glow" />
        <span className="t-label !tracking-[0.28em] text-muted">
          Race complete — graded against result
        </span>
      </div>
    );
  }

  return (
    <div>
      <span className="t-label text-faint">Lights out in</span>
      <div className="mt-2 flex items-end gap-5">
        <Unit v={t.days} label="Days" />
        <span className="mb-6 h-8 w-px bg-line" />
        <Unit v={t.hours} label="Hrs" />
        <Unit v={t.mins} label="Min" />
        <Unit v={t.secs} label="Sec" live />
      </div>
    </div>
  );
}
