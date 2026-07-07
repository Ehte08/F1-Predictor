"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { spring } from "@/lib/motion";
import { Chevron } from "./icons";

export interface Option {
  value: string;
  label: string;
  hint?: string;
}

// Fully custom, theme-matched select (never native). Keyboard + click-away.
export default function Select({
  value,
  options,
  onChange,
  ariaLabel,
  className = "",
}: {
  value: string;
  options: Option[];
  onChange: (v: string) => void;
  ariaLabel: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o.value === value);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  useEffect(() => {
    if (open) {
      const idx = options.findIndex((o) => o.value === value);
      setActive(idx < 0 ? 0 : idx);
    }
  }, [open, value, options]);

  function onKey(e: React.KeyboardEvent) {
    if (!open && (e.key === "Enter" || e.key === " " || e.key === "ArrowDown")) {
      e.preventDefault();
      setOpen(true);
      return;
    }
    if (!open) return;
    if (e.key === "Escape") setOpen(false);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, options.length - 1));
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    }
    if (e.key === "Enter") {
      e.preventDefault();
      onChange(options[active].value);
      setOpen(false);
    }
  }

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onKeyDown={onKey}
        onClick={() => setOpen((o) => !o)}
        className="no-native-select flex w-full items-center justify-between gap-3 rounded-[3px] border border-line bg-white/[0.02] px-4 py-3 text-left transition-[border-color,background-color] duration-200 ease-quart hover:border-lineBright hover:bg-white/[0.04]"
      >
        <span className="truncate">
          <span className="font-text text-[15px] font-semibold leading-none tracking-tight text-chalk">
            {current?.label ?? "Select"}
          </span>
          {current?.hint && (
            <span className="t-data ml-2 text-xs text-muted">{current.hint}</span>
          )}
        </span>
        <motion.span
          animate={{ rotate: open ? 180 : 0 }}
          transition={spring}
          className="text-rosso"
        >
          <Chevron size={13} />
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: "top center" }}
            className="absolute z-50 mt-2 max-h-72 w-full overflow-auto rounded-[3px] border border-line bg-ink2/95 p-1.5 shadow-card backdrop-blur-xl"
          >
            {options.map((o, i) => {
              const selected = o.value === value;
              return (
                <li key={o.value} role="option" aria-selected={selected}>
                  <button
                    type="button"
                    onMouseEnter={() => setActive(i)}
                    onClick={() => {
                      onChange(o.value);
                      setOpen(false);
                    }}
                    className={`flex w-full items-center justify-between gap-3 rounded-[2px] px-3 py-2 text-left text-sm transition-colors ${
                      i === active ? "bg-rosso/15 text-chalk" : "bg-transparent"
                    } ${selected ? "text-chalk" : "text-muted hover:text-chalk"}`}
                  >
                    <span className="truncate">{o.label}</span>
                    {o.hint && <span className="t-data text-xs opacity-70">{o.hint}</span>}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
