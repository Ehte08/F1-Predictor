"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import { spring } from "@/lib/motion";
import { forwardRef } from "react";

type Props = HTMLMotionProps<"button"> & {
  active?: boolean;
  variant?: "solid" | "ghost";
};

const PressButton = forwardRef<HTMLButtonElement, Props>(function PressButton(
  { children, className = "", active = false, variant = "ghost", ...rest },
  ref,
) {
  const base =
    "select-none rounded-[3px] px-3.5 py-2 font-mono text-[11px] font-medium uppercase tracking-[0.14em] transition-[color,background-color,box-shadow] duration-200 ease-quart";
  const skin =
    variant === "solid"
      ? "bg-rosso text-white shadow-glow hover:bg-rossoDark"
      : active
        ? "bg-rosso/[0.14] text-chalk ring-1 ring-rosso/60"
        : "bg-white/[0.02] text-muted ring-1 ring-line hover:text-chalk hover:ring-lineBright";
  return (
    <motion.button
      ref={ref}
      whileTap={{ scale: 0.96 }}
      transition={spring}
      className={`${base} ${skin} ${className}`}
      {...rest}
    >
      {children}
    </motion.button>
  );
});

export default PressButton;
