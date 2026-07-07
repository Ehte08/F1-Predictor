"use client";

import { motion } from "framer-motion";
import { enterBlur } from "@/lib/motion";

// Scroll-into-view blur/translate reveal. Respects reduced motion via framer's
// global reducer (users with prefers-reduced-motion get instant states).
export default function Reveal({
  children,
  className,
  delay = 0,
  once = true,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  once?: boolean;
}) {
  return (
    <motion.div
      className={className}
      variants={enterBlur}
      initial="hidden"
      whileInView="show"
      viewport={{ once, amount: 0.25 }}
      transition={{ delay }}
    >
      {children}
    </motion.div>
  );
}
