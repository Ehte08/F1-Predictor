import type { Variants, Transition } from "framer-motion";

// Emil-style strong easing curves, mirrored from globals.css so JS + CSS agree.
export const EASE_OUT_QUART = [0.25, 1, 0.5, 1] as const;
export const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;
export const EASE_IO = [0.65, 0, 0.35, 1] as const;

// Snappy press spring — for buttons and interactive feedback.
export const spring: Transition = {
  type: "spring",
  stiffness: 420,
  damping: 32,
  mass: 0.7,
};

// Softer settle — for layout re-order in the playground.
export const layoutSpring: Transition = {
  type: "spring",
  stiffness: 260,
  damping: 30,
};

// Reveal: blur + rise. Duration/ease deliberately distinct from the press
// spring so entrances and interactions never feel like the same gesture.
export const enterBlur: Variants = {
  hidden: { opacity: 0, y: 16, filter: "blur(5px)" },
  show: {
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: { duration: 0.7, ease: EASE_OUT_EXPO },
  },
  exit: {
    opacity: 0,
    y: -10,
    filter: "blur(4px)",
    transition: { duration: 0.22, ease: EASE_OUT_QUART },
  },
};

// Faster, no-blur rise for dense lists where blur would smear numbers.
export const riseItem: Variants = {
  hidden: { opacity: 0, y: 10 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: EASE_OUT_QUART },
  },
};

export const staggerParent: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05, delayChildren: 0.04 } },
};

export const pressProps = {
  whileTap: { scale: 0.97 },
  transition: spring,
};
