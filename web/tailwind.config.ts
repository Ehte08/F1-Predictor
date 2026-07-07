import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        rosso: "#DC0000",
        rossoDark: "#9E0000",
        ink: "#0a0a0a",
        ink2: "#0e0e11",
        surface: "#131317",
        carbon: "#141416",
        line: "#232329",
        lineBright: "#33333b",
        chalk: "#f4f4f5",
        muted: "#8a8a92",
        faint: "#5a5a62",
        steel: "#6b7280",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        text: ["var(--font-text)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        glow: "0 0 32px -10px rgba(220,0,0,0.6)",
        card: "0 24px 70px -34px rgba(0,0,0,0.92)",
        lift: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 20px 50px -30px rgba(0,0,0,0.9)",
      },
      transitionTimingFunction: {
        quart: "cubic-bezier(0.25, 1, 0.5, 1)",
        expo: "cubic-bezier(0.16, 1, 0.3, 1)",
        drawer: "cubic-bezier(0.32, 0.72, 0, 1)",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "sweep-in": {
          "0%": { transform: "scaleX(0)" },
          "100%": { transform: "scaleX(1)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};

export default config;
