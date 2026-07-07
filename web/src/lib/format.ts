export function pct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function fixed(x: number, digits = 2): string {
  return x.toFixed(digits);
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

export function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

// Color scale for |delta| between predicted and actual finish.
export function deltaColor(absDelta: number): string {
  if (absDelta <= 3) return "#2ec16b";
  if (absDelta <= 6) return "#f4b740";
  if (absDelta <= 9) return "#f4772e";
  return "#dc0000";
}

export function podiumColor(pos: number): string | null {
  if (pos === 1) return "#FFD24A";
  if (pos === 2) return "#C9D1D9";
  if (pos === 3) return "#CD7F32";
  return null;
}
