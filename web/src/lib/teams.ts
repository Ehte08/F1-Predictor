// Team colors — 2026 grid plus legacy teams that appear in 2025 artifacts.
export const TEAM_COLORS: Record<string, string> = {
  Mercedes: "#27F4D2",
  McLaren: "#FF8000",
  Ferrari: "#DC0000",
  "Red Bull": "#3671C6",
  "Aston Martin": "#358C75",
  Alpine: "#FF87BC",
  Williams: "#64C4FF",
  "Racing Bulls": "#6692FF",
  "Haas F1 Team": "#B6BABD",
  Audi: "#E8002D",
  Cadillac: "#003087",
  // legacy / 2025 names
  "Kick Sauber": "#52E252",
  Sauber: "#52E252",
  RB: "#6692FF",
  AlphaTauri: "#6692FF",
  Renault: "#FFF500",
};

export function teamColor(team: string): string {
  return TEAM_COLORS[team] ?? "#9aa0a6";
}

// Readable text color on top of a team swatch.
export function onTeamColor(team: string): string {
  const light = new Set([
    "Mercedes",
    "Williams",
    "Haas F1 Team",
    "Racing Bulls",
    "RB",
    "AlphaTauri",
    "Kick Sauber",
    "Sauber",
    "Renault",
  ]);
  return light.has(team) ? "#0a0a0a" : "#f4f4f5";
}
