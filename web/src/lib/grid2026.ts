import type { PlaygroundGridEntry } from "./types";

// 2026 grid defaults for the what-if playground (driver -> team), in a
// plausible baseline grid order. Users can reshuffle starts in the editor.
export const GRID_2026: PlaygroundGridEntry[] = [
  { driver: "Norris", team: "McLaren", start: 1 },
  { driver: "Piastri", team: "McLaren", start: 2 },
  { driver: "Verstappen", team: "Red Bull", start: 3 },
  { driver: "Leclerc", team: "Ferrari", start: 4 },
  { driver: "Hamilton", team: "Ferrari", start: 5 },
  { driver: "Russell", team: "Mercedes", start: 6 },
  { driver: "Antonelli", team: "Mercedes", start: 7 },
  { driver: "Hadjar", team: "Red Bull", start: 8 },
  { driver: "Alonso", team: "Aston Martin", start: 9 },
  { driver: "Stroll", team: "Aston Martin", start: 10 },
  { driver: "Gasly", team: "Alpine", start: 11 },
  { driver: "Colapinto", team: "Alpine", start: 12 },
  { driver: "Sainz", team: "Williams", start: 13 },
  { driver: "Albon", team: "Williams", start: 14 },
  { driver: "Lawson", team: "Racing Bulls", start: 15 },
  { driver: "Lindblad", team: "Racing Bulls", start: 16 },
  { driver: "Bearman", team: "Haas F1 Team", start: 17 },
  { driver: "Ocon", team: "Haas F1 Team", start: 18 },
  { driver: "Hulkenberg", team: "Audi", start: 19 },
  { driver: "Bortoleto", team: "Audi", start: 20 },
  { driver: "Perez", team: "Cadillac", start: 21 },
  { driver: "Bottas", team: "Cadillac", start: 22 },
];

export const TEAM_DRIVERS_2026: Record<string, string[]> = {
  Mercedes: ["Antonelli", "Russell"],
  McLaren: ["Norris", "Piastri"],
  Ferrari: ["Leclerc", "Hamilton"],
  "Red Bull": ["Verstappen", "Hadjar"],
  "Aston Martin": ["Alonso", "Stroll"],
  Alpine: ["Gasly", "Colapinto"],
  Williams: ["Sainz", "Albon"],
  "Racing Bulls": ["Lawson", "Lindblad"],
  "Haas F1 Team": ["Bearman", "Ocon"],
  Audi: ["Hulkenberg", "Bortoleto"],
  Cadillac: ["Perez", "Bottas"],
};
