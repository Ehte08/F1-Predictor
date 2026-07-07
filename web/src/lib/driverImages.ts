// Driver image mapping. Real PNGs live in web/public/drivers/<slug>.png;
// drivers without an entry fall back to a generated initials-disc avatar
// (see `driverAvatarDataUri`).
import { teamColor } from "./teams";

export const DRIVER_IMAGES: Record<string, string> = {
  Leclerc: "/drivers/leclerc.png",
  Hamilton: "/drivers/hamilton.png",
  Antonelli: "/drivers/antonelli.png",
  Russell: "/drivers/russell.png",
  Norris: "/drivers/norris.png",
  Piastri: "/drivers/piastri.png",
};

export function driverImage(driver: string): string | null {
  return DRIVER_IMAGES[driver] ?? null;
}

export function driverInitials(driver: string): string {
  const clean = driver.trim();
  if (!clean) return "?";
  // Surname-only data -> first three letters, uppercase (e.g. VER, HAM).
  return clean.slice(0, 3).toUpperCase();
}

// Placeholder driver plate — a squared timing-tower tag: dark plate, team-colour
// spine on the left, three-letter code set in a technical face. Reads as an F1
// broadcast driver entry rather than a generic avatar disc. Returned as a data
// URI (zero network cost). Real PNG headshots override this via DRIVER_IMAGES.
export function driverAvatarDataUri(driver: string, team: string): string {
  const bar = teamColor(team);
  const initials = driverInitials(driver);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">
  <rect width="240" height="240" fill="#131317"/>
  <rect width="240" height="240" fill="none" stroke="#2a2a30" stroke-width="4"/>
  <rect x="0" y="0" width="26" height="240" fill="${bar}"/>
  <text x="140" y="150" text-anchor="middle" font-family="'Arial Narrow', Arial, sans-serif" font-size="94" font-weight="800" letter-spacing="-2" fill="#f4f4f5">${initials}</text>
</svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}
