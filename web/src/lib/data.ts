import type { RaceArtifact, SiteIndex } from "./types";

// All data is served statically from /data (copied by scripts/copy-data.mjs).
const base = "/data";

// "no-cache" = always revalidate with the server (ETag) so a fresh deploy is
// picked up immediately; predictions change after every quali/race.
export async function fetchIndex(): Promise<SiteIndex> {
  const res = await fetch(`${base}/index.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`index.json ${res.status}`);
  return res.json();
}

export async function fetchRace(slug: string): Promise<RaceArtifact> {
  const res = await fetch(`${base}/races/${slug}.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`race ${slug} ${res.status}`);
  return res.json();
}

export const MODEL_URL = `${base}/model/browser_model.json`;
