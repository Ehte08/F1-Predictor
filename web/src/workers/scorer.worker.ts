/// <reference lib="webworker" />
// Lazy-loads the 16 MB browser model on first `init`, then scores what-if grids
// entirely off the main thread.
import {
  buildFeatureRows,
  scoreBooster,
  dnfProbability,
  simulateRace,
  rankAscending,
} from "./lgbm";
import type {
  BrowserModel,
  PlaygroundResultRow,
  WorkerRequest,
  WorkerResponse,
} from "@/lib/types";

let model: BrowserModel | null = null;
let loading: Promise<BrowserModel> | null = null;

const MODEL_URL = "/data/model/browser_model.json";

const post = (msg: WorkerResponse) => (self as unknown as Worker).postMessage(msg);

async function loadModel(): Promise<BrowserModel> {
  if (model) return model;
  if (loading) return loading;
  loading = (async () => {
    // no-cache = ETag revalidation; 304 when unchanged, fresh model after a retrain deploy
    const res = await fetch(MODEL_URL, { cache: "no-cache" });
    if (!res.ok || !res.body) throw new Error(`model fetch ${res.status}`);
    const total = Number(res.headers.get("Content-Length") ?? 0);
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (total) post({ type: "progress", loaded, total });
    }
    const blob = new Blob(chunks as BlobPart[]);
    const text = await blob.text();
    model = JSON.parse(text) as BrowserModel;
    return model;
  })();
  return loading;
}

async function handleScore(input: import("@/lib/types").PlaygroundInput) {
  const m = await loadModel();
  const rows = buildFeatureRows(input.grid, m.snapshot, {
    year: input.year,
    circuitName: input.circuitName,
    rainfall: input.rainfall,
    avg_track_temp: input.avg_track_temp,
    min_humidity: input.min_humidity,
  });

  const scores = rows.map((r) => scoreBooster(m.booster, r, m.categorical_features));
  const pDnf = rows.map((r) => dnfProbability(m.dnf_booster, r, m.categorical_features));

  const sim = simulateRace(scores, pDnf, m.pl.tau, input.nSims, 7);
  const ranks = rankAscending(sim.expected_pos);

  const out: PlaygroundResultRow[] = input.grid.map((g, i) => ({
    driver: g.driver,
    team: g.team,
    start: g.start,
    score: scores[i],
    p_dnf: sim.p_dnf[i],
    p_win: sim.p_win[i],
    p_podium: sim.p_podium[i],
    p_points: sim.p_points[i],
    expected_pos: sim.expected_pos[i],
    pred_finish: ranks[i],
  }));
  out.sort((a, b) => a.pred_finish - b.pred_finish);
  post({ type: "result", rows: out });
}

self.addEventListener("message", async (e: MessageEvent<WorkerRequest>) => {
  try {
    const msg = e.data;
    if (msg.type === "init") {
      await loadModel();
      post({ type: "ready" });
    } else if (msg.type === "score") {
      await handleScore(msg.input);
    }
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : String(err) });
  }
});
