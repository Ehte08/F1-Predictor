// Pure-TS LightGBM `dump_model` scorer + feature builder + Plackett-Luce sim.
// No DOM/worker globals here so it can be unit-tested under Node (vitest).
//
// Validated to float precision (<1e-14) against Python lightgbm on the browser
// model booster — see lgbm.test.ts.

import type {
  Booster,
  BrowserModel,
  ModelSnapshot,
  PlaygroundGridEntry,
  TreeNode,
} from "@/lib/types";

export type FeatureRow = Record<string, number | string>;

const ELO_INIT = 1500.0;

// Map an input value to its category index. LightGBM `dump_model` categorical
// thresholds are indices into the training category list. Values may arrive as
// strings ("Ferrari") or numbers (0.5 sentinel for GP_name/circuit_name), and
// the category list itself can hold either form, so compare loosely.
export function categoryIndex(
  categories: (string | number)[] | undefined,
  val: number | string,
): number {
  if (!categories) return -1;
  for (let i = 0; i < categories.length; i++) {
    const c = categories[i];
    if (c === val) return i;
    if (String(c) === String(val)) return i;
  }
  return -1;
}

// Traverse one tree to its leaf value.
export function scoreTree(
  root: TreeNode,
  row: FeatureRow,
  featureNames: string[],
  categoricalFeatures: Record<string, (string | number)[]>,
): number {
  let node = root;
  while (node.leaf_value === undefined) {
    const fname = featureNames[node.split_feature as number];
    const val = row[fname];
    if (node.decision_type === "==") {
      const idx = categoryIndex(categoricalFeatures[fname], val);
      const thr = String(node.threshold)
        .split("||")
        .map((x) => parseInt(x, 10));
      node = thr.includes(idx)
        ? (node.left_child as TreeNode)
        : (node.right_child as TreeNode);
    } else {
      const missing =
        val === undefined ||
        val === null ||
        (typeof val === "number" && Number.isNaN(val));
      if (missing) {
        node = node.default_left
          ? (node.left_child as TreeNode)
          : (node.right_child as TreeNode);
      } else {
        node =
          (val as number) <= (node.threshold as number)
            ? (node.left_child as TreeNode)
            : (node.right_child as TreeNode);
      }
    }
  }
  return node.leaf_value;
}

export function scoreBooster(
  booster: Booster,
  row: FeatureRow,
  categoricalFeatures: Record<string, (string | number)[]>,
): number {
  const feats = booster.feature_names;
  let sum = 0;
  const trees = booster.tree_info;
  for (let t = 0; t < trees.length; t++) {
    sum += scoreTree(trees[t].tree_structure, row, feats, categoricalFeatures);
  }
  return sum;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function dnfProbability(
  dnfBooster: Booster,
  row: FeatureRow,
  categoricalFeatures: Record<string, (string | number)[]>,
): number {
  return sigmoid(scoreBooster(dnfBooster, row, categoricalFeatures));
}

// ── Feature building — mirrors src/predict/prep.build_race_features ────────
// plus src/site_build.predict_upcoming's neutral (0.5) fill for the
// GP_name / circuit_name categorical columns that inference never populates.
function qualiGapFor(snap: ModelSnapshot, start: number): number {
  const slot = snap.quali_slot_gap ?? {};
  const v = slot[String(start)] ?? slot[start as unknown as string];
  if (v === undefined || v === null) return snap.quali_global_gap ?? 5.0;
  return v;
}

export function buildFeatureRows(
  grid: PlaygroundGridEntry[],
  snap: ModelSnapshot,
  opts: {
    year: number;
    circuitName: string;
    rainfall: number;
    avg_track_temp: number;
    min_humidity: number;
  },
): FeatureRow[] {
  const circuit = snap.circuit_map?.[opts.circuitName] ?? opts.circuitName;

  const gaps = grid.map((g) => qualiGapFor(snap, g.start));

  // teammate averages by team (matches the pandas groupby transform)
  const teamSum: Record<string, number> = {};
  const teamCnt: Record<string, number> = {};
  grid.forEach((g, i) => {
    teamSum[g.team] = (teamSum[g.team] ?? 0) + gaps[i];
    teamCnt[g.team] = (teamCnt[g.team] ?? 0) + 1;
  });

  return grid.map((g, i) => {
    const d = snap.driver_features?.[g.driver];
    const t = snap.team_features?.[g.team];
    const gap = gaps[i];
    const cnt = teamCnt[g.team];
    const teammateAvg =
      cnt > 1 ? (teamSum[g.team] - gap) / (cnt - 1) : gap;

    const row: FeatureRow = {
      year: opts.year,
      GP_name: 0.5, // inference sentinel (never a real category at predict time)
      start: g.start,
      team: g.team,
      driver: g.driver,
      circuit_name: 0.5, // inference sentinel
      age_at_race: snap.driver_ages?.[g.driver] ?? 28,
      driver_active: 1,
      team_active: 1,
      rainfall: opts.rainfall,
      min_humidity: opts.min_humidity,
      avg_track_temp: opts.avg_track_temp,
      driver_reliability_ewm10: d?.reliability ?? 0.9,
      team_reliability_ewm10: t?.reliability ?? 0.9,
      finish_ewm3: d?.finish_ewm3 ?? 10.0,
      driver_circuit_avg: d?.circuit_avgs?.[circuit] ?? 10.0,
      driver_elo: d?.elo ?? ELO_INIT,
      team_pace_ewm5: t?.pace_ewm5 ?? 10.5,
      quali_gap_pct: gap,
      teammate_quali_delta: gap - teammateAvg,
    };
    return row;
  });
}

// ── Plackett-Luce Monte-Carlo — mirrors src/models/simulate.simulate_race ──
function standardize(scores: number[]): number[] {
  const n = scores.length;
  const mean = scores.reduce((a, b) => a + b, 0) / n;
  let variance = 0;
  for (const s of scores) variance += (s - mean) ** 2;
  const sd = Math.sqrt(variance / n);
  if (sd < 1e-9) return scores.map((s) => s - mean);
  return scores.map((s) => (s - mean) / sd);
}

// Gumbel(0,1) via inverse-CDF of a uniform.
function gumbel(rng: () => number): number {
  let u = rng();
  if (u <= 0) u = 1e-12;
  return -Math.log(-Math.log(u));
}

// Small deterministic PRNG (mulberry32) so results are stable per input.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SimResult {
  p_win: number[];
  p_podium: number[];
  p_points: number[];
  p_dnf: number[];
  expected_pos: number[];
}

export function simulateRace(
  scores: number[],
  pDnf: number[],
  tau: number,
  nSims: number,
  seed = 0,
): SimResult {
  const n = scores.length;
  const z = standardize(scores);
  const t = Math.max(tau, 1e-6);
  const utilities = z.map((v) => v / t);
  const rng = mulberry32(seed || 1);

  const positionCounts: number[][] = Array.from({ length: n }, () =>
    new Array(n).fill(0),
  );
  const dnfCounts = new Array(n).fill(0);

  const key = new Array(n);
  const order = new Array(n);

  for (let s = 0; s < nSims; s++) {
    for (let d = 0; d < n; d++) {
      const isDnf = rng() < pDnf[d];
      if (isDnf) dnfCounts[d] += 1;
      const noisy = utilities[d] + gumbel(rng);
      key[d] = isDnf ? noisy - 1e6 : noisy;
      order[d] = d;
    }
    // sort driver indices by key desc: position 0 = best
    order.sort((a, b) => key[b] - key[a]);
    for (let pos = 0; pos < n; pos++) {
      positionCounts[order[pos]][pos] += 1;
    }
  }

  const p_win = new Array(n);
  const p_podium = new Array(n);
  const p_points = new Array(n);
  const p_dnf = new Array(n);
  const expected_pos = new Array(n);
  const podN = Math.min(3, n);
  const ptsN = Math.min(10, n);

  for (let d = 0; d < n; d++) {
    const probs = positionCounts[d].map((c) => c / nSims);
    p_win[d] = Math.min(1, probs[0]);
    let pod = 0;
    for (let p = 0; p < podN; p++) pod += probs[p];
    p_podium[d] = Math.min(1, pod);
    let pts = 0;
    for (let p = 0; p < ptsN; p++) pts += probs[p];
    p_points[d] = Math.min(1, pts);
    let exp = 0;
    for (let p = 0; p < n; p++) exp += probs[p] * (p + 1);
    expected_pos[d] = exp;
    p_dnf[d] = Math.min(1, dnfCounts[d] / nSims);
  }

  return { p_win, p_podium, p_points, p_dnf, expected_pos };
}

// Rank helper: 1 = best expected finishing position (ties broken by order).
export function rankAscending(values: number[]): number[] {
  const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const ranks = new Array(values.length);
  idx.forEach((origIdx, r) => {
    ranks[origIdx] = r + 1;
  });
  return ranks;
}

export type { BrowserModel };
