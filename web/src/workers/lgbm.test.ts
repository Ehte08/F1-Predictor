import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  scoreBooster,
  dnfProbability,
  simulateRace,
  rankAscending,
  type FeatureRow,
} from "./lgbm";
import type { Booster } from "@/lib/types";

// The fixture holds exact per-driver feature rows plus the reference scores
// produced by Python lightgbm (ranker) and sklearn (DNF classifier) on the same
// rows — see scripts that generated web/src/workers/__fixtures__/parity.json.
interface FixtureRow {
  driver: string;
  features: FeatureRow;
  ref_score: number;
  ref_pdnf: number;
}
interface Fixture {
  race: string;
  feature_names: string[];
  rows: FixtureRow[];
}

const fixture: Fixture = JSON.parse(
  readFileSync(
    resolve(__dirname, "__fixtures__/parity.json"),
    "utf8",
  ),
);

// The full browser model on disk (copied to public by the build, but the source
// of truth lives in ../data/site/model).
const model = JSON.parse(
  readFileSync(
    resolve(__dirname, "..", "..", "..", "data", "site", "model", "browser_model.json"),
    "utf8",
  ),
) as {
  booster: Booster;
  dnf_booster: Booster;
  categorical_features: Record<string, (string | number)[]>;
};

describe("LightGBM tree traversal scorer", () => {
  it("matches Python lightgbm ranker scores to float precision", () => {
    let maxErr = 0;
    for (const row of fixture.rows) {
      const s = scoreBooster(model.booster, row.features, model.categorical_features);
      expect(Number.isFinite(s)).toBe(true);
      maxErr = Math.max(maxErr, Math.abs(s - row.ref_score));
    }
    // eslint-disable-next-line no-console
    console.log(`[parity] ranker max abs error = ${maxErr.toExponential(3)}`);
    expect(maxErr).toBeLessThan(1e-9);
  });

  it("matches sklearn DNF probabilities to float precision", () => {
    let maxErr = 0;
    for (const row of fixture.rows) {
      const p = dnfProbability(model.dnf_booster, row.features, model.categorical_features);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
      maxErr = Math.max(maxErr, Math.abs(p - row.ref_pdnf));
    }
    // eslint-disable-next-line no-console
    console.log(`[parity] dnf max abs error = ${maxErr.toExponential(3)}`);
    expect(maxErr).toBeLessThan(1e-9);
  });

  it("produces a sane finishing order from the Plackett-Luce sim", () => {
    const scores = fixture.rows.map((r) =>
      scoreBooster(model.booster, r.features, model.categorical_features),
    );
    const pDnf = fixture.rows.map((r) =>
      dnfProbability(model.dnf_booster, r.features, model.categorical_features),
    );
    const sim = simulateRace(scores, pDnf, 0.5, 3000, 42);

    // probabilities are valid & win probs sum to ~1
    const winSum = sim.p_win.reduce((a, b) => a + b, 0);
    expect(winSum).toBeGreaterThan(0.9);
    expect(winSum).toBeLessThan(1.1);
    for (let i = 0; i < scores.length; i++) {
      expect(sim.p_win[i]).toBeLessThanOrEqual(sim.p_podium[i] + 1e-9);
      expect(sim.p_podium[i]).toBeLessThanOrEqual(sim.p_points[i] + 1e-9);
    }

    // the highest-scoring surviving driver should win the pred_finish=1 slot
    const ranks = rankAscending(sim.expected_pos);
    const topByScore = scores.indexOf(Math.max(...scores));
    // top scorer should land in the front third of the grid
    expect(ranks[topByScore]).toBeLessThanOrEqual(Math.ceil(scores.length / 3));
  });
});
