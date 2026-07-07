// Mirrors the on-disk data contract in data/site/*.

export interface RaceIndexEntry {
  slug: string;
  race_name: string;
  race_date: string;
  year: number;
  round: number;
  has_actual: boolean;
  locked: boolean;
}

export interface TrackRecordEntry {
  slug: string;
  race_name: string;
  race_date: string;
  spearman: number;
  ndcg3: number;
  winner_correct: boolean;
  podium_hits: number;
  mean_abs_delta: number;
}

export interface SiteIndex {
  updated_at: string;
  model_version: string;
  races: RaceIndexEntry[];
  next_race: { slug: string; race_name: string; race_date: string } | null;
  track_record: TrackRecordEntry[];
}

export interface Weather {
  rainfall: number;
  avg_track_temp: number;
  min_humidity: number;
  source: string;
}

export interface GridEntry {
  driver: string;
  team: string;
  start: number;
}

export interface Prediction {
  driver: string;
  team: string;
  start: number;
  pred_finish: number;
  pred_score: number;
  p_win: number;
  p_podium: number;
  p_points: number;
  p_dnf: number;
  position_probs: number[];
}

export interface ShapFeature {
  feature: string;
  value: number;
  shap: number;
}

export interface ShapBlock {
  base_value: number;
  drivers: Record<string, ShapFeature[]>;
}

export interface ActualResult {
  driver: string;
  team: string;
  finish: number;
  status: string;
}

export interface RaceMetrics {
  spearman: number;
  ndcg3: number;
  ndcg10: number;
  ndcg20: number;
  winner_correct: boolean;
  podium_hits: number;
  mean_abs_delta: number;
}

export interface RaceArtifact {
  race_name: string;
  race_date: string;
  year: number;
  round: number;
  circuit: string;
  model_version: string;
  generated_at: string;
  locked: boolean;
  weather: Weather;
  grid: GridEntry[];
  predictions: Prediction[];
  shap: ShapBlock;
  actual: ActualResult[] | null;
  metrics: RaceMetrics | null;
}

// ── Browser model (16 MB) ─────────────────────────────────────────────
export interface TreeNode {
  split_index?: number;
  split_feature?: number;
  threshold?: number | string;
  decision_type?: "<=" | "==";
  default_left?: boolean;
  missing_type?: string;
  left_child?: TreeNode;
  right_child?: TreeNode;
  leaf_value?: number;
  leaf_index?: number;
}

export interface TreeInfo {
  tree_index: number;
  num_leaves: number;
  num_cat: number;
  shrinkage: number;
  tree_structure: TreeNode;
}

export interface Booster {
  feature_names: string[];
  objective: string;
  tree_info: TreeInfo[];
}

export interface DriverSnapshot {
  reliability: number;
  finish_ewm3: number;
  elo: number;
  circuit_avgs: Record<string, number>;
}

export interface TeamSnapshot {
  reliability: number;
  pace_ewm5: number;
}

export interface ModelSnapshot {
  driver_features: Record<string, DriverSnapshot>;
  team_features: Record<string, TeamSnapshot>;
  driver_ages: Record<string, number>;
  circuit_map: Record<string, string>;
  quali_slot_gap: Record<string, number>;
  quali_global_gap: number;
}

export interface BrowserModel {
  model_version: string;
  trained_through: string;
  feature_names: string[];
  categorical_features: Record<string, (string | number)[]>;
  booster: Booster;
  dnf_booster: Booster;
  pl: { tau: number; n_sims_recommended: number };
  snapshot: ModelSnapshot;
}

// ── Playground worker protocol ────────────────────────────────────────
export interface PlaygroundGridEntry {
  driver: string;
  team: string;
  start: number;
}

export interface PlaygroundInput {
  grid: PlaygroundGridEntry[];
  year: number;
  circuitName: string;
  rainfall: number;
  avg_track_temp: number;
  min_humidity: number;
  nSims: number;
}

export interface PlaygroundResultRow {
  driver: string;
  team: string;
  start: number;
  score: number;
  p_dnf: number;
  p_win: number;
  p_podium: number;
  p_points: number;
  expected_pos: number;
  pred_finish: number;
}

export type WorkerRequest =
  | { type: "init" }
  | { type: "score"; input: PlaygroundInput };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "progress"; loaded: number; total: number }
  | { type: "result"; rows: PlaygroundResultRow[] }
  | { type: "error"; message: string };
