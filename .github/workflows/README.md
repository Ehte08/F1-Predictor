# Workflows

## `ci.yml`
Lint (`ruff`) + test (`pytest`) gate for every push/PR to `main`.
Ruff now also covers `scripts/` (the update/predict entry points below).
GitHub Actions natively skips `push`/`pull_request` runs whose head commit
message contains `[skip ci]`, so the automated commits made by
`update-predictions.yml` don't re-trigger this workflow — no extra config
needed here.

## `update-predictions.yml`
Keeps the published site data fresh without a human in the loop:

1. `python scripts/update_after_race.py` — pulls any newly-completed
   session(s) via FastF1, retrains, regenerates the affected race artifacts
   with actuals/metrics, and refreshes `index.json` /
   `data/site/model/browser_model.json` / the model pickles (pickles stay
   local, they're git-ignored).
2. `python scripts/predict_next.py` — finds the next race, builds a grid
   (real quali if it exists yet, else a current-form placeholder), pulls an
   Open-Meteo weather forecast, and writes/locks the next prediction
   artifact.

Both scripts are idempotent and self-healing (they no-op quickly if there's
nothing new), so it's always safe to just re-run this workflow.

### Schedule

F1 sessions run Friday-Sunday and results/quali typically settle within a
few hours, so the workflow polls every 6 hours across that window, plus one
Wednesday check as a mid-week safety net (missed run, delayed results, a
transient FastF1/network hiccup, etc.):

| Cron (UTC)              | When                                   |
|--------------------------|-----------------------------------------|
| `0 0,6,12,18 * * 5`      | Friday, every 6h                        |
| `0 0,6,12,18 * * 6`      | Saturday, every 6h                      |
| `0 0,6,12,18 * * 0`      | Sunday, every 6h                        |
| `0 0,6,12 * * 1`         | Monday, through midday                  |
| `0 12 * * 3`             | Wednesday midday (safety net)           |

Runs are serialized via a `concurrency` group (`update-predictions`) so two
overlapping scheduled/manual runs can't race each other's commits.

### Manual trigger

Actions tab -> **Update Predictions** -> **Run workflow** (uses
`workflow_dispatch`, no inputs required). Useful right after a session
finishes if you don't want to wait for the next cron tick.

### Publishing

If `data/site/**` or `f1_datasets/season_recent.json` changed, the workflow
commits as `f1-predictor-bot` with message `chore: update predictions
[skip ci]` and pushes straight to `main`. `models/*.pkl` and `cache_folder/`
are never committed (both git-ignored) — only the site-facing artifacts and
the incremental race-data cache. Vercel is configured to redeploy
automatically on pushes to `main`, so a successful run **is** the deploy.
If nothing changed, the commit step exits `0` without pushing.

### Repo/GitHub settings required (one-time, by the repo owner)

- **Settings -> Actions -> General -> Workflow permissions**: set to
  "Read and write permissions" (the workflow also declares
  `permissions: contents: write`, but the repo-level default must allow it
  too, otherwise the push step will be rejected with a 403).
- Confirm the Vercel project is connected to auto-deploy on pushes to
  `main` (no action needed here beyond the commit/push above).
- FastF1 cache (`cache_folder/`) is cached via `actions/cache` keyed on
  `github.run_id` with a `fastf1-cache-` restore-key prefix, so it warms
  from the most recent prior run even though the exact key never matches.
  It will grow over time; GitHub evicts old cache entries automatically
  once the repo's cache quota is hit, so no manual cleanup is required.
