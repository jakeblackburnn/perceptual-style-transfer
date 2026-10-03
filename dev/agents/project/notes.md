# Project notes

Durable facts that outlive a session. Checked 2026-10-03 at `dfbac70`.

## Commands
- Tests: `.venv/bin/python -m pytest` (21 tests, about 3 s). Python is 3.14 in `.venv`.
- Codex workers: `~/.claude/skills/codex/worker.sh <name> <prompt-file> -m gpt-6-astra -c model_reasoning_effort=max`.

## Traps
- `.gitignore` ignores itself, `CLAUDE.md`, `demo/`, `models/` and most of `artifacts/`. A Codex
  worker clone has none of them and no untracked `dev/`; inline what it needs and give it the
  real checkout's absolute path to read.
- A Codex worker that hits the account usage limit exits 1 without committing. Its files may
  still be complete in `$TMPDIR/codex-workers/<repo>-<name>/repo`; check before rerunning.
  Four parallel `gpt-6-astra` runs at `max` used the whole allowance in about 20 minutes.
- Checkpoint lookup differs by entry point: `train.py:105` writes `models/<name>/`,
  `inference.py:59` prefers `artifacts/models/<name>/`.
- The live demo API is the URL in `demo/web_ui/public/script.js:4` (region `us-east4`), not the
  `style-transfer-api` / `us-central1` example in `demo/docs/deploy.md`.

## Pointers
- Trained models and metrics: `artifacts/models/<name>/` (final `<name>.pth`, `metrics.csv`,
  `metrics_config.json`, `checkpoints/`). Sample outputs: `artifacts/outputs/<name>/`.
- Style images: `artifacts/images/singles/` and `artifacts/images/style/<set>/`. Content:
  `Impressionism/`, `Pointillism/`, `Ukiyo_e/`, `VOC2012/`, test photos in `content/`.
