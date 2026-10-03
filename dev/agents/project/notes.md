# Project notes

Durable facts that outlive a session. Checked 2026-10-03 at `fc5b4fa`.

## Commands
- Tests: `.venv/bin/python -m pytest` (108 tests, about 3 s). Python is 3.14 in `.venv`.
- CLI without installing: `.venv/bin/python -m style_transfer list|stylize|train|serve`.
- Codex workers: `~/.claude/skills/codex/worker.sh <name> <prompt-file> -m gpt-6-astra -c model_reasoning_effort=max`.

## Traps
- `.gitignore` ignores itself, `CLAUDE.md`, `demo/`, `models/` and most of `artifacts/`. A Codex
  worker clone has none of them and no untracked `dev/`; inline what it needs and give it the
  real checkout's absolute path to read.
- A Codex worker that hits the account usage limit exits 1 without committing. Its files may
  still be complete in `$TMPDIR/codex-workers/<repo>-<name>/repo`; check before rerunning.
  Four parallel `gpt-6-astra` runs at `max` used the whole allowance in about 20 minutes.
- A model is visible to the CLI and web UI only if `artifacts/models/<name>/model.json` exists;
  `scripts/write_model_json.py NAME` writes one for an archived model.
- Models trained before 2026-10-03 used the old loss (`"loss": "legacy-mean"`); style weights
  since then are about 85,000 times smaller and not comparable.
- Agent worktrees may start at `origin/main`, not local `main`; have agents check their base.
- The live demo API is the URL in `demo/web_ui/public/script.js:4` (region `us-east4`), not the
  `style-transfer-api` / `us-central1` example in `demo/docs/deploy.md`.

## Pointers
- Trained models and metrics: `artifacts/models/<name>/` (final `<name>.pth`, `model.json`, `metrics.csv`,
  `metrics_config.json`, `checkpoints/`). Sample outputs: `artifacts/outputs/<name>/`.
- Style images: `artifacts/images/singles/` and `artifacts/images/style/<set>/`. Content:
  `Impressionism/`, `Pointillism/`, `Ukiyo_e/`, `VOC2012/`, test photos in `content/`.
