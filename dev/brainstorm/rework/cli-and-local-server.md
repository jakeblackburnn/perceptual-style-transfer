# CLI and local server: options for the workshop

**Recommendation, not a decision:** one installed `style-transfer` command, a shared model resolver, and a small tracked FastAPI app that serves both the local page and inference. Keep the activation viewer as a separate optional research command.

Code references are to clone `dfbac70`; `demo/`, `.gitignore`, and `CLAUDE.md` references are to the read-only checkout at `/Users/jackblackburn/code/main/audio_video/style-transfer/py`. The public shutdown and CLI-first/localhost-only direction are given constraints. The dependency direction also agrees with `CLAUDE.md:7` and the shared-inference goal at `CLAUDE.md:13`.

Assumptions: one person per local server; image batches and training belong in the terminal; no compatibility promise to public API clients. These are workshop assumptions, not facts about users. I did not inspect live cloud resources, verify a weight-download service, assess model aesthetics, or run training. Source inspection and small read-only CPU probes are the evidence here; no test-suite result is claimed.

## Ranked findings

1. **Resolve a trained artifact, not an experiment name plus a guessed location.** The CLI searches `artifacts/models` before `models`, relative to the working directory (`style_transfer/inference.py:52`). Training writes to `models/<name>` (`style_transfer/train.py:105`), while the menu's apply action only checks that latter location (`run_experiment.py:41`). Thus a retrain can be hidden by an older archived checkpoint when switching entry points. Inference gets architecture size from today's experiment definition (`style_transfer/inference.py:108`), but final training output saves only a state dict (`style_transfer/train.py:173`). The demo independently supplies three names, sizes, and filenames (`demo/core/config.py:53`). Recommendation: one explicit model root and artifact metadata shared by every caller; keep experiment recipes separate from installed weights.

2. **Replace the demo's three-request workflow with one upload/result request.** The browser keeps `uploadedImageId` until the file input changes and skips repeat uploads (`demo/web_ui/public/script.js:77`, `demo/web_ui/public/script.js:108`). The backend deletes that input after a successful transfer (`demo/api/routes/style.py:227`). Code-derived consequence: applying another style to the same selected file can fail with an absent upload; I did not reproduce this in a browser. Results are also deleted after the download endpoint responds (`demo/api/routes/style.py:312`). Returning image bytes directly removes upload IDs, result URLs, retention, and this state mismatch together.

3. **Localizing requires changing the frontend URL as well as the bind address.** The backend defaults to `0.0.0.0` (`demo/core/config.py:18`); the frontend chooses the public API for every hostname except literal `localhost`, including `127.0.0.1` (`demo/web_ui/public/script.js:2`). Express only serves the static directory/index and adds CORS (`demo/web_ui/server.js:8`). Recommendation: one loopback-bound FastAPI process, relative API URLs, and no Node runtime or cross-origin configuration. Merely deleting deployment files would not take the existing services offline.

4. **Make image/output behavior part of the command contract.** Directory inference takes only immediate children and writes every image to `<stem>.jpg`, without an overwrite check (`style_transfer/inference.py:69`, `style_transfer/inference.py:128`). `a.jpg` and `a.png` therefore target the same output. The model downsamples twice then doubles twice (`style_transfer/models.py:51`, `style_transfer/models.py:69`); `stylize_image` returns that size unchanged (`style_transfer/inference.py:36`). A read-only CPU probe confirmed `(65,97) -> (68,100)`, `(64,64) -> (64,64)`, and a padding error for `(1,1)`. The existing shape test uses only 64×64 (`tests/test_models.py:6`). Recommendation: shared padding/cropping, explicit output paths, collision checks, and tests for odd and tiny dimensions before advertising arbitrary dimensions.

5. **Keep useful research tools, but repair them before recommending them.** Dash normalizes before calling a VGG wrapper that normalizes again (`style_transfer/utils/visualize.py:312`, `style_transfer/feature_extractors/vgg.py:23`). It also hardcodes image paths and enables debug mode (`style_transfer/utils/visualize.py:369`, `style_transfer/utils/visualize.py:272`). ONNX export reconstructs/loading weights separately, fixes the source/output roots, and merely warns when its checker fails (`style_transfer/utils/convert_to_onnx.py:9`, `style_transfer/utils/convert_to_onnx.py:19`, `style_transfer/utils/convert_to_onnx.py:62`). Recommendation: optional commands with explicit inputs; no folding the research viewer into the everyday image UI.

The central numerical path is already shared: the menu, apply script, and demo call library inference (`run_experiment.py:9`, `style_transfer/apply_style.py:6`, `demo/core/inference.py:10`, `demo/core/model_manager.py:11`). Preserve this accomplishment. The rework should remove callers' inconsistent policy, not introduce another preprocessing engine.

## Two complete alternatives

Both options use the model scheme, commands, and direct-response local UI described below; neither keeps public serving.

| Option | Installation and invocation | Model/data policy and local UI | Benefit and cost |
| --- | --- | --- | --- |
| A. Repository tool | Install core requirements, optional UI requirements; run `python -m style_transfer_cli models list`, `... stylize`, `... train`, `... serve` from the checkout. | Same explicit model root and manifests as B. Track `style_transfer_web/`; serve its static files from FastAPI. Keep optional Dash separate. | Least packaging work; honest choice if this remains one person's checkout. Requires the checkout to locate the command, and separate dependency files remain part of setup. |
| B. Installable command **(recommended)** | `python -m pip install -e .`; optional `'.[ui]'`. Console script `style-transfer` and identical `python -m style_transfer_cli` entry point. | Same model scheme; absolute paths after resolution allow invocation from any directory. Package the tracked static UI with the optional server. Keep optional Dash separate. | One extra packaging task buys a stable everyday invocation and clean optional dependencies. Must test a built wheel as well as editable installation; otherwise static assets can accidentally work only in the checkout. |

For B, use a single top-level `style_transfer_cli.py` with argparse. In `pyproject.toml`, declare `style-transfer = "style_transfer_cli:main"`; no command framework or plugin discovery. The dispatcher imports training/UI/visualization/export code only when that command runs. PyPA documents [console scripts and dependency extras](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

Proposed dependency direction: `style_transfer_cli.py -> style_transfer/` and, only for `serve`, `style_transfer_cli.py -> style_transfer_web/ -> style_transfer/`. The library must import neither application module. This outer dispatcher removes the need to put a web-app import inside `style_transfer/`; it replaces the existing script collection rather than adding another application layer.

Base installation should cover core training/inference. Extras: `ui` for FastAPI/Uvicorn/python-multipart, `viz` for Dash/Plotly, `export` for the tested ONNX exporter/runtime, and development/research extras for tests/notebooks/diffusion. Keep those dependencies out of ordinary inference startup. Today's root requirements mix all these uses (`requirements.txt:1`); the demo also freezes unrelated notebook/research packages alongside its server (`demo/requirements.txt:14`, `demo/requirements.txt:30`, `demo/requirements.txt:41`). Pick supported versions by an installation check, not by copying that freeze wholesale.

## Proposed command contract

All commands below are proposed interfaces, not commands available at `dfbac70`.

| Task | Proposed command/example |
| --- | --- |
| Discover styles/models | `style-transfer models list` — ID, label, size, state, resolved path; `--available` filters usable local weights; `--json` supports scripts. A style label may have several model IDs. |
| Inspect a model | `style-transfer models show mini_kanagawa` — manifest, checksum, path, acquisition guidance. |
| Acquire reviewed weights | `style-transfer models fetch mini_kanagawa` — explicit download, only after a pinned release/catalog entry exists. |
| Register a local checkpoint | `style-transfer models import /path/to/mini_kanagawa.pth --id mini_kanagawa --size small` |
| One image | `style-transfer stylize ./photo.jpg --model mini_kanagawa --output ./styled.png --device cpu` |
| Directory | `style-transfer stylize ./photos --model mini_kanagawa --output-dir ./styled --recursive --limit 20` |
| Explicit checkpoint | `style-transfer stylize ./photo.jpg --checkpoint ./trial.pth --size small --output ./trial.png` |
| List training recipes | `style-transfer experiments list`; these are recipes, not claims that weights exist. |
| Validate an experiment | `style-transfer train --experiment kanagawa_dry_run --run wave-study --data-root "$PWD" --device mps --dry-run` |
| Run that experiment | Same command without `--dry-run`. For other data, add `--content-dir /data/photos --style-image /data/style.jpg`; `--style-dir` is the alternative for a style dataset. |
| Local image UI | `style-transfer serve --port 8000 --device auto` — print `http://127.0.0.1:8000`; optional `--open` opens it. |
| Activation research | `style-transfer visualize ./photo.jpg --layer-preset standard --port 8050` |
| ONNX utility | `style-transfer export-onnx --model mini_kanagawa --output ./mini_kanagawa.onnx` |

Global `--models-dir PATH` goes before the subcommand; it overrides `STYLE_TRANSFER_MODELS_DIR`, which overrides `~/.style-transfer/models`. Expand/resolve once. Do not search parent directories, infer a repository, or fall back to a second model root. `models list` and `--help` should not create directories, load tensors, or access the network.

Inference details worth fixing as a contract:

- Require exactly one of `--model ID` and `--checkpoint PATH`; require `--size` only for a legacy explicit checkpoint without metadata. Reject contradictory flags. Check availability of an explicitly requested device; do not silently fall back. `auto` can preserve MPS, then CUDA, then CPU selection (`style_transfer/inference.py:44`).
- Require `--output FILE` for a single result or `--output-dir DIR` for a batch. Single-file encoding follows `.png`/`.jpg`/`.jpeg`; directory results default to PNG. Use `<relative input filename>.png`, e.g. `birds/a.jpg.png`, to keep distinct source extensions from colliding. Default directory traversal is shallow; `--recursive` is explicit. Sort and collect regular JPEG/PNG files before inference; require a positive limit.
- Reject an output directory inside the input directory, input/output identity, and existing target files unless `--overwrite`. Preflight the destination set, load the model once, and save each image atomically. Fail on the first decode/inference error, report completed outputs, and exit nonzero; do not print an unconditional success banner.
- Put orientation handling, RGB conversion, and dimensional behavior in `stylize_image`, shared by UI and CLI. Proposal: apply EXIF orientation, flatten transparency onto white, keep `[0,1]` tensors, pad on the right/bottom to multiples of four with at least eight pixels per side, then crop back. Use replication for dimensions too small for reflection. Existing divisible-by-four RGB images should retain their numerical path. This changes the behavior observed in finding 4 and needs tests.
- Preserve oriented input dimensions by default. Only `--max-side N` requests an aspect-preserving downsize, applied before padding; report the resulting dimensions. Do not imply unlimited-resolution memory capacity or add tiling before measuring its effect on style.
- Exit 0 for complete success, 2 for invalid arguments/missing prerequisites, 1 for processing failure, 130 for interruption. Logs/progress go to stderr; JSON listing remains machine-readable stdout. Do not launch a folder automatically or force process termination as the menu currently does (`run_experiment.py:86`, `run_experiment.py:258`).

Training details:

- Keep existing Python experiment recipes; avoid adding a second YAML configuration language. Their current assembly is explicit imports and dictionary updates (`style_transfer/config/styles/__init__.py:1`). Deep-copy a selected recipe before applying path overrides.
- Resolve recipe-relative dataset paths against `--data-root`, defaulting explicitly to the invocation directory; CLI-supplied paths are relative to that invocation directory. Print absolute resolved inputs. This is independent of the model-root setting. The Kanagawa recipe currently names `artifacts/images/...` paths (`style_transfer/config/styles/kanagawa.py:19`, `style_transfer/config/styles/kanagawa.py:26`).
- `--dry-run` validates paths, nonempty selected datasets/fractions, curriculum, and device; prints configuration/output paths without downloading VGG or constructing loaders/models. A recipe whose name contains `dry_run` is still a training recipe (`style_transfer/config/styles/kanagawa.py:62`); distinguish that from the no-training flag.
- Use `--run ID` to select a new artifact directory; default to `<experiment>-<timestamp>`. Refuse an existing run ID, including an unfinished run. Save resolved configuration before training; write the final model manifest only on successful completion. No implicit resume or overwriting a published model. Add an explicit output-directory argument to `train_model`, rather than temporarily changing the process working directory around its hardcoded output (`style_transfer/train.py:80`, `style_transfer/train.py:105`).
- A successful run prints the exact subsequent `stylize --model <run-id>` command. Train-and-apply becomes two composable commands. The UI gets no training endpoint in this proposal.

### One model registry/lookup scheme

Use a small JSON schema and ordinary directories, not a database or registry service. Add `style_transfer/registry.py` as the sole resolver, plus a packaged `style_transfer/model_catalog.json` for reviewed downloadable artifacts. The catalog is a download menu using the same record schema as local manifests; it is not a second loader or a web-specific model list.

Proposed local layout:

```text
<models-dir>/
  mini_kanagawa/
    model.json
    mini_kanagawa.pth
  wave-study/
    model.json                 # appears only after successful training
    wave-study.pth
    resolved_config.json
    metrics.csv
    checkpoints/...
```

Minimum manifest: schema version, immutable model ID, human label, family (`johnson`), architecture preset, preprocessing contract (`rgb01-v1`), relative weight filename, SHA-256, and optional source experiment. Catalog records additionally have a version-pinned download URL. Never infer a saved architecture from the mutable `Models` recipe map. A model ID identifies specific weights; retraining gets a new ID.

Resolution: enumerate `<root>/*/model.json` and the packaged catalog, merge identical IDs only when their immutable fields agree, and reject conflicts. A catalog-only record is `missing`, not runnable; an absent/corrupt local weight is `broken`. `models list` distinguishes these states, and `resolve_model` verifies the checksum and compatibility before loading. Keep malformed manifests visible as errors rather than silently hiding them. Raw unregistered directories are not guessed into models; the import command produces their manifest.

`models import` validates the chosen size by strict state-dict loading, computes the digest, and writes the manifest. Copy into `<root>/<id>/<id>.pth` unless the source already is that exact file, in which case register it in place. Support the raw and `model_state_dict` formats handled by current inference (`style_transfer/inference.py:27`); reject unsupported structures with a useful message. Restrict IDs/manifest-relative paths to that model directory. `--checkpoint` remains the explicit escape hatch for examining intermediate checkpoints without registration.

For the owner's existing archive, proposal: set `STYLE_TRANSFER_MODELS_DIR` to the real `artifacts/models` directory and import wanted finals in place. Do not copy all epoch checkpoints into a new cache. Use canonical `mini_colors1`, not the demo-only `mini_colors` name: their relationship is documented at `demo/api/models/README.md:12` and the recipe is `mini_colors1` (`style_transfer/config/styles/colors1.py:30`). Read-only SHA-256 comparisons in this review confirmed all three demo copies match the documented archived source files (`demo/api/models/README.md:11`). Those copies loaded as raw tensor dictionaries; this is not an assessment of their visual quality.

Fresh-clone acquisition has to be delivered, not implied:

1. Before advertising `models fetch`, publish a small, versioned starter release with the owner-approved final weights, manifests/checksums, and provenance. Candidate IDs are `mini_kanagawa`, `mini_colors1`, and `high_starry_night`; choosing them for convenience does not rank their quality. A static release asset does not require keeping the public inference website.
2. Put the actual immutable URLs and SHA-256 values into the catalog. Fetch to a temporary file, verify, then install atomically; preserve an existing valid artifact on interruption or mismatch. No silent downloads during import, listing, stylizing, or serving.
3. **Unverified dependency:** no release URLs were established by this review. Until the owner publishes that release, the README must say that weights require a supplied local checkpoint or training. A fresh clone on this machine can import from the read-only checkout's existing archive; a stranger cannot be promised those files. Do not ship invented URLs or claim that the current clone is inference-ready.

This scheme adds metadata files but removes the demo model dictionary, inference's dependence on training recipes, duplicated weight copies, and competing path searches. It also lets local training runs appear in both interfaces without editing UI source.

### Existing entry points after the rework

| Current entry point/evidence | Recommended destination |
| --- | --- |
| `style_transfer/inference.py:78` | Keep library functions; move argparse behavior to `style_transfer_cli.py`. Its old module invocation may emit a short migration error without importing the CLI/web app. Keep the legacy `find_checkpoint` function temporarily for its diffusion caller at `run_diffusion_experiment.py:8`; new Johnson commands must not use it. |
| `style_transfer/apply_style.py:9`, `style_transfer/apply_style.py:95` | Retire the hardcoded batch/checkpoint driver. Normal use becomes `stylize`; comparing intermediate weights uses an explicit shell loop over `--checkpoint`, with separate output directories. |
| `style_transfer/build.py:8` | Retire the hardcoded training launcher; use `train --experiment ...`. |
| `run_experiment.py:91`, `run_experiment.py:157` | Retire interactive menus; `experiments list`, `train`, and `stylize` cover their actions. Update the README in the same release; external script consumers have not been surveyed. |
| `style_transfer/utils/convert_to_onnx.py:7`, `style_transfer/utils/convert_to_onnx.py:74` | Keep a parameterized export function using the shared loader; replace the hardcoded module main with `export-onnx`. Require successful checker/runtime validation before reporting success. |
| `style_transfer/utils/visualize.py:369` | Keep parameterized viewer functions behind `visualize`; remove hardcoded inputs and the standalone publicized entry point. |
| `demo/server.py:113`, `demo/web_ui/server.js:18` | Replace both launches with `serve`; no separate npm process. |
| `run_diffusion_experiment.py:10` | Leave this experimental runner separate for now and document it as research. Do not pretend the Johnson `train` command already supports diffusion. |

## Proposed local UI and file disposition

Track `style_transfer_web/__init__.py`, `app.py`, and `static/{index.html,script.js,styles.css}` as a sibling of the library. Copy only reviewed source from ignored `demo/` into it. Track a root ignore policy for weights, local outputs, environments, and caches; do not force-add the whole demo tree. Currently the real checkout ignores `demo/`, weights, and even its own ignore file (`.gitignore:1`, `.gitignore:11`, `.gitignore:15`).

One `create_app(models_dir, device)` factory owns the selected device and one cached model for its lifetime. Use the library resolver/loader, and replace the cached model when the selected ID changes. One inference slot is enough initially: run blocking work off the event loop; reject overlap with a clear busy response rather than implementing a job queue. Keep health/list endpoints responsive. This removes the separate manager/engine/config/storage layering while retaining model reuse.

Proposed HTTP contract:

| Route | Contract |
| --- | --- |
| `GET /` and `/assets/...` | Serve the packaged HTML/CSS/JS from paths relative to the installed package, never the working directory. No CDN assets. |
| `GET /api/models` | Same registry records/statuses as the CLI; UI enables only available models and explains missing weights. |
| `GET /api/health` | Ready/busy and selected device; no claim that all catalog weights are installed. |
| `POST /api/stylize` | Multipart `file`, `model`, optional `max_side`; respond with PNG bytes and processing-time/model headers, or a JSON `detail` error. No paths or checkpoint uploads accepted through HTTP. |

FastAPI can [serve static files directly](https://fastapi.tiangolo.com/tutorial/static-files/); a `FileResponse` for `/` plus a `/assets` mount avoids swallowing API routes. Its [UploadFile already provides a spooled temporary file](https://fastapi.tiangolo.com/tutorial/request-files/), so request-scoped upload handling can replace the application upload store. Decode from that file, validate JPEG/PNG and decoded pixel count, stylize once, encode the response, and close resources even on failure. The browser owns the result blob for preview and repeated downloads, revoking old object URLs when replacing results.

Bind only `127.0.0.1`; no `--host`/LAN mode in this design. One Uvicorn worker, reload/debug off, configurable port; print an actionable port-in-use error. Remove CORS middleware because page and API share an origin. Also validate the local Host and reject foreign or null browser Origins on POST; permit ordinary local CLI clients without an Origin. This replaces deployment-origin/auth settings with a small local boundary, not an account system. Disable CDN-backed API documentation pages by default; `/openapi.json` can remain for development.

Do retain resource limits and decoding errors on localhost. Proposed starting limits: 10 MiB upload and 8 million decoded pixels, visible in UI errors and configurable via `serve --max-upload-mib`/`--max-pixels`. These are starting policy choices, not measured safe hardware limits. The CLI remains the route for large jobs and directory input. Neither interface silently resizes; the user requests `max_side` explicitly. Use 400/422 for invalid input, 404 for unknown model, 409 for missing model/busy, 413 for limits, and 500 for inference errors; retain useful server tracebacks without the existing checkpoint-by-checkpoint log chatter (`demo/api/routes/style.py:67`, `demo/storage/file_handler.py:114`).

File-by-file disposition of application files inspected under `demo/` follows. “Delete” means omit from the tracked replacement and retire the local legacy copy after migration; this notes task deletes nothing.

| Existing file and evidence | Recommendation |
| --- | --- |
| `demo/server.py:22` | **Simplify/move** into `style_transfer_web/app.py`: factory/lifespan, static serving, loopback launch. Remove global initialization and CORS at `demo/server.py:55`. |
| `demo/api/__init__.py:1` | **Delete** old namespace marker; use the new package's one initializer. |
| `demo/api/routes/__init__.py:1` | **Delete**; a separate router package is unnecessary for three small endpoints. |
| `demo/api/routes/health.py:21` | **Simplify** into app health endpoint; remove duplicate loaded-model count/uptime schema. |
| `demo/api/routes/models.py:40` | **Simplify** into app model listing backed by the shared resolver, rather than settings' hardcoded names. |
| `demo/api/routes/style.py:37`, `demo/api/routes/style.py:151`, `demo/api/routes/style.py:242` | **Replace** three routes with the single multipart response; delete upload-ID/download-ID state and background deletion. |
| `demo/api/schemas.py:9`, `demo/api/schemas.py:33`, `demo/api/schemas.py:46` | **Simplify**: retain only needed model/status/error response validation inline in the app; delete UploadResponse, job/output-URL fields, and image-ID aliases. |
| `demo/core/__init__.py:1` | **Delete** obsolete package marker. |
| `demo/core/config.py:10` | **Delete** app-wide settings hierarchy; command/factory arguments cover local settings. Remove model catalog, paths, public origins, TTL, and import-time directory creation at `demo/core/config.py:83`. |
| `demo/core/model_manager.py:20`, `demo/core/model_manager.py:119` | **Simplify** caching into app-lifetime state; remove duplicate device detection, three-entry LRU/preload machinery, and app-specific weight lookup. |
| `demo/core/inference.py:28`, `demo/core/inference.py:112` | **Delete wrapper class**; retain useful image validation in the request boundary, call library inference directly, encode the response once. No disk-output naming policy. |
| `demo/storage/__init__.py:1` | **Delete** obsolete package marker. |
| `demo/storage/file_handler.py:98`, `demo/storage/file_handler.py:162` | **Delete storage abstraction**; retain byte/decode limits as small request checks. UUID saves and extension searches disappear. `cleanup_old_files` is defined at `demo/storage/file_handler.py:271`; no caller was found in the inspected app, so do not describe TTL cleanup as a working feature. |
| `demo/api/models/README.md:3` | **Move useful provenance** into the registry migration/release notes; replace instructions to copy weights into the app. |
| `demo/api/models/mini_kanagawa.pth` (source mapping: `demo/api/models/README.md:11`) | **Delete duplicate after verifying registry use**; preserve the archived original. |
| `demo/api/models/mini_colors.pth` (source mapping: `demo/api/models/README.md:12`) | **Delete duplicate after migration**; register canonical `mini_colors1`. |
| `demo/api/models/high_starry_night.pth` (source mapping: `demo/api/models/README.md:13`) | **Delete duplicate after migration**; preserve the archived original. |
| `demo/web_ui/public/index.html:14` | **Keep/simplify** upload, style selector, preview/download in tracked static HTML; label it local, explain model availability and resource errors, align accepted image formats. |
| `demo/web_ui/public/styles.css:13`, `demo/web_ui/public/styles.css:66` | **Keep** the small layout/preview stylesheet; redesign is unnecessary for this rework. |
| `demo/web_ui/public/script.js:2`, `demo/web_ui/public/script.js:85` | **Simplify** to relative fetch URLs and one request per Apply. Keep local preview/blob download; show backend errors and keep them visible until action/dismissal. |
| `demo/web_ui/server.js:1` | **Delete** Express/CORS/static server; FastAPI takes its only application role. |
| `demo/web_ui/package.json:5` | **Delete** npm runtime/development setup; no frontend build step is proposed. |
| `demo/web_ui/package-lock.json:7` | **Delete** matching Express/CORS/nodemon lock; package metadata was also parsed read-only. Do not migrate installed `node_modules/`. |
| `demo/web_ui/firebase.json:2` | **Delete** hosting configuration after operational shutdown is recorded. |
| `demo/web_ui/.firebaserc:2` | **Delete** Firebase project binding after shutdown; it is not local UI configuration. |
| `demo/web_ui/.gitignore:10`, `demo/web_ui/.gitignore:47` | **Delete** Node/Firebase-specific ignore list; local package needs only root artifact/cache ignores. |
| `demo/web_ui/webui.md:16` | **Delete/merge** useful usage into README; remove the npm launch instruction. |
| `demo/docs/webui.md:5`, `demo/docs/webui.md:10` | **Simplify/merge** useful upload/select/download requirements into README; drop the public docs/blog ambitions from this scope. |
| `demo/docs/api.md:28`, `demo/docs/api.md:87`, `demo/docs/api.md:163` | **Replace** with the concise local HTTP contract. The spec's scale-to-255 and auto-resize text conflicts with current shared inference/rejection (`style_transfer/inference.py:17`, `demo/core/inference.py:61`). API-key authentication/rate limiting appear as documentation proposals; no implementation was found in the reviewed settings/router dependencies (`demo/core/config.py:42`, `demo/api/routes/style.py:23`). Do not claim to remove working authentication. |
| `demo/docs/deploy.md:246`, `demo/docs/deploy.md:357` | **Delete** Cloud Run deployment/secret-management guide after capturing a brief shutdown record. No cloud-auth flow survives in the localhost product. |
| `demo/docs/fastapi-example.py:7` | **Delete** unrelated Item/price tutorial. |
| `demo/Dockerfile:7`, `demo/Dockerfile:28` | **Delete** Cloud Run image/Gunicorn launch; containers are not needed to use this local Python app. Revisit only for a concrete portability problem. |
| `demo/Dockerfile.dockerignore:1` | **Delete** with the Docker build. |
| `demo/requirements.txt:1` | **Replace** with packaging extras and a tested dependency policy; no second environment freeze. |
| `demo/web_ui/.firebase/hosting.cHVibGlj.cache:1` | **Delete/exclude** generated deployment cache. |
| `demo/web_ui/.claude/settings.local.json:2` | **Exclude** machine-local editor permissions from the migrated app. |
| `demo/api/.DS_Store`, `demo/api/models/.DS_Store` | **Exclude** binary filesystem metadata; these inventory items have no source-line behavior to cite. |

### Activation visualizer choice

Keep it separate and optional, behind the same command surface. Its research purpose is already described in `README.md:21`, and its controls inspect image/layer/channel activations (`style_transfer/utils/visualize.py:45`, `style_transfer/utils/visualize.py:145`). Folding it into FastAPI would add WSGI/ASGI integration or a new rendering implementation without simplifying everyday stylization. Dropping it would remove a useful way to compare feature choices, though usage frequency is unverified.

Recommended repairs: explicit image inputs, chosen layer preset, normalization exactly once, loopback/debug-off launch, and no disk activation cache initially. The existing cache key contains only parent-directory name and image stem (`style_transfer/utils/visualize.py:327`), so changing contents/preprocessing can reuse stale activations. Drop that cache instead of designing a new cache subsystem. Use the VGG wrapper's selected outputs rather than hooks on every layer (`style_transfer/feature_extractors/vgg.py:30`, `style_transfer/utils/activation_extractor.py:13`); this also permits retiring `activation_extractor.py`. Test with a tiny fake extractor to avoid downloading VGG. Activation inspection may still need pretrained VGG weights on first real use (`style_transfer/feature_extractors/vgg.py:18`).

## README usage draft for the recommended option

This is proposed replacement copy. Publish the fetch example only when the release/catalog described above exists; before then make checkpoint import the first weights step. Replace the current public-site links and menu-based usage (`README.md:3`, `README.md:19`, `README.md:35`).

````markdown
## Install and choose a model

From a checkout, create an environment and install the command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
style-transfer --help
```

Each trained model applies one learned style. Experiment recipes describe how to
train; they do not supply weights. Download a released model explicitly:

```bash
style-transfer models list
style-transfer models fetch mini_kanagawa
style-transfer models list --available
```

Or import a checkpoint you already have:

```bash
style-transfer models import /path/to/mini_kanagawa.pth --id mini_kanagawa --size small
```

Weights live in `~/.style-transfer/models`. To use another location, set
`STYLE_TRANSFER_MODELS_DIR` or pass `--models-dir PATH` before the subcommand.
Listing and stylizing do not download anything. All commands also work as
`python -m style_transfer_cli ...`.

## Stylize images

```bash
style-transfer stylize ./photo.jpg --model mini_kanagawa --output ./styled.png
style-transfer stylize ./photos --model mini_kanagawa --output-dir ./styled --recursive
```

Batch results preserve relative paths and append `.png` to the input filename.
Existing files are protected unless you pass `--overwrite`. Images keep their
oriented dimensions unless you request `--max-side 1600`. Use `--device cpu`,
`cuda`, or `mps` to choose explicitly; the default is `auto`.

## Train an experiment

Training datasets and style source images are separate from the code and weights.
Supply their paths or put them at the locations required by your selected recipe.

```bash
style-transfer experiments list
style-transfer train --experiment kanagawa_dry_run --run wave-study \
  --content-dir /path/to/content-images --style-image /path/to/style.jpg --dry-run
style-transfer train --experiment kanagawa_dry_run --run wave-study \
  --content-dir /path/to/content-images --style-image /path/to/style.jpg
style-transfer stylize ./photo.jpg --model wave-study --output ./study.png
```

`--dry-run` only validates and prints the plan. Removing it starts training.
The run directory contains the resolved configuration, metrics, checkpoints,
and final registered model. Choose a new run name for another experiment.
Training uses pretrained VGG features and may need a first-use weight download.

## Optional local UI

```bash
python -m pip install -e '.[ui]'
style-transfer serve --open
```

Open `http://127.0.0.1:8000`, select an installed model and a JPEG/PNG, then Apply
and Download. Images are processed on this computer. Use Ctrl-C in the terminal
to stop. For another port, use `serve --port 8001`. Batch processing stays in the CLI.

For research tools, install `'.[viz]'` and run
`style-transfer visualize ./photo.jpg --layer-preset standard`, or install
`'.[export]'` and run
`style-transfer export-onnx --model mini_kanagawa --output ./model.onnx`.
````

## Proposed implementation slices with disjoint file ownership

Agree these small contracts first: `registry.list_models(root)` / `resolve_model(root, id)` return plain model records; `inference.load_model` and `stylize_image` remain the numerical path; `train_model(..., output_dir=...)` returns the final path; the web factory owns app state and the HTTP contract above. No generic service interface is needed. Disjoint ownership permits parallel implementation; it does not mean the components have no integration dependencies. Individual checks can use fixtures/fakes until their real dependency lands.

| Slice / owned files | Independent check and completion evidence |
| --- | --- |
| 1. Registry: new `style_transfer/registry.py`, `style_transfer/model_catalog.json`, `tests/test_model_registry.py` | Temporary model root; raw/wrapped tiny checkpoints; import in place; checksum failure, malformed metadata, ID conflict, explicit-root precedence, interrupted fetch. Network mocked; listing never loads weights/downloads. Fetch is not releasable until actual assets/URLs exist. |
| 2. Image contract: `style_transfer/inference.py`, new `tests/test_inference_io.py` | Existing preprocessing tests plus real small-model 64×64, 65×97, 1×1 and narrow images; EXIF, grayscale/alpha and requested downsize. Confirm old divisible-by-four RGB output parity. Preserve `find_checkpoint` for the diffusion caller; migrate only its Johnson CLI entry. |
| 3. Training paths: `style_transfer/train.py`, `style_transfer/utils/metrics.py`, new `tests/test_training_paths.py` | Fake VGG/epoch work verifies all outputs remain under the passed directory and the final path is returned; no writes to implicit `models/`. Failure creates no successful-model marker. Use synthetic inputs; no real experiment training required for this slice. |
| 4. Local backend: new `style_transfer_web/__init__.py`, `style_transfer_web/app.py`, `tests/test_local_server.py` | Factory with fixture model provider and temporary static directory; one POST returns a decodable image; absent/bad models, limits, corrupt images, busy response, foreign Origin/Host, and cleanup on errors. A slow fake inference must not block health. No legacy upload/output directories appear. |
| 5. Local frontend: new `style_transfer_web/static/index.html`, `script.js`, `styles.css` | Serve against the agreed fixture API: apply two different styles to the same file, download repeatedly, replace an image, show missing models and readable errors. Browser network inspection shows only same-origin requests on both localhost and 127.0.0.1; test an empty model list. No npm runtime. |
| 6. CLI/install: new `style_transfer_cli.py`, `pyproject.toml`, `style_transfer/__init__.py`, `tests/test_cli.py`; `requirements.txt`; retire `run_experiment.py`, `style_transfer/apply_style.py`, `style_transfer/build.py` | Stub command targets to prove flag dispatch, missing extras, no prompts, exit codes, positive limits, collision/overwrite policy and dry-run behavior. Build/install a wheel in a clean environment and invoke from outside the checkout; assert static assets/catalog ship. Core install imports without FastAPI/Dash/ONNX. UI launch proves a loopback listener and clean Ctrl-C. |
| 7. Research utilities: `style_transfer/utils/visualize.py`, `activation_extractor.py`, `convert_to_onnx.py`; new focused utility tests | Fake VGG proves one normalization and selected layers without stale disk cache. Export checker failures are fatal; ONNX Runtime/PyTorch parity for even and odd rectangular dimensions is required before claiming usable dynamic dimensions. Keep this independent of the local image UI. |
| 8. Migration/docs: `README.md`, tracked root `.gitignore`, a concise new `docs/local-ui.md`; retirement inventory for ignored `demo/` | README commands exercised against a fixture artifact and temporary data; trained-artifact registration tested using a fake trainer. Check only intended UI source/assets are tracked, no weights/uploads/Node tree. Remove stale public links. Record which legacy files were migrated before removing local copies. |

The CLI slice owns final manifest registration after successful training; the trainer slice only accepts an explicit output directory and returns its artifact. Frontend/backend checks may run independently against fixtures; the acceptance gate must use their real implementations together. Packaging may be prepared early but its wheel check waits for the new package/assets.

Final acceptance proposal: a fresh environment can install, acquire/import one artifact, list it, stylize a single image and a directory, and use the same model through the local UI. Compare decoded PNG pixels from CLI and HTTP on the same CPU inputs. With UI dependencies absent, training/inference still import. Run the existing suite plus the new targeted checks; no full training run is necessary to verify command/web wiring. The package must neither import `demo` nor contact its former API.

Operational shutdown is a separate already-required action, not a design alternative: identify and retire the actual Firebase Hosting deployment and Cloud Run service, then verify the old URLs no longer serve the product. The checked-in examples in the deployment guide are not authoritative resource identities (`demo/docs/deploy.md:254`); the frontend and Firebase binding give leads (`demo/web_ui/public/script.js:4`, `demo/web_ui/.firebaserc:3`). Capture shutdown evidence without deleting a whole potentially shared cloud project. This review made no cloud changes, and deleting local deployment files cannot be counted as shutdown verification.
