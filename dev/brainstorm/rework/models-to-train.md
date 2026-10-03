# Models to train next: options for the workshop

Written by Claude (the Codex worker for this angle hit its usage limit before writing). Options
with a recommendation; nothing here is decided and nothing was trained.

Evidence: the registered configs, every `artifacts/models/*/metrics.csv`, and contact sheets I
built from `artifacts/outputs/` (two frog images per model) and from the style folders. Visual
judgments are mine from 230 px thumbnails of two content images, so treat them as a first pass.

## What exists

27 experiments are registered (`style_transfer/config/styles/__init__.py:10-15`); 23 have a final
checkpoint under `artifacts/models/`. Not trained: `kanagawa_voc`, `kanagawa_large_content`,
`kanagawa_custom_layers`, `super_collioure`. All 23 archived names are still in the config.

| Style | Experiment | Size | Layer preset | Style weight | Res × epochs | Loss first → last | Look (my read) |
|---|---|---|---|---|---|---|---|
| Kanagawa | `kanagawa` | medium | standard | 4e4 | 256 × 6 | 14.0 → 7.5 | palette shift, light texture |
| | `weighted_kanagawa` | medium | standard_weighted | 4e4 | 256 × 6 | 13.9 → 7.9 | same as `kanagawa` |
| | `high_kanagawa` | medium | standard | 8e4 | 256 × 6 | 21.8 → 10.2 | stronger hatching |
| | `small_kanagawa` | small | standard | 4e4 | 256 × 6 | 13.9 → 7.5 | close to `kanagawa` |
| | `small_high_kanagawa` | small | standard | 8e4 | 256 × 6 | 19.2 → 9.9 | close to `high_kanagawa` |
| | `mini_kanagawa` | small | standard | 2e5 | 256 × 6 | 25.7 → 12.9 | strongest line work; best of the family |
| | `shallow_kanagawa` | medium | shallow | 4e4 | 256 × 6 | 20.7 → 9.3 | colour only, content intact |
| | `deep_kanagawa` | medium | deep | 4e4 | 256 × 6 | 9.2 → 3.6 | content destroyed |
| | `deep_weighted_kanagawa` | medium | deep_weighted | 4e4 | 256 × 6 | 15.4 → 7.5 | mild, clean |
| | `big_kanagawa` | big | standard | 4e4 | 256 × 6 | 16.5 → 7.6 | no outputs archived |
| | `super_kanagawa` | big | standard | 8e4 | 256 × 6 | 20.7 → 8.9 | patterned artifacts at edges |
| Collioure | `port_of_collioure` | medium | standard | 8e4 | 256 × 6 | 33.1 → 16.4 | pointillist dots, content mostly lost |
| | `shallow_collioure` | small | shallow | 8e4 | 256 × 6 | 34.4 → 12.9 | same problem |
| | `mini_collioure` | small | standard_x_shallow | 2e5 | 256 × 6 | 58.7 → 22.3 | same problem |
| Starry Night | `starry_night` | medium | standard | 4e4 | 256 × 6 | 11.4 → 6.1 | blue tint, almost no swirl |
| | `high_starry_night` | medium | standard | 2e5 | 256 × 6 | 24.8 → 10.7 | visible strokes, dark |
| | `mini_starry_night` | small | standard | 8e4 | 364 × 12 | 12.8 → 5.7 | blue tint, almost no swirl |
| colors1 | `colors1` | medium | standard | 4e4 | 256 × 6 | 26.0 → 15.6 | vivid, clean |
| | `mini_colors1` | small | standard | 8e4 | 364 × 12 | 17.0 → 8.4 | vivid, clean |
| | `super_colors1` | big | standard | 8e4 | 256 × 6 | 18.5 → 9.9 | red/blue blob artifacts |
| colors2 | `colors2` | medium | standard | 8e4 | 256 × 6 | 15.2 → 9.2 | warm impasto, good |
| | `mini_colors2` | small | standard | 8e4 | 364 × 12 | 11.9 → 6.7 | good |
| (test) | `kanagawa_dry_run` | small | standard | default | 32 × 4 | 334.6 → 193.6 | not a model |

Losses are not comparable across rows with different style weights or presets.

## What the evidence says

1. **Every model is trained for very few steps.** The content set is 7–15% of 13,060
   Impressionism images, so 914–1,959 images, batch 4, 6 epochs: roughly 1,400–2,900 optimizer
   steps. Johnson et al. train about 40,000 steps at batch 4. Recorded epoch times add up to
   roughly 3 to 20 minutes per run, so the short schedules were not forced by cost. This
   is the cheapest untested lever: no archived model answers "what does 10–20× more training
   look like".
2. **The layer-weight experiments tested nothing.** `style_layer_weights` is read at
   `style_transfer/loss.py:29` and never used, so `standard_weighted` and `standard_x_shallow`
   are the same objective as `standard`. `weighted_kanagawa` matches `kanagawa` in both metrics
   and output, as that predicts. `mini_collioure` and `kanagawa_large_content` are affected the
   same way. Any new weighted run waits on that fix.
3. **`kanagawa_custom_layers` cannot train.** Its preset lives only in
   `style_transfer/config/styles/kanagawa.py:7`, and `train.py:101` never passes that dictionary
   to `initialize_vgg`, so it raises `KeyError`.
4. **Bigger did not help.** Both big models with outputs show artifacts the small and medium
   ones do not, at 7× the parameters (11.4 M against 1.7 M and 0.78 M) and 45.6 MB per file.
   With under 3,000 steps that may be undertraining, not capacity. Small is as good as medium on
   Kanagawa and colors.
5. **Style weight is the dial that visibly matters.** Kanagawa and Starry Night only show
   stroke structure at 2e5; at 4e4 they are colour filters.
6. **Content set is always paintings.** Only the untrained `kanagawa_voc` uses photographs, yet
   every test image is a photograph. Training on paintings and applying to photos is an
   untested mismatch.
7. **Style images are squashed to a square.** `SingleImageDataset` resizes the style image to
   `res × res` (`style_transfer/dataset.py:97`). The Wave is 3:2 and Starry Night 5:4, so
   strokes are distorted, and stroke scale is fixed by the training resolution.
8. **No multi-image style model exists.** `ImageDataset` supports a style folder
   (`train.py:126`) but all 27 experiments set `single: True`. The style folders below have
   never been used.

Redundant today: `weighted_kanagawa` (identical objective to `kanagawa`), `small_kanagawa` and
`small_high_kanagawa` (same look as their medium twins), `starry_night` and `mini_starry_night`
(both fail the same way).

## Unused style material

| Source | Count | What it is (viewed) |
|---|---|---|
| `artifacts/images/singles/lines.jpg` | 1 | black-and-white flowing line art |
| `singles/snakeskin.jpg`, `python.jpg` | 2 | scale patterns; snakeskin is a strip of four swatches |
| `singles/cantelope.jpg` | 1 | green rind with a cream net pattern |
| `singles/iris13.png`, `iris18.png` | 2 | colour-inverted iris photos |
| `style/rainbows/` | 8 | saturated false-colour mountain landscapes |
| `style/iris/`, `orchid/`, `rose/` | 26, 9, 6 | recoloured flower photos, one subject per set |
| `style/*-swaps/` | 27 each | channel-swap variants of the same flowers |
| `Ukiyo_e/` | 1,167 | woodblock prints, many artists |

## Default set for the CLI and local UI

The demo ships `mini_kanagawa`, `mini_colors` (a copy of `mini_colors1`) and `high_starry_night`.

Recommendation: ship small models only (3.1 MB each), and ship four: `mini_kanagawa`,
`mini_colors1`, `mini_colors2`, plus `high_starry_night` as a placeholder until a better Starry
Night exists. Leave Collioure, every big model and the deep variants out. The alternative is to
ship nothing until the retrained set below exists; that delays a usable CLI for no gain.

## Shortlist to train, ranked

Each run is minutes on this machine at current settings; the long ones are about 1–2 hours.
Timings are extrapolated from the archived `metrics.csv` files, not measured for these configs.

| # | Model | Style source | Setup | Why | Worked if |
|---|---|---|---|---|---|
| 1 | `kanagawa_long` | Wave | small, standard, 2e5, VOC content, about 20,000 steps | answers finding 1 and 6 at once on the best-understood style; becomes the reference | visibly more wave texture than `mini_kanagawa` on the frog set, content still readable |
| 2 | `starry_night_v2` | Starry Night | small, 2e5, style image kept at native aspect, long schedule | the most recognisable style is the weakest model | swirls visible in flat regions, not just a blue cast |
| 3 | `lines` | `lines.jpg` | small, standard, start at 2e5 | pure structure, no colour: the clearest test of whether the model learns strokes | output reads as line drawing |
| 4 | `rainbows` | `style/rainbows/` (8) | small, multi-image style, batch 4 | first multi-image model; images share a palette so the averaged Gram target is coherent | consistent palette across content images |
| 5 | `ukiyo_e` | `Ukiyo_e/` (1,167) | small, multi-image style | "woodblock in general" against the single Wave | flat colour and outlines without Wave-specific foam |
| 6 | `snakeskin` | `python.jpg` | small, shallow preset | fine regular texture is what the shallow preset is for | scales at a consistent size |
| 7 | `collioure_v2` | Collioure | small, 4e4, long schedule | three attempts all lost the content; lower weight is untried | frog recognisable with dots |
| 8 | `kanagawa_multires` | Wave | small, stages at 256, 384, 512 | the only one aimed at "arbitrary dimension" | stroke size stays similar on a 512 and a 1536 px input |

Do first: 1, then 2 and 3. Run 1 decides the schedule for all the others, so the rest should not
start before it is looked at.

Skip for now: more big models (finding 4), any `*_weighted` variant until the loss fix, the
`-swaps` sets (27 near-duplicates of one image give a muddy averaged target), and `iris`,
`orchid`, `rose` as styles (they are photographs; the model would learn a colour cast).

## Arbitrary dimensions

The network is fully convolutional, so any size runs, but two things limit it:
- Output size is rounded up to a multiple of four: 65×97 in gives 68×100 out (I ran this). That
  is an inference fix (pad and crop), not a training matter.
- Every model has seen one resolution, 256 or 364. Stroke size is fixed in pixels, so a 3000 px
  photo gets tiny strokes and a 200 px one gets coarse ones. Options: train with a resolution
  curriculum (run 8; the curriculum format already takes `res` per stage,
  `config/curricula.py:46`), or leave training alone and add a `--max-side` option so the user
  picks the stroke scale. The second is nearly free and worth doing regardless.

## What a saved model should carry

Today the final file is a bare state dict (`train.py:174`) and the size comes from the current
config (`inference.py:111`), so renaming or editing an experiment breaks loading an old file.
Minimum next to each final checkpoint: model size, layer preset as resolved values, style
weight, style source, content source and count, steps trained, and the commit. See
`cli-and-local-server.md` for the manifest proposal and `code-review.md` finding 4 for the
checkpoint-envelope alternative; the workshop should pick one.

## Housekeeping (owner's call)

`artifacts/models/` is 4.4 GB, of which the 23 final files are about 0.24 GB. The rest is
per-epoch checkpoints that also store optimizer state. If nothing resumes from them, they can go.
