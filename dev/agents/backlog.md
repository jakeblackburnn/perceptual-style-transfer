# Backlog: models to train and experiments

From the 2026-10-03 workshop (`dev/workshop/rework.md`). Nothing here has been run. Record runs
with `/experiment`. Detail and evidence: `dev/brainstorm/rework/models-to-train.md` and
`future-experiments.md`.

All new runs use the paper-matching loss; style weights are untested starting points (old value
÷ 85,000) and total variation starts at 1e-6. Time estimates are extrapolated from archived
`metrics.csv` files on this machine, not measured.

## Models, in order
| # | Model | Hypothesis | Config it needs | Est. time | Worked if |
|---|---|---|---|---|---|
| 1 | `kanagawa_long` | 10× more steps on photo content gives real wave texture, and a style weight near 2.5 is balanced under the new loss | small, `standard`, VOC content, about 20,000 steps (registered by slice A) | 1–2 h | more texture than `mini_kanagawa` on the frog set with the frog still readable; if not, sweep style weight 0.5 / 2.5 / 10 |
| 2 | `starry_night_v2` | the weak Starry Night models are undertrained and under-weighted | small, schedule from run 1; needs the style image kept at native aspect (`dataset.py:97` squashes it) | as run 1 | swirls in flat regions |
| 3 | `lines` | a colourless line-art style shows whether the model learns strokes | `singles/lines.jpg`, small | as run 1 | output reads as a line drawing |
| 4 | `rainbows` | a small coherent style set trains a usable palette model | `style/rainbows/` (8), multi-image style | as run 1 | consistent palette across content |
| 5 | `ukiyo_e` | a large style set gives "woodblock in general" | `Ukiyo_e/` (1,167) | longer | flat colour and outlines without Wave foam |
| 6 | `snakeskin` | the `shallow` preset suits fine regular texture | `singles/python.jpg`, `shallow` | as run 1 | scales at a consistent size |
| 7 | `collioure_v2` | lower style weight keeps the content the three old runs lost | small, low weight | as run 1 | frog recognisable with dots |
| 8 | `kanagawa_multires` | a resolution curriculum makes stroke size hold across input sizes | stages at 256, 384, 512 | 2–4 h | similar strokes on 512 and 1536 px inputs |

After run 1: retrain the four default models under the new loss so the shipped set is consistent.

## Experiments (from `future-experiments.md`; its numbering)
| # | Experiment | Hypothesis | Needs | Kills it |
|---|---|---|---|---|
| E1 | Fixed benchmark | a fixed content set plus logged content/style/TV terms makes runs comparable | component logging in `train.py`; a frozen content list | – (infrastructure) |
| E2 | Layer weighting | early-heavy and late-heavy weights give visibly different texture scales | nothing new once slice A lands | outputs indistinguishable from unit weights |
| E6 | Resolution | multi-resolution training keeps style across output scales | model 8 above | no difference from single-resolution |
| E7 | Style sets | a coherent set beats one painting; an incoherent one averages away | models 4 and 5 above | set model is muddier than the single |
| E4 | Style statistics | Gram alternatives (mean/variance) are enough | new loss option | visibly worse texture |
| E11 | Diffusion objective | the denoising target fights the style term when content is the clean target | a controlled run in `diffuser/` | – |
| E8 | One model, many styles | conditional instance norm gives selectable styles in one model | new generator variant | per-style quality drops |
| E9 | Video | a temporal term removes flicker | video pipeline | – |
| E10 | Fast local inference | ONNX or quantized inference is worth it | repaired exporter | no meaningful speedup |
| E12 | Pretrained diffusion guidance | perceptual-loss guidance stylizes a pretrained denoiser | the parked `PerceptualLoss` object | – |

Order suggested there by learning per hour: E1, E2, E11, E6, E7. By "cool result": E8, E9, E12.

## Parked decisions that gate experiments
- `PerceptualLoss` object owning its VGG (workshop thread 5): needed before E12.
