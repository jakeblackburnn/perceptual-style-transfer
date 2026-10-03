# Journal
## Oct 3 2026
reworking style transfer - I wanna remove the website from the internet, and switch the project
to just be cli based only, or maybe make the UI locally run only for ease of use. 

today I wanna run a brainstorm and workshop session for
this rework of the project, critically assess the whole codebase 
then make decisions about what to change and where the project goes, 
and what additional models to train, and also to make notes about future 
experiments to make. 

the process should use codex with maximally capable models to perform brainstorming.
the workshopping agent should spawn the codex models for brainstorming
and use additional subagents for implementation when decisions are set.

basic end goal: make the project easy to use with cli and a local web server
and refactor the code to be better, and make some cool models.

## results:

## Results · 2026-10-03 16:22 (agent)
1. Remove the website from the internet: not started — yours to run; commands in `dev/agents/journal.md` Next 1. README links are gone.
2. CLI-based, UI local only: done — `style-transfer list|stylize|train|serve`, `style_transfer_web/` on 127.0.0.1 (`fc5b4fa`); `pip install -e .` not yet tried.
3. Brainstorm and workshop, critically assess the codebase: done — `dev/brainstorm/rework/`, `dev/workshop/rework.md`.
4. Decide what to change and where the project goes: done — five decided, one parked (`dev/workshop/rework.md`).
5. What additional models to train: done as a list, none trained — `dev/agents/backlog.md`; first is `kanagawa_long`.
6. Notes on future experiments: done — `dev/agents/backlog.md`, `dev/brainstorm/rework/future-experiments.md`.
7. Codex for brainstorming, subagents for implementation: partial — Codex hit its usage limit; three notes are Codex's, one is Claude's; implementation by three Claude subagents.
8. Refactor the code: partial — loss, training output, inference, CLI, packaging done; follow-ups in `dev/workshop/rework.md`.
9. Make some cool models: not started — no training today, by your choice.
