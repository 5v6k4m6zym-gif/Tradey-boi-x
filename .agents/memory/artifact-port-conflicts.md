---
name: Artifact workflow port conflicts in multi-artifact projects
description: How a failed artifact workflow turned out to be a port collision with an unrelated app, not broken code
---

A registered artifact's `artifact.toml` can end up with a dev command or `localPort`
copy-pasted from a different artifact in the same project (e.g. via templating or
manual edits), causing it to try to bind the same port as an already-running,
unrelated service (EADDRINUSE on startup).

**Why:** In one project, `artifacts/api-server` (an Express/Node scaffold, id
`3B4_FFSkEVBkAeYMFRJ2e`) had its dev command wrongly set to launch the project's
actual Streamlit dashboard (`cd tradey-boi-x && streamlit run dashboard.py`).
Fixing the command alone still failed because the corrected Express server then
tried to bind port 5000, which is hardcoded in the real dashboard's
`.streamlit/config.toml` and already in use.

**How to apply:** When an artifact workflow fails, check both (1) that its dev
command actually belongs to that artifact, and (2) that its `localPort` doesn't
collide with another already-running service's hardcoded port. Give unrelated/
scaffold artifacts their own distinct port rather than touching the working app's
port config.
