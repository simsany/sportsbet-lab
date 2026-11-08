# ProphitBet Heroku launcher

This repository is a deployment wrapper. At image-build time it downloads the original
ProphitBet repository at a pinned upstream commit and runs its original PySide6 desktop GUI
inside a virtual X11 display. noVNC streams that desktop to the browser.

**The upstream `src/core` backend is not patched, copied over, or reimplemented.**
The GUI is also the original Qt GUI, not a browser rewrite.

## Upload to your GitHub `sportsbet-lab` main branch
Replace the contents of that repository with the contents of this ZIP:
- `Dockerfile`
- `heroku.yml`
- `.dockerignore`
- `deploy/entrypoint.sh`
- `deploy/README-HEROKU.md`

The Docker build downloads the original ProphitBet source itself, pinned to commit
`49fa5ff14783af5dbbac7919ec743ab41628fcac`. No ProphitBet source files need to be copied
into this launcher repository.

## Deploy to Heroku
Create/select your Heroku app, then set a VNC password (at least 8 characters):
`heroku config:set VNC_PASSWORD='Choose-A-Strong-Password' -a YOUR_APP`

From the repository root:
`heroku login`
`heroku container:login`
`heroku container:push web -a YOUR_APP`
`heroku container:release web -a YOUR_APP`

Open `https://YOUR_APP.herokuapp.com`; noVNC should display the original ProphitBet desktop
and prompt for the VNC password.

Optional display resolution:
`heroku config:set SCREEN_SIZE=1920x1080x24 -a YOUR_APP`

## Critical limitations to resolve before relying on it
- Heroku filesystem is ephemeral. ProphitBet's local SQLite databases, datasets, trained
  models, settings and exports can disappear on dyno restart/replacement/redeploy. This
  launcher deliberately does not change the backend to add persistence. Keep backups and
  do not treat Heroku local storage as durable.
- Dependencies include TensorFlow, PyTorch, XGBoost, LightGBM and Qt. Container build time,
  image size and RAM use need validation on your Heroku plan. Training may require a larger
  dyno. Do not reduce model settings to hide resource failures.
- Upstream specifies 64-bit CPython 3.11; the image uses Python 3.11.
- Graphviz binaries are installed in addition to its Python package.
- Secure the Heroku app and use a unique VNC password. Do not publish the URL/password.
- First verify app startup, opening a league, and a small training/evaluation run. Compare
  the same data, configuration, seed and dependency versions with desktop output.
