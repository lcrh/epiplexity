# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **static client-side demo** (vanilla HTML/CSS/JS ES modules + TensorFlow.js from CDN). There is no package manager, build step, backend, or test suite.

### Run

Serve the repo root over HTTP (ES modules will not load via `file://`):

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`.

### Lint / test / build

- No lint, unit-test, or build tooling is configured in this repository.
- Manual verification: load the page, click **Play**, then **Randomize NCA** (and optionally **Mutate**) and confirm the canvas pattern updates.

### Runtime notes

- TensorFlow.js is loaded from jsDelivr (`@tensorflow/tfjs` and `@tensorflow/tfjs-backend-webgpu`). The app tries WebGPU, then WebGL, and may fall back to **CPU** in headless/cloud environments without a usable GPU.
- Epiplexity estimation (**Estimate Epiplexity** / **Auto-Evolve**) trains a transformer in-browser and can be slow on CPU; NCA Play/Randomize/Mutate is enough for a basic smoke check.
- External CDN access (fonts.googleapis.com, fonts.gstatic.com, cdn.jsdelivr.net) is required for full functionality.
