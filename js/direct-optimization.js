import { ContinuousNCA, optimizeStep } from './continuous-nca.js';
import { ReservoirEpiplexity } from './reservoir-epiplexity.js';

/** Owns the continuous experiment; discrete estimates and zoo retain their units. */
export class DirectOptimization {
  constructor() {
    this.root = document.getElementById('direct-mode');
    this.canvas = document.getElementById('continuous-canvas');
    this.graph = document.getElementById('reservoir-graph');
    this.status = document.getElementById('direct-status');
    this.startButton = document.getElementById('optimize-btn');
    this.playButton = document.getElementById('continuous-play');
    this.seed = 1;
    this.running = false;
    this.busy = false;
    this.playing = false;
    this.history = [];
    this.evaluations = [];
    this.stepCount = 0;
    this.startButton.addEventListener('click', () => this.running ? this.stop() : this.start());
    this.playButton.addEventListener('click', () => this.playing ? this.pause() : this.play());
    document.getElementById('continuous-reset').addEventListener('click', () => this.resetPreview());
    document.getElementById('continuous-randomize').addEventListener('click', () => { this.seed++; this.newExperiment(); });
    document.getElementById('export-continuous').addEventListener('click', () => this.exportRule());
    for (const id of ['direct-horizon', 'direct-width', 'direct-features', 'direct-rate',
      'continuous-channels', 'continuous-depth', 'continuous-kernel', 'reservoir-depth', 'reservoir-kernel']) {
      document.getElementById(id).addEventListener('change', () => this.newExperiment());
    }
  }

  get config() {
    return {
      horizon: Number(document.getElementById('direct-horizon').value),
      size: Number(document.getElementById('direct-width').value),
      channels: Number(document.getElementById('direct-features').value),
      reservoirDepth: Number(document.getElementById('reservoir-depth').value),
      reservoirKernel: Number(document.getElementById('reservoir-kernel').value),
      ncaChannels: Number(document.getElementById('continuous-channels').value),
      ncaDepth: Number(document.getElementById('continuous-depth').value),
      ncaKernel: Number(document.getElementById('continuous-kernel').value),
      learningRate: Number(document.getElementById('direct-rate').value),
      batch: 2, burnIn: 32, reservoirSeed: 42, lambda: 0.3
    };
  }

  show() { if (!this.model) this.newExperiment(); }
  hide() { this.stop(); this.pause(); }

  newExperiment() {
    if (this.busy) return;
    this.pause();
    this.model?.dispose(); this.observer?.dispose(); this.optimizer?.dispose();
    const c = this.config;
    this.model = new ContinuousNCA({ seed: this.seed, channels: c.ncaChannels, depth: c.ncaDepth, kernelSize: c.ncaKernel });
    this.observer = new ReservoirEpiplexity({ channels: c.channels, depth: c.reservoirDepth,
      kernelSize: c.reservoirKernel, seed: c.reservoirSeed, lambda: c.lambda });
    this.optimizer = tf.train.adam(c.learningRate);
    this.stepCount = 0; this.history = []; this.evaluations = [];
    this.startButton.textContent = 'Optimize Epiplexity';
    document.getElementById('direct-export').hidden = true;
    document.getElementById('direct-export-json').value = '';
    if (this.exportUrl) URL.revokeObjectURL(this.exportUrl);
    this.exportUrl = null;
    this.status.textContent = `Rule seed ${this.seed} · Ready. Settings start a new experiment.`;
    document.getElementById('direct-score').textContent = '—';
    document.getElementById('direct-evaluation').textContent = '—';
    document.getElementById('direct-step').textContent = '0';
    const parameters = this.model.weights.reduce((sum, w) => sum + w.size, 0);
    const field = 1 + (c.reservoirDepth - 1) * (c.reservoirKernel - 1);
    document.getElementById('continuous-architecture').textContent = `${parameters.toLocaleString()} trainable parameters · ${c.ncaKernel} × ${c.ncaKernel} neighborhood per step`;
    document.getElementById('reservoir-architecture').textContent = `${c.channels} features per cell · ${field} × ${field} receptive field · frozen weights`;
    document.getElementById('prediction-summary').textContent = `Current neighborhood → ${c.horizon} future steps × 2 state channels`;
    this.resetPreview(); this.drawGraph();
  }

  resetPreview() {
    this.state?.dispose();
    this.state = this.model.randomState(1, 48, this.seed + 90000);
    this.canvas.width = 48; this.canvas.height = 48;
    this.drawPreview();
  }

  drawPreview() {
    const pixels = this.state.dataSync();
    const ctx = this.canvas.getContext('2d');
    const image = ctx.createImageData(48, 48);
    for (let i = 0; i < pixels.length / 2; i++) {
      const angle = Math.atan2(pixels[2 * i + 1], pixels[2 * i]);
      for (let c = 0; c < 3; c++) image.data[4 * i + c] = Math.round(128 + 110 * Math.cos(angle + c * 2 * Math.PI / 3));
      image.data[4 * i + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  advancePreview() {
    const next = this.model.step(this.state);
    this.state.dispose(); this.state = next;
    this.drawPreview();
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this.playButton.textContent = 'Pause';
    const tick = () => {
      if (!this.playing) return;
      this.advancePreview();
      this.playTimer = setTimeout(tick, 100);
    };
    tick();
  }

  pause() {
    this.playing = false; clearTimeout(this.playTimer);
    this.playButton.textContent = 'Play';
  }

  lock(locked) {
    this.root.querySelectorAll('select, input, button').forEach(el => {
      if (el !== this.startButton) el.disabled = locked;
    });
  }

  stop() {
    this.running = false;
    if (this.busy) {
      this.startButton.disabled = true;
      this.startButton.textContent = 'Stopping…';
    }
  }

  evaluate() {
    // Fixed independent seed, never used by the optimizer. Re-burn under the
    // current rule so the score describes its current attractor dynamics.
    const state = this.model.warmState({ ...this.config, seed: 700000 });
    try {
      const score = tf.tidy(() => this.observer.scoreFeatures(
        this.observer.features(state), this.model.rollout(state, this.config.horizon)).dataSync()[0]);
      if (!Number.isFinite(score)) throw new Error('Non-finite evaluation score.');
      this.evaluations.push({ step: this.stepCount, score });
      const initial = this.evaluations[0].score;
      document.getElementById('direct-evaluation').textContent = `${score.toFixed(2)} bits (initial ${initial.toFixed(2)})`;
    } finally { state.dispose(); }
  }

  async start() {
    if (this.busy) return;
    this.running = true; this.busy = true; this.lock(true);
    this.startButton.textContent = 'Stop Optimization';
    const steps = Number(document.getElementById('direct-steps').value);
    const target = this.stepCount + steps;
    let failed = false;
    try {
      // Use the same playback timer throughout training and after it finishes.
      this.play();
      this.status.textContent = 'Evaluating the starting rule…';
      await tf.nextFrame();
      if (!this.running) return;
      if (!this.evaluations.length) this.evaluate();
      while (this.running && this.stepCount < target) {
        this.status.textContent = `Optimizing rule ${this.seed} · ${this.stepCount + 1} / ${target}`;
        await tf.nextFrame();
        if (!this.running) break;
        const started = performance.now();
        const state = this.model.warmState({ ...this.config, seed: this.seed * 10000 + 2 * this.stepCount });
        let result;
        try { result = optimizeStep(this.model, this.observer, this.optimizer, state, this.config.horizon); }
        finally { state.dispose(); }
        this.stepCount++;
        this.history.push({ step: this.stepCount, ...result });
        document.getElementById('direct-score').textContent = result.score.toFixed(2);
        document.getElementById('direct-step').textContent = String(this.stepCount);
        if (this.stepCount % 10 === 0 || this.stepCount === target) this.evaluate();
        this.drawGraph();
        this.status.textContent = `Step ${this.stepCount} · ${(performance.now() - started).toFixed(0)} ms · gradient norm ${result.gradientNorm.toFixed(4)}`;
      }
      if (this.evaluations.at(-1)?.step !== this.stepCount) this.evaluate();
      this.drawGraph();
    } catch (error) {
      failed = true;
      this.pause();
      console.error('Direct optimization failed', error);
      this.status.textContent = `Optimization stopped: ${error.message}`;
    } finally {
      this.running = false; this.busy = false; this.lock(false);
      this.startButton.disabled = false;
      this.startButton.textContent = this.stepCount ? 'Continue Optimization' : 'Optimize Epiplexity';
      if (!failed) {
        this.status.textContent = `${this.stepCount >= target ? 'Finished' : 'Paused'} at step ${this.stepCount}. ${this.playing ? 'Preview playing.' : 'Play to inspect the learned rule.'} Continue training to optimize further.`;
      }
    }
  }

  drawGraph() {
    const ctx = this.graph.getContext('2d');
    const w = this.graph.width = 384, h = this.graph.height = 230;
    ctx.fillStyle = '#0a0a15'; ctx.fillRect(0, 0, w, h);
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = '#a3acba';
    ctx.fillText('Reservoir epiplexity (bits) ↑', 14, 20);
    const max = Math.max(1, ...this.history.map(p => p.score), ...this.evaluations.map(p => p.score)) * 1.15;
    const end = Math.max(10, this.stepCount);
    for (let i = 0; i <= 4; i++) {
      const y = 190 - i * 37;
      ctx.strokeStyle = '#252a3a'; ctx.beginPath(); ctx.moveTo(42, y); ctx.lineTo(370, y); ctx.stroke();
      ctx.fillStyle = '#8892a0'; ctx.fillText((max * i / 4).toFixed(1), 8, y + 4);
    }
    for (const [points, color] of [[this.history, '#4ecdc4'], [this.evaluations, '#ffd166']]) {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      points.forEach((p, i) => {
        const x = 42 + 328 * p.step / end, y = 190 - 148 * p.score / max;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (points.length === 1) {
        ctx.fillStyle = color; ctx.beginPath();
        ctx.arc(42 + 328 * points[0].step / end, 190 - 148 * points[0].score / max, 3, 0, 2 * Math.PI); ctx.fill();
      }
    }
    ctx.fillStyle = '#8892a0'; ctx.fillText('0', 42, 210); ctx.fillText(`Step ${end}`, 312, 210);
  }

  exportRule() {
    const checkpoint = {
      format: 'continuous-nca-reservoir-v1', tensorflowVersion: tf.version.tfjs,
      seed: this.seed, config: this.config,
      step: this.stepCount,
      weights: this.model.weights.map(w => ({ shape: w.shape, values: Array.from(w.dataSync()) })),
      history: this.history, evaluations: this.evaluations
    };
    const json = JSON.stringify(checkpoint);
    if (this.exportUrl) URL.revokeObjectURL(this.exportUrl);
    this.exportUrl = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const link = document.getElementById('direct-export-download');
    link.href = this.exportUrl; link.download = `nca-rule-${this.seed}-step-${this.stepCount}.json`;
    document.getElementById('direct-export-json').value = json;
    const details = document.getElementById('direct-export');
    details.hidden = false; details.open = true;
  }
}
