/**
 * Main Application Module
 *
 * Initializes the NCA simulation and epiplexity estimation.
 */

import { NCAModel } from './nca-model.js';
import { Renderer } from './renderer.js';
import { EpiplexityModel } from './epiplexity-model.js';
import { LossGraph } from './loss-graph.js';

class NCAApp {
  constructor() {
    // NCA Configuration
    this.ncaConfig = {
      width: 64,
      height: 64,
      numConvLayers: 1,
      internalChannels: 4,
      kernelSize: 3,
      targetFps: 10
    };

    // Epiplexity Configuration
    this.epiConfig = {
      burnInSteps: 50,
      predictionHorizon: 5,
      maxTrainSteps: 500,
      earlyStopPatience: 100,
      dModel: 64,
      numLayers: 2,
      minFinalLoss: 0.05
    };

    // NCA State
    this.state = null;
    this.ncaModel = null;
    this.renderer = null;
    this.isRunning = false;
    this.animationId = null;

    // Animation timing
    this.lastFrameTime = 0;

    // Epiplexity State
    this.epiplexityModel = null;
    this.lossGraph = null;
    this.isTraining = false;
    this.trainingStep = 0;
    this.trainingAnimationId = null;
    this.trainingData = []; // Pre-built dataset of {input, target} pairs
    this.bestSmoothedLoss = Infinity;
    this.stepsSinceImprovement = 0;
    this.lastTrainedEpiplexity = null;
    this.lastTrainedWeights = null;
    this.lastTrainedConfig = null;
    this.burnInSnapshot = null; // Canvas snapshot after burn-in

    // Zoo state
    this.zoo = [];
    this.maxZooSize = 16;
    this.isAutoEvolving = false;
    this.mutationChance = 0.5;
    this.autoEvolveGeneration = 0;
    this.autoEvolveTrainResolve = null;
    this.lastTrainWasBoring = false; // True if final loss was below threshold

    // Bind methods
    this.step = this.step.bind(this);
    this.animate = this.animate.bind(this);
    this.trainStep = this.trainStep.bind(this);
  }

  /**
   * Initialize the application
   */
  async init() {
    // Try WebGPU first, fall back to WebGL
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
    } catch (e) {
      console.log('WebGPU not available, falling back to WebGL');
      await tf.setBackend('webgl');
      await tf.ready();
    }
    console.log('TensorFlow.js backend:', tf.getBackend());

    // Get NCA DOM elements
    this.canvas = document.getElementById('nca-canvas');
    this.playPauseBtn = document.getElementById('play-pause-btn');
    this.resetBtn = document.getElementById('reset-btn');
    this.randomizeBtn = document.getElementById('randomize-btn');
    this.mutateBtn = document.getElementById('mutate-btn');
    this.mutateStrengthSlider = document.getElementById('mutate-strength');
    this.mutateStrengthValue = document.getElementById('mutate-strength-value');
    this.convLayersSlider = document.getElementById('conv-layers');
    this.convLayersValue = document.getElementById('conv-layers-value');
    this.internalChannelsSlider = document.getElementById('internal-channels');
    this.internalChannelsValue = document.getElementById('internal-channels-value');
    this.kernelSizeSelect = document.getElementById('kernel-size');
    this.targetFpsSlider = document.getElementById('target-fps');
    this.targetFpsValue = document.getElementById('target-fps-value');

    // Get Epiplexity DOM elements
    this.lossGraphCanvas = document.getElementById('loss-graph');
    this.epiplexityDisplay = document.getElementById('epiplexity-display');
    this.trainBtn = document.getElementById('train-btn');
    this.burnInStepsSlider = document.getElementById('burn-in-steps');
    this.burnInStepsValue = document.getElementById('burn-in-steps-value');
    this.predictionHorizonSlider = document.getElementById('prediction-horizon');
    this.predictionHorizonValue = document.getElementById('prediction-horizon-value');
    this.maxTrainStepsSlider = document.getElementById('max-train-steps');
    this.maxTrainStepsValue = document.getElementById('max-train-steps-value');
    this.dModelSlider = document.getElementById('d-model');
    this.dModelValue = document.getElementById('d-model-value');
    this.numLayersSlider = document.getElementById('num-layers');
    this.numLayersValue = document.getElementById('num-layers-value');
    this.earlyStopSlider = document.getElementById('early-stop');
    this.earlyStopValue = document.getElementById('early-stop-value');
    this.smoothingSlider = document.getElementById('smoothing');
    this.smoothingValue = document.getElementById('smoothing-value');
    this.minFinalLossSlider = document.getElementById('min-final-loss');
    this.minFinalLossValue = document.getElementById('min-final-loss-value');
    this.trainStepDisplay = document.getElementById('train-step-display');
    this.trainMaxDisplay = document.getElementById('train-max-display');
    this.trainLossDisplay = document.getElementById('train-loss-display');

    // Get Zoo DOM elements
    this.saveZooBtn = document.getElementById('save-zoo-btn');
    this.zooGrid = document.getElementById('zoo-grid');
    this.zooEmpty = document.getElementById('zoo-empty');
    this.autoEvolveBtn = document.getElementById('auto-evolve-btn');
    this.mutationChanceSlider = document.getElementById('mutation-chance');
    this.mutationChanceValue = document.getElementById('mutation-chance-value');
    this.zooStatus = document.getElementById('zoo-status');

    // Initialize NCA renderer
    this.renderer = new Renderer(this.canvas);
    this.renderer.setSize(this.ncaConfig.width, this.ncaConfig.height);

    // Initialize NCA model
    this.ncaModel = new NCAModel({
      numConvLayers: this.ncaConfig.numConvLayers,
      internalChannels: this.ncaConfig.internalChannels,
      kernelSize: this.ncaConfig.kernelSize
    });

    // Initialize NCA state
    this.resetState();

    // Initialize loss graph
    this.lossGraph = new LossGraph(this.lossGraphCanvas);
    this.lossGraph.setMaxSteps(this.epiConfig.maxSteps);

    // Set up event listeners
    this.setupNCAEventListeners();
    this.setupEpiplexityEventListeners();
    this.setupZooEventListeners();

    // Initial render
    await this.renderer.draw(this.state);

    console.log('App initialized');
  }

  /**
   * Set up NCA UI event listeners
   */
  setupNCAEventListeners() {
    this.playPauseBtn.addEventListener('click', () => {
      this.toggleRunning();
    });

    this.resetBtn.addEventListener('click', () => {
      this.resetState();
      this.renderer.draw(this.state);
    });

    this.randomizeBtn.addEventListener('click', () => {
      this.ncaModel.randomize();
      this.resetState();
      this.renderer.draw(this.state);
      console.log('NCA weights and state randomized');
    });

    this.mutateBtn.addEventListener('click', () => {
      const strength = parseFloat(this.mutateStrengthSlider.value);
      this.mutateNCA(strength);
    });

    this.mutateStrengthSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.mutateStrengthValue.textContent = value.toFixed(2);
    });

    this.convLayersSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.convLayersValue.textContent = value;
      this.ncaConfig.numConvLayers = value;
      this.rebuildNCAModel();
    });

    this.internalChannelsSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.internalChannelsValue.textContent = value;
      this.ncaConfig.internalChannels = value;
      this.rebuildNCAModel();
    });

    this.kernelSizeSelect.addEventListener('change', (e) => {
      const value = parseInt(e.target.value);
      this.ncaConfig.kernelSize = value;
      this.rebuildNCAModel();
    });

    this.targetFpsSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.targetFpsValue.textContent = value;
      this.ncaConfig.targetFps = value;
    });
  }

  /**
   * Set up Epiplexity UI event listeners
   */
  setupEpiplexityEventListeners() {
    this.trainBtn.addEventListener('click', () => {
      if (this.isTraining) {
        this.stopTraining();
      } else {
        this.startTraining();
      }
    });

    this.burnInStepsSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.burnInStepsValue.textContent = value;
      this.epiConfig.burnInSteps = value;
    });

    this.predictionHorizonSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.predictionHorizonValue.textContent = value;
      this.epiConfig.predictionHorizon = value;
    });

    this.maxTrainStepsSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.maxTrainStepsValue.textContent = value;
      this.epiConfig.maxTrainSteps = value;
      this.lossGraph.setMaxSteps(value);
      this.trainMaxDisplay.textContent = value;
    });

    this.dModelSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.dModelValue.textContent = value;
      this.epiConfig.dModel = value;
    });

    this.numLayersSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.numLayersValue.textContent = value;
      this.epiConfig.numLayers = value;
    });

    this.earlyStopSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.earlyStopValue.textContent = value;
      this.epiConfig.earlyStopPatience = value;
    });

    this.smoothingSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.smoothingValue.textContent = value.toFixed(2);
      this.lossGraph.setSmoothingFactor(value);
    });

    this.minFinalLossSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.minFinalLossValue.textContent = value.toFixed(2);
      this.epiConfig.minFinalLoss = value;
    });

    this.saveZooBtn.addEventListener('click', () => {
      this.saveToZoo();
    });
  }

  /**
   * Rebuild NCA model with current config
   */
  rebuildNCAModel() {
    this.ncaModel.updateConfig({
      numConvLayers: this.ncaConfig.numConvLayers,
      internalChannels: this.ncaConfig.internalChannels,
      kernelSize: this.ncaConfig.kernelSize
    });
    console.log('NCA model rebuilt');
  }

  /**
   * Reset NCA state to random noise
   */
  resetState() {
    if (this.state) {
      this.state.dispose();
    }
    this.state = this.renderer.createRandomState(
      this.ncaConfig.width,
      this.ncaConfig.height,
      4 // Fixed: 0=dead, 1-3=alive colors
    );
  }

  /**
   * Toggle NCA running state
   */
  toggleRunning() {
    this.isRunning = !this.isRunning;
    this.playPauseBtn.textContent = this.isRunning ? 'Pause' : 'Play';

    if (this.isRunning) {
      this.animate();
    } else if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }

  /**
   * Run one NCA simulation step
   */
  step() {
    const newState = this.ncaModel.step(this.state);
    this.state.dispose();
    this.state = newState;
  }

  /**
   * NCA animation loop
   */
  async animate() {
    if (!this.isRunning) return;

    const now = performance.now();
    const frameInterval = 1000 / this.ncaConfig.targetFps;

    // Throttle to target FPS
    if (now - this.lastFrameTime >= frameInterval) {
      this.lastFrameTime = now;
      this.step();
      await this.renderer.draw(this.state);
    }

    this.animationId = requestAnimationFrame(this.animate);
  }

  /**
   * Start epiplexity training
   */
  async startTraining() {
    // Pause NCA visualization if running
    if (this.isRunning) {
      this.toggleRunning();
    }

    // Dispose old epiplexity model if exists
    if (this.epiplexityModel) {
      this.epiplexityModel.dispose();
    }

    // Dispose old training data
    this.disposeTrainingData();

    // Update UI to show building dataset
    this.trainBtn.textContent = 'Stop';
    this.trainBtn.disabled = true;
    this.saveZooBtn.disabled = true;
    this.epiplexityDisplay.textContent = '--';
    this.trainStepDisplay.textContent = '0';
    this.trainMaxDisplay.textContent = this.epiConfig.maxTrainSteps;
    this.lossGraph.showMessage('Building dataset...');

    console.log('Building training dataset...');

    // Build dataset (use setTimeout to allow UI update)
    await new Promise(resolve => setTimeout(resolve, 50));
    this.buildTrainingDataset(this.epiConfig.maxTrainSteps);

    // Shuffle dataset
    for (let i = this.trainingData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.trainingData[i], this.trainingData[j]] = [this.trainingData[j], this.trainingData[i]];
    }

    console.log(`Dataset built and shuffled: ${this.trainingData.length} samples`);

    // Create new epiplexity model
    this.epiplexityModel = new EpiplexityModel({
      gridSize: this.ncaConfig.width,
      dModel: this.epiConfig.dModel,
      numLayers: this.epiConfig.numLayers,
      numStates: 4
    });

    // Reset graph and training state
    this.lossGraph.reset();
    this.lossGraph.setMaxSteps(this.epiConfig.maxTrainSteps);
    this.trainingStep = 0;
    this.bestSmoothedLoss = Infinity;
    this.stepsSinceImprovement = 0;
    this.isTraining = true;

    // Update UI
    this.trainBtn.textContent = 'Stop';
    this.trainBtn.disabled = false;
    this.trainBtn.classList.add('training');

    console.log('Started epiplexity training');

    // Start training loop
    this.trainStep();
  }

  /**
   * Stop epiplexity training
   */
  stopTraining() {
    this.isTraining = false;

    // Also stop auto-evolve if running
    if (this.isAutoEvolving) {
      this.stopAutoEvolve();
    }

    if (this.trainingAnimationId) {
      cancelAnimationFrame(this.trainingAnimationId);
      this.trainingAnimationId = null;
    }

    // Finalize graph and compute epiplexity
    const epiplexity = this.lossGraph.finalize();
    this.epiplexityDisplay.textContent = epiplexity.toFixed(2);

    // Save NCA weights and epiplexity for potential zoo save
    this.lastTrainedEpiplexity = epiplexity;
    this.lastTrainedWeights = this.ncaModel.getWeightsData();
    this.lastTrainedConfig = { ...this.ncaConfig };

    // Update UI
    this.trainBtn.textContent = 'Estimate Epiplexity';
    this.trainBtn.classList.remove('training');
    this.saveZooBtn.disabled = false;

    console.log('Training stopped. Epiplexity:', epiplexity.toFixed(2));
  }

  /**
   * Dispose all training data tensors
   */
  disposeTrainingData() {
    for (const pair of this.trainingData) {
      pair.input.dispose();
      pair.target.dispose();
    }
    this.trainingData = [];
  }

  /**
   * Build training dataset by running NCA simulations
   * Extracts multiple pairs per simulation for efficiency
   * @param {number} numSamples - Total number of training pairs needed
   */
  buildTrainingDataset(numSamples) {
    const pairsPerRun = 8;
    const horizon = this.epiConfig.predictionHorizon;
    const pairOffset = 1;

    const numRuns = Math.ceil(numSamples / pairsPerRun);

    for (let run = 0; run < numRuns; run++) {
      // Create random initial state
      let state = this.renderer.createRandomState(
        this.ncaConfig.width,
        this.ncaConfig.height,
        4
      );

      // Run burn-in steps
      for (let i = 0; i < this.epiConfig.burnInSteps; i++) {
        const newState = this.ncaModel.step(state);
        state.dispose();
        state = newState;
      }

      // Capture snapshot after burn-in on first run
      if (run === 0) {
        this.captureBurnInSnapshot(state);
      }

      // Extract pairs from this run
      for (let p = 0; p < pairsPerRun; p++) {
        // Stop if we have enough samples
        if (this.trainingData.length >= numSamples) break;

        // Take input snapshot
        const input = tf.tidy(() => {
          const stateSquashed = state.squeeze([0]); // [H, W, 4]
          return tf.argMax(stateSquashed, 2).reshape([-1]); // [H*W]
        });

        // Run horizon steps
        for (let i = 0; i < horizon; i++) {
          const newState = this.ncaModel.step(state);
          state.dispose();
          state = newState;
        }

        // Take target snapshot
        const target = tf.tidy(() => {
          const stateSquashed = state.squeeze([0]); // [H, W, 4]
          return tf.argMax(stateSquashed, 2).reshape([-1]); // [H*W]
        });

        this.trainingData.push({ input, target });

        // Run offset steps (except after last pair)
        if (p < pairsPerRun - 1) {
          for (let i = 0; i < pairOffset; i++) {
            const newState = this.ncaModel.step(state);
            state.dispose();
            state = newState;
          }
        }
      }

      // Clean up final state
      state.dispose();
    }
  }

  /**
   * One step of epiplexity training
   */
  trainStep() {
    if (!this.isTraining) return;

    // Get training sample from pre-built dataset
    const pair = this.trainingData[this.trainingStep];

    // Perform training step
    const loss = this.epiplexityModel.trainStep(pair.input, pair.target);

    // Record loss
    this.lossGraph.addPoint(this.trainingStep, loss);
    this.trainingStep++;

    // Check for early stopping based on smoothed loss
    const smoothedLoss = this.lossGraph.getCurrentSmoothedLoss();
    if (smoothedLoss !== null) {
      if (smoothedLoss < this.bestSmoothedLoss) {
        this.bestSmoothedLoss = smoothedLoss;
        this.stepsSinceImprovement = 0;
      } else {
        this.stepsSinceImprovement++;
      }
    }

    // Update UI
    this.trainStepDisplay.textContent = this.trainingStep;
    this.trainLossDisplay.textContent = loss.toFixed(3);

    // Check if done (max steps, out of data, or early stopping)
    if (this.trainingStep >= this.trainingData.length ||
        this.trainingStep >= this.epiConfig.maxTrainSteps) {
      this.stopTraining();
      return;
    }

    if (this.stepsSinceImprovement >= this.epiConfig.earlyStopPatience) {
      console.log(`Early stopping at step ${this.trainingStep}`);
      this.stopTraining();
      return;
    }

    // Schedule next step
    this.trainingAnimationId = requestAnimationFrame(this.trainStep);
  }

  /**
   * Reset epiplexity state
   */
  resetEpiplexity() {
    if (this.isTraining) {
      this.stopTraining();
    }

    if (this.epiplexityModel) {
      this.epiplexityModel.dispose();
      this.epiplexityModel = null;
    }

    this.disposeTrainingData();
    this.lossGraph.reset();
    this.trainingStep = 0;

    // Update UI
    this.epiplexityDisplay.textContent = '--';
    this.trainStepDisplay.textContent = '0';
    this.trainLossDisplay.textContent = '--';
    this.trainBtn.textContent = 'Estimate Epiplexity';
    this.trainBtn.classList.remove('training');

    console.log('Epiplexity reset');
  }

  /**
   * Capture burn-in snapshot for zoo thumbnail
   * @param {tf.Tensor} state - Current NCA state
   */
  captureBurnInSnapshot(state) {
    // Create a small canvas for the thumbnail
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Render state to canvas
    const stateData = state.squeeze([0]).arraySync(); // [H, W, 4]
    const imageData = ctx.createImageData(this.ncaConfig.width, this.ncaConfig.height);

    const palette = [
      [12, 12, 24],
      [255, 0, 110],
      [0, 255, 198],
      [255, 234, 0]
    ];

    for (let y = 0; y < this.ncaConfig.height; y++) {
      for (let x = 0; x < this.ncaConfig.width; x++) {
        const idx = (y * this.ncaConfig.width + x) * 4;
        const cell = stateData[y][x];
        const stateIdx = cell.indexOf(Math.max(...cell));
        const color = palette[stateIdx];
        imageData.data[idx] = color[0];
        imageData.data[idx + 1] = color[1];
        imageData.data[idx + 2] = color[2];
        imageData.data[idx + 3] = 255;
      }
    }

    // Draw to temp canvas at original size, then scale
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.ncaConfig.width;
    tempCanvas.height = this.ncaConfig.height;
    tempCanvas.getContext('2d').putImageData(imageData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, size, size);

    this.burnInSnapshot = canvas.toDataURL();
  }

  /**
   * Save current NCA to zoo
   */
  saveToZoo() {
    if (!this.lastTrainedEpiplexity || !this.lastTrainedWeights || !this.burnInSnapshot) {
      console.warn('No trained NCA to save');
      return;
    }

    const entry = {
      id: Date.now(),
      epiplexity: this.lastTrainedEpiplexity,
      weights: this.lastTrainedWeights,
      config: this.lastTrainedConfig,
      thumbnail: this.burnInSnapshot
    };

    this.zoo.push(entry);
    this.zoo.sort((a, b) => b.epiplexity - a.epiplexity); // Descending (highest first)

    this.saveZooBtn.disabled = true;
    this.renderZoo();

    console.log('Saved to zoo. Epiplexity:', entry.epiplexity.toFixed(2));
  }

  /**
   * Load NCA from zoo entry
   * @param {number} id - Zoo entry ID
   */
  loadFromZoo(id) {
    const entry = this.zoo.find(e => e.id === id);
    if (!entry) return;

    // Stop any running simulation
    if (this.isRunning) {
      this.toggleRunning();
    }

    // Update config
    this.ncaConfig.numConvLayers = entry.config.numConvLayers;
    this.ncaConfig.internalChannels = entry.config.internalChannels;
    this.ncaConfig.kernelSize = entry.config.kernelSize;

    // Update UI sliders
    this.convLayersSlider.value = entry.config.numConvLayers;
    this.convLayersValue.textContent = entry.config.numConvLayers;
    this.internalChannelsSlider.value = entry.config.internalChannels;
    this.internalChannelsValue.textContent = entry.config.internalChannels;
    this.kernelSizeSelect.value = entry.config.kernelSize;

    // Rebuild model with correct config and set weights
    this.ncaModel.updateConfig({
      numConvLayers: entry.config.numConvLayers,
      internalChannels: entry.config.internalChannels,
      kernelSize: entry.config.kernelSize
    });
    this.ncaModel.setWeightsData(entry.weights);

    // Reset state, render, and start playing
    this.resetState();
    this.renderer.draw(this.state);

    // Start playing if not already
    if (!this.isRunning) {
      this.toggleRunning();
    }

    console.log('Loaded from zoo. Epiplexity:', entry.epiplexity.toFixed(2));
  }

  /**
   * Delete entry from zoo
   * @param {number} id - Zoo entry ID
   */
  deleteFromZoo(id) {
    this.zoo = this.zoo.filter(e => e.id !== id);
    this.renderZoo();
  }

  /**
   * Render zoo grid
   */
  renderZoo() {
    this.zooGrid.innerHTML = '';

    if (this.zoo.length === 0) {
      this.zooEmpty.style.display = 'block';
      return;
    }

    this.zooEmpty.style.display = 'none';

    this.zoo.forEach((entry, index) => {
      const item = document.createElement('div');
      item.className = 'zoo-item';

      // Thumbnail
      const img = document.createElement('img');
      img.src = entry.thumbnail;
      img.style.width = '100%';
      img.style.aspectRatio = '1';
      img.style.imageRendering = 'pixelated';
      item.appendChild(img);

      // Rank badge
      const rank = document.createElement('div');
      rank.className = 'zoo-item-rank';
      if (index === 0) {
        rank.classList.add('gold');
        rank.textContent = '#1';
      } else if (index === 1) {
        rank.classList.add('silver');
        rank.textContent = '#2';
      } else if (index === 2) {
        rank.classList.add('bronze');
        rank.textContent = '#3';
      } else {
        rank.textContent = `#${index + 1}`;
      }
      item.appendChild(rank);

      // Info overlay
      const info = document.createElement('div');
      info.className = 'zoo-item-info';
      const epiValue = document.createElement('div');
      epiValue.className = 'zoo-item-epiplexity';
      epiValue.textContent = entry.epiplexity.toFixed(2);
      info.appendChild(epiValue);
      item.appendChild(info);

      // Delete button
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'zoo-item-delete';
      deleteBtn.innerHTML = '&times;';
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.deleteFromZoo(entry.id);
      });
      item.appendChild(deleteBtn);

      // Click to load
      item.addEventListener('click', () => {
        this.loadFromZoo(entry.id);
      });

      this.zooGrid.appendChild(item);
    });
  }

  /**
   * Set up Zoo UI event listeners
   */
  setupZooEventListeners() {
    this.autoEvolveBtn.addEventListener('click', () => {
      if (this.isAutoEvolving) {
        this.stopAutoEvolve();
      } else {
        this.startAutoEvolve();
      }
    });

    this.mutationChanceSlider.addEventListener('input', (e) => {
      const value = parseInt(e.target.value);
      this.mutationChanceValue.textContent = `${value}%`;
      this.mutationChance = value / 100;
    });
  }

  /**
   * Start auto-evolve genetic algorithm loop
   */
  async startAutoEvolve() {
    this.isAutoEvolving = true;
    this.autoEvolveGeneration = 0;
    this.autoEvolveBtn.textContent = 'Stop';
    this.autoEvolveBtn.classList.add('evolving');
    this.zooStatus.classList.add('active');

    console.log('Starting auto-evolve');
    this.runAutoEvolveIteration();
  }

  /**
   * Stop auto-evolve loop
   */
  stopAutoEvolve() {
    this.isAutoEvolving = false;
    this.autoEvolveBtn.textContent = 'Auto-Evolve';
    this.autoEvolveBtn.classList.remove('evolving');
    this.zooStatus.classList.remove('active');
    this.zooStatus.textContent = '';

    // Stop ongoing training
    if (this.isTraining) {
      this.isTraining = false;
      this.trainBtn.textContent = 'Estimate Epiplexity';
      this.trainBtn.classList.remove('training');
      this.saveZooBtn.disabled = false;
    }

    // Stop NCA playback
    if (this.isRunning) {
      this.toggleRunning();
    }

    // Resolve any pending promise
    if (this.autoEvolveTrainResolve) {
      this.autoEvolveTrainResolve();
      this.autoEvolveTrainResolve = null;
    }

    console.log('Auto-evolve stopped');
  }

  /**
   * Run one iteration of auto-evolve
   */
  async runAutoEvolveIteration() {
    if (!this.isAutoEvolving) return;

    this.autoEvolveGeneration++;
    this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Preparing NCA...`;

    // Decide whether to mutate or create new random NCA
    const shouldMutate = Math.random() < this.mutationChance && this.zoo.length > 0;

    if (shouldMutate) {
      // Pick zoo entry proportional to exp(epiplexity / temp) - higher epiplexity = more likely to be selected
      const temp = 15;
      const weights = this.zoo.map(e => Math.exp(e.epiplexity / temp));
      const totalWeight = weights.reduce((a, b) => a + b, 0);
      let r = Math.random() * totalWeight;
      let selectedIdx = 0;
      for (let i = 0; i < weights.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          selectedIdx = i;
          break;
        }
      }
      const randomEntry = this.zoo[selectedIdx];

      // Update config from zoo entry
      this.ncaConfig.numConvLayers = randomEntry.config.numConvLayers;
      this.ncaConfig.internalChannels = randomEntry.config.internalChannels;
      this.ncaConfig.kernelSize = randomEntry.config.kernelSize;

      // Update UI sliders
      this.convLayersSlider.value = randomEntry.config.numConvLayers;
      this.convLayersValue.textContent = randomEntry.config.numConvLayers;
      this.internalChannelsSlider.value = randomEntry.config.internalChannels;
      this.internalChannelsValue.textContent = randomEntry.config.internalChannels;
      this.kernelSizeSelect.value = randomEntry.config.kernelSize;

      // Rebuild model and set weights
      this.ncaModel.updateConfig({
        numConvLayers: randomEntry.config.numConvLayers,
        internalChannels: randomEntry.config.internalChannels,
        kernelSize: randomEntry.config.kernelSize
      });
      this.ncaModel.setWeightsData(randomEntry.weights);

      // Mutate with current strength
      const strength = parseFloat(this.mutateStrengthSlider.value);
      this.mutateNCA(strength);

      this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Mutating specimen (epi: ${randomEntry.epiplexity.toFixed(2)})...`;
    } else {
      // Create new random NCA
      this.ncaModel.randomize();
      this.resetState();
      this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: New random NCA...`;
    }

    // Start playing the NCA in the simulator
    if (!this.isRunning) {
      this.toggleRunning();
    }

    // Give UI time to update
    await new Promise(resolve => setTimeout(resolve, 50));

    // Run training with a completion callback
    await this.runAutoEvolveTraining();
  }

  /**
   * Run training as part of auto-evolve and handle completion
   */
  async runAutoEvolveTraining() {
    if (!this.isAutoEvolving) return;

    this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Building dataset...`;

    // Keep NCA playing for visual feedback during auto-evolve

    // Dispose old epiplexity model if exists
    if (this.epiplexityModel) {
      this.epiplexityModel.dispose();
    }

    // Dispose old training data
    this.disposeTrainingData();

    // Update UI
    this.trainBtn.textContent = 'Stop';
    this.trainBtn.disabled = true;
    this.saveZooBtn.disabled = true;
    this.epiplexityDisplay.textContent = '--';
    this.trainStepDisplay.textContent = '0';
    this.trainMaxDisplay.textContent = this.epiConfig.maxTrainSteps;
    this.lossGraph.showMessage('Building dataset...');

    // Build dataset
    await new Promise(resolve => setTimeout(resolve, 50));
    this.buildTrainingDataset(this.epiConfig.maxTrainSteps);

    // Shuffle dataset
    for (let i = this.trainingData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.trainingData[i], this.trainingData[j]] = [this.trainingData[j], this.trainingData[i]];
    }

    // Create new epiplexity model
    this.epiplexityModel = new EpiplexityModel({
      gridSize: this.ncaConfig.width,
      dModel: this.epiConfig.dModel,
      numLayers: this.epiConfig.numLayers,
      numStates: 4
    });

    // Reset graph and training state
    this.lossGraph.reset();
    this.lossGraph.setMaxSteps(this.epiConfig.maxTrainSteps);
    this.trainingStep = 0;
    this.bestSmoothedLoss = Infinity;
    this.stepsSinceImprovement = 0;
    this.isTraining = true;

    // Update UI
    this.trainBtn.textContent = 'Stop';
    this.trainBtn.disabled = false;
    this.trainBtn.classList.add('training');

    this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Training...`;

    // Run training loop with promise
    await new Promise(resolve => {
      this.autoEvolveTrainResolve = resolve;
      this.autoEvolveTrainStep();
    });

    // Training complete - add to zoo if auto-evolving
    if (this.isAutoEvolving && this.lastTrainedEpiplexity !== null) {
      this.addToZooFromAutoEvolve();

      // Continue to next iteration
      setTimeout(() => this.runAutoEvolveIteration(), 100);
    }
  }

  /**
   * Training step for auto-evolve (similar to trainStep but resolves promise on completion)
   */
  autoEvolveTrainStep() {
    if (!this.isTraining) {
      if (this.autoEvolveTrainResolve) {
        this.autoEvolveTrainResolve();
        this.autoEvolveTrainResolve = null;
      }
      return;
    }

    // Get training sample from pre-built dataset
    const pair = this.trainingData[this.trainingStep];

    // Perform training step
    const loss = this.epiplexityModel.trainStep(pair.input, pair.target);

    // Record loss
    this.lossGraph.addPoint(this.trainingStep, loss);
    this.trainingStep++;

    // Check for early stopping based on smoothed loss
    const smoothedLoss = this.lossGraph.getCurrentSmoothedLoss();
    let isBoring = false;
    if (smoothedLoss !== null) {
      if (smoothedLoss < this.bestSmoothedLoss) {
        this.bestSmoothedLoss = smoothedLoss;
        this.stepsSinceImprovement = 0;
      } else {
        this.stepsSinceImprovement++;
      }

      // Check if loss is below minimum threshold (boring/constant pattern)
      if (smoothedLoss < this.epiConfig.minFinalLoss) {
        isBoring = true;
      }
    }

    // Update UI
    this.trainStepDisplay.textContent = this.trainingStep;
    this.trainLossDisplay.textContent = loss.toFixed(3);

    // Check if done
    const shouldStop = this.trainingStep >= this.trainingData.length ||
        this.trainingStep >= this.epiConfig.maxTrainSteps ||
        this.stepsSinceImprovement >= this.epiConfig.earlyStopPatience ||
        isBoring;

    if (shouldStop) {
      // Finalize and compute epiplexity
      const epiplexity = this.lossGraph.finalize();
      this.epiplexityDisplay.textContent = epiplexity.toFixed(2);

      // Save NCA weights and epiplexity
      this.lastTrainedEpiplexity = epiplexity;
      this.lastTrainedWeights = this.ncaModel.getWeightsData();
      this.lastTrainedConfig = { ...this.ncaConfig };
      this.lastTrainWasBoring = isBoring;

      // Update UI
      this.trainBtn.textContent = 'Estimate Epiplexity';
      this.trainBtn.classList.remove('training');
      this.isTraining = false;

      if (this.autoEvolveTrainResolve) {
        this.autoEvolveTrainResolve();
        this.autoEvolveTrainResolve = null;
      }
      return;
    }

    // Schedule next step
    requestAnimationFrame(() => this.autoEvolveTrainStep());
  }

  /**
   * Add current NCA to zoo from auto-evolve (handles max size)
   */
  addToZooFromAutoEvolve() {
    if (!this.lastTrainedEpiplexity || !this.lastTrainedWeights || !this.burnInSnapshot) {
      return;
    }

    // Skip boring patterns (loss below threshold)
    if (this.lastTrainWasBoring) {
      this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Skipped (boring pattern, loss too low)`;
      console.log(`Auto-evolve skipped boring specimen`);
      return;
    }

    const entry = {
      id: Date.now(),
      epiplexity: this.lastTrainedEpiplexity,
      weights: this.lastTrainedWeights,
      config: this.lastTrainedConfig,
      thumbnail: this.burnInSnapshot
    };

    this.zoo.push(entry);
    this.zoo.sort((a, b) => b.epiplexity - a.epiplexity); // Descending (highest first)

    // Trim to max size (keep highest epiplexity)
    while (this.zoo.length > this.maxZooSize) {
      this.zoo.pop();
    }

    this.renderZoo();

    this.zooStatus.textContent = `Generation ${this.autoEvolveGeneration}: Added (epi: ${entry.epiplexity.toFixed(2)}). Zoo size: ${this.zoo.length}`;
    console.log(`Auto-evolve added specimen. Epiplexity: ${entry.epiplexity.toFixed(2)}, Zoo size: ${this.zoo.length}`);
  }

  /**
   * Mutate NCA weights with Gaussian noise
   * @param {number} strength - Mutation strength (stddev)
   */
  mutateNCA(strength) {
    const weightsData = this.ncaModel.getWeightsData();

    // Add Gaussian noise to each weight
    weightsData.kernels = weightsData.kernels.map(kernel => {
      return tf.tidy(() => {
        const t = tf.tensor(kernel);
        const noise = tf.randomNormal(t.shape, 0, strength);
        return t.add(noise).arraySync();
      });
    });

    weightsData.biases = weightsData.biases.map(bias => {
      return tf.tidy(() => {
        const t = tf.tensor(bias);
        const noise = tf.randomNormal(t.shape, 0, strength);
        return t.add(noise).arraySync();
      });
    });

    this.ncaModel.setWeightsData(weightsData);
    this.resetState();
    this.renderer.draw(this.state);

    console.log('Mutated NCA with strength:', strength);
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    if (this.trainingAnimationId) {
      cancelAnimationFrame(this.trainingAnimationId);
    }
    if (this.state) {
      this.state.dispose();
    }
    if (this.ncaModel) {
      this.ncaModel.dispose();
    }
    if (this.epiplexityModel) {
      this.epiplexityModel.dispose();
    }
    this.disposeTrainingData();
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  const app = new NCAApp();
  await app.init();

  // Expose for debugging
  window.ncaApp = app;
});
