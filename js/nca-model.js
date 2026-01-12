/**
 * Neural Cellular Automata Model
 *
 * A discrete NCA with Game-of-Life style dynamics:
 * - Dead cells (black) can stay dead or come alive as any color
 * - Alive cells can stay alive (same color) or die (become black)
 */

export class NCAModel {
  /**
   * @param {Object} config
   * @param {number} config.numConvLayers - Number of convolutional layers (default: 1)
   * @param {number} config.internalChannels - Hidden channels between layers (default: 4)
   * @param {number} config.kernelSize - Convolution kernel size (default: 3)
   */
  constructor(config = {}) {
    this.numConvLayers = config.numConvLayers ?? 1;
    this.internalChannels = config.internalChannels ?? 4;
    this.kernelSize = config.kernelSize ?? 3;
    this.numStates = 4; // Fixed: 0=dead, 1-3=alive colors

    this.kernels = [];
    this.biases = [];

    this.build();
  }

  /**
   * Build the model with random weights
   */
  build() {
    this.dispose();

    this.kernels = [];
    this.biases = [];

    for (let i = 0; i < this.numConvLayers; i++) {
      const isFirst = i === 0;
      const isLast = i === this.numConvLayers - 1;

      const inChannels = isFirst ? this.numStates : this.internalChannels;
      const outChannels = isLast ? this.numStates : this.internalChannels;

      // Xavier initialization
      const stddev = Math.sqrt(2.0 / (this.kernelSize * this.kernelSize * inChannels));

      const kernel = tf.variable(
        tf.randomNormal([this.kernelSize, this.kernelSize, inChannels, outChannels], 0, stddev)
      );
      const bias = tf.variable(tf.zeros([outChannels]));

      this.kernels.push(kernel);
      this.biases.push(bias);
    }
  }

  /**
   * Apply circular (toroidal) padding to a tensor
   */
  circularPad(x, pad) {
    const top = x.slice([0, x.shape[1] - pad, 0, 0], [-1, pad, -1, -1]);
    const bottom = x.slice([0, 0, 0, 0], [-1, pad, -1, -1]);
    x = tf.concat([top, x, bottom], 1);

    const left = x.slice([0, 0, x.shape[2] - pad, 0], [-1, -1, pad, -1]);
    const right = x.slice([0, 0, 0, 0], [-1, -1, pad, -1]);
    return tf.concat([left, x, right], 2);
  }

  /**
   * Run one step of the NCA
   * @param {tf.Tensor4D} state - Input state [batch, height, width, 4] (one-hot)
   * @returns {tf.Tensor4D} - Updated state (one-hot)
   */
  step(state) {
    return tf.tidy(() => {
      let x = state;

      // Apply convolutional layers
      for (let i = 0; i < this.numConvLayers; i++) {
        const isLast = i === this.numConvLayers - 1;

        const pad = Math.floor(this.kernelSize / 2);
        x = this.circularPad(x, pad);
        x = tf.conv2d(x, this.kernels[i], 1, 'valid');
        x = tf.add(x, this.biases[i]);

        if (!isLast) {
          x = tf.relu(x);
        }
      }

      // x is now logits [batch, height, width, 4]
      // State 0 = dead (black), States 1-3 = alive (colors)

      // Create transition mask based on current state
      // Dead cells (state 0): can go to any state (0, 1, 2, 3)
      // Alive cells (state 1, 2, 3): can only stay same or die (go to 0)
      const isDead = state.slice([0, 0, 0, 0], [-1, -1, -1, 1]); // [batch, H, W, 1]
      const isAlive = tf.sub(1, isDead); // [batch, H, W, 1]

      // For alive cells, mask out transitions to OTHER alive states
      // Allow: current state and state 0
      const currentState = state; // one-hot of current state
      const deathAllowed = tf.concat([tf.ones(isDead.shape), tf.zeros([...isDead.shape.slice(0, 3), 3])], 3);
      const aliveMask = currentState.add(deathAllowed).clipByValue(0, 1);

      // Dead cells can transition to anything (mask = all 1s)
      // Alive cells use the restricted mask
      const fullMask = tf.ones(x.shape);
      const mask = isDead.mul(fullMask).add(isAlive.mul(aliveMask));

      // Apply mask: set disallowed transitions to -infinity
      const maskedLogits = x.add(mask.sub(1).mul(1e9));

      // Argmax (temperature = 0, deterministic)
      const indices = tf.argMax(maskedLogits, 3);

      // Convert to one-hot
      return tf.oneHot(indices, this.numStates).toFloat();
    });
  }

  /**
   * Randomize all weights
   */
  randomize() {
    this.build();
  }

  /**
   * Update configuration and rebuild
   */
  updateConfig(config) {
    if (config.numConvLayers !== undefined) this.numConvLayers = config.numConvLayers;
    if (config.internalChannels !== undefined) this.internalChannels = config.internalChannels;
    if (config.kernelSize !== undefined) this.kernelSize = config.kernelSize;

    this.build();
  }

  /**
   * Dispose of all tensors
   */
  dispose() {
    for (const kernel of this.kernels) {
      kernel.dispose();
    }
    for (const bias of this.biases) {
      bias.dispose();
    }
    this.kernels = [];
    this.biases = [];
  }

  /**
   * Get current configuration
   */
  getConfig() {
    return {
      numConvLayers: this.numConvLayers,
      internalChannels: this.internalChannels,
      kernelSize: this.kernelSize,
      numStates: this.numStates
    };
  }

  /**
   * Get weights as serializable arrays
   * @returns {Object} - { kernels: [...], biases: [...] }
   */
  getWeightsData() {
    return {
      kernels: this.kernels.map(k => k.arraySync()),
      biases: this.biases.map(b => b.arraySync())
    };
  }

  /**
   * Set weights from serializable arrays
   * @param {Object} data - { kernels: [...], biases: [...] }
   */
  setWeightsData(data) {
    // Dispose existing weights
    this.dispose();

    this.kernels = data.kernels.map(k => tf.variable(tf.tensor(k)));
    this.biases = data.biases.map(b => tf.variable(tf.tensor(b)));
  }
}
