/**
 * Convolutional Epiplexity Model
 *
 * A circular-padded ConvNet that predicts future NCA states from current states.
 * Same local inductive bias as the NCA generator — single-step prediction should
 * therefore have very low epiplexity when capacity is sufficient.
 */

export class ConvEpiplexityModel {
  /**
   * @param {Object} config
   * @param {number} config.gridSize - Size of input grid (default: 64)
   * @param {number} config.numStates - Number of discrete states (default: 4)
   * @param {number} config.numConvLayers - Number of convolutional layers (default: 3)
   * @param {number} config.channels - Hidden channels between layers (default: 24)
   * @param {number} config.kernelSize - Convolution kernel size (default: 5)
   * @param {number} config.learningRate - Learning rate (default: 1e-3)
   */
  constructor(config = {}) {
    this.gridSize = config.gridSize ?? 64;
    this.numStates = config.numStates ?? 4;
    this.numConvLayers = config.numConvLayers ?? 3;
    this.channels = config.channels ?? 24;
    this.kernelSize = config.kernelSize ?? 5;
    this.learningRate = config.learningRate ?? 1e-3;

    this.seqLen = this.gridSize * this.gridSize;

    this.kernels = [];
    this.biases = [];
    this.optimizer = null;

    this.build();
  }

  /**
   * Build model weights
   */
  build() {
    this.dispose();

    this.kernels = [];
    this.biases = [];

    for (let i = 0; i < this.numConvLayers; i++) {
      const isFirst = i === 0;
      const isLast = i === this.numConvLayers - 1;

      const inChannels = isFirst ? this.numStates : this.channels;
      const outChannels = isLast ? this.numStates : this.channels;

      // Xavier / He-style initialization
      const stddev = Math.sqrt(2.0 / (this.kernelSize * this.kernelSize * inChannels));

      this.kernels.push(
        tf.variable(
          tf.randomNormal(
            [this.kernelSize, this.kernelSize, inChannels, outChannels],
            0,
            stddev
          )
        )
      );
      this.biases.push(tf.variable(tf.zeros([outChannels])));
    }

    this.optimizer = tf.train.adam(this.learningRate);
  }

  /**
   * Apply circular (toroidal) padding
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
   * Forward pass
   * @param {tf.Tensor} inputIndices - [seqLen] or [batch, seqLen] state indices
   * @returns {tf.Tensor} - [seqLen, numStates] or [batch, seqLen, numStates] logits
   */
  forward(inputIndices) {
    return tf.tidy(() => {
      const isBatched = inputIndices.shape.length === 2;
      const batchSize = isBatched ? inputIndices.shape[0] : 1;
      const flatIndices = isBatched ? inputIndices : inputIndices.expandDims(0);

      // One-hot spatial grid: [batch, H, W, numStates]
      let x = tf.oneHot(flatIndices, this.numStates).toFloat();
      x = x.reshape([batchSize, this.gridSize, this.gridSize, this.numStates]);

      const pad = Math.floor(this.kernelSize / 2);

      for (let i = 0; i < this.numConvLayers; i++) {
        const isLast = i === this.numConvLayers - 1;
        x = this.circularPad(x, pad);
        x = tf.conv2d(x, this.kernels[i], 1, 'valid');
        x = tf.add(x, this.biases[i]);
        if (!isLast) {
          x = tf.relu(x);
        }
      }

      // [batch, H, W, numStates] -> [batch, seqLen, numStates]
      const logits = x.reshape([batchSize, this.seqLen, this.numStates]);
      return isBatched ? logits : logits.squeeze([0]);
    });
  }

  /**
   * Compute cross-entropy loss
   * @param {tf.Tensor} logits - [seqLen, numStates] or [batch, seqLen, numStates]
   * @param {tf.Tensor} targets - [seqLen] or [batch, seqLen]
   * @returns {tf.Scalar}
   */
  computeLoss(logits, targets) {
    return tf.tidy(() => {
      const isBatched = logits.shape.length === 3;
      if (isBatched) {
        const flatLogits = logits.reshape([-1, this.numStates]);
        const flatTargets = targets.reshape([-1]);
        const oneHotTargets = tf.oneHot(flatTargets, this.numStates);
        return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
      }
      const oneHotTargets = tf.oneHot(targets, this.numStates);
      return tf.losses.softmaxCrossEntropy(oneHotTargets, logits);
    });
  }

  /**
   * Perform one training step
   * @param {tf.Tensor} inputIndices
   * @param {tf.Tensor} targetIndices
   * @returns {number}
   */
  trainStep(inputIndices, targetIndices) {
    const lossValue = this.optimizer.minimize(() => {
      const logits = this.forward(inputIndices);
      return this.computeLoss(logits, targetIndices);
    }, true);

    const loss = lossValue.dataSync()[0];
    lossValue.dispose();
    return loss;
  }

  /**
   * Dispose all tensors
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
    this.optimizer = null;
  }
}
