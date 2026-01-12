/**
 * Epiplexity Model
 *
 * A transformer that predicts future NCA states from current states.
 * Given a snapshot at time T, predicts the snapshot at time T+k.
 * Uses trainable token embeddings and 2D positional encodings.
 */

export class EpiplexityModel {
  /**
   * @param {Object} config
   * @param {number} config.gridSize - Size of input grid (default: 64)
   * @param {number} config.numStates - Number of discrete states (default: 4)
   * @param {number} config.dModel - Model dimension (default: 64)
   * @param {number} config.numHeads - Number of attention heads (default: 4)
   * @param {number} config.numLayers - Number of transformer layers (default: 2)
   * @param {number} config.ffnDim - Feed-forward dimension (default: 128)
   * @param {number} config.learningRate - Learning rate (default: 1e-3)
   */
  constructor(config = {}) {
    this.gridSize = config.gridSize ?? 64;
    this.numStates = config.numStates ?? 4;
    this.dModel = config.dModel ?? 64;
    this.numHeads = config.numHeads ?? 4;
    this.numLayers = config.numLayers ?? 2;
    this.ffnDim = config.ffnDim ?? 128;
    this.learningRate = config.learningRate ?? 1e-3;

    this.seqLen = this.gridSize * this.gridSize;

    this.weights = {};
    this.optimizer = null;
    this.posEncoding = null;

    this.build();
  }

  /**
   * Create 2D sinusoidal positional encodings
   * @returns {tf.Tensor2D} - [seqLen, dModel]
   */
  create2DPositionalEncoding() {
    const encodings = [];
    const numFreqs = Math.floor(this.dModel / 4);

    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        const posEnc = [];
        for (let i = 0; i < numFreqs; i++) {
          const freq = 1 / Math.pow(10000, (2 * i) / this.dModel);
          posEnc.push(Math.sin(x * freq));
          posEnc.push(Math.cos(x * freq));
          posEnc.push(Math.sin(y * freq));
          posEnc.push(Math.cos(y * freq));
        }
        // Pad to dModel if needed
        while (posEnc.length < this.dModel) {
          posEnc.push(0);
        }
        encodings.push(posEnc.slice(0, this.dModel));
      }
    }

    return tf.tensor2d(encodings);
  }

  /**
   * Build model weights
   */
  build() {
    this.dispose();

    // Create positional encodings
    this.posEncoding = this.create2DPositionalEncoding();

    // Trainable token embeddings: [numStates, dModel]
    this.weights.tokenEmbed = tf.variable(
      tf.randomNormal([this.numStates, this.dModel], 0, 0.02)
    );

    // Transformer layers
    this.weights.layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.weights.layers.push({
        // Self-attention
        qProj: tf.variable(tf.randomNormal([this.dModel, this.dModel], 0, 0.02)),
        kProj: tf.variable(tf.randomNormal([this.dModel, this.dModel], 0, 0.02)),
        vProj: tf.variable(tf.randomNormal([this.dModel, this.dModel], 0, 0.02)),
        outProj: tf.variable(tf.randomNormal([this.dModel, this.dModel], 0, 0.02)),

        // Layer norm 1
        ln1Gamma: tf.variable(tf.ones([this.dModel])),
        ln1Beta: tf.variable(tf.zeros([this.dModel])),

        // FFN
        ffn1: tf.variable(tf.randomNormal([this.dModel, this.ffnDim], 0, 0.02)),
        ffn1Bias: tf.variable(tf.zeros([this.ffnDim])),
        ffn2: tf.variable(tf.randomNormal([this.ffnDim, this.dModel], 0, 0.02)),
        ffn2Bias: tf.variable(tf.zeros([this.dModel])),

        // Layer norm 2
        ln2Gamma: tf.variable(tf.ones([this.dModel])),
        ln2Beta: tf.variable(tf.zeros([this.dModel]))
      });
    }

    // Output projection to numStates logits
    this.weights.outputProj = tf.variable(
      tf.randomNormal([this.dModel, this.numStates], 0, 0.02)
    );
    this.weights.outputBias = tf.variable(tf.zeros([this.numStates]));

    // Create optimizer
    this.optimizer = tf.train.adam(this.learningRate);
  }

  /**
   * Layer normalization
   */
  layerNorm(x, gamma, beta, epsilon = 1e-6) {
    const mean = x.mean(-1, true);
    const variance = x.sub(mean).square().mean(-1, true);
    const normalized = x.sub(mean).div(variance.add(epsilon).sqrt());
    return normalized.mul(gamma).add(beta);
  }

  /**
   * Multi-head self-attention (bidirectional - no causal mask)
   */
  selfAttention(x, qProj, kProj, vProj, outProj) {
    const seqLen = x.shape[0];
    const headDim = Math.floor(this.dModel / this.numHeads);

    // Project Q, K, V
    let q = tf.matMul(x, qProj);
    let k = tf.matMul(x, kProj);
    let v = tf.matMul(x, vProj);

    // Reshape for multi-head: [seqLen, numHeads, headDim]
    q = q.reshape([seqLen, this.numHeads, headDim]);
    k = k.reshape([seqLen, this.numHeads, headDim]);
    v = v.reshape([seqLen, this.numHeads, headDim]);

    // Transpose to [numHeads, seqLen, headDim]
    q = q.transpose([1, 0, 2]);
    k = k.transpose([1, 0, 2]);
    v = v.transpose([1, 0, 2]);

    // Attention scores: [numHeads, seqLen, seqLen]
    const scale = Math.sqrt(headDim);
    const scores = tf.matMul(q, k.transpose([0, 2, 1])).div(scale);

    // Softmax (no mask - bidirectional attention)
    const attnWeights = tf.softmax(scores, -1);

    // Apply attention to values
    let attnOutput = tf.matMul(attnWeights, v);

    // Reshape back: [numHeads, seqLen, headDim] -> [seqLen, dModel]
    attnOutput = attnOutput.transpose([1, 0, 2]).reshape([seqLen, this.dModel]);

    // Output projection
    return tf.matMul(attnOutput, outProj);
  }

  /**
   * Feed-forward network
   */
  ffn(x, w1, b1, w2, b2) {
    let h = tf.matMul(x, w1).add(b1);
    h = tf.relu(h);
    return tf.matMul(h, w2).add(b2);
  }

  /**
   * Forward pass (supports batched input)
   * @param {tf.Tensor} inputIndices - [seqLen] or [batch, seqLen] state indices (0-3)
   * @returns {tf.Tensor} - [seqLen, numStates] or [batch, seqLen, numStates] logits
   */
  forward(inputIndices) {
    return tf.tidy(() => {
      const isBatched = inputIndices.shape.length === 2;
      const batchSize = isBatched ? inputIndices.shape[0] : 1;

      // Flatten for gather if batched
      const flatIndices = isBatched ? inputIndices.reshape([-1]) : inputIndices;

      // Get token embeddings
      let x = tf.gather(this.weights.tokenEmbed, flatIndices);

      // Reshape to [batch, seqLen, dModel] if batched
      if (isBatched) {
        x = x.reshape([batchSize, this.seqLen, this.dModel]);
      }

      // Add positional encoding (broadcast over batch)
      x = x.add(this.posEncoding);

      // Process each sample independently through transformer
      // For simplicity, we process unbatched and stack results
      if (isBatched) {
        const results = [];
        for (let b = 0; b < batchSize; b++) {
          const sample = x.slice([b, 0, 0], [1, -1, -1]).squeeze([0]);
          const output = this.transformerForward(sample);
          results.push(output);
        }
        return tf.stack(results);
      } else {
        return this.transformerForward(x);
      }
    });
  }

  /**
   * Transformer forward pass for single sample
   * @param {tf.Tensor2D} x - [seqLen, dModel]
   * @returns {tf.Tensor2D} - [seqLen, numStates]
   */
  transformerForward(x) {
    // Transformer layers
    for (const layer of this.weights.layers) {
      // Self-attention with residual
      const attnOut = this.selfAttention(
        x, layer.qProj, layer.kProj, layer.vProj, layer.outProj
      );
      x = x.add(attnOut);
      x = this.layerNorm(x, layer.ln1Gamma, layer.ln1Beta);

      // FFN with residual
      const ffnOut = this.ffn(x, layer.ffn1, layer.ffn1Bias, layer.ffn2, layer.ffn2Bias);
      x = x.add(ffnOut);
      x = this.layerNorm(x, layer.ln2Gamma, layer.ln2Beta);
    }

    // Output projection: [seqLen, numStates] logits
    return tf.matMul(x, this.weights.outputProj).add(this.weights.outputBias);
  }

  /**
   * Compute cross-entropy loss (supports batched input)
   * @param {tf.Tensor} logits - [seqLen, numStates] or [batch, seqLen, numStates]
   * @param {tf.Tensor} targets - [seqLen] or [batch, seqLen] true class indices
   * @returns {tf.Scalar} - Mean cross-entropy loss
   */
  computeLoss(logits, targets) {
    return tf.tidy(() => {
      const isBatched = logits.shape.length === 3;

      if (isBatched) {
        // Flatten batch and sequence dimensions
        const flatLogits = logits.reshape([-1, this.numStates]);
        const flatTargets = targets.reshape([-1]);
        const oneHotTargets = tf.oneHot(flatTargets, this.numStates);
        return tf.losses.softmaxCrossEntropy(oneHotTargets, flatLogits);
      } else {
        const oneHotTargets = tf.oneHot(targets, this.numStates);
        return tf.losses.softmaxCrossEntropy(oneHotTargets, logits);
      }
    });
  }

  /**
   * Perform one training step (supports batched input)
   * @param {tf.Tensor} inputIndices - [seqLen] or [batch, seqLen] input state indices
   * @param {tf.Tensor} targetIndices - [seqLen] or [batch, seqLen] target state indices
   * @returns {number} - Loss value
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
    if (this.posEncoding) {
      this.posEncoding.dispose();
    }

    if (this.weights.tokenEmbed) {
      this.weights.tokenEmbed.dispose();
      this.weights.outputProj.dispose();
      this.weights.outputBias.dispose();

      for (const layer of this.weights.layers || []) {
        layer.qProj.dispose();
        layer.kProj.dispose();
        layer.vProj.dispose();
        layer.outProj.dispose();
        layer.ln1Gamma.dispose();
        layer.ln1Beta.dispose();
        layer.ffn1.dispose();
        layer.ffn1Bias.dispose();
        layer.ffn2.dispose();
        layer.ffn2Bias.dispose();
        layer.ln2Gamma.dispose();
        layer.ln2Beta.dispose();
      }
    }

    this.weights = {};
  }
}

