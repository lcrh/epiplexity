import { circularPad } from './reservoir-epiplexity.js';

/** A bounded, two-channel 2D adaptation of the paper's continuous NCA. */
export class ContinuousNCA {
  constructor({ channels = 16, depth = 3, kernelSize = 3, seed = 1 } = {}) {
    this.channels = channels;
    this.depth = depth;
    this.kernelSize = kernelSize;
    this.seed = seed;
    this.weights = tf.tidy(() => {
      const weights = [];
      for (let i = 0; i < depth; i++) {
        const size = i === 0 ? kernelSize : 1;
        const input = i === 0 ? 2 : channels;
        const output = i === depth - 1 ? 2 : channels;
        const scale = i === depth - 1 ? 0.1 : 1 / Math.sqrt(size * size * input);
        weights.push(tf.variable(tf.randomNormal([size, size, input, output], 0, scale, 'float32', seed + i)));
        weights.push(tf.variable(tf.zeros([output])));
      }
      return weights;
    });
  }

  randomState(batch, size, seed) {
    return tf.tidy(() => {
      const angle = tf.randomUniform([batch, size, size, 1], -Math.PI, Math.PI, 'float32', seed);
      return tf.concat([angle.cos(), angle.sin()], 3);
    });
  }

  step(state) {
    return tf.tidy(() => {
      let x = state;
      for (let i = 0; i < this.depth; i++) {
        const kernel = this.weights[2 * i];
        x = tf.conv2d(circularPad(x, Math.floor(kernel.shape[0] / 2)), kernel, 1, 'valid').add(this.weights[2 * i + 1]);
        if (i < this.depth - 1) x = tf.elu(x);
      }
      x = state.add(x);
      return x.div(x.square().sum(3, true).add(1e-8).sqrt());
    });
  }

  warmState({ batch = 2, size = 16, burnIn = 32, seed = 1 } = {}) {
    // Called outside the gradient tape: burn-in has bounded memory.
    let state = this.randomState(batch, size, seed);
    for (let i = 0; i < burnIn; i++) {
      const next = this.step(state);
      state.dispose(); state = next;
    }
    const noisy = tf.tidy(() => state.add(tf.randomNormal(state.shape, 0, 0.1, 'float32', seed + 1)));
    state.dispose();
    return noisy;
  }

  rollout(state, horizon = 8) {
    return tf.tidy(() => {
      const frames = [];
      for (let t = 0; t < horizon; t++) { state = this.step(state); frames.push(state); }
      return tf.concat(frames, 3);
    });
  }

  dispose() { tf.dispose(this.weights); }
}

/** One ascent step; the observer and detached starting state are held fixed. */
export function optimizeStep(model, observer, optimizer, state, horizon = 8) {
  return tf.tidy(() => {
    const features = observer.features(state);
    const { value, grads } = tf.variableGrads(() =>
      observer.scoreFeatures(features, model.rollout(state, horizon)).div(-100), model.weights);
    const norm = tf.addN(Object.values(grads).map(g => g.square().sum())).sqrt();
    const score = -100 * value.dataSync()[0];
    const gradientNorm = norm.dataSync()[0];
    if (!Number.isFinite(score) || !Number.isFinite(gradientNorm)) throw new Error('Non-finite gradient. Start a new rule or lower the learning rate.');
    const scale = tf.minimum(tf.scalar(1), tf.scalar(0.5).div(norm.add(1e-12)));
    optimizer.applyGradients(Object.fromEntries(Object.entries(grads).map(([name, g]) => [name, g.mul(scale)])));
    return { score, gradientNorm };
  });
}
