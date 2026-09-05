/**
 * Zhang & Levin (2026), equations 7–9 and Appendix E.
 * Independent implementation: augmented Householder QR in float64, with an
 * implicit derivative of the ridge optimum. No finite-difference estimator,
 * explicit matrix inverse, or differentiation through a predictor training run.
 */
export function ridgeScore(h, y, n, m, d, lambda = 0.3, eta = 1) {
  if (!(lambda > 0) || !(eta > 0)) throw new Error('Ridge and resolution must be positive.');
  if (h.length !== n * m || y.length !== n * d) throw new Error('Invalid readout dimensions.');
  const rows = n + m;
  const a = new Float64Array(rows * m);
  const b = new Float64Array(rows * d);
  a.set(h); b.set(y);
  for (let j = 0; j < m; j++) a[(n + j) * m + j] = Math.sqrt(lambda);
  // Apply each Householder reflection to both [H; sqrt(lambda) I] and [Y; 0].
  for (let k = 0; k < m; k++) {
    let norm = 0;
    for (let i = k; i < rows; i++) norm = Math.hypot(norm, a[i * m + k]);
    const alpha = a[k * m + k] >= 0 ? -norm : norm;
    const v = new Float64Array(rows - k);
    for (let i = k; i < rows; i++) v[i - k] = a[i * m + k];
    v[0] -= alpha;
    let vv = 0;
    for (const x of v) vv += x * x;
    const beta = 2 / vv;
    for (let j = k + 1; j < m; j++) {
      let dot = 0;
      for (let i = k; i < rows; i++) dot += v[i - k] * a[i * m + j];
      for (let i = k; i < rows; i++) a[i * m + j] -= beta * dot * v[i - k];
    }
    for (let j = 0; j < d; j++) {
      let dot = 0;
      for (let i = k; i < rows; i++) dot += v[i - k] * b[i * d + j];
      for (let i = k; i < rows; i++) b[i * d + j] -= beta * dot * v[i - k];
    }
    a[k * m + k] = alpha;
  }
  const w = new Float64Array(m * d);
  for (let i = m - 1; i >= 0; i--) {
    for (let j = 0; j < d; j++) {
      let x = b[i * d + j];
      for (let k = i + 1; k < m; k++) x -= a[i * m + k] * w[k * d + j];
      w[i * d + j] = x / a[i * m + i];
    }
  }
  // det(I + eta WW^T) = det(I + eta W^T W). The latter is small
  // (two state channels per horizon). Cholesky avoids SVD degeneracy at zero.
  const l = new Float64Array(d * d);
  let score = 0;
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let x = i === j ? 1 : 0;
      for (let k = 0; k < m; k++) x += eta * w[k * d + i] * w[k * d + j];
      for (let k = 0; k < j; k++) x -= l[i * d + k] * l[j * d + k];
      if (i === j) {
        if (!(x > 0) || !Number.isFinite(x)) throw new Error('Non-finite reservoir score. Try a new rule.');
        l[i * d + j] = Math.sqrt(x);
        score += Math.log(x) / (2 * Math.LN2);
      } else l[i * d + j] = x / l[j * d + j];
    }
  }
  // Q = dS/dW = eta/ln(2) W (I + eta W^T W)^-1.
  const q = new Float64Array(m * d);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < d; j++) {
      let x = eta / Math.LN2 * w[i * d + j];
      for (let k = 0; k < j; k++) x -= l[j * d + k] * q[i * d + k];
      q[i * d + j] = x / l[j * d + j];
    }
    for (let j = d - 1; j >= 0; j--) {
      let x = q[i * d + j];
      for (let k = j + 1; k < d; k++) x -= l[k * d + j] * q[i * d + k];
      q[i * d + j] = x / l[j * d + j];
    }
  }
  // U = (H^T H + lambda I)^-1 Q via R^T and R, never forming H^T H.
  const u = new Float64Array(q);
  for (let i = 0; i < m; i++) for (let j = 0; j < d; j++) {
    let x = u[i * d + j];
    for (let k = 0; k < i; k++) x -= a[k * m + i] * u[k * d + j];
    u[i * d + j] = x / a[i * m + i];
  }
  for (let i = m - 1; i >= 0; i--) for (let j = 0; j < d; j++) {
    let x = u[i * d + j];
    for (let k = i + 1; k < m; k++) x -= a[i * m + k] * u[k * d + j];
    u[i * d + j] = x / a[i * m + i];
  }
  return { score, w, u };
}

export function differentiableRidgeScore(h, y, lambda = 0.3, eta = 1) {
  return tf.customGrad((features, targets, save) => {
    const [n, m] = features.shape;
    const d = targets.shape[1];
    const { score, w, u } = ridgeScore(features.dataSync(), targets.dataSync(), n, m, d, lambda, eta);
    save([features, targets, tf.tensor2d(Float32Array.from(w), [m, d]), tf.tensor2d(Float32Array.from(u), [m, d])]);
    return {
      value: tf.scalar(score),
      gradFunc: (dy, saved) => tf.tidy(() => {
        const [h, y, w, u] = saved;
        const hu = h.matMul(u);
        const residual = y.sub(h.matMul(w));
        return [residual.matMul(u, false, true).sub(hu.matMul(w, false, true)).mul(dy), hu.mul(dy)];
      })
    };
  })(h, y);
}

export function circularPad(x, pad) {
  if (!pad) return x;
  const h = x.shape[1], w = x.shape[2];
  const vertical = tf.concat([x.slice([0, h - pad, 0, 0], [-1, pad, -1, -1]), x,
    x.slice([0, 0, 0, 0], [-1, pad, -1, -1])], 1);
  return tf.concat([vertical.slice([0, 0, w - pad, 0], [-1, -1, pad, -1]), vertical,
    vertical.slice([0, 0, 0, 0], [-1, -1, pad, -1])], 2);
}

export class ReservoirEpiplexity {
  constructor({ channels = 32, depth = 4, kernelSize = 3, seed = 42, lambda = 0.3 } = {}) {
    this.channels = channels;
    this.lambda = lambda;
    this.kernels = [];
    this.biases = [];
    for (let i = 0; i < depth; i++) {
      const size = i === depth - 1 ? 1 : kernelSize;
      const input = i === 0 ? 2 : channels;
      const bound = 1 / Math.sqrt(size * size * input);
      this.kernels.push(tf.randomUniform([size, size, input, channels], -bound, bound, 'float32', seed + 2 * i));
      this.biases.push(tf.randomUniform([channels], -bound, bound, 'float32', seed + 2 * i + 1));
    }
  }

  features(state) {
    return tf.tidy(() => {
      let x = state;
      this.kernels.forEach((kernel, i) => {
        x = tf.conv2d(circularPad(x, Math.floor(kernel.shape[0] / 2)), kernel, 1, 'valid').add(this.biases[i]);
        if (i < this.kernels.length - 1) {
          const { mean, variance } = tf.moments(x, 3, true);
          x = tf.elu(x.sub(mean).div(variance.add(1e-5).sqrt()));
        }
      });
      const h = x.reshape([-1, this.channels]);
      const { mean, variance } = tf.moments(h, 0, true);
      // A variance floor makes constant fields differentiable in float32.
      return h.sub(mean).div(variance.add(1e-8).sqrt().mul(Math.sqrt(this.channels)));
    });
  }

  scoreFeatures(features, future) {
    return tf.tidy(() => {
      const y = future.reshape([-1, future.shape[3]]);
      return differentiableRidgeScore(features, y.sub(y.mean(0, true)), this.lambda);
    });
  }

  dispose() { tf.dispose([...this.kernels, ...this.biases]); }
}
