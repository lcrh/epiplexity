// Run: node tests/direct-optimization.mjs /absolute/path/to/tf.min.cjs
// Uses exactly the app's TF.js 4.17.0 bundle; no package manager required.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { ridgeScore, differentiableRidgeScore, ReservoirEpiplexity } from '../js/reservoir-epiplexity.js';
import { ContinuousNCA, optimizeStep } from '../js/continuous-nca.js';

if (!process.argv[2]) throw new Error('Pass the local TensorFlow.js 4.17.0 CommonJS bundle path.');
globalThis.tf = (await import(pathToFileURL(process.argv[2]))).default;
await tf.setBackend('cpu'); await tf.ready();
const close = (actual, expected, tolerance = 1e-5) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected} (tol ${tolerance})`);
const h = Float64Array.from({ length: 21 }, (_, i) => Math.sin(i * 0.7) + i % 3 * 0.1);
const y = Float64Array.from({ length: 14 }, (_, i) => Math.cos(i * 0.4));
const reference = ridgeScore(h, y, 7, 3, 2);
// Independent NumPy float64 reference: lstsq([H; sqrt(.3)I], [Y; 0]),
// followed by .5 * sum(log2(1 + svd(W)**2)).
close(reference.score, 0.040890706971484746, 1e-12);
const expectedWeights = [-0.003702940696244131, -0.06480689226352623,
  0.137617875582089, 0.1480392806598384, 0.06204734917475146, 0.09646755478124314];
reference.w.forEach((w, i) => close(w, expectedWeights[i], 1e-12));

// Independently check that the QR solution satisfies the ridge stationarity
// equation; include a rank-deficient feature matrix and an underdetermined case.
for (const [hh, yy, n, m, d] of [[h, y, 7, 3, 2], [new Float64Array(21).fill(1), y, 7, 3, 2],
  [h.slice(0, 12), y.slice(0, 4), 2, 6, 2]]) {
  const { w } = ridgeScore(hh, yy, n, m, d);
  for (let a = 0; a < m; a++) for (let b = 0; b < d; b++) {
    let normalResidual = 0.3 * w[a * d + b];
    for (let i = 0; i < n; i++) {
      let residual = -yy[i * d + b];
      for (let c = 0; c < m; c++) residual += hh[i * m + c] * w[c * d + b];
      normalResidual += hh[i * m + a] * residual;
    }
    close(normalResidual, 0, 1e-12);
  }
}
close(ridgeScore(h, new Float64Array(14), 7, 3, 2).score, 0, 1e-12);
assert.throws(() => ridgeScore(h, y, 7, 3, 2, 0));

// Validate both branches of the custom backward against central differences.
const ht = tf.tensor2d(Float32Array.from(h), [7, 3]);
const yt = tf.tensor2d(Float32Array.from(y), [7, 2]);
const gradients = tf.grads((a, b) => differentiableRidgeScore(a, b))([ht, yt]);
for (const [values, gradient, which] of [[h, gradients[0], 0], [y, gradients[1], 1]]) {
  const analytic = gradient.dataSync();
  for (let i = 0; i < values.length; i++) {
    const plus = Float64Array.from(values), minus = Float64Array.from(values);
    plus[i] += 1e-5; minus[i] -= 1e-5;
    const numerical = which === 0
      ? (ridgeScore(plus, y, 7, 3, 2).score - ridgeScore(minus, y, 7, 3, 2).score) / 2e-5
      : (ridgeScore(h, plus, 7, 3, 2).score - ridgeScore(h, minus, 7, 3, 2).score) / 2e-5;
    close(analytic[i], numerical, 2e-6);
  }
}
tf.dispose([ht, yt, ...gradients]);
console.log('PASS: ridge residuals, rank deficiency, zero targets, analytical gradients');

const model = new ContinuousNCA({ channels: 4, seed: 3 });
const observer = new ReservoirEpiplexity({ channels: 8 });
const state = model.warmState({ batch: 2, size: 4, burnIn: 3, seed: 22 });
const score = () => tf.tidy(() => observer.scoreFeatures(observer.features(state), model.rollout(state, 3)).dataSync()[0]);
const before = score();
const frozen = observer.kernels.map(k => Array.from(k.dataSync()));
// End-to-end gradient includes temporal unrolling and target centering.
const { value, grads } = tf.variableGrads(() => observer.scoreFeatures(observer.features(state), model.rollout(state, 3)), model.weights);
const weight = model.weights[0];
const original = Float32Array.from(weight.dataSync());
const analytical = grads[weight.name].dataSync();
for (const index of [0, 7, 20]) {
  const sample = sign => {
    const changed = Float32Array.from(original); changed[index] += sign * 0.001;
    tf.tidy(() => weight.assign(tf.tensor(changed, weight.shape)));
    return score();
  };
  close(analytical[index], (sample(1) - sample(-1)) / 0.002, 0.002);
}
tf.tidy(() => weight.assign(tf.tensor(original, weight.shape)));
tf.dispose([value, ...Object.values(grads)]);
const optimizer = tf.train.adam(0.001);
optimizeStep(model, observer, optimizer, state, 3); // Allocate Adam slots.
const memory = tf.memory().numTensors;
for (let step = 0; step < 20; step++) optimizeStep(model, observer, optimizer, state, 3);
assert.equal(tf.memory().numTensors, memory, 'Training must not leak tensors');
const after = score();
assert.ok(after > before, `Expected ascent on a fixed batch: ${before} -> ${after}`);
observer.kernels.forEach((k, i) => assert.deepEqual(Array.from(k.dataSync()), frozen[i], 'Observer must remain frozen'));
const unitError = tf.tidy(() => model.step(state).square().sum(3).sub(1).abs().max().dataSync()[0]);
assert.ok(unitError < 1e-5, 'Continuous states must stay bounded');
const constantScore = tf.tidy(() => {
  const constant = tf.ones([2, 4, 4, 2]);
  return observer.scoreFeatures(observer.features(constant), constant).dataSync()[0];
});
close(constantScore, 0, 1e-8);
state.dispose(); model.dispose(); observer.dispose(); optimizer.dispose();
// Exercise non-default depths, widths, and spatial kernels through backprop.
const customModel = new ContinuousNCA({ channels: 8, depth: 4, kernelSize: 5 });
const customObserver = new ReservoirEpiplexity({ channels: 16, depth: 3, kernelSize: 5 });
const customState = customModel.warmState({ size: 6, batch: 2, burnIn: 2 });
const customOptimizer = tf.train.adam(0.0003);
const customResult = optimizeStep(customModel, customObserver, customOptimizer, customState, 4);
assert.ok(Number.isFinite(customResult.score) && customResult.gradientNorm > 0);
assert.equal(customModel.weights.length, 8);
assert.equal(customObserver.kernels.length, 3);
customState.dispose(); customModel.dispose(); customObserver.dispose(); customOptimizer.dispose();
assert.equal(tf.memory().numTensors, 0, 'All owned tensors must be disposed');
console.log(`PASS: end-to-end gradient, ascent ${before.toFixed(4)} -> ${after.toFixed(4)}, frozen observer, bounded states, no tensor leaks`);
console.log('PASS: custom architectures and independent NumPy reference');
