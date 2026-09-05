# epiplexity

Static browser experiments with epiplexity estimation and optimization. Serve
this directory with `python3 -m http.server 8000`, then open
[localhost:8000](http://localhost:8000/). TensorFlow.js 4.17.0 loads from jsDelivr;
there is no build step or package manager.

## Experiment tabs

- **Measure & Evolve:** discrete four-state NCA, trained convolutional or
  transformer observer, learning-curve estimates, mutation search, and zoo.
- **Direct Optimization:** continuous two-channel NCA, frozen random CNN
  reservoir, and gradient ascent on a reservoir epiplexity approximation.
  Play inspects the learned rule on a 48 × 48 grid. Export provides downloadable
  or copyable JSON with its weights,
  architecture, seeds, prediction settings, training scores, and evaluation scores.

Continuous mode exposes the NCA's hidden width, convolutional depth, and
neighborhood size, plus the observer's feature width, depth, and kernel size.
Parameter counts and the observer's receptive field update with the controls.
Changing architecture, prediction, or learning-rate settings starts a new
experiment. Stop/Continue preserves the rule and Adam state; switching tabs
pauses work. Larger grids, horizons, and networks take longer per step.

## What is predicted?

For each optimization step:

1. Draw two random continuous grids and burn in the current NCA for 32 steps
   outside the gradient tape. Add Gaussian noise with standard deviation 0.1
   so a collapsed state has a perturbed neighborhood to respond to.
2. Apply a frozen, circular-padded CNN to the starting state. Each cell gets
   `m` local features. Pool cells and grids as rows of a shared regression.
3. Roll out the NCA for `T` differentiable steps (default 8). The target at
   each cell is its two-channel trajectory: `2T` columns, rather than just
   its last state. No future frame is fed to the observer.
4. Standardize feature columns and divide by `sqrt(m)`. Center each target
   column without dividing by its empirical standard deviation (target unit 1).
5. Fit `W = argmin ||Y - HW||² + 0.3 ||W||²` and score
   `S = 0.5 log₂ det(I + WᵀW)`. Minimize `-S/100` with Adam and global
   gradient clipping at 0.5, updating only the NCA parameters.

This implements the ridge/spectral estimator in equations 7–9 and the stable
solve in Appendix E of [Zhang & Levin, *Intelligence from Learnable Novelty*](https://arxiv.org/html/2607.18433v1).
The [authors' reference code](https://github.com/Zhangyanbo/learnable-novelty)
describes the same prediction task in 1D. Our implementation is independently
written from the mathematics.

The browser adaptation uses 2D neighborhoods, smaller networks/batches,
ELU hidden layers in the NCA without batch normalization, and Adam with a
constant learning rate. The NCA applies one spatial convolution followed by
pointwise layers, a residual update, and per-cell unit normalization. The
observer uses spatial convolutions except for its final pointwise projection;
hidden layers normalize across channels before ELU. A `1e-8` variance floor
keeps feature normalization finite on constant fields. These are adaptations,
not a reproduction of the paper's soliton experiment.

The readout uses augmented Householder QR in JavaScript float64 rather than
forming normal equations. A small Cholesky factorization evaluates the
equivalent log-determinant without differentiating through repeated singular
values. A TensorFlow.js custom gradient implements the implicit derivative:
with `Q = ∂S/∂W` and `U = (HᵀH + λI)⁻¹Q`, solved through the QR factors,
`∂S/∂Y = HU` and `∂S/∂H = (Y-HW)Uᵀ - HUWᵀ`. Convolutions and rollout
gradients run on TensorFlow.js's selected backend. The small readout solve
synchronizes to CPU once per score.

The gold curve re-evaluates the rule every ten steps and on stop using a fixed,
independent initialization seed; those grids never enter the optimizer. It uses
the same frozen observer, so it checks fresh initial conditions, not transfer to
other observers. Training scores are measured immediately before each update;
evaluation scores are measured after the update.

Scores depend on observer capacity, horizon, sampling, and target units. They
are kept separate from the discrete tab's learning-curve estimates. A higher
score alone does not establish traveling structures or computation: inspect
long rollouts and compare observers and horizons to investigate those behaviors.

## Numerical checks

Download the same TF.js bundle used by the page, then run the focused checks:

```sh
curl -L --fail https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js -o /tmp/epiplexity-tf.cjs
node tests/direct-optimization.mjs /tmp/epiplexity-tf.cjs
```

Checks cover an independent NumPy least-squares/SVD reference, rank-deficient
and underdetermined solves, zero targets, finite-difference gradients for both
readout inputs and NCA weights, ascent on a fixed batch, a frozen observer,
bounded states, custom architectures, and tensor disposal. Browser smoke checks:
Play/Randomize in the discrete tab; optimize ten steps, stop/continue, change
architectures, switch tabs, and inspect continuous playback.
