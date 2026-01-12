/**
 * Renderer Module
 *
 * Handles canvas rendering and state initialization for discrete NCA visualization.
 * Maps 4 discrete states to a cyberpunk color palette.
 */

// 4-color cyberpunk palette
const PALETTE = [
  [12, 12, 24],     // 0: Near-black (background)
  [255, 0, 110],    // 1: Hot pink
  [0, 255, 198],    // 2: Electric cyan
  [255, 234, 0],    // 3: Bright yellow
];

export const PALETTE_SIZE = 4;

export class Renderer {
  /**
   * @param {HTMLCanvasElement} canvas - The canvas element to render to
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Precompute palette tensor for efficient GPU color mapping
    this.paletteTensor = tf.tensor2d(PALETTE, [PALETTE_SIZE, 3]).div(255);
  }

  /**
   * Set canvas dimensions
   * @param {number} width
   * @param {number} height
   */
  setSize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
  }

  /**
   * Create an initial state tensor with random one-hot vectors
   * @param {number} width
   * @param {number} height
   * @param {number} numStates - Number of discrete states (default 8)
   * @returns {tf.Tensor4D} - State tensor [1, height, width, numStates] with one-hot encoding
   */
  createRandomState(width, height, numStates = 8) {
    return tf.tidy(() => {
      // Generate random class indices [height, width]
      const randomIndices = tf.randomUniformInt([height, width], 0, numStates);
      // Convert to one-hot [height, width, numStates] and cast to float32
      const oneHot = tf.oneHot(randomIndices, numStates).toFloat();
      // Add batch dimension [1, height, width, numStates]
      return oneHot.expandDims(0);
    });
  }

  /**
   * Draw a discrete state tensor to the canvas
   * Maps one-hot encoded states to palette colors (modulo PALETTE_SIZE for hidden states)
   * @param {tf.Tensor4D} state - State tensor [1, height, width, numStates] (one-hot)
   * @returns {Promise<void>}
   */
  async draw(state) {
    const imageData = tf.tidy(() => {
      // Remove batch dimension: [height, width, numStates]
      const squeezed = state.squeeze([0]);

      // Get class indices via argmax: [height, width]
      let indices = tf.argMax(squeezed, 2);

      // Apply modulo to map to palette (for states >= PALETTE_SIZE)
      indices = tf.mod(indices, PALETTE_SIZE).toInt();

      // Map indices to RGB colors using gather
      // indices: [height, width] -> flatten to [height*width]
      const [h, w] = indices.shape;
      const flatIndices = indices.reshape([-1]);

      // Gather colors from palette: [height*width, 3]
      const flatColors = tf.gather(this.paletteTensor, flatIndices);

      // Reshape back to [height, width, 3]
      return flatColors.reshape([h, w, 3]);
    });

    await tf.browser.toPixels(imageData, this.canvas);
    imageData.dispose();
  }

  /**
   * Clear the canvas
   * @param {string} color - Fill color (default: black)
   */
  clear(color = '#000000') {
    this.ctx.fillStyle = color;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Dispose of resources
   */
  dispose() {
    this.paletteTensor.dispose();
  }
}
