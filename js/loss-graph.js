/**
 * Loss Graph Module
 *
 * Canvas-based visualization for training loss curves and epiplexity calculation.
 */

export class LossGraph {
  /**
   * @param {HTMLCanvasElement} canvas - The canvas element to render to
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // Set actual canvas resolution
    this.width = 384;
    this.height = 250;
    this.canvas.width = this.width;
    this.canvas.height = this.height;

    // Graph margins
    this.margin = { top: 20, right: 20, bottom: 30, left: 50 };
    this.plotWidth = this.width - this.margin.left - this.margin.right;
    this.plotHeight = this.height - this.margin.top - this.margin.bottom;

    // Data
    this.losses = [];
    this.maxSteps = 500;
    this.smoothingFactor = 0.8; // EMA smoothing (0 = no smoothing, 0.99 = very smooth)

    // Styling
    this.colors = {
      background: '#0a0a15',
      grid: '#1a1a2e',
      axis: '#444',
      axisLabel: '#666',
      lineRaw: 'rgba(78, 205, 196, 0.3)',
      lineSmooth: '#4ecdc4',
      area: 'rgba(78, 205, 196, 0.3)',
      finalLine: '#e74c3c'
    };

    this.reset();
  }

  /**
   * Reset the graph
   */
  reset() {
    this.losses = [];
    this.draw();
  }

  /**
   * Show a message on the graph
   * @param {string} message
   */
  showMessage(message) {
    const ctx = this.ctx;

    // Clear
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, this.width, this.height);

    // Draw message centered
    ctx.fillStyle = this.colors.lineSmooth;
    ctx.font = '16px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(message, this.width / 2, this.height / 2);
  }

  /**
   * Set max steps (for x-axis scaling)
   * @param {number} maxSteps
   */
  setMaxSteps(maxSteps) {
    this.maxSteps = maxSteps;
    this.draw();
  }

  /**
   * Set smoothing factor
   * @param {number} factor - 0 = no smoothing, 0.99 = very smooth
   */
  setSmoothingFactor(factor) {
    this.smoothingFactor = Math.max(0, Math.min(0.99, factor));
    this.draw();
  }

  /**
   * Compute smoothed losses using exponential moving average
   * @returns {Array} - Array of {step, loss} with smoothed values
   */
  getSmoothedLosses() {
    if (this.losses.length === 0) return [];

    const smoothed = [];
    let ema = this.losses[0].loss;

    for (const point of this.losses) {
      ema = this.smoothingFactor * ema + (1 - this.smoothingFactor) * point.loss;
      smoothed.push({ step: point.step, loss: ema });
    }

    return smoothed;
  }

  /**
   * Add a data point
   * @param {number} step - Training step
   * @param {number} loss - Loss value
   */
  addPoint(step, loss) {
    this.losses.push({ step, loss });
    this.draw();
  }

  /**
   * Get current smoothed (EMA) loss value
   * @returns {number|null}
   */
  getCurrentSmoothedLoss() {
    const smoothed = this.getSmoothedLosses();
    if (smoothed.length === 0) return null;
    return smoothed[smoothed.length - 1].loss;
  }

  /**
   * Get the minimum smoothed loss value
   * @returns {number|null}
   */
  getMinLoss() {
    const smoothed = this.getSmoothedLosses();
    if (smoothed.length === 0) return null;
    return Math.min(...smoothed.map(p => p.loss));
  }

  /**
   * Find the index where smoothed loss reaches its minimum value
   * @returns {number} - Index of minimum point
   */
  getFirstConvergenceIndex() {
    const smoothed = this.getSmoothedLosses();
    if (smoothed.length < 2) return smoothed.length - 1;

    // Find the index of the minimum value in the smoothed curve
    let minIdx = 0;
    let minVal = smoothed[0].loss;

    for (let i = 1; i < smoothed.length; i++) {
      if (smoothed[i].loss < minVal) {
        minVal = smoothed[i].loss;
        minIdx = i;
      }
    }

    return minIdx;
  }

  /**
   * Compute epiplexity (area under smoothed curve above min loss, until first convergence)
   * @returns {number}
   */
  computeEpiplexity() {
    const smoothed = this.getSmoothedLosses();
    if (smoothed.length < 2) return 0;

    const minLoss = this.getMinLoss();
    const convergenceIdx = this.getFirstConvergenceIndex();
    let area = 0;

    // Only integrate up to first convergence
    for (let i = 1; i <= convergenceIdx; i++) {
      const prev = smoothed[i - 1];
      const curr = smoothed[i];

      // Trapezoidal integration of (loss - minLoss), only positive values
      const h1 = Math.max(0, prev.loss - minLoss);
      const h2 = Math.max(0, curr.loss - minLoss);
      const width = curr.step - prev.step;

      area += (h1 + h2) * width / 2;
    }

    return area;
  }

  /**
   * Draw the graph
   * @param {boolean} showArea - Whether to show the epiplexity area
   */
  draw(showArea = false) {
    const ctx = this.ctx;

    // Clear
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, this.width, this.height);

    // Find y-axis range
    let maxLoss = 5; // Default max
    if (this.losses.length > 0) {
      maxLoss = Math.max(...this.losses.map(p => p.loss)) * 1.1;
    }
    const minLoss = 0;

    // Helper functions for coordinate conversion
    const xScale = (step) => {
      return this.margin.left + (step / this.maxSteps) * this.plotWidth;
    };
    const yScale = (loss) => {
      return this.margin.top + this.plotHeight - ((loss - minLoss) / (maxLoss - minLoss)) * this.plotHeight;
    };

    // Draw grid
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;

    // Horizontal grid lines
    const numYTicks = 5;
    for (let i = 0; i <= numYTicks; i++) {
      const y = this.margin.top + (i / numYTicks) * this.plotHeight;
      ctx.beginPath();
      ctx.moveTo(this.margin.left, y);
      ctx.lineTo(this.margin.left + this.plotWidth, y);
      ctx.stroke();
    }

    // Vertical grid lines
    const numXTicks = 5;
    for (let i = 0; i <= numXTicks; i++) {
      const x = this.margin.left + (i / numXTicks) * this.plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, this.margin.top);
      ctx.lineTo(x, this.margin.top + this.plotHeight);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = this.colors.axis;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(this.margin.left, this.margin.top);
    ctx.lineTo(this.margin.left, this.margin.top + this.plotHeight);
    ctx.lineTo(this.margin.left + this.plotWidth, this.margin.top + this.plotHeight);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = this.colors.axisLabel;
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';

    // X-axis labels
    for (let i = 0; i <= numXTicks; i++) {
      const step = (i / numXTicks) * this.maxSteps;
      const x = xScale(step);
      ctx.fillText(Math.round(step).toString(), x, this.height - 8);
    }

    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= numYTicks; i++) {
      const loss = minLoss + ((numYTicks - i) / numYTicks) * (maxLoss - minLoss);
      const y = this.margin.top + (i / numYTicks) * this.plotHeight;
      ctx.fillText(loss.toFixed(1), this.margin.left - 5, y + 3);
    }

    // Graph title
    ctx.fillStyle = this.colors.axisLabel;
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loss', this.width / 2, 12);

    // If no data, we're done
    if (this.losses.length === 0) return;

    const smoothed = this.getSmoothedLosses();

    // Draw epiplexity area if requested (using smoothed curve)
    if (showArea && smoothed.length > 1) {
      const minLoss = this.getMinLoss();
      const convergenceIdx = this.getFirstConvergenceIndex();

      // Only shade area from start to first convergence, above minLoss line
      ctx.fillStyle = this.colors.area;
      ctx.beginPath();

      // Start at the minLoss line
      ctx.moveTo(xScale(smoothed[0].step), yScale(minLoss));

      // Draw up to convergence point, clamping to stay above minLoss
      for (let i = 0; i <= convergenceIdx; i++) {
        const point = smoothed[i];
        const clampedLoss = Math.max(point.loss, minLoss); // Never go below minLoss
        ctx.lineTo(xScale(point.step), yScale(Math.min(clampedLoss, maxLoss)));
      }

      // Close back to the minLoss line
      ctx.lineTo(xScale(smoothed[convergenceIdx].step), yScale(minLoss));
      ctx.closePath();
      ctx.fill();

      // Draw min loss line (dashed)
      ctx.strokeStyle = this.colors.finalLine;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(this.margin.left, yScale(minLoss));
      ctx.lineTo(this.margin.left + this.plotWidth, yScale(minLoss));
      ctx.stroke();
      ctx.setLineDash([]);

      // Mark convergence point
      const convPoint = smoothed[convergenceIdx];
      ctx.fillStyle = this.colors.finalLine;
      ctx.beginPath();
      ctx.arc(xScale(convPoint.step), yScale(convPoint.loss), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw raw loss curve (faint)
    ctx.strokeStyle = this.colors.lineRaw;
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < this.losses.length; i++) {
      const point = this.losses[i];
      const x = xScale(point.step);
      const y = yScale(Math.min(point.loss, maxLoss));

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw smoothed loss curve (bold)
    if (smoothed.length > 0) {
      ctx.strokeStyle = this.colors.lineSmooth;
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < smoothed.length; i++) {
        const point = smoothed[i];
        const x = xScale(point.step);
        const y = yScale(Math.min(point.loss, maxLoss));

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
  }

  /**
   * Finalize the graph (show area and compute epiplexity)
   * @returns {number} - The computed epiplexity value
   */
  finalize() {
    this.draw(true);
    return this.computeEpiplexity();
  }
}
