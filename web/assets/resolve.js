/* ============================================================================
   Resolve — renders any source through a sensor grid.

   Samples a source (text, image, or live video) down to a coarse grid of
   luminance values, then redraws each cell as a block or a glyph chosen from a
   density ramp. Dropping the grid resolution is what makes the image fall
   apart; raising it is what makes it resolve.

   Deliberately plain 2D canvas: every knob is a number, so the look can be
   tuned by eye in real time rather than recompiled.
   ========================================================================= */

export const RAMPS = {
  blocks:   ' ░▒▓█',
  ascii:    ' .:-=+*#%@',
  asciiLong:' .\'`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
  dots:     ' ·∶∷⁘⁙●',
  binary:   ' 01',
  minimal:  ' ▁▂▃▄▅▆▇█',
  shades:   ' ⠁⠃⠇⠧⠷⠿',
};

export const DEFAULTS = {
  // grid
  cell: 12,            // cell height in CSS pixels
  charAspect: 0.58,    // width/height of one cell; 0.58 suits monospace glyphs
  // rendering
  mode: 'ascii',       // 'ascii' | 'blocks'
  ramp: 'ascii',
  font: 'IBM Plex Mono',
  fontScale: 1.05,     // glyph size relative to cell height
  // tone
  brightness: 0,       // -1..1
  contrast: 1,         // 0..3
  gamma: 1,            // 0.2..3
  invert: false,
  threshold: 0.02,     // cells darker than this are skipped entirely
  // colour
  fg: '#e8edf2',
  bg: '#07080a',
  accent: '#37e2f0',
  accentAt: 0.92,      // luminance above which a cell takes the accent colour
  tint: false,         // blend fg->accent continuously instead of a hard cut
  floor: 0.15,         // alpha of the faintest visible cell
  // motion
  jitter: 0,           // px of per-cell noise, adds a live sensor shimmer

  // melt: cells detach, fall under gravity, and converge on a set of targets
  melt: 0,             // 0 = intact, 1 = fully poured
  meltTargets: [0.125, 0.375, 0.625, 0.875],  // x fractions the cells pour into
  meltFloor: 0.62,     // y fraction the piles land on
  meltPile: 0.10,      // how much landing heights vary, as a fraction of height
  meltStagger: 0.55,   // spread of per-cell start times; 0 = everything at once
  meltConverge: 1,     // 0 = falls straight down, 1 = fully gathers into targets
  meltSpread: 0.06,    // lateral scatter within a pile, as a fraction of width
  meltFade: 0.35,      // how much cells dim once landed
  morphScatter: 0.05,  // how loosely cells pick their target slot
  morphSag: 0.05,      // how far the stream dips on its way across
};

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const PALETTE_STEPS = 32;
const ALPHA_STEPS = 6;

/** Smooth ease-out, so cells gather toward their target before they land. */
const ease = (t) => 1 - Math.pow(1 - t, 3);

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export class Resolve {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = { ...DEFAULTS, ...options };

    // The source is drawn into a small offscreen canvas whose pixel grid IS
    // the sensor grid, so one drawImage does all the downsampling for us and
    // each pixel arrives pre-averaged.
    this.sample = document.createElement('canvas');
    this.sampleCtx = this.sample.getContext('2d', { willReadFrequently: true });

    this.source = null;
    this.sourceAspect = 1;
    this.cols = 0;
    this.rows = 0;
    this.time = 0;

    // Per-cell random values, held stable for the life of a grid so a cell's
    // fall timing and landing spot do not reshuffle every frame.
    this.noise = null;

    // Optional second image the cells reassemble into during the melt.
    this.morphSource = null;
    this.morph = null;        // Float32Array of [px, py] pairs
    this.morphCount = 0;
    this.morphKey = '';
    this.spread = null;       // column -> share of lit cells to its left
    this.spreadKey = '';
  }

  /**
   * Give the melt somewhere to reassemble into.
   *
   * Without this the cells fall into loose piles, which reads as sediment.
   * With it they redistribute into the lit pixels of a second image, so the
   * text visibly becomes the thing it is turning into.
   */
  setMorphTarget(source) {
    this.morphSource = source;
    this.morphKey = '';
    return this;
  }

  /**
   * Work out how the source's lit cells are distributed across its columns.
   *
   * Targets were assigned by raw horizontal position, which assumes the ink
   * is spread evenly. It is not: a short centred line and a long wide one
   * feed wildly different amounts of material to each target, so some
   * destinations ended up dense and others nearly empty. Mapping through the
   * cumulative distribution instead sends an equal share of cells to each.
   */
  _buildSpread(data, cols, rows, o) {
    const key = `${cols}x${rows}:${o.threshold}:${o.contrast}:${o.gamma}`;
    if (this.spreadKey === key) return;
    this.spreadKey = key;

    const perCol = new Float32Array(cols);
    let total = 0;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4;
        const a = data[i + 3] / 255;
        if (a === 0) continue;
        let lum = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255 * a;
        lum = clamp((lum - 0.5) * o.contrast + 0.5 + o.brightness, 0, 1);
        if (o.gamma !== 1) lum = Math.pow(lum, o.gamma);
        if (o.invert) lum = 1 - lum;
        if (lum <= o.threshold) continue;
        perCol[x]++; total++;
      }
    }

    const cdf = new Float32Array(cols);
    let running = 0;
    for (let x = 0; x < cols; x++) {
      running += perCol[x];
      cdf[x] = total > 0 ? (running - perCol[x] * 0.5) / total : x / cols;
    }
    this.spread = cdf;
  }

  /** Collect the lit cells of the morph target as screen positions. */
  _buildMorph(cols, rows, cellW, cellH) {
    const key = `${cols}x${rows}`;
    if (this.morphKey === key) return;
    this.morphKey = key;

    const c = document.createElement('canvas');
    c.width = cols; c.height = rows;
    const cx = c.getContext('2d', { willReadFrequently: true });
    cx.drawImage(this.morphSource, 0, 0, cols, rows);
    const d = cx.getImageData(0, 0, cols, rows).data;

    const pts = [];
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4;
        const lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255 * (d[i + 3] / 255);
        if (lum > 0.35) pts.push(x, y);
      }
    }
    // Ordered left to right so cells travel to the nearest cluster rather
    // than criss-crossing the screen to reach a randomly assigned target.
    const pairs = [];
    for (let i = 0; i < pts.length; i += 2) pairs.push([pts[i], pts[i + 1]]);
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

    this.morph = new Float32Array(pairs.length * 2);
    pairs.forEach(([x, y], i) => {
      this.morph[i * 2] = x * cellW;
      this.morph[i * 2 + 1] = y * cellH;
    });
    this.morphCount = pairs.length;
  }

  /** Point the renderer at a canvas, image, or video element. */
  setSource(source) {
    this.source = source;
    const w = source.videoWidth || source.naturalWidth || source.width;
    const h = source.videoHeight || source.naturalHeight || source.height;
    this.sourceAspect = w && h ? w / h : 1;
    return this;
  }

  set(key, value) { this.opts[key] = value; return this; }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return false;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = w;
    this.h = h;
    return true;
  }

  /**
   * Draw one frame.
   * `resolution` in 0..1 scales the grid: 0 is maximally coarse, 1 is the
   * configured cell size. This is the parameter scroll drives.
   */
  render(resolution = 1) {
    const o = this.opts;
    if (!this.w || !this.h) { if (!this.resize()) return; }

    // Coarse end of the range is 9x the cell size; that is far enough to be
    // genuinely unreadable without the grid collapsing to a handful of cells.
    const eased = clamp(resolution, 0, 1);
    const cellH = o.cell * (9 - eased * 8);
    const cellW = cellH * o.charAspect;

    const cols = Math.max(1, Math.ceil(this.w / cellW));
    const rows = Math.max(1, Math.ceil(this.h / cellH));

    if (cols !== this.cols || rows !== this.rows) {
      this.sample.width = cols;
      this.sample.height = rows;
      this.cols = cols;
      this.rows = rows;
      this.noise = new Float32Array(cols * rows * 2);
      for (let i = 0; i < this.noise.length; i++) this.noise[i] = Math.random();
    }

    // --- sample the source into the grid ---------------------------------
    this.sampleCtx.clearRect(0, 0, cols, rows);
    if (this.source) {
      // Letterbox the source so it keeps its aspect ratio inside the grid.
      const gridAspect = (cols * cellW) / (rows * cellH);
      let dw = cols, dh = rows, dx = 0, dy = 0;
      if (this.sourceAspect > gridAspect) {
        dh = cols / this.sourceAspect * (cellW / cellH);
        dy = (rows - dh) / 2;
      } else {
        dw = rows * this.sourceAspect / (cellW / cellH);
        dx = (cols - dw) / 2;
      }
      try {
        this.sampleCtx.drawImage(this.source, dx, dy, dw, dh);
      } catch (e) {
        // A video that has not produced a frame yet throws; skip this frame.
        return;
      }
    }

    const data = this.sampleCtx.getImageData(0, 0, cols, rows).data;
    const ramp = RAMPS[o.ramp] || RAMPS.ascii;
    const last = ramp.length - 1;

    // --- paint ------------------------------------------------------------
    const ctx = this.ctx;
    ctx.fillStyle = o.bg;
    ctx.fillRect(0, 0, this.w, this.h);

    const useGlyphs = o.mode === 'ascii';
    if (useGlyphs) {
      ctx.font = `${cellH * o.fontScale}px "${o.font}", ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
    }

    if (this.morphSource && o.melt > 0) {
      this._buildMorph(cols, rows, cellW, cellH);
      this._buildSpread(data, cols, rows, o);
    }

    const palette = this._palette();
    let currentStyle = null;
    this.time += 0.016;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4;
        const a = data[i + 3] / 255;
        if (a === 0) continue;

        // Rec. 601 luma, which tracks perceived brightness far better than a
        // flat average of the channels.
        let lum = (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) / 255;
        lum *= a;

        lum = clamp((lum - 0.5) * o.contrast + 0.5 + o.brightness, 0, 1);
        if (o.gamma !== 1) lum = Math.pow(lum, o.gamma);
        if (o.invert) lum = 1 - lum;
        if (lum <= o.threshold) continue;

        let px = x * cellW;
        let py = y * cellH;
        let alphaScale = 1;

        if (o.melt > 0) {
          const ni = (y * cols + x) * 2;
          const r1 = this.noise[ni];
          const r2 = this.noise[ni + 1];

          // Stagger the start so the text sags and tears apart rather than
          // dropping as one solid block. Lower rows go first, which reads as
          // the bottom of the letters giving way under the weight above.
          const bias = (y / rows) * 0.45 + r1 * 0.55;
          const delay = bias * o.meltStagger;
          const span = Math.max(1 - delay, 1e-3);
          const t = clamp((o.melt - delay) / span, 0, 1);

          if (t > 0) {
            const fall = t * t;             // accelerating, like gravity
            const gather = ease(t);         // converge sooner than they land

            if (this.morphCount) {
              // Reassemble into the target image. The cell's horizontal rank
              // picks roughly where it belongs, with a little scatter, so the
              // left of the old word feeds the left of the new one.
              // Rank by share of ink to the left, not by raw position, so
              // every target receives a comparable amount of material.
              const u = this.spread ? this.spread[x] : (px + cellW * 0.5) / this.w;
              const jitterN = this.morphCount * o.morphScatter;
              let idx = Math.floor(u * this.morphCount + (r2 - 0.5) * jitterN);
              idx = ((idx % this.morphCount) + this.morphCount) % this.morphCount;

              const tx = this.morph[idx * 2];
              const ty = this.morph[idx * 2 + 1];
              // Sag downward first, then swing into place, so it reads as
              // falling material being caught rather than a straight tween.
              px += (tx - px) * gather;
              py += (ty - py) * gather;
              py += Math.sin(t * Math.PI) * o.morphSag * this.h;
              alphaScale = 1 - t * o.meltFade;
            } else {
              // No target: fall into loose piles on the floor.
              const u = (px + cellW * 0.5) / this.w;
              const n = o.meltTargets.length;
              const slot = Math.min(n - 1, Math.max(0, Math.floor(u * n)));
              const tx = (o.meltTargets[slot] + (r2 - 0.5) * o.meltSpread) * this.w;
              const landY = (o.meltFloor - r1 * o.meltPile) * this.h;
              px += (tx - px) * gather * o.meltConverge;
              py += (landY - py) * fall;
              alphaScale = 1 - t * o.meltFade;
            }
          }
        }

        if (o.jitter) {
          px += (Math.sin(this.time * 3 + x * 0.7 + y) * o.jitter);
          py += (Math.cos(this.time * 2.6 + y * 0.9 - x) * o.jitter);
        }

        // Colours come from a small lookup table rather than being built per
        // cell. Setting fillStyle is the expensive call in this loop, and a
        // continuous tint would otherwise give every one of fifteen thousand
        // cells its own unique style string.
        const bucket = Math.round(clamp(alphaScale, 0, 1) * (ALPHA_STEPS - 1));
        if (bucket === 0) continue;   // fully faded out; nothing to paint
        const style = palette[bucket][Math.round(lum * (PALETTE_STEPS - 1))];
        if (style !== currentStyle) {
          ctx.fillStyle = style;
          currentStyle = style;
        }

        if (useGlyphs) {
          const ch = ramp[Math.round(lum * last)];
          if (ch !== ' ') ctx.fillText(ch, px + cellW / 2, py + cellH / 2);
        } else {
          const s = lum;
          ctx.fillRect(px, py, Math.max(1, cellW * s), Math.max(1, cellH * s));
        }
      }
    }
  }

  /**
   * Build the luminance -> colour ramp for this frame.
   *
   * `tint` blends foreground to accent continuously across the range, so the
   * two colours read as one gradient; without it the accent is a hard cut at
   * `accentAt`. Alpha is folded in here too, fading the darkest cells out so
   * the field has depth instead of a binary on/off edge.
   */
  _palette() {
    const o = this.opts;
    const fg = hexToRgb(o.fg);
    const ac = hexToRgb(o.accent);

    // Two dimensions: luminance picks the colour, and a separate opacity axis
    // fades a cell out. Folding the two together would mean a fading cell
    // walked back down the colour ramp and changed hue on its way out, which
    // is why melted cells were turning green instead of simply vanishing.
    const lut = new Array(ALPHA_STEPS);
    for (let a = 0; a < ALPHA_STEPS; a++) {
      const opacity = a / (ALPHA_STEPS - 1);
      const row = new Array(PALETTE_STEPS);
      for (let i = 0; i < PALETTE_STEPS; i++) {
        const lum = i / (PALETTE_STEPS - 1);
        let r, g, b;
        if (o.tint) {
          const t = clamp(lum / Math.max(o.accentAt, 0.001), 0, 1);
          r = fg[0] + (ac[0] - fg[0]) * t;
          g = fg[1] + (ac[1] - fg[1]) * t;
          b = fg[2] + (ac[2] - fg[2]) * t;
        } else {
          [r, g, b] = lum >= o.accentAt ? ac : fg;
        }
        const alpha = clamp(o.floor + lum * (1 - o.floor + 0.1), 0, 1) * opacity;
        row[i] = `rgba(${r | 0},${g | 0},${b | 0},${alpha.toFixed(3)})`;
      }
      lut[a] = row;
    }
    return lut;
  }

  /** Cell count of the current grid, useful for a readout. */
  get gridSize() { return { cols: this.cols, rows: this.rows }; }
}

/* ------------------------------------------------------------------ sources */

/** Render a line (or lines) of text into a canvas for use as a source. */
export function textSource(lines, {
  width = 1600, height = 900, font = 'IBM Plex Mono',
  weight = 600, fill = '#ffffff', tracking = 0, fit = 0.86, softness = 0.55,
  padBottom = 0,
} = {}) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = fill;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const rows = Array.isArray(lines) ? lines : [lines];
  // padBottom reserves a strip at the foot of the frame, so a caller can keep
  // the composition clear of anything anchored to the bottom of the viewport.
  const usable = height * (1 - padBottom);
  const lineHeight = usable / (rows.length + 0.35);

  // Find the largest size at which the widest line still fits the target
  // width, so the type always fills the frame regardless of what it says.
  let size = lineHeight;
  for (let i = 0; i < 40; i++) {
    ctx.font = `${weight} ${size}px "${font}", monospace`;
    const widest = Math.max(...rows.map((t) => ctx.measureText(t).width + tracking * t.length));
    if (widest <= width * fit) break;
    size *= 0.94;
  }

  ctx.font = `${weight} ${size}px "${font}", monospace`;
  if (tracking) ctx.letterSpacing = `${tracking}px`;

  // Crisp type is pure white on pure black, which is binary: every lit cell
  // lands at the top of the density ramp, so the ramp does nothing and the
  // glyphs never modulate. A slight blur gives the edges a real gradient, so
  // letterforms resolve through the middle of the ramp instead of snapping on.
  if (softness > 0) ctx.filter = `blur(${Math.max(1, size * softness * 0.06)}px)`;

  rows.forEach((text, i) => {
    ctx.fillText(text, width / 2, lineHeight * (i + 0.72));
  });
  ctx.filter = 'none';
  return canvas;
}

/** Ask for the camera and return a playing <video> to use as a source. */
export async function webcamSource() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 540, facingMode: 'user' }, audio: false,
  });
  const video = document.createElement('video');
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  return video;
}
