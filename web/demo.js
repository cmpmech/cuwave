// The cuwave demo: the 2D scalar wave equation on a grid the size of the window,
// stepped by cuwave's own kernels compiled to WebAssembly (solver.js, build.sh),
// on a pool of threads where the page is cross-origin isolated (coi.js), or by
// their WGSL port (gpu.js) with ?gpu. Classic scripts, so it also runs opened
// as a file.

// ------------------------------------- settings
// discretization
const COURANT = 0.5; // c dt / dx, below the 2D limit at order 4 (0.61)
const WAVELENGTH = 36; // CSS px of a burst's wavelength, where the grid resolves it
const RESOLVED = { 1: 20, 2: 10 }; // fewest nodes a wavelength, by stencil radius
const CYCLES = 3; // periods of a burst
// absorption
const SPONGE = 2; // wavelengths of margin behind an absorbing face
const BETA = 0.07; // sponge strength, 1.8 % reflected at 2 wavelengths
const FADE = 2.5; // seconds over which a wave falls by 1/e
// pace
const SPEED = 0.25; // window diagonals a wave travels per second
const FPS = 30; // frames drawn a second, the time between them the solver's
const BUDGET = 0.6; // share of a frame the solver may take
const SMALLEST = 2; // finest CPU cell, CSS px
const SETTLE = 1200; // ms the animation runs on the empty field before input
const BEHIND = 0.85; // share of the due steps below which the grid coarsens
const THREADS = Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) >> 1));
// interaction
const BRUSH = 10; // wall radius, CSS px
const DRAG = 6; // CSS px a press moves before it draws instead of a burst
const MODES = ["neumann", "dirichlet", "absorbing"]; // solver.cpp's face modes
// look: the colormaps of the README animations, the field clipped at LIMIT
const THEMES = {
  light: { colors: seismic(), wall: [0, 0, 0], limit: 0.03 },
  dark: { colors: berlin(), wall: [255, 255, 255], limit: 0.02 },
};

// ---------------------------------------- setup
const canvas = document.querySelector("canvas");
const help = document.getElementById("help");
const badge = document.getElementById("backend");
const edges = [...document.querySelectorAll(".edge")];
const dark = matchMedia("(prefers-color-scheme: dark)");
const root = document.documentElement;
const mode = () => root.dataset.theme || (dark.matches ? "dark" : "light"); // the toggle, else the system
const modes = [0, 0, 0, 2]; // top, bottom, left, right
let solver, backend, rate, cell, rows, cols, lambda, margin, theme, lut, wallColor;
let stepsPerSecond, msPerStep, dirty = null, debt = 0, last = performance.now();
let quiet = 0, redraw = true; // when the last burst has faded, and whether to draw anyway
const pace = { due: 0, done: 0 }; // steps owed and taken while stepping, for keepUp

document.querySelector(".theme").addEventListener("click", () => {
  root.dataset.theme = mode() === "dark" ? "light" : "dark";
  try {
    localStorage.setItem("theme", root.dataset.theme);
  } catch (e) {}
  applyTheme();
});
if (matchMedia("(pointer: coarse)").matches)
  for (const line of help.children)
    if (line.textContent.includes("click")) line.textContent = line.textContent.replace("click", "tap");
script(crossOriginIsolated ? "solver-threads.js" : "solver.js")
  .then(() => Solver())
  .then(start)
  .catch((e) => (help.textContent = `the demo could not start: ${e}`));

async function start(module) {
  solver = module;
  const threads = solver._threads(THREADS);
  if (new URLSearchParams(location.search).has("gpu"))
    backend = await gpuBackend(solver, canvas, COURANT).catch(() => null);
  backend ??= cpuBackend();
  modes.forEach((mode, face) => solver._boundary(face, mode));
  rate = await calibrate(threads);
  cell = cellSize();
  document.getElementById("pitch").hidden = backend.name === "webgpu";
  applyTheme();
  resize();
  dark.addEventListener("change", applyTheme);
  addEventListener("resize", () => resize());
  document.fonts.ready.then(fit);
  requestAnimationFrame(animate);
  quiet = performance.now() + SETTLE; // the empty field stepped, so keepUp can refit
  await new Promise((resolve) => setTimeout(resolve, SETTLE));
  badge.textContent = backend.name === "webgpu" ? "gpu" : `cpu \u00d7${threads}`;
  badge.title = `${(rate / 1e6).toFixed(0)} M node updates/s, ${cell.toFixed(1)} px cells`;
  listen();
}

// -------------------------------------- helpers
function script(src) {
  return new Promise((resolve, reject) => {
    const element = Object.assign(document.createElement("script"), { src });
    element.onload = resolve;
    element.onerror = () => reject(new Error(`${src} did not load`));
    document.head.append(element);
  });
}

// nodes a wavelength and margin nodes at `size` CSS px a node
function geometry(size) {
  const lambda = Math.max(RESOLVED[solver._radius()], WAVELENGTH / size);
  return { lambda, margin: Math.round(SPONGE * lambda) };
}

// node updates per second on a scratch grid, timed as the animation steps:
// for BUDGET of a frame, then idle, which runs slower than steps back to back
// where cores clock down or other work evicts the grid from the cache
async function calibrate(threads) {
  const [r, c] = backend.name === "webgpu" ? [1000, 1500] : [200 * threads, 300];
  const { margin } = geometry(4);
  solver._configure(COURANT, 0, margin, BETA);
  solver._resize(r, c);
  backend.resize(r, c);
  solver._burst(r / 2, c / 2, 1, 0.1, CYCLES);
  const frame = 1e3 / FPS;
  let steps = 0, busy = 0;
  for (let k = 0; k < 10; ++k) {
    const start = performance.now();
    let n = 0;
    while (performance.now() - start < BUDGET * frame) {
      backend.advance(2);
      await backend.finished();
      n += 2;
    }
    const elapsed = performance.now() - start;
    if (k >= 2) {
      steps += n; // the first frames warm up
      busy += elapsed;
    }
    await new Promise((resolve) => setTimeout(resolve, frame - elapsed));
  }
  solver._clear();
  backend.clear();
  return (steps * backend.cost(r, c, margin)) / (busy / 1e3);
}

// the finest cell whose steps at SPEED fit the BUDGET of the measured rate
function cellSize() {
  const reach = SPEED * Math.hypot(screen.width, screen.height);
  let size = backend.smallest;
  while (size < 32) {
    const r = Math.ceil(screen.height / size), c = Math.ceil(screen.width / size);
    const cost = backend.cost(r, c, geometry(size).margin) * reach / (size * COURANT);
    if (cost <= BUDGET * rate) break;
    size *= 1.05;
  }
  return size;
}

function applyTheme() {
  theme = THEMES[mode()];
  lut = new Uint32Array(256);
  for (let i = 0; i < 256; ++i) lut[i] = rgba(theme.colors(i / 255));
  wallColor = rgba(theme.wall);
  const [r, g, b] = theme.colors(0.5);
  document.documentElement.style.setProperty("--bg", `rgb(${r}, ${g}, ${b})`);
  if (rows) backend.view(rows, cols, cell * (devicePixelRatio || 1), theme.limit, lut, wallColor);
  redraw = true;
}

// a color packed as the little-endian bytes of ImageData
function rgba([r, g, b]) {
  return ((255 << 24) | (b << 16) | (g << 8) | r) >>> 0;
}

// the grid at the window, cells `scale` times the old ones, the field and
// walls resampled onto it
function resize(scale = 1) {
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.round(innerWidth * dpr);
  canvas.height = Math.round(innerHeight * dpr);
  rows = Math.ceil(innerHeight / cell);
  cols = Math.ceil(innerWidth / cell);
  ({ lambda, margin } = geometry(cell));
  stepsPerSecond = (SPEED * Math.hypot(innerWidth, innerHeight)) / (cell * COURANT);
  msPerStep = (1e3 * backend.cost(rows, cols, margin)) / rate;
  const decay = 2 / (FADE * stepsPerSecond * COURANT); // u ~ exp(-decay t / 2)
  solver._configure(COURANT, decay, margin, BETA);
  solver._resize(rows, cols, scale);
  backend.resize(rows, cols);
  backend.view(rows, cols, cell * dpr, theme.limit, lut, wallColor);
  pace.due = pace.done = 0;
  redraw = true;
  fit();
}

// a coarser grid where a second of stepping fell behind, at the rate the
// animation itself reached: drawing, a busy machine and clocked-down cores
// make it slower than the calibration, and change over time
function keepUp() {
  if (pace.due < stepsPerSecond) return;
  if (pace.done < BEHIND * pace.due && cell < 32) {
    rate = (1e3 * backend.cost(rows, cols, margin)) / msPerStep;
    const old = cell;
    cell = Math.max(1.1 * cell, cellSize());
    resize(cell / old);
    badge.title = `${(rate / 1e6).toFixed(0)} M node updates/s, ${cell.toFixed(1)} px cells`;
  }
  pace.due = pace.done = 0;
}

// the bar stacked where it would cover the top edge's label
function fit() {
  const bar = document.getElementById("bar");
  document.body.classList.remove("narrow");
  const a = edges[0].getBoundingClientRect(), b = bar.getBoundingClientRect();
  const hit = a.right > b.left && a.left < b.right && a.bottom > b.top && a.top < b.bottom;
  document.body.classList.toggle("narrow", hit);
}

function animate(now) {
  requestAnimationFrame(animate);
  if (now - last < 1e3 / FPS - 4) return; // a 60 Hz display draws every other frame
  if (dirty) {
    solver._repaint(dirty[0], dirty[1]);
    const R = solver._radius(), top = margin + 1;
    backend.upload(top + dirty[0] - R, top + dirty[1] + R, false);
    dirty = null;
    redraw = true;
  }
  // the steps due since the last frame, at most what fits the budget
  const elapsed = Math.min(0.1, (now - last) / 1e3);
  last = now;
  const cap = Math.max(1, Math.floor((BUDGET * elapsed * 1e3) / msPerStep));
  debt = Math.min(debt + elapsed * stepsPerSecond, cap);
  const steps = now < quiet ? Math.floor(debt) : 0; // a faded field is not stepped
  debt -= Math.floor(debt);
  if (now < quiet) {
    pace.due += elapsed * stepsPerSecond;
    pace.done += steps;
  }
  if (steps) {
    const start = performance.now();
    backend.advance(steps);
    const record = () => {
      msPerStep = 0.9 * msPerStep + (0.1 * (performance.now() - start)) / steps;
    };
    const done = backend.finished();
    done ? done.then(record) : record(); // the CPU's steps are done on return
  }
  if (steps || redraw) backend.draw();
  redraw = false;
  keepUp();
}

// the faces' modes rebuilt into the grid
function rebuild() {
  modes.forEach((mode, face) => solver._boundary(face, mode));
  solver._rebuild();
  backend.upload(0, solver._extent(), true);
  edges.forEach((edge, face) => (edge.textContent = MODES[modes[face]]));
  redraw = true;
  fit();
}

// --------------------------------------- backend
// cuwave's kernels on the CPU, drawn into a canvas a node a pixel
function cpuBackend() {
  const context = canvas.getContext("2d");
  const image = document.createElement("canvas");
  const imageContext = image.getContext("2d");
  let frame, pixels;
  return {
    name: "wasm",
    cost: (rows, cols, margin) => rows * (cols + margin), // margins off the domain are skipped
    smallest: SMALLEST,
    resize(rows, cols) {
      image.width = cols;
      image.height = rows;
      frame = imageContext.createImageData(cols, rows);
      pixels = new Uint32Array(frame.data.buffer);
    },
    upload() {},
    view() {},
    advance: (steps) => solver._advance(steps),
    draw() {
      const u = solver.HEAPF32, walls = solver.HEAPU8;
      const origin = solver._origin(), stride = solver._stride();
      const field = (solver._field() >> 2) + origin, solid = solver._walls() + origin;
      const scale = 127.5 / theme.limit;
      for (let r = 0; r < rows; ++r)
        for (let c = 0, row = r * stride; c < cols; ++c) {
          if (walls[solid + row + c]) {
            pixels[r * cols + c] = wallColor;
            continue;
          }
          const i = 127.5 + scale * u[field + row + c];
          pixels[r * cols + c] = lut[i < 0 ? 0 : i > 255 ? 255 : i | 0];
        }
      imageContext.putImageData(frame, 0, 0);
      const dpr = canvas.width / innerWidth;
      context.drawImage(image, 0, 0, cols * cell * dpr, rows * cell * dpr);
    },
    clear() {},
    finished: () => null,
  };
}

// --------------------------------------- pointer
// a press that stays put is a burst; one that moves draws walls, or erases
// them where it started on one
function listen() {
  const presses = new Map();
  const node = (x, y) => [Math.floor(y / cell), Math.floor(x / cell)];

  const nearWall = (x, y) => {
    const solid = solver._walls() + solver._origin(), stride = solver._stride();
    const reach = Math.ceil(BRUSH / 2 / cell), [r0, c0] = node(x, y);
    for (let r = Math.max(0, r0 - reach); r <= Math.min(rows - 1, r0 + reach); ++r)
      for (let c = Math.max(0, c0 - reach); c <= Math.min(cols - 1, c0 + reach); ++c)
        if (solver.HEAPU8[solid + r * stride + c]) return true;
    return false;
  };

  // walls set to `value` along the stroke from (x0, y0) to (x1, y1)
  const paint = (x0, y0, x1, y1, value) => {
    const solid = solver._walls() + solver._origin(), stride = solver._stride();
    const reach = BRUSH / cell;
    const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (cell / 2)));
    for (let k = 0; k <= steps; ++k) {
      const x = (x0 + ((x1 - x0) * k) / steps) / cell;
      const y = (y0 + ((y1 - y0) * k) / steps) / cell;
      const first = Math.max(0, Math.floor(y - reach)), last = Math.min(rows - 1, Math.ceil(y + reach));
      for (let r = first; r <= last; ++r)
        for (let c = Math.max(0, Math.floor(x - reach)); c <= Math.min(cols - 1, x + reach); ++c)
          if ((c + 0.5 - x) ** 2 + (r + 0.5 - y) ** 2 <= reach * reach)
            solver.HEAPU8[solid + r * stride + c] = value;
      dirty = [Math.min(dirty?.[0] ?? first, first), Math.max(dirty?.[1] ?? last + 1, last + 1)];
    }
  };

  const dismiss = () => help.classList.add("gone");

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture?.(e.pointerId);
    const erase = nearWall(e.clientX, e.clientY);
    presses.set(e.pointerId, { x: e.clientX, y: e.clientY, erase, drawing: false });
    dismiss();
  });

  canvas.addEventListener("pointermove", (e) => {
    const press = presses.get(e.pointerId);
    if (!press) return;
    if (!press.drawing && Math.hypot(e.clientX - press.x, e.clientY - press.y) < DRAG) return;
    press.drawing = true;
    paint(press.x, press.y, e.clientX, e.clientY, press.erase ? 0 : 1);
    press.x = e.clientX;
    press.y = e.clientY;
  });

  canvas.addEventListener("pointerup", (e) => {
    const press = presses.get(e.pointerId);
    presses.delete(e.pointerId);
    if (press && !press.drawing) {
      const [r, c] = node(press.x, press.y);
      solver._burst(r, c, 1, 1 / lambda, CYCLES);
      quiet = performance.now() + 6e3 * FADE; // e^-6 of the burst left, below a color step
    }
  });

  canvas.addEventListener("pointercancel", (e) => presses.delete(e.pointerId));

  edges.forEach((edge, face) =>
    edge.addEventListener("click", () => {
      modes[face] = (modes[face] + 1) % MODES.length;
      rebuild();
      dismiss();
    }),
  );

  document.getElementById("reset").addEventListener("click", () => {
    const walls = solver._walls();
    solver.HEAPU8.fill(0, walls, walls + solver._extent() * solver._stride());
    rebuild();
    solver._clear();
    backend.clear();
  });
  rebuild();
}

// ------------------------------------ colormaps
// matplotlib's seismic, piecewise linear
function seismic() {
  const stops = [
    [0, 0, 77],
    [0, 0, 255],
    [255, 255, 255],
    [255, 0, 0],
    [128, 0, 0],
  ];
  return (x) => interpolate(stops, x);
}

// matplotlib's berlin at 65 stops
function berlin() {
  const hex =
    "9eb0ff95affb8caef682adf279abed6fa9e865a7e25ba4db519fd3489aca4194c13b8db7" +
    "3685ad327ea32e76992b6f8f28688625607c22597320526a1d4b611a4458183d4f163747" +
    "14303e122a3611242e101f26111a20111519121214150e0e190c091d0b05220c02260d01" +
    "2b0e01300f003510003b11004112014714014d16025419055b1d0863210c6c2711752d17" +
    "7d341e863b268e422e964a369e513fa65948ae6051b6685abe6f63c6776cce7f76d7877f" +
    "df8f89e79792f09f9cf9a7a6ffadad";
  const stops = [];
  for (let i = 0; i < hex.length; i += 6)
    stops.push([0, 2, 4].map((k) => parseInt(hex.slice(i + k, i + k + 2), 16)));
  return (x) => interpolate(stops, x);
}

function interpolate(stops, x) {
  const t = Math.min(1, Math.max(0, x)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), w = t - i;
  return stops[i].map((v, k) => Math.round(v + w * (stops[i + 1][k] - v)));
}
