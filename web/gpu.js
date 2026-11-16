// The demo's GPU path: fd_kernel of scalar.cu under USE_DOMAIN and USE_DAMPING
// (stiffness 1), excitation_kernel and homogeneous_neumann_kernel ported to
// WGSL, stepping the cells, damping and source plan solver.js builds.

const STEP_WGSL = /* wgsl */ `
struct Grid { n0: i32, n1: i32, f: f32, dt: f32, w: vec4f }
override R: i32 = 1;
const DEEP: i32 = 1 << 30;
const HELD: i32 = 7;
@group(0) @binding(0) var<storage, read_write> u0: array<f32>; // u2 in place
@group(0) @binding(1) var<storage, read> u1: array<f32>;
@group(0) @binding(2) var<storage, read> cells: array<i32>;
@group(0) @binding(3) var<storage, read> damping: array<f32>;
@group(0) @binding(4) var<uniform> g: Grid;

fn W(r: i32, k: i32) -> f32 { return g.w[(r - 1) * 2 + k - 1]; }

// u1 at node a + j of the line through idx, past a wall off its mirror
fn tap(idx: i32, s: i32, a: i32, n: i32, j: i32) -> f32 {
  let b = a + j;
  let m = select(select(b, 2 * (n - 2) - b, b > n - 1), 2 - b, b < 0);
  return u1[idx + (m - a) * s];
}

// flux_divergence_axis at the full radius, both harmonic means 1/2
fn axis(idx: i32, s: i32, a: i32, n: i32, uc: f32) -> f32 {
  var dp = tap(idx, s, a, n, 1) - uc;
  var dm = uc - tap(idx, s, a, n, -1);
  if (R > 1) {
    dp = W(R, 1) * dp;
    dm = W(R, 1) * dm;
  }
  for (var k = 2; k <= R; k++) {
    dp = fma(W(R, k), tap(idx, s, a, n, k) - tap(idx, s, a, n, 1 - k), dp);
    dm = fma(W(R, k), tap(idx, s, a, n, k - 1) - tap(idx, s, a, n, -k), dm);
  }
  return fma(dp, 0.5, -(dm * 0.5));
}

// flux_divergence_cells: each cell at its own radius, a held one onto zero
fn cells_axis(idx: i32, s: i32, uc: f32, rp: i32, rm: i32) -> f32 {
  var dp = 0.0;
  var dm = 0.0;
  if (rp == HELD) {
    dp = -uc;
  } else if (rp > 0) {
    dp = W(rp, 1) * (u1[idx + s] - uc);
    for (var k = 2; k <= R; k++) {
      if (k <= rp) { dp += W(rp, k) * (u1[idx + k * s] - u1[idx - (k - 1) * s]); }
    }
    dp *= 0.5;
  }
  if (rm == HELD) {
    dm = uc;
  } else if (rm > 0) {
    dm = W(rm, 1) * (uc - u1[idx - s]);
    for (var k = 2; k <= R; k++) {
      if (k <= rm) { dm += W(rm, k) * (u1[idx + (k - 1) * s] - u1[idx - k * s]); }
    }
    dm *= 0.5;
  }
  return g.f * (dp - dm);
}

@compute @workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) id: vec3u) {
  let a0 = i32(id.y);
  let a1 = i32(id.x);
  if (a0 < 1 || a0 > g.n0 - 2 || a1 < 1 || a1 > g.n1 - 2) { return; }
  let idx = a0 * g.n1 + a1;
  let code = cells[idx];
  if (code == 0) { // off the domain the field is zero
    u0[idx] = 0.0;
    return;
  }
  let uc = u1[idx];
  var laplacian: f32;
  if (code == DEEP) {
    let d0 = axis(idx, g.n1, a0, g.n0, uc);
    let d1 = axis(idx, 1, a1, g.n1, uc);
    laplacian = select(fma(g.f, d1, g.f * d0), fma(g.f, d0, g.f * d1), R == 1);
  } else {
    laplacian = cells_axis(idx, g.n1, uc, code & 7, (code >> 3) & 7) +
                cells_axis(idx, 1, uc, (code >> 6) & 7, (code >> 9) & 7);
  }
  let beta = 0.5 * damping[idx] * g.dt;
  u0[idx] = (2.0 * uc - u0[idx] * (1.0 - beta) + laplacian) / (1.0 + beta);
}
`;

const EXCITE_WGSL = /* wgsl */ `
struct Source { node: i32, value: f32 }
@group(0) @binding(0) var<storage, read_write> u: array<f32>;
@group(0) @binding(1) var<storage, read> plan: array<Source>;
@group(0) @binding(2) var<storage, read_write> counter: array<u32>;

// the sources of the step the counter names, in the plan's order
@compute @workgroup_size(1)
fn excite() {
  let k = counter[0];
  for (var i = 0u; i < 16u; i++) {
    let source = plan[k * 16u + i];
    if (source.node >= 0) { u[source.node] += source.value; }
  }
  counter[0] = k + 1u;
}
`;

const REFILL_WGSL = /* wgsl */ `
struct Grid { n0: i32, n1: i32, f: f32, dt: f32, w: vec4f }
@group(0) @binding(0) var<storage, read_write> u: array<f32>;
@group(0) @binding(1) var<uniform> g: Grid;

// each face ghost takes its mirror two nodes in, corners left alone
@compute @workgroup_size(64)
fn refill(@builtin(global_invocation_id) id: vec3u) {
  var t = i32(id.x);
  let l1 = g.n1 - 2;
  let l0 = g.n0 - 2;
  var ghost: i32;
  var normal: i32;
  if (t < 2 * l1) {
    let high = t >= l1;
    ghost = select(0, (g.n0 - 1) * g.n1, high) + 1 + t % l1;
    normal = select(g.n1, -g.n1, high);
  } else {
    t -= 2 * l1;
    if (t >= 2 * l0) { return; }
    let high = t >= l0;
    ghost = (1 + t % l0) * g.n1 + select(0, g.n1 - 1, high);
    normal = select(1, -1, high);
  }
  u[ghost] = u[ghost + 2 * normal];
}
`;

const RENDER_WGSL = /* wgsl */ `
struct View { cols: i32, rows: i32, origin: i32, stride: i32, cell: f32, limit: f32, wall: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read> cells: array<i32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<uniform> v: View;

@vertex
fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn at(c: vec2i) -> i32 {
  let k = clamp(c, vec2i(0), vec2i(v.cols - 1, v.rows - 1));
  return v.origin + k.y * v.stride + k.x;
}

// the field between the four nearest nodes, a wall where the nearest is one
@fragment
fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let x = position.xy / v.cell - 0.5;
  if (cells[at(vec2i(round(x)))] == 0) { return unpack4x8unorm(v.wall); }
  let c = vec2i(floor(x));
  let w = x - floor(x);
  let top = mix(u[at(c)], u[at(c + vec2i(1, 0))], w.x);
  let bottom = mix(u[at(c + vec2i(0, 1))], u[at(c + vec2i(1, 1))], w.x);
  let i = clamp(127.5 + 127.5 * mix(top, bottom, w.y) / v.limit, 0.0, 255.0);
  return unpack4x8unorm(lut[u32(i)]);
}
`;

const MAX_STEPS = 64; // steps a submission, the plan's rows
const SLOTS = 16; // sources a step, solver.cpp's SOURCES

// the GPU backend over `solver`'s grid on `canvas` at Courant number `courant`,
// or null without WebGPU
async function gpuBackend(solver, canvas, courant) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const R = solver._radius();
  const compute = (code, entryPoint, constants) =>
    device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint, constants },
    });
  const step = compute(STEP_WGSL, "step", { R });
  const excite = compute(EXCITE_WGSL, "excite");
  const refill = compute(REFILL_WGSL, "refill");
  const format = navigator.gpu.getPreferredCanvasFormat();
  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const render = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "vertex" },
    fragment: { module: renderModule, entryPoint: "fragment", targets: [{ format }] },
  });
  const error = await device.popErrorScope();
  if (error) throw new Error(error.message);
  const context = canvas.getContext("webgpu");
  context.configure({ device, format, alphaMode: "opaque" });

  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const buffer = (size, usage = STORAGE) => device.createBuffer({ size, usage });
  const plan = buffer(MAX_STEPS * SLOTS * 8);
  const counter = buffer(4);
  const lut = buffer(256 * 4);
  const grid = buffer(32, UNIFORM);
  const view = buffer(32, UNIFORM);
  let n0, n1, U, cells, damping, groups, ghosts, s = 0;

  const bind = (pipeline, buffers) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
    });

  return {
    name: "webgpu",
    // node updates a step: every node of the grid is dispatched
    cost: (rows, cols, margin) => (rows + 2 * margin) * (cols + 2 * margin),
    smallest: 1 / (devicePixelRatio || 1),

    // buffers for the solver's current grid, the field zeroed
    resize() {
      for (const b of [U, cells, damping].flat()) b?.destroy();
      n0 = solver._extent();
      n1 = solver._stride();
      U = [buffer(n0 * n1 * 4), buffer(n0 * n1 * 4)];
      cells = buffer(n0 * n1 * 4);
      damping = buffer(n0 * n1 * 4);
      ghosts = 2 * (n0 - 2) + 2 * (n1 - 2);
      const f = 2 * courant * courant;
      const w = [1, 2].flatMap((r) => [1, 2].map((k) => (k <= r && r <= R ? solver._coefficient(r, k) : 0)));
      const header = new DataView(new ArrayBuffer(32));
      header.setInt32(0, n0, true);
      header.setInt32(4, n1, true);
      header.setFloat32(8, f, true);
      header.setFloat32(12, courant, true);
      w.forEach((x, i) => header.setFloat32(16 + 4 * i, x, true));
      device.queue.writeBuffer(grid, 0, header.buffer);
      groups = [0, 1].map((p) => ({
        step: bind(step, [U[p], U[1 - p], cells, damping, grid]),
        excite: bind(excite, [U[p], plan, counter]),
        refill: bind(refill, [U[p], grid]),
        render: bind(render, [U[p], cells, lut, view]),
      }));
      s = 0;
      this.upload(0, n0, true);
    },

    // grid rows `first` to `last` of the cells, and the damping on a rebuild
    upload(first, last, rebuilt) {
      const a = Math.max(0, first) * n1 * 4, b = Math.min(n0, last) * n1 * 4;
      device.queue.writeBuffer(cells, a, solver.HEAPU8, solver._codes() + a, b - a);
      if (rebuilt) device.queue.writeBuffer(damping, 0, solver.HEAPU8, solver._dampings(), n0 * n1 * 4);
    },

    // the visible `rows` by `cols` nodes at `pixels` device px each, colored by
    // `colors` (256 packed RGBA) up to `limit`, walls in `wall`
    view(rows, cols, pixels, limit, colors, wall) {
      const data = new DataView(new ArrayBuffer(32));
      data.setInt32(0, cols, true);
      data.setInt32(4, rows, true);
      data.setInt32(8, solver._origin(), true);
      data.setInt32(12, n1, true);
      data.setFloat32(16, pixels, true);
      data.setFloat32(20, limit, true);
      data.setUint32(24, wall >>> 0, true);
      device.queue.writeBuffer(view, 0, data.buffer);
      device.queue.writeBuffer(lut, 0, colors);
    },

    advance(steps) {
      while (steps > 0) {
        const n = Math.min(steps, MAX_STEPS);
        const schedule = solver._plan(n);
        device.queue.writeBuffer(plan, 0, solver.HEAPU8, schedule, n * SLOTS * 8);
        device.queue.writeBuffer(counter, 0, new Uint32Array([0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        for (let k = 0; k < n; ++k, ++s) {
          const group = groups[s % 2];
          pass.setPipeline(step);
          pass.setBindGroup(0, group.step);
          pass.dispatchWorkgroups(Math.ceil(n1 / 16), Math.ceil(n0 / 16));
          pass.setPipeline(excite);
          pass.setBindGroup(0, group.excite);
          pass.dispatchWorkgroups(1);
          pass.setPipeline(refill);
          pass.setBindGroup(0, group.refill);
          pass.dispatchWorkgroups(Math.ceil(ghosts / 64));
        }
        pass.end();
        device.queue.submit([encoder.finish()]);
        steps -= n;
      }
    },

    // the latest level, U[(s + 1) % 2], drawn over the canvas
    draw() {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: context.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store" },
        ],
      });
      pass.setPipeline(render);
      pass.setBindGroup(0, groups[(s + 1) % 2].render);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },

    clear() {
      const encoder = device.createCommandEncoder();
      for (const u of U) encoder.clearBuffer(u);
      device.queue.submit([encoder.finish()]);
    },

    finished: () => device.queue.onSubmittedWorkDone(),
  };
}
