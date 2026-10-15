/// <reference types="@webgpu/types" preserve="true" />
import { elementCount, equalShape, matrixShape } from "./types.js";
import type { BinaryOperation, ComputeBackend, Tensor, UnaryOperation } from "./types.js";

// WebGPU constants are defined by the specification; avoid browser-only globals
// so the same module works with browser devices and Node's Dawn provider.
const USAGE = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 };
const READ = 1;

class WebGpuTensor implements Tensor {
  readonly shape: readonly number[];
  readonly size: number;
  disposed = false;
  constructor(readonly backend: WebGpuBackend, readonly buffer: GPUBuffer, shape: readonly number[]) {
    this.shape = Object.freeze([...shape]);
    this.size = elementCount(shape);
  }
  dispose(): void {
    if (!this.disposed) { this.buffer.destroy(); this.disposed = true; this.backend.release(this); }
  }
}

/** Shared browser/Node WGSL compute. The caller owns the supplied GPUDevice. */
export class WebGpuBackend implements ComputeBackend {
  readonly kind = "webgpu";
  private tensors = new Set<WebGpuTensor>();
  private pipelines = new Map<string, GPUComputePipeline>();
  private disposed = false;
  private lost: string | undefined;
  private validationError: string | undefined;
  private pendingChecks = new Set<Promise<void>>();

  constructor(readonly device: GPUDevice, private readonly ownsDevice = false) {
    void device.lost.then(info => { this.lost = info.message || info.reason; });
  }
  private check(): void {
    if (this.disposed) throw new Error("Backend has been disposed");
    if (this.lost !== undefined) throw new Error(`WebGPU device lost: ${this.lost}`);
    if (this.validationError !== undefined) throw new Error(`WebGPU validation failed: ${this.validationError}`);
  }
  release(tensor: WebGpuTensor): void { this.tensors.delete(tensor); }
  private allocate(shape: readonly number[]): WebGpuTensor {
    this.check();
    const bytes = elementCount(shape) * 4;
    if (bytes > this.device.limits.maxStorageBufferBindingSize || bytes > this.device.limits.maxBufferSize) {
      throw new RangeError(`Tensor requires ${bytes} bytes, exceeding this WebGPU device's buffer limits`);
    }
    const buffer = this.device.createBuffer({ size: bytes, usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST });
    const tensor = new WebGpuTensor(this, buffer, shape);
    this.tensors.add(tensor);
    return tensor;
  }
  tensor(data: ArrayLike<number>, shape: readonly number[]): Tensor {
    if (data.length !== elementCount(shape)) throw new RangeError("Data length does not match shape");
    const tensor = this.allocate(shape);
    this.device.queue.writeBuffer(tensor.buffer, 0, Float32Array.from(data));
    return tensor;
  }
  private buffer(tensor: Tensor): GPUBuffer {
    this.check();
    if (!(tensor instanceof WebGpuTensor) || tensor.backend !== this) throw new TypeError("Tensor belongs to another backend");
    if (tensor.disposed) throw new Error("Tensor has been disposed");
    return tensor.buffer;
  }
  async read(tensor: Tensor): Promise<Float32Array> {
    const buffer = this.buffer(tensor);
    const staging = this.device.createBuffer({ size: tensor.size * 4, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, staging, 0, tensor.size * 4);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(READ);
      await Promise.all(this.pendingChecks);
      this.check();
      return new Float32Array(staging.getMappedRange()).slice();
    } finally { staging.destroy(); }
  }
  private dispatch(inputs: Tensor[], shape: readonly number[], body: string, invocations = elementCount(shape)): Tensor {
    const buffers = inputs.map(t => this.buffer(t));
    const workgroups = Math.ceil(invocations / 64);
    if (workgroups > this.device.limits.maxComputeWorkgroupsPerDimension) throw new RangeError("Compute dispatch exceeds device limits");
    const declarations = inputs.map((_, i) => `@group(0) @binding(${i}) var<storage, read> x${i}: array<f32>;`).join("\n");
    const code = `${declarations}
@group(0) @binding(${inputs.length}) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= ${invocations}u) { return; }
  ${body}
}`;
    this.device.pushErrorScope("validation");
    try {
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" } });
      // Bound specialization cache for workloads whose tensor shapes vary.
      if (this.pipelines.size >= 128) this.pipelines.delete(this.pipelines.keys().next().value!);
      this.pipelines.set(code, pipeline);
    }
    const output = this.allocate(shape);
    try {
      const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...buffers, output.buffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(workgroups); pass.end();
      this.device.queue.submit([encoder.finish()]);
      return output;
    } catch (error) { output.dispose(); throw error; }
    } finally {
      const pending = this.device.popErrorScope().then(error => {
        if (error) this.validationError = error.message;
      }, error => { this.validationError = String(error); });
      this.pendingChecks.add(pending);
      void pending.then(() => this.pendingChecks.delete(pending));
    }
  }
  unary(operation: UnaryOperation, tensor: Tensor): Tensor {
    const expressions = { neg: "-x0[i]", exp: "exp(x0[i])", log: "log(x0[i])", sqrt: "sqrt(x0[i])" };
    return this.dispatch([tensor], tensor.shape, `y[i] = ${expressions[operation]};`);
  }
  binary(operation: BinaryOperation, left: Tensor, right: Tensor): Tensor {
    equalShape(left, right);
    const symbols = { add: "+", sub: "-", mul: "*", div: "/" };
    return this.dispatch([left, right], left.shape, `y[i] = x0[i] ${symbols[operation]} x1[i];`);
  }
  matmul(left: Tensor, right: Tensor): Tensor {
    const [m, k] = matrixShape(left), [rk, n] = matrixShape(right);
    if (k !== rk) throw new RangeError("Matrix inner dimensions do not match");
    return this.dispatch([left, right], [m, n], `
let row = i / ${n}u; let col = i % ${n}u;
var value = 0.0;
for (var j = 0u; j < ${k}u; j++) { value += x0[row * ${k}u + j] * x1[j * ${n}u + col]; }
y[i] = value;`);
  }
  transpose(tensor: Tensor): Tensor {
    const [m, n] = matrixShape(tensor);
    return this.dispatch([tensor], [n, m], `y[i] = x0[(i % ${m}u) * ${n}u + i / ${m}u];`);
  }
  softmax(tensor: Tensor): Tensor {
    const [rows, cols] = matrixShape(tensor);
    return this.dispatch([tensor], tensor.shape, `
let offset = i * ${cols}u;
var maximum = x0[offset];
for (var j = 1u; j < ${cols}u; j++) { maximum = max(maximum, x0[offset + j]); }
var total = 0.0;
for (var j = 0u; j < ${cols}u; j++) { total += exp(x0[offset + j] - maximum); }
for (var j = 0u; j < ${cols}u; j++) { y[offset + j] = exp(x0[offset + j] - maximum) / total; }`, rows);
  }
  sum(tensor: Tensor): Tensor {
    return this.dispatch([tensor], [], `var total = 0.0; for (var j = 0u; j < ${tensor.size}u; j++) { total += x0[j]; } y[0] = total;`);
  }
  broadcastScalar(tensor: Tensor, shape: readonly number[]): Tensor {
    if (tensor.size !== 1) throw new RangeError("Expected a scalar");
    return this.dispatch([tensor], shape, "y[i] = x0[0];");
  }
  dispose(): void {
    for (const tensor of this.tensors) tensor.dispose();
    this.pipelines.clear();
    this.disposed = true;
    if (this.ownsDevice) this.device.destroy();
  }
}

export async function createBrowserBackend(): Promise<WebGpuBackend> {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu) throw new Error("WebGPU is unavailable; use a secure context and a WebGPU-capable browser");
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter available");
  return new WebGpuBackend(await adapter.requestDevice(), true);
}
