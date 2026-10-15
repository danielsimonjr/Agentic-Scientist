import { WebGpuBackend } from "./webgpu.js";

/** Native WebGPU implementation only; no Python, CUDA API, or Bun runtime API. */
export async function createNodeBackend(options: string[] = []): Promise<WebGpuBackend & { close(): void }> {
  const { create } = await import("webgpu");
  let provider: GPU | undefined = create(options);
  const adapter = await provider.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { provider = undefined; throw new Error("No WebGPU adapter available on this host"); }
  const device = await adapter.requestDevice();
  const backend = new WebGpuBackend(device);
  return Object.assign(backend, {
    close() {
      backend.dispose();
      device.destroy();
      // Retain the Dawn provider until GPU work has ended, then release it so
      // the Node process is free to exit. Never attach it to globalThis.
      provider = undefined;
    },
  });
}
