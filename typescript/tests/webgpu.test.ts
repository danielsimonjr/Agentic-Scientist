import { it } from "node:test";
import { createNodeBackend } from "../compute/node.js";
import { checkCompute, checkAutograd, checkTraining } from "./compute-helpers.js";

// Opt-in GPU tests fail on missing drivers. Ordinary CPU tests don't pretend
// to exercise shaders and do not silently substitute CPU for requested WebGPU.
it("executes WGSL compute, gradients, and training on WebGPU", { skip: process.env.AGSCI_TEST_WEBGPU !== "1" }, async () => {
  const backend = await createNodeBackend();
  backend.device.pushErrorScope("validation");
  try {
    await checkCompute(backend);
    await checkAutograd(backend);
    await checkTraining(backend);
    const error = await backend.device.popErrorScope();
    if (error) throw new Error(error.message);
  } finally { backend.close(); }
});
