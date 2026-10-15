import assert from "node:assert/strict";
import { Tape, sgd } from "../compute/autograd.js";
import type { ComputeBackend } from "../compute/types.js";

export function closeArray(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-5): void {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i]! - expected[i]!) <= tolerance * Math.max(1, Math.abs(expected[i]!)), `index ${i}: ${actual[i]} vs ${expected[i]}`);
}

export async function checkCompute(backend: ComputeBackend): Promise<void> {
  const a = backend.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
  const b = backend.tensor([7, 8, 9, 10, 11, 12], [3, 2]);
  closeArray(await backend.read(backend.matmul(a, b)), [58, 64, 139, 154]);
  closeArray(await backend.read(backend.transpose(a)), [1, 4, 2, 5, 3, 6]);
  closeArray(await backend.read(backend.sum(a)), [21]);
  closeArray(await backend.read(backend.binary("mul", a, a)), [1, 4, 9, 16, 25, 36]);
  closeArray(await backend.read(backend.binary("sub", a, a)), [0, 0, 0, 0, 0, 0]);
  closeArray(await backend.read(backend.binary("add", a, a)), [2, 4, 6, 8, 10, 12]);
  closeArray(await backend.read(backend.binary("div", a, a)), [1, 1, 1, 1, 1, 1]);
  closeArray(await backend.read(backend.unary("sqrt", backend.tensor([1, 4, 9], [3]))), [1, 2, 3]);
  const probabilities = backend.softmax(backend.tensor([1000, 1001, 1002, -1000, -1000, -1000], [2, 3]));
  closeArray(await backend.read(probabilities), [0.09003057, 0.24472847, 0.66524096, 1 / 3, 1 / 3, 1 / 3]);
  closeArray(await backend.read(backend.broadcastScalar(backend.tensor([3], []), [2, 2])), [3, 3, 3, 3]);
  // Exercise a partial final workgroup; guards must prevent out-of-bounds writes.
  const uneven = backend.tensor(new Float32Array(67).fill(2), [67]);
  closeArray(await backend.read(backend.unary("neg", uneven)), new Float32Array(67).fill(-2));
}

export async function checkAutograd(backend: ComputeBackend): Promise<void> {
  const tape = new Tape(backend);
  try {
    const a = tape.constant([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tape.constant([7, 8, 9, 10, 11, 12], [3, 2]);
    const product = a.matmul(b);
    const loss = product.mul(product).sum();
    tape.backward(loss);
    // Independent analytic derivative: 2*(A B)*B^T and 2*A^T*(A B).
    closeArray(await backend.read(a.gradient!), [1836, 2324, 2812, 4410, 5582, 6754]);
    closeArray(await backend.read(b.gradient!), [1228, 1360, 1622, 1796, 2016, 2232]);
  } finally { tape.dispose(); }
}

export async function checkTraining(backend: ComputeBackend): Promise<void> {
  let weights = backend.tensor([0], [1, 1]);
  try {
    for (let step = 0; step < 40; step++) {
      const tape = new Tape(backend);
      try {
        const parameter = tape.variable(weights);
        const x = tape.constant([1, 2, 3], [3, 1]);
        const y = tape.constant([2, 4, 6], [3, 1]);
        const error = x.matmul(parameter).sub(y);
        tape.backward(error.mul(error).sum());
        const next = sgd(parameter, 0.02);
        weights.dispose(); weights = next;
      } finally { tape.dispose(); }
    }
    closeArray(await backend.read(weights), [2], 1e-4);
  } finally { weights.dispose(); }
}
