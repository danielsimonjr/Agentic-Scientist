import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { CpuBackend } from "../compute/cpu.js";
import { Tape } from "../compute/autograd.js";

import { closeArray, checkCompute, checkAutograd, checkTraining } from "./compute-helpers.js";

describe("CPU reference compute", () => {
  it("computes dense operations and stable softmax", async () => {
    const backend = new CpuBackend();
    try { await checkCompute(backend); } finally { backend.dispose(); }
  });
  it("differentiates a shared matrix product", async () => {
    const backend = new CpuBackend();
    try { await checkAutograd(backend); } finally { backend.dispose(); }
  });
  it("learns a linear model with reverse-mode gradients", async () => {
    const backend = new CpuBackend();
    try { await checkTraining(backend); } finally { backend.dispose(); }
  });
  it("agrees with central finite differences for exp/log and reused variables", async () => {
    const backend = new CpuBackend(), tape = new Tape(backend);
    try {
      const values = [0.3, 0.7, 1.2];
      const x = tape.constant(values, [3]);
      tape.backward(tape.log(tape.exp(x).add(x.mul(x))).sum());
      const f = (v: number) => Math.log(Math.exp(v) + v * v), h = 1e-5;
      closeArray(await backend.read(x.gradient!), values.map(v => (f(v + h) - f(v - h)) / (2 * h)), 1e-5);
    } finally { tape.dispose(); backend.dispose(); }
  });
  it("rejects shape mismatch, foreign tensors, and use after disposal", async () => {
    const a = new CpuBackend(), b = new CpuBackend();
    try {
      assert.throws(() => a.tensor([1], [0]));
      assert.throws(() => a.tensor([1], [2]));
      const x = a.tensor([1, 2], [2]), y = a.tensor([1, 2], [1, 2]);
      assert.throws(() => a.binary("add", x, y));
      assert.throws(() => a.matmul(x, y));
      await assert.rejects(() => b.read(x));
      x.dispose(); await assert.rejects(() => a.read(x));
      a.dispose(); assert.throws(() => a.tensor([1], []));
    } finally { a.dispose(); b.dispose(); }
  });
  it("rejects cross-tape graphs and repeated backward", () => {
    const backend = new CpuBackend(), a = new Tape(backend), b = new Tape(backend);
    try {
      const x = a.constant([2], []), y = b.constant([3], []);
      assert.throws(() => x.add(y));
      a.backward(x.mul(x));
      assert.throws(() => a.backward(x));
    } finally { a.dispose(); b.dispose(); backend.dispose(); }
  });
});
