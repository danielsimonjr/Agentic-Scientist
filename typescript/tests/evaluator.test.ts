import { it } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { evaluateCirclePacking, runSolution } from "../science/evaluate-circle-packing.js";
import { solution } from "../science/circle-packing.js";
import { unzipSync } from "fflate";

it("evaluates a JS candidate without Python and preserves artifact keys", async () => {
  const directory = await mkdtemp(join(tmpdir(), "agsci-evaluate-"));
  try {
    const program = join(directory, "program.mjs"), results = join(directory, "results");
    await writeFile(program, `export function solution() { console.log('evaluating'); return ${JSON.stringify(solution())}; }`);
    const output = await evaluateCirclePacking(program, results);
    assert.equal(output.validation.correct, true);
    const metrics = JSON.parse(await readFile(join(results, "metrics.json"), "utf8")) as { combined_score: number };
    assert.ok(metrics.combined_score > 0);
    assert.match(await readFile(join(results, "stdout.log"), "utf8"), /evaluating/);
    const arrays = unzipSync(await readFile(join(results, "extra.npz")));
    assert.deepEqual(Object.keys(arrays).sort(), ["centers.npy", "radii.npy", "reported_sum.npy"]);
    const score = arrays["reported_sum.npy"]!;
    const view = new DataView(score.buffer, score.byteOffset, score.byteLength);
    assert.equal(view.getFloat64(10 + view.getUint16(8, true), true), metrics.combined_score);
    await writeFile(program, "export function solution() { return null; }");
    assert.equal((await evaluateCirclePacking(program, results)).validation.correct, false);
    await assert.rejects(() => readFile(join(results, "extra.npz")));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

it("terminates infinite loops, excessive logs and candidates with dangling timers", async () => {
  const directory = await mkdtemp(join(tmpdir(), "agsci-timeout-"));
  try {
    const program = join(directory, "program.mjs");
    await writeFile(program, "export function solution() { while (true) {} }");
    assert.match((await runSolution(program, { timeoutMs: 200 })).error!, /Timed out/);
    await writeFile(program, "export function solution() { console.log('x'.repeat(10000)); return []; }");
    assert.match((await runSolution(program, { maxLogBytes: 100 })).error!, /log bytes/);
    await writeFile(program, "export function solution() { setInterval(() => {}, 1000); return [42]; }");
    assert.deepEqual((await runSolution(program, { timeoutMs: 2000 })).result, [42]);
    await writeFile(program, "export function solution() { throw new Error('broken'); }");
    assert.match((await runSolution(program)).error!, /broken/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
