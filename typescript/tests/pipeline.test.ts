import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { derivePaths, generateRunId } from "../pipeline/paths.js";
import { resolveCycleAgent, splitCycleBudget } from "../investigation/agent-selection.js";
import { validatedBillingMetadata } from "../investigation/billing-metadata.js";

describe("pipeline behavior ports", () => {
  it("preserves run-envelope paths and independent overrides", () => {
    assert.equal(generateRunId(new Date("2026-10-15T13:00:00-05:00")), "20261015_180000");
    const p = derivePaths({ task: { dir: "examples/circpack" }, agent: { name: "claude" }, run_id: "test", paths: { results_root: "results" } });
    const root = join("results", "circpack", "claude", "test");
    assert.equal(p.run_dir, root);
    assert.equal(p.investigation_dir, join(root, "investigation"));
    assert.equal(p.final_dir, join(root, "final_evaluation"));
    assert.equal(p.ablations_aggregated_task_dir, join(root, "ablations", "aggregated", "circpack"));
    const overridden = derivePaths({ task: { dir: "examples/circpack" }, agent: { name: "claude" }, paths: { results_root: "ignored", run_dir: "saved", paper_dir: "manuscript" } });
    assert.equal(overridden.paper_dir, "manuscript");
    assert.equal(overridden.task_dir, join("saved", "task"));
  });
  it("persists random cycle selection and applies per-agent model overrides", () => {
    const state: Record<string, unknown> = {}, config = { name: "claude", pool: ["claude", "codex"], model: "global", models: { codex: " specific " } };
    assert.deepEqual(resolveCycleAgent(config, state, 1, () => 0.9), { agent: "codex", model: "specific" });
    assert.deepEqual(resolveCycleAgent(config, state, 1, () => { throw new Error("must not resample"); }), { agent: "codex", model: "specific" });
    assert.deepEqual(splitCycleBudget(7, 3), [3, 2, 2]);
    assert.deepEqual(splitCycleBudget(2, 4), [1, 1, 0, 0]);
    assert.throws(() => splitCycleBudget(1, 0));
  });
  it("bounds billing provenance and strips arbitrary fields", () => {
    const metadata = { attempts: [{ route: "native", secret: "must not copy", usage: { usageStatus: "reported", pricingStatus: "native", costBasis: "native-reported", pricingSource: "native", outputTokens: 42, cost: { total: 0.01, arbitrary: "drop" } } }] };
    const clean = validatedBillingMetadata(metadata);
    assert.equal(clean.attempts[0]!.secret, undefined);
    const usage = clean.attempts[0]!.usage as Record<string, unknown>;
    assert.deepEqual(usage.cost, { total: 0.01 });
    assert.equal(usage.outputTokens, 42);
    for (const invalid of [-1, NaN, Infinity, true, 1.5, 1e16]) {
      assert.throws(() => validatedBillingMetadata({ attempts: [{ route: "native", usage: { ...metadata.attempts[0]!.usage, outputTokens: invalid } }] }));
    }
    assert.throws(() => validatedBillingMetadata({ attempts: [] }));
    assert.throws(() => validatedBillingMetadata({ attempts: [{ route: "native", usage: {} }] }));
  });
});
