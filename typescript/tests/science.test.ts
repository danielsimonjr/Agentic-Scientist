import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { solution, validatePacking, packingMetrics } from "../science/circle-packing.js";
import { groupNormalizedRewards, grpoClipLoss } from "../science/grpo.js";
import golden from "./fixtures/circle-packing-baseline.json" with { type: "json" };

describe("circle packing", () => {
  it("preserves the original 26-circle construction and score", () => {
    const result = solution();
    assert.equal(result[0].length, 26);
    assert.deepEqual(result[0][25], [0.01, 0.01]);
    assert.equal(validatePacking(result).correct, true);
    assert.equal(packingMetrics(result).combined_score, result[2]);
    assert.ok(Math.abs(result[2] - golden.reported_sum) < 1e-14);
    result[0].forEach((center, i) => center.forEach((value, j) => assert.ok(Math.abs(value - golden.centers[i]![j]!) < 1e-14)));
    result[1].forEach((radius, i) => assert.ok(Math.abs(radius - golden.radii[i]!) < 1e-14));
  });
  it("rejects NaN, infinity, overlaps, negative radii, wrong sums and malformed outputs", () => {
    assert.equal(validatePacking(null).correct, false);
    for (const bad of [NaN, Infinity, -Infinity, "0.5"]) {
      const result = structuredClone(solution());
      (result[0][0] as unknown as unknown[])[0] = bad;
      assert.equal(validatePacking(result).correct, false);
    }
    const [centers, radii, sum] = solution();
    assert.equal(validatePacking([centers, radii, sum + 1]).correct, false);
    assert.equal(validatePacking([centers, radii.map(() => 0.2), 5.2]).correct, false);
    assert.equal(validatePacking([centers, [-1, ...radii.slice(1)], sum]).correct, false);
  });
});

describe("GRPO numerical behavior", () => {
  it("uses sample standard deviation and centers each group", () => {
    const a = groupNormalizedRewards([0, 1, 2, 2], 2);
    assert.ok(Math.abs(a[0]! + 1 / Math.sqrt(2)) < 1e-5);
    assert.ok(Math.abs(a[1]! - 1 / Math.sqrt(2)) < 1e-5);
    assert.equal(a[2], 0); assert.equal(a[3], 0);
    assert.deepEqual([...groupNormalizedRewards([0, 2], 2, 1e-6, false)], [-1, 1]);
  });
  it("clips both positive and negative advantages and zeros clipped gradients", () => {
    const result = grpoClipLoss([1, -1], [Math.log(1.5), Math.log(0.5)], [0, 0], 1, 0.2);
    assert.ok(Math.abs(result.loss[0]! + 1.2) < 1e-6);
    assert.ok(Math.abs(result.loss[1]! - 0.8) < 1e-6);
    assert.deepEqual([...result.gradient], [0, 0]);
    assert.equal(result.clipFraction, 1);
  });
  it("matches finite differences for an unclipped log-probability gradient", () => {
    const h = 1e-3, x = 0.02;
    const f = (p: number) => grpoClipLoss([0.4], [p], [0], 1, 0.2).loss[0]!;
    const result = grpoClipLoss([0.4], [x], [0], 1, 0.2);
    assert.ok(Math.abs(result.gradient[0]! - (f(x + h) - f(x - h)) / (2 * h)) < 1e-4);
  });
  it("rejects incomplete groups, invalid normalization and overflow", () => {
    assert.throws(() => groupNormalizedRewards([1, 2, 3], 2));
    assert.throws(() => groupNormalizedRewards([1], 1));
    assert.throws(() => groupNormalizedRewards([1, NaN], 2));
    assert.throws(() => grpoClipLoss([1], [1000], [0], 1, 0.2));
  });
});
