/** Numerical components ported from examples/grpo/program_0/src/program.py.
 * These functions do not imply a port of its transformer or sampling runtime. */
export function groupNormalizedRewards(rewards: readonly number[], groupSize: number, epsilon = 1e-6, normalizeByStd = true): Float32Array {
  if (!Number.isInteger(groupSize) || groupSize < (normalizeByStd ? 2 : 1) || rewards.length === 0 || rewards.length % groupSize !== 0) {
    throw new RangeError("Rewards must contain complete groups; sample standard deviation requires at least two values");
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0 || rewards.some(r => !Number.isFinite(r))) throw new RangeError("Rewards and epsilon must be finite, epsilon positive");
  const result = new Float32Array(rewards.length);
  for (let start = 0; start < rewards.length; start += groupSize) {
    const group = rewards.slice(start, start + groupSize);
    const mean = group.reduce((a, b) => a + b, 0) / groupSize;
    // torch.std defaults to correction=1 (sample variance).
    const std = normalizeByStd ? Math.sqrt(group.reduce((s, r) => s + (r - mean) ** 2, 0) / (groupSize - 1)) + epsilon : 1;
    group.forEach((r, i) => { result[start + i] = (r - mean) / std; });
  }
  return result;
}

export function grpoClipLoss(advantages: ArrayLike<number>, policy: ArrayLike<number>, previous: ArrayLike<number>, sequenceLength: number, clipRange: number) {
  if (!Number.isInteger(sequenceLength) || sequenceLength < 1 || advantages.length === 0 || policy.length !== advantages.length * sequenceLength || policy.length !== previous.length) throw new RangeError("Incompatible GRPO shapes");
  if (!Number.isFinite(clipRange) || clipRange < 0 || clipRange >= 1) throw new RangeError("Clip range must be in [0, 1)");
  const loss = new Float32Array(policy.length), gradient = new Float32Array(policy.length), clipped = new Uint8Array(policy.length);
  let clippedCount = 0;
  for (let i = 0; i < policy.length; i++) {
    const advantage = advantages[Math.floor(i / sequenceLength)]!, logp = policy[i]!, old = previous[i]!;
    if (![advantage, logp, old].every(Number.isFinite)) throw new RangeError("Non-finite policy input");
    const ratio = Math.exp(logp - old);
    if (!Number.isFinite(ratio)) throw new RangeError("Policy ratio overflow");
    const raw = ratio * advantage, limited = Math.max(1 - clipRange, Math.min(1 + clipRange, ratio)) * advantage;
    loss[i] = -Math.min(raw, limited);
    clipped[i] = limited < raw ? 1 : 0;
    clippedCount += clipped[i]!;
    gradient[i] = clipped[i] ? 0 : -ratio * advantage;
  }
  return { loss, gradient, clipped, clipFraction: clippedCount / policy.length };
}
