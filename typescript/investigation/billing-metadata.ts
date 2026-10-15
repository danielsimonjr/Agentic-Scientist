/** Diagnostic provenance only; billing metadata must never authorize retries. */
const ROUTES = new Set(["subscription", "openai-api", "bedrock", "native"]);
const REASONS = new Set(["subscription-quota", "subscription-model-unsupported"]);
const COUNTS = ["inputTokens", "cacheReadTokens", "cacheWriteTokens", "outputTokens", "reasoningOutputTokens", "totalTokens"];
const ENUMS: Record<string, readonly (string | null)[]> = {
  usageStatus: ["reported", "missing"], costBasis: ["native-reported", "api-list-price-estimate", null],
  pricingSource: ["native", "models.dev", null], pricingStatus: ["native", "priced", "missing"],
};
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError("Invalid billing object");
  return value as Record<string, unknown>;
}
function number(value: unknown, maximum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > maximum) throw new RangeError("Invalid billing usage number");
  return value;
}
function usage(raw: unknown): Record<string, unknown> {
  const input = record(raw), clean: Record<string, unknown> = {};
  for (const field of ["agent", "model", "provider"]) {
    const value = input[field];
    if (value == null) continue;
    if (typeof value !== "string" || !/^[A-Za-z0-9._:/-]{1,240}$/.test(value) || value.includes("\n")) throw new TypeError("Invalid billing usage label");
    clean[field] = value;
  }
  for (const [field, choices] of Object.entries(ENUMS)) {
    const value = input[field] ?? null;
    if ((typeof value !== "string" && value !== null) || !choices.includes(value)) throw new TypeError("Invalid billing usage provenance");
    clean[field] = value;
  }
  for (const field of COUNTS) if (input[field] != null) {
    const count = number(input[field], 1e15);
    if (!Number.isInteger(count)) throw new RangeError("Invalid billing token count");
    clean[field] = count;
  }
  if (input.cost == null) clean.cost = null;
  else {
    const inputCost = record(input.cost), cost: Record<string, number | null> = {};
    for (const key of ["input", "cacheRead", "cacheWrite", "output", "total"]) if (Object.hasOwn(inputCost, key)) cost[key] = inputCost[key] === null ? null : number(inputCost[key], 1e9);
    clean.cost = cost;
  }
  return clean;
}

export function validatedBillingMetadata(raw: unknown): { attempts: Record<string, unknown>[] } {
  const attempts = record(raw).attempts;
  if (!Array.isArray(attempts) || attempts.length < 1 || attempts.length > 5) throw new RangeError("Invalid billing attempt count");
  return { attempts: attempts.map(rawAttempt => {
    const attempt = record(rawAttempt);
    if (typeof attempt.route !== "string" || !ROUTES.has(attempt.route)) throw new TypeError("Invalid billing route");
    const clean: Record<string, unknown> = { route: attempt.route, usage: usage(attempt.usage) };
    if (attempt.reason != null) {
      if (typeof attempt.reason !== "string" || !REASONS.has(attempt.reason)) throw new TypeError("Invalid billing reason");
      clean.reason = attempt.reason;
    }
    if (attempt.retryReason != null) {
      if (attempt.retryReason !== "model-capacity") throw new TypeError("Invalid billing retry reason");
      clean.retryReason = attempt.retryReason;
    }
    return clean;
  }) };
}
