export type Center = readonly [number, number];
export type Packing = readonly [readonly Center[], readonly number[], number];
export interface Validation { correct: boolean; message: string }

/** Port of the original sequential pairwise radius reduction. Ordering matters. */
export function computeMaxRadii(centers: readonly Center[]): number[] {
  const radii = centers.map(([x, y]) => Math.min(x, y, 1 - x, 1 - y));
  for (let i = 0; i < centers.length; i++) for (let j = i + 1; j < centers.length; j++) {
    const a = centers[i]!, b = centers[j]!;
    const distance = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (radii[i]! + radii[j]! > distance) {
      const scale = distance / (radii[i]! + radii[j]!);
      radii[i] = radii[i]! * scale;
      radii[j] = radii[j]! * scale;
    }
  }
  return radii;
}

export function solution(): Packing {
  // The original leaves circle 25 at (0,0), subsequently clipped to (.01,.01).
  const centers: [number, number][] = Array.from({ length: 26 }, () => [0, 0]);
  centers[0] = [0.5, 0.5];
  for (let i = 0; i < 8; i++) {
    const angle = 2 * Math.PI * i / 8;
    centers[i + 1] = [0.5 + 0.3 * Math.cos(angle), 0.5 + 0.3 * Math.sin(angle)];
  }
  for (let i = 0; i < 16; i++) {
    const angle = 2 * Math.PI * i / 16;
    centers[i + 9] = [0.5 + 0.7 * Math.cos(angle), 0.5 + 0.7 * Math.sin(angle)];
  }
  for (const center of centers) {
    center[0] = Math.max(0.01, Math.min(0.99, center[0]));
    center[1] = Math.max(0.01, Math.min(0.99, center[1]));
  }
  const radii = computeMaxRadii(centers);
  return [centers, radii, radii.reduce((a, b) => a + b, 0)];
}

/** Reject non-finite values explicitly; NaN must never bypass constraints. */
export function validatePacking(output: unknown, atol = 1e-6): Validation {
  if (!Number.isFinite(atol) || atol < 0) throw new RangeError("Invalid tolerance");
  const fail = (message: string): Validation => ({ correct: false, message });
  if (!Array.isArray(output) || output.length !== 3) return fail("Expected [centers, radii, reported_sum]");
  const [centers, radii, reported] = output as unknown[];
  if (!Array.isArray(centers) || centers.length !== 26 || centers.some(c => !Array.isArray(c) || c.length !== 2)) return fail("Expected centers shape (26, 2)");
  if (!Array.isArray(radii) || radii.length !== 26) return fail("Expected radii shape (26,)");
  if (typeof reported !== "number" || !Number.isFinite(reported) ||
      centers.some(c => c.some((x: unknown) => typeof x !== "number" || !Number.isFinite(x))) ||
      radii.some(r => typeof r !== "number" || !Number.isFinite(r))) return fail("Packing contains non-finite or non-numeric values");
  if (radii.some(r => r < 0)) return fail("Negative radius");
  const sum = radii.reduce((a: number, b: number) => a + b, 0);
  if (Math.abs(sum - reported) > atol + 1e-5 * Math.abs(reported)) return fail("Sum of radii does not match reported sum");
  for (let i = 0; i < 26; i++) {
    const [x, y] = centers[i] as number[], r = radii[i] as number;
    if (x! - r < -atol || x! + r > 1 + atol || y! - r < -atol || y! + r > 1 + atol) return fail(`Circle ${i} is outside unit square`);
    for (let j = i + 1; j < 26; j++) {
      if (Math.hypot(x! - centers[j][0], y! - centers[j][1]) < r + radii[j] - atol) return fail(`Circles ${i} & ${j} overlap`);
    }
  }
  return { correct: true, message: "The circles are placed correctly. There are no overlaps or any circles outside the unit square." };
}

export function packingMetrics(packing: Packing) {
  const [centers, , reported] = packing;
  return {
    combined_score: reported,
    public: {
      centers_str: centers.map(([x, y], i) => `  centers[${i}] = (${x.toFixed(4)}, ${y.toFixed(4)})`).join("\n"),
      num_circles: centers.length,
    },
    private: { reported_sum_of_radii: reported },
  };
}
