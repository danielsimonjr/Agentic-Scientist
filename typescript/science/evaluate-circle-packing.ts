import { fork } from "node:child_process";
import { mkdir, writeFile, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { packingMetrics, validatePacking } from "./circle-packing.js";
import type { Packing } from "./circle-packing.js";
import { encodeNpz } from "./numpy-format.js";

export interface ExecutionResult { result?: unknown; error?: string; stdout: string; stderr: string }
export interface EvaluationOptions { timeoutMs?: number; maxLogBytes?: number }

/** Run a candidate in a killable subprocess. This is process isolation, not a
 * security sandbox; production execution still needs the pipeline container. */
export async function runSolution(program: string, options: EvaluationOptions = {}): Promise<ExecutionResult> {
  const timeoutMs = options.timeoutMs ?? 300_000, maxLogBytes = options.maxLogBytes ?? 4 * 1024 * 1024;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647) throw new RangeError("Invalid timeout");
  if (!Number.isSafeInteger(maxLogBytes) || maxLogBytes < 1) throw new RangeError("Invalid log limit");
  const path = resolve(program);
  const extension = import.meta.url.endsWith(".ts") ? "ts" : "js";
  const worker = fileURLToPath(new URL(`./candidate-worker.${extension}`, import.meta.url));
  return new Promise(resolveResult => {
    const child = fork(worker, [path], { cwd: dirname(path), detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe", "ipc"], execArgv: [] });
    let stdout = "", stderr = "", bytes = 0, settled = false;
    let result: unknown, received = false, failure: string | undefined;
    const stop = () => {
      try {
        if (process.platform !== "win32" && child.pid) process.kill(-child.pid, "SIGKILL");
        else child.kill("SIGKILL");
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ESRCH") child.kill("SIGKILL");
      }
    };
    const finish = () => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      const execution: ExecutionResult = { stdout, stderr };
      if (failure !== undefined) execution.error = failure;
      else if (received) execution.result = result;
      else execution.error = "Imported program finished without returning any result";
      resolveResult(execution);
    };
    const timer = setTimeout(() => { failure = `Timed out after ${timeoutMs} ms while running imported program`; stop(); }, timeoutMs);
    const append = (chunk: Buffer, stream: "stdout" | "stderr") => {
      const available = Math.max(0, maxLogBytes - bytes);
      const text = chunk.subarray(0, available).toString("utf8");
      if (stream === "stdout") stdout += text; else stderr += text;
      bytes += chunk.byteLength;
      if (bytes > maxLogBytes && failure === undefined) { failure = `Candidate exceeded ${maxLogBytes} log bytes`; stop(); }
    };
    child.stdout!.on("data", (chunk: Buffer) => append(chunk, "stdout"));
    child.stderr!.on("data", (chunk: Buffer) => append(chunk, "stderr"));
    child.on("message", (message: unknown) => {
      if (received || failure !== undefined) return;
      if (!message || typeof message !== "object" || !("ok" in message)) { failure = "Malformed candidate response"; stop(); return; }
      if (message.ok === true && "result" in message) { result = message.result; received = true; }
      else failure = "error" in message ? String(message.error) : "Candidate did not return a result";
      // Terminate timers and subprocesses left alive after the solution returns.
      stop();
    });
    child.on("error", error => { failure = error.message; stop(); finish(); });
    child.on("close", (code, signal) => {
      if (!received && failure === undefined) failure = `Imported program exited without a result (code=${code}, signal=${signal})`;
      // Also reap descendants if the direct worker exited before replying.
      stop(); finish();
    });
  });
}

export async function evaluateCirclePacking(program: string, resultsDirectory: string, options: EvaluationOptions = {}) {
  await mkdir(resultsDirectory, { recursive: true });
  const execution = await runSolution(program, options);
  const validation = execution.error === undefined
    ? validatePacking(execution.result)
    : { correct: false, message: `Execution error: ${execution.error}` };
  // Invalid candidates never receive a positive score in the new evaluator.
  const metrics = validation.correct
    ? packingMetrics(execution.result as Packing)
    : { combined_score: 0, error: validation.message };
  await Promise.all([
    writeFile(join(resultsDirectory, "stdout.log"), execution.stdout),
    writeFile(join(resultsDirectory, "stderr.log"), execution.stderr),
    writeFile(join(resultsDirectory, "correct.json"), JSON.stringify(validation, null, 2) + "\n"),
    writeFile(join(resultsDirectory, "metrics.json"), JSON.stringify(metrics, null, 2) + "\n"),
  ]);
  if (validation.correct) {
    const [centers, radii, reported_sum] = execution.result as Packing;
    await writeFile(join(resultsDirectory, "extra.npz"), encodeNpz({
      centers: { data: centers.flat(), shape: [centers.length, 2] },
      radii: { data: radii, shape: [radii.length] },
      reported_sum: { data: [reported_sum], shape: [] },
    }));
  } else {
    await rm(join(resultsDirectory, "extra.npz"), { force: true });
  }
  return { validation, metrics };
}
