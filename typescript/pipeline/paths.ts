import { basename, join, normalize } from "node:path";
import { homedir } from "node:os";

export interface PathConfig {
  task: { dir: string };
  agent: { name: string };
  run_id?: string | null;
  paths: {
    results_root: string;
    run_dir?: string | null;
    ablations_dir?: string | null;
    paper_dir?: string | null;
    posthoc_dir?: string | null;
    final_evaluation_dir?: string | null;
  };
}

function expandUser(value: string): string {
  if (value === "~") return homedir();
  if (value.startsWith("~/") || value.startsWith("~\\")) return join(homedir(), value.slice(2));
  if (value.startsWith("~")) throw new Error("Named-user home expansion is not supported; supply an explicit path");
  return normalize(value);
}

export function generateRunId(date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(date.getUTCDate())}_${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(date.getUTCSeconds())}`;
}

/** Pure path derivation, preserving the established run-envelope layout. */
export function derivePaths(config: PathConfig, taskName?: string) {
  const sourceTask = expandUser(config.task.dir);
  const name = taskName || basename(sourceTask);
  const runId = config.run_id || generateRunId();
  const run = config.paths.run_dir ? expandUser(config.paths.run_dir) : join(expandUser(config.paths.results_root), name, config.agent.name, runId);
  const ablations = config.paths.ablations_dir ? expandUser(config.paths.ablations_dir) : join(run, "ablations");
  const paper = config.paths.paper_dir ? expandUser(config.paths.paper_dir) : join(run, "paper");
  const posthoc = config.paths.posthoc_dir ? expandUser(config.paths.posthoc_dir) : join(run, "posthoc");
  const final = config.paths.final_evaluation_dir ? expandUser(config.paths.final_evaluation_dir) : join(run, "final_evaluation");
  return Object.freeze({
    task_name: name, run_id: runId, source_task_dir: sourceTask, task_dir: join(run, "task"),
    investigation_results_dir: run, investigation_workspace_dir: join(run, "investigation"),
    ablations_output_dir: ablations, ablations_aggregated_task_dir: join(ablations, "aggregated", name),
    final_output_dir: final, posthoc_output_dir: posthoc, paper_task_dir: join(run, "task"), paper_output_dir: paper,
    run_dir: run, investigation_dir: join(run, "investigation"), task_drafting_dir: join(run, "task_drafting"),
    task_drafting_output_dir: join(run, "task_drafting"), ablations_dir: ablations, final_dir: final,
    posthoc_dir: posthoc, paper_dir: paper, revision_dir: join(run, "revision"),
  });
}
