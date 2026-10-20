import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { join } from "node:path";

type ExtensionAPI = { registerTool(tool: any): void };

const ENGINE_PROFILES_PATH = process.env.OE_PRIVATE_ENGINE_PROFILES_PATH
	?? join(process.cwd(), "search_history", "engine_profiles.json");
const PRIVATE_HISTORY_ROOT = process.env.OE_PRIVATE_HISTORY_ROOT?.trim() || null;
const BRANCH_CATALOG_PATH = join(process.cwd(), "workspace", "meta", "branch_catalog.json");

function seqFromId(value: string): string {
	const match = value.match(/^(?:attempt:seq:)?(\d{1,8})$/);
	if (!match) throw new Error(`Invalid attempt ID: ${value}`);
	return match[1].padStart(8, "0");
}

function finite(value: unknown): number | null {
	return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function similarity(left: string, right: string): number | null {
	const normalize = (text: string) => text.toLowerCase().replace(/\s+/g, " ").trim();
	const a = normalize(left);
	const b = normalize(right);
	if (!a || !b) return null;
	const counts = (text: string) => {
		const result = new Map<string, number>();
		for (let i = 0; i < text.length - 1; i++) {
			const pair = text.slice(i, i + 2);
			result.set(pair, (result.get(pair) ?? 0) + 1);
		}
		return result;
	};
	const ca = counts(a);
	const cb = counts(b);
	let overlap = 0;
	for (const [pair, count] of ca) overlap += Math.min(count, cb.get(pair) ?? 0);
	return Number((2 * overlap / Math.max(1, a.length + b.length - 2)).toFixed(6));
}

async function optionalJson(path: string): Promise<any | null> {
	try {
		return JSON.parse(await readFile(path, "utf8"));
	} catch (error: any) {
		if (error?.code === "ENOENT") return null;
		throw error;
	}
}

function profileDelta(left: any, right: any): any | null {
	if (!left || !right) return null;
	const result: any = {};
	for (const field of [
		"cycles",
		"emitted_bundle_count",
		"scratch_words",
		"max_engine_slot_lower_bound",
		"dependency_critical_path_lower_bound",
	]) {
		const before = finite(left[field]);
		const after = finite(right[field]);
		if (before !== null && after !== null) result[`delta_${field}`] = after - before;
	}
	for (const field of ["engine_op_counts", "engine_occupied_bundle_counts", "engine_slot_lower_bounds"]) {
		if (!left[field] || !right[field]) continue;
		const delta: any = {};
		for (const engine of Object.keys(right[field])) {
			const before = finite(left[field][engine]);
			const after = finite(right[field][engine]);
			if (before !== null && after !== null) delta[engine] = after - before;
		}
		if (Object.keys(delta).length) result[`delta_${field}`] = delta;
	}
	return result;
}

async function loadAttempt(id: string, profiles: any) {
	const seq = seqFromId(id);
	let root = PRIVATE_HISTORY_ROOT === null
		? join(process.cwd(), "recent_attempts", `seq_${seq}`)
		: join(PRIVATE_HISTORY_ROOT, `seq_${seq}`);
	let trajectoryPath = join(root, "trajectory.jsonl");
	try {
		await readFile(trajectoryPath, "utf8");
	} catch (error: any) {
		if (error?.code !== "ENOENT" || PRIVATE_HISTORY_ROOT !== null) throw error;
		root = join(process.cwd(), "search_history", `seq_${seq}`);
		trajectoryPath = join(root, "trajectory.jsonl");
	}
	const codePath = join(root, "code.py");
	const trajectory = await readFile(trajectoryPath, "utf8");
	const events = trajectory.trim().split("\n").map((line) => JSON.parse(line));
	const scheduled = events.find((event) => event.event === "attempt_scheduled") ?? {};
	const completion = events.find((event) => event.event === "canonical_completion") ?? {};
	const committed = events.find((event) => event.event === "program_committed") ?? {};
	const child = completion.child ?? {};
	let code: string | null = null;
	try {
		code = await readFile(codePath, "utf8");
	} catch (error: any) {
		if (error?.code !== "ENOENT") throw error;
	}
	const cycles = finite(completion.metrics?.cycles) ?? finite(child.metrics?.cycles);
	const parentCycles = finite(scheduled.parent_metrics?.cycles) ?? finite(child.parent_metrics?.cycles);
	const programId = child.id ?? committed.program_id ?? completion.program_id ?? null;
	return {
		kind: "attempt",
		id: `attempt:seq:${seq}`,
		trajectoryPath,
		codePath: code === null ? null : codePath,
		code,
		changes: committed.metadata?.changes ?? child.metadata?.changes ?? "",
		profile: typeof programId === "string" ? profiles?.profiles?.[programId] ?? null : null,
		readPaths: [
			`search_history/seq_${seq}/trajectory.jsonl`,
			...(code === null ? [] : [`search_history/seq_${seq}/code.py`]),
		],
		privateReadPaths: [trajectoryPath, ...(code === null ? [] : [codePath])],
		summary: {
			cycles,
			parent_cycles: parentCycles,
			parent_gain_cycles: cycles === null || parentCycles === null ? null : parentCycles - cycles,
			program_id: programId,
			parent_id: scheduled.parent_id ?? completion.parent_id ?? null,
			island: scheduled.target_island ?? completion.target_island ?? null,
			valid: completion.evaluation_valid ?? null,
			code_sha256: code === null ? null : createHash("sha256").update(code).digest("hex"),
		},
	};
}

async function loadBranch(id: string, catalog: any, profiles: any) {
	const row = (catalog?.programs ?? []).find((item: any) => String(item.id) === id);
	if (!row) throw new Error(`Unknown attempt or live branch ID: ${id}`);
	const codePath = join(process.cwd(), String(row.code_path));
	const code = await readFile(codePath, "utf8");
	return {
		kind: "live_branch",
		id,
		trajectoryPath: null,
		codePath,
		code,
		changes: "",
		profile: profiles?.profiles?.[id] ?? null,
		readPaths: [BRANCH_CATALOG_PATH, codePath],
		summary: {
			cycles: finite(row.metrics?.cycles),
			parent_cycles: null,
			parent_gain_cycles: null,
			program_id: id,
			parent_id: row.parent_id ?? null,
			island: row.island ?? null,
			valid: true,
			is_global_best: row.is_global_best === true,
			is_island_best: row.is_island_best === true,
			code_sha256: createHash("sha256").update(code).digest("hex"),
		},
	};
}

async function loadItem(id: string, catalog: any, profiles: any) {
	return /^(?:attempt:seq:)?\d{1,8}$/.test(id)
		? loadAttempt(id, profiles)
		: loadBranch(id, catalog, profiles);
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "compare_attempts",
		label: "Compare attempts or branches",
		description: "Read-only comparison of two attempt:seq:* evidence IDs or live branch IDs. Returns measured cycles, exact code diff, and available deterministic engine-profile deltas; it never makes a causal claim or decides compatibility.",
		parameters: {
			type: "object",
			required: ["left", "right"],
			properties: {
				left: { type: "string", description: "First attempt:seq:* or live branch ID" },
				right: { type: "string", description: "Second attempt:seq:* or live branch ID" },
			},
			additionalProperties: false,
		},
		async execute(_toolCallId, params) {
			const needsCatalog = !/^(?:attempt:seq:)?\d{1,8}$/.test(params.left)
				|| !/^(?:attempt:seq:)?\d{1,8}$/.test(params.right);
			const profiles = await optionalJson(ENGINE_PROFILES_PATH);
			const catalog = needsCatalog ? await optionalJson(BRANCH_CATALOG_PATH) : null;
			const left = await loadItem(params.left, catalog, profiles);
			const right = await loadItem(params.right, catalog, profiles);
			let codeDiff = "Code unavailable for one or both attempts.";
			if (left.codePath && right.codePath) {
				const result = spawnSync("/usr/bin/diff", ["-u", left.codePath, right.codePath], {
					encoding: "utf8",
					maxBuffer: 2 * 1024 * 1024,
				});
				codeDiff = result.stdout || "Codes are identical.";
				if (codeDiff.length > 12000) codeDiff = `${codeDiff.slice(0, 12000)}\n... [diff truncated]`;
			}
			const readPaths = [...left.readPaths, ...right.readPaths];
			const privateReadPaths = [
				...(left.privateReadPaths ?? []),
				...(right.privateReadPaths ?? []),
			];
			if (profiles) privateReadPaths.push(ENGINE_PROFILES_PATH);
			const leftCycles = finite(left.summary.cycles);
			const rightCycles = finite(right.summary.cycles);
			const output = {
				comparison_direction: "profile deltas are right minus left; they are not causal claims",
				measured_cycle_delta_right_minus_left:
					leftCycles === null || rightCycles === null ? null : rightCycles - leftCycles,
				left: { kind: left.kind, ...left.summary, changes: left.changes.slice(0, 4000), profile: left.profile },
				right: { kind: right.kind, ...right.summary, changes: right.changes.slice(0, 4000), profile: right.profile },
				same_code: left.summary.code_sha256 !== null && left.summary.code_sha256 === right.summary.code_sha256,
				change_text_similarity: similarity(left.changes, right.changes),
				profile_delta_right_minus_left: profileDelta(left.profile, right.profile),
				code_diff: codeDiff,
			};
			return {
				content: [{ type: "text", text: JSON.stringify(output, null, 2) }],
				details: {
					readPaths: [...new Set(readPaths)],
					privateReadPaths: [...new Set(privateReadPaths)],
				},
			};
		},
	});
}
