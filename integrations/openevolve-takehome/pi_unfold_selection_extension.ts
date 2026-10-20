import { readFile } from "node:fs/promises";
import { join } from "node:path";

type ExtensionAPI = { registerTool(tool: any): void };

const ATTEMPT_ID = /^attempt:seq:(\d{1,8})$/;
const MAX_REASON_LENGTH = 500;
const PRIVATE_HISTORY_ROOT = process.env.OE_PRIVATE_HISTORY_ROOT?.trim() || null;

function normalizeSelections(value: unknown): any[] {
	if (!Array.isArray(value) || value.length === 0) {
		throw new Error("selections must be a nonempty list");
	}
	const seen = new Set<string>();
	return value.map((selection: any) => {
		if (
			typeof selection !== "object" || selection === null ||
			Object.keys(selection).sort().join(",") !==
				["attempt_id", "metric_ids", "selection_reason"].sort().join(",")
		) {
			throw new Error(
				"each selection must contain exactly attempt_id, metric_ids, and selection_reason",
			);
		}
		const attemptMatch = typeof selection.attempt_id === "string"
			? ATTEMPT_ID.exec(selection.attempt_id)
			: null;
		if (attemptMatch === null) {
			throw new Error(`invalid attempt_id: ${selection.attempt_id}`);
		}
		const attemptId = `attempt:seq:${attemptMatch[1].padStart(8, "0")}`;
		if (seen.has(attemptId)) {
			throw new Error(`duplicate attempt_id in batch: ${attemptId}`);
		}
		seen.add(attemptId);
		if (
			!Array.isArray(selection.metric_ids) ||
			selection.metric_ids.some((item: unknown) => typeof item !== "string" || !item.trim()) ||
			new Set(selection.metric_ids).size !== selection.metric_ids.length
		) {
			throw new Error("metric_ids must be a list of distinct nonempty strings");
		}
		if (
			typeof selection.selection_reason !== "string" ||
			!selection.selection_reason.trim() ||
			selection.selection_reason.length > MAX_REASON_LENGTH
		) {
			throw new Error(`selection_reason must contain 1-${MAX_REASON_LENGTH} characters`);
		}
		return {
			attempt_id: attemptId,
			metric_ids: selection.metric_ids,
			selection_reason: selection.selection_reason.trim(),
		};
	});
}

export default function (pi: ExtensionAPI) {
	const registered = new Set<string>();

	pi.registerTool({
		name: "register_unfold_selection",
		label: "Register selective unfolding",
		description: "Record why historical attempts were selected before reading or comparing their raw evidence.",
		parameters: {
			type: "object",
			required: ["selections"],
			properties: {
				selections: {
					type: "array",
					minItems: 1,
					items: {
						type: "object",
						required: ["attempt_id", "metric_ids", "selection_reason"],
						properties: {
							attempt_id: { type: "string" },
							metric_ids: { type: "array", items: { type: "string" } },
							selection_reason: { type: "string" },
						},
						additionalProperties: false,
					},
				},
			},
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: any) {
			const selections = normalizeSelections(params?.selections);
			for (const selection of selections) registered.add(selection.attempt_id);
			return {
				content: [{
					type: "text",
					text: `Registered ${selections.length} historical attempt selection(s).`,
				}],
				details: { selections },
			};
		},
	});

	pi.registerTool({
		name: "read_registered_attempt",
		label: "Read registered historical attempt",
		description: "Read trajectory and optionally code for an attempt previously registered with register_unfold_selection.",
		parameters: {
			type: "object",
			required: ["attempt_id", "include_code"],
			properties: {
				attempt_id: { type: "string" },
				include_code: { type: "boolean" },
			},
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: any) {
			const match = typeof params?.attempt_id === "string"
				? ATTEMPT_ID.exec(params.attempt_id)
				: null;
			if (match === null) throw new Error(`invalid attempt_id: ${params?.attempt_id}`);
			const attemptId = `attempt:seq:${match[1].padStart(8, "0")}`;
			if (!registered.has(attemptId)) {
				throw new Error(`${attemptId} must be registered before reading`);
			}
			if (PRIVATE_HISTORY_ROOT === null) {
				throw new Error("OE_PRIVATE_HISTORY_ROOT is not configured");
			}
			const seq = match[1].padStart(8, "0");
			const root = join(PRIVATE_HISTORY_ROOT, `seq_${seq}`);
			const trajectory = await readFile(join(root, "trajectory.jsonl"), "utf8");
			let code: string | null = null;
			if (params.include_code === true) {
				try {
					code = await readFile(join(root, "code.py"), "utf8");
				} catch (error: any) {
					if (error?.code !== "ENOENT") throw error;
				}
			}
			const logicalPaths = [
				`search_history/seq_${seq}/trajectory.jsonl`,
				...(code === null ? [] : [`search_history/seq_${seq}/code.py`]),
			];
			return {
				content: [{
					type: "text",
					text: JSON.stringify({
						attempt_id: attemptId,
						trajectory_jsonl: trajectory,
						code,
					}, null, 2),
				}],
				details: {
					attemptId,
					readPaths: logicalPaths,
					privateReadPaths: logicalPaths.map((path) =>
						join(PRIVATE_HISTORY_ROOT, path.replace("search_history/", ""))
					),
				},
			};
		},
	});
}
