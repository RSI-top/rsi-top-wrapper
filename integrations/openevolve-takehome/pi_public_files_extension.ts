import { readFile, readdir, realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";

type ExtensionAPI = { registerTool(tool: any): void };

const PUBLIC_ROOT = resolve(process.cwd());
const DEFAULT_LINE_LIMIT = 2000;
const MAX_LINE_LIMIT = 10000;

async function publicPath(raw: unknown): Promise<{ path: string; logical: string }> {
	if (typeof raw !== "string" || !raw.trim() || isAbsolute(raw)) {
		throw new Error("path must be a nonempty relative public path");
	}
	if (raw.split(/[\\/]+/).includes("..")) {
		throw new Error("parent traversal is not allowed");
	}
	const candidate = resolve(PUBLIC_ROOT, raw);
	const logical = relative(PUBLIC_ROOT, candidate).replaceAll("\\", "/") || ".";
	if (logical === ".." || logical.startsWith("../")) {
		throw new Error("path escapes the public planner workspace");
	}
	const canonical = await realpath(candidate);
	const canonicalRelative = relative(PUBLIC_ROOT, canonical).replaceAll("\\", "/") || ".";
	if (canonicalRelative === ".." || canonicalRelative.startsWith("../")) {
		throw new Error("path resolves outside the public planner workspace");
	}
	return { path: canonical, logical: canonicalRelative };
}

export default function (pi: ExtensionAPI) {
	pi.registerTool({
		name: "read_public_file",
		label: "Read public planner file",
		description: "Read a UTF-8 file inside the public planner workspace. Absolute paths, parent traversal, and escaping symlinks are rejected.",
		parameters: {
			type: "object",
			required: ["path"],
			properties: {
				path: { type: "string" },
				offset: { type: "integer", minimum: 1 },
				limit: { type: "integer", minimum: 1, maximum: MAX_LINE_LIMIT },
			},
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: any) {
			const selected = await publicPath(params?.path);
			const lines = (await readFile(selected.path, "utf8")).split("\n");
			const offset = Number.isInteger(params?.offset) ? params.offset : 1;
			const limit = Number.isInteger(params?.limit)
				? Math.min(params.limit, MAX_LINE_LIMIT)
				: DEFAULT_LINE_LIMIT;
			const start = Math.max(0, offset - 1);
			const end = Math.min(lines.length, start + limit);
			const text = lines.slice(start, end)
				.map((line, index) => `${start + index + 1}: ${line}`)
				.join("\n");
			return {
				content: [{ type: "text", text }],
				details: {
					readPaths: [selected.logical],
					offset,
					linesReturned: end - start,
					totalLines: lines.length,
				},
			};
		},
	});

	pi.registerTool({
		name: "list_public_directory",
		label: "List public planner directory",
		description: "List one directory inside the public planner workspace without accessing controller-private evidence.",
		parameters: {
			type: "object",
			required: ["path"],
			properties: { path: { type: "string" } },
			additionalProperties: false,
		},
		async execute(_toolCallId: string, params: any) {
			const selected = await publicPath(params?.path);
			const entries = await readdir(selected.path, { withFileTypes: true });
			const names = entries.slice(0, 1000).map((entry) =>
				`${entry.name}${entry.isDirectory() ? "/" : ""}`
			);
			return {
				content: [{ type: "text", text: names.join("\n") }],
				details: { path: selected.logical, entryCount: entries.length },
			};
		},
	});
}
