"""State components required for a reproducible checkpoint."""

EXACT_T0_RULE = "Exact t0 pins every durable or in-memory state component that the baseline will read, or that can influence the very next ordinary scheduling, parent/context selection, prompt, proposal, evaluation, budget decision, or concurrency behaviour. A target candidate tree alone is NOT an exact checkpoint of a multi-agent search. Restoring some components and restarting the rest from their own records is an approximate reconstruction that can change future search, and is never called forkable."
REQUIRED_STATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "openevolve": (
        "archive",
        "budget_state",
        "candidate_state",
        "concurrency_state",
        "evaluator_closure",
        "island_feature_maps",
        "island_state",
        "iteration_counter",
        "prompt_state",
        "search_population",
        "settings",
    ),
    "shinka": (
        "archive",
        "beam_search_parent",
        "budget_state",
        "candidate_state",
        "concurrency_state",
        "evaluator_closure",
        "generation_counter",
        "island_state",
        "prompt_state",
        "search_population",
        "settings",
    ),
}
