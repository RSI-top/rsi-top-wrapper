from typing import List, Optional, Tuple, Literal
import numpy as np
from shinka.database import Program
from shinka.database.inspirations import InspirationContextBuilder
from shinka.prompts import (
    construct_eval_history_msg,
    perf_str,
    format_text_feedback_section,
    BASE_SYSTEM_MSG,
    DIFF_SYS_FORMAT,
    DIFF_ITER_MSG,
    FULL_ITER_MSG,
    FULL_SYS_FORMATS,
    CROSS_SYS_FORMAT,
    CROSS_ITER_MSG,
    get_cross_component,
    FIX_SYS_FORMAT,
    FIX_ITER_MSG,
    format_error_output_section,
)
from shinka.prompts.prompts_init import INIT_SYSTEM_MSG, INIT_USER_MSG
from shinka.defaults import default_patch_type_probs, default_patch_types
import logging

logger = logging.getLogger(__name__)


class PromptSampler:
    def __init__(
        self,
        task_sys_msg: Optional[str] = None,
        language: str = "python",
        patch_types: Optional[List[str]] = None,
        patch_type_probs: Optional[List[float]] = None,
        use_text_feedback: bool = False,
        inspiration_sort_order: Literal[
            "ascending", "chronological", "none"
        ] = "ascending",
    ):
        if patch_types is None:
            patch_types = default_patch_types()
        if patch_type_probs is None:
            patch_type_probs = default_patch_type_probs()

        self.task_sys_msg = task_sys_msg
        self.language = language
        self.patch_types = patch_types
        self.patch_type_probs = patch_type_probs
        # Check if probabilities sum to 1.0 w. tolerance for errors
        prob_sum = np.sum(patch_type_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Coding type probabilities must sum to 1.0, got {prob_sum:.6f}"
            )
        # Whether to use text feedback in the prompt
        self.use_text_feedback = use_text_feedback
        # Context builder for sorting inspirations (least-to-most by default)
        self.context_builder = InspirationContextBuilder(
            sort_order=inspiration_sort_order
        )

    def initial_program_prompt(self) -> Tuple[str, str]:
        """Generate the prompt for the initial program."""
        if self.task_sys_msg is None:
            sys_msg = INIT_SYSTEM_MSG
            task_description = "The user has not provided a task description."
        else:
            sys_msg = self.task_sys_msg
            task_description = self.task_sys_msg

        user_msg = INIT_USER_MSG.format(
            language=self.language,
            task_description=task_description,
        )
        return sys_msg, user_msg

    def select_patch_type(
        self,
        archive_inspirations: List[Program],
        top_k_inspirations: List[Program],
    ) -> str:
        """Native patch-mode policy — the ONE selector for ordinary ``sample``
        and the meta-audit intervention path, so the two cannot drift.

        Mechanically extracted from ``sample``: filter out crossover when both
        inspiration sets are empty, renormalize (with the zero-probability
        uniform fallback), and draw once with ``np.random.choice``.
        """
        # Sample coding type
        # Filter out crossover if no inspirations
        if len(archive_inspirations) == 0 and len(top_k_inspirations) == 0:
            valid_types = [t for t in self.patch_types if t != "cross"]
            valid_probs = [
                p
                for t, p in zip(self.patch_types, self.patch_type_probs)
                if t != "cross"
            ]
            # Renormalize probabilities
            prob_sum = sum(valid_probs)
            if prob_sum > 0:
                valid_probs = [p / prob_sum for p in valid_probs]
            else:
                # Fallback: uniform distribution if all probs are zero
                if len(valid_types) > 0:
                    valid_probs = [1.0 / len(valid_types)] * len(valid_types)
                else:
                    # No valid types, fall back to original patch types
                    valid_types = self.patch_types
                    valid_probs = self.patch_type_probs
            patch_type = np.random.choice(valid_types, p=valid_probs)
        else:
            patch_type = np.random.choice(
                self.patch_types,
                p=self.patch_type_probs,
            )
        return patch_type

    def sample(
        self,
        parent: Program,
        archive_inspirations: List[Program],
        top_k_inspirations: List[Program],
        meta_recommendations: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        if self.task_sys_msg is None:
            sys_msg = BASE_SYSTEM_MSG
        else:
            sys_msg = self.task_sys_msg

        patch_type = self.select_patch_type(archive_inspirations, top_k_inspirations)

        # Add meta-recommendations BEFORE format instructions (if provided)
        if meta_recommendations not in [None, "none"] and patch_type != "cross":
            sys_msg += "\n\n# Potential Recommendations"
            sys_msg += (
                "\nThe following are potential recommendations for the "
                "next program generation:\n"
            )
            sys_msg += f"\n{meta_recommendations}"
            logger.info(
                f"Added meta recommendation to system prompt: "
                f"{meta_recommendations[:80]}..."
            )
        else:
            logger.debug(
                f"No meta recommendation added: "
                f"meta_recommendations={bool(meta_recommendations)}, "
                f"patch_type={patch_type}"
            )

        # Add format instructions AFTER meta-recommendations
        if patch_type == "diff":
            sys_msg += DIFF_SYS_FORMAT
        elif patch_type == "full":
            # Randomly sample from different full rewrite variants
            full_variant_idx = np.random.randint(0, len(FULL_SYS_FORMATS))
            selected_format = FULL_SYS_FORMATS[full_variant_idx]
            sys_msg += selected_format
        elif patch_type == "cross":
            sys_msg += CROSS_SYS_FORMAT

        # Build sorted inspiration context (combines archive + top-k)
        sorted_inspirations = self.context_builder.build_context(
            archive_inspirations, top_k_inspirations
        )

        if len(sorted_inspirations) > 0:
            eval_history_msg = construct_eval_history_msg(
                sorted_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
        else:
            eval_history_msg = ""

        # Format text feedback section for current program
        text_feedback_section = ""
        if self.use_text_feedback:
            text_feedback_section = "\n" + format_text_feedback_section(
                parent.text_feedback
            )

        if patch_type == "diff":
            iter_msg = DIFF_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "full":
            iter_msg = FULL_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "cross":
            iter_msg = CROSS_ITER_MSG.format(
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf_str(
                    parent.combined_score, parent.public_metrics
                ),
                text_feedback_section=text_feedback_section,
            )
            iter_msg += "\n\n" + get_cross_component(
                archive_inspirations,
                top_k_inspirations,
                language=self.language,
            )
        elif patch_type == "paper":
            raise NotImplementedError("Paper edit not implemented.")
        else:
            raise ValueError(f"Invalid patch type: {patch_type}")

        return (
            sys_msg,
            eval_history_msg + "\n" + iter_msg,
            patch_type,
        )

    #: Patch modes the meta-audit intervention path accepts. The mode itself
    #: is chosen NATIVELY (``select_patch_type`` / ``needs_fix``), never by
    #: the adapter or the judge.
    INTERVENTION_PATCH_TYPES = ("diff", "full", "cross", "fix")

    INTERVENTION_HEADER = "ONE-SHOT META-INTERVENTION TASK"

    #: Every native format template consists of ACTION-OBJECTIVE lines
    #: followed by response mechanics that start with this exact line prefix.
    #: The intervention builder keeps everything FROM this anchor (mechanics)
    #: and drops what precedes it (the ordinary objective); templates that
    #: drift away from the anchor fail loudly rather than delivering two
    #: objectives. ``DIFF_SYS_FORMAT`` carries no objective lines at all and
    #: must start at the anchor.
    INTERVENTION_MECHANICS_ANCHOR = "\nYou MUST respond using"

    #: The native user-side "# Task" ACTION OBJECTIVE section of
    #: ``FULL_ITER_MSG`` the canonical task REPLACES (the trailing
    #: "IMPORTANT: ..." same-inputs/outputs line is response mechanics and
    #: stays, minus its improve clause — see below).
    INTERVENTION_ITER_OBJECTIVE_SECTION = (
        "# Task\n\n"
        "Rewrite the program to improve its performance on the specified metrics.\n"
        "Provide the complete new program code.\n\n"
    )

    #: Neutral native role sentence used when no task-specific
    #: ``task_sys_msg`` exists. ``BASE_SYSTEM_MSG``'s own wording ("tasked
    #: with improving ... suggest improvements ...") is an ordinary action
    #: objective, so the dedicated builder never renders it.
    INTERVENTION_NEUTRAL_ROLE = "You are an expert software engineer."

    #: ``FULL_ITER_MSG``'s current-program intro is an objective ("we are
    #: trying to improve ... you will need to propose ... improved internal
    #: implementation"); the dedicated builder renders the factual variant,
    #: keeping the same-inputs/outputs mechanics without the improve clause.
    INTERVENTION_ITER_INTRO_OBJECTIVE = (
        "Here is the current program we are trying to improve (you will need "
        "to propose a new program with the same inputs and outputs as the "
        "original program, but with improved internal implementation):"
    )
    INTERVENTION_ITER_INTRO_FACTUAL = (
        "Here is the current program (any new program must keep the same "
        "inputs and outputs as the original program):"
    )

    #: The improve clause inside the retained response mechanics (format
    #: template bullet + user "IMPORTANT:" line). Same-I/O is mechanical and
    #: stays; the improve directive is an objective and is dropped.
    INTERVENTION_IMPROVE_MECHANIC = (
        "maintains the same inputs and outputs as the original program, "
        "but with improved internal implementation."
    )
    INTERVENTION_NEUTRAL_MECHANIC = (
        "maintains the same inputs and outputs as the original program."
    )

    #: ``DIFF_ITER_MSG``'s intro and "# Task" ACTION OBJECTIVE. The
    #: "# Instructions" SEARCH/REPLACE consistency notes and the
    #: "IMPORTANT: ... targeted improvements." diff-scope constraint are
    #: mechanics and stay.
    INTERVENTION_DIFF_INTRO_OBJECTIVE = (
        "Here is the current program we are trying to improve (you will need "
        "to propose a modification to it below):"
    )
    INTERVENTION_DIFF_INTRO_FACTUAL = (
        "Here is the current program (any modification you propose applies "
        "to it below):"
    )
    INTERVENTION_DIFF_OBJECTIVE_SECTION = (
        "# Task\n\n"
        "Suggest a new idea to improve the performance of the code that is "
        "inspired by your expert knowledge of the considered subject.\n"
        "Your goal is to maximize the `combined_score` of the program.\n"
        "Describe each change with a SEARCH/REPLACE block.\n\n"
    )

    #: ``CROSS_ITER_MSG``'s "# Task" crossover ACTION OBJECTIVE (its intro and
    #: IMPORTANT line reuse the FULL literals above).
    INTERVENTION_CROSS_OBJECTIVE_SECTION = (
        "# Task\n\n"
        "Perform a cross-over between the code script above and the one "
        "below. Aim to combine the best parts of both code implementations "
        "that improves the score.\n"
        "Provide the complete new program code.\n\n"
    )

    #: ``FIX_ITER_MSG``'s trailing "# Task" fix ACTION OBJECTIVE (analysis
    #: steps + correctness-first priority advice). The incorrect program,
    #: validation-failure statement, and error output above it are factual
    #: context and stay.
    INTERVENTION_FIX_OBJECTIVE_SECTION = (
        "# Task\n\n"
        "Analyze the error output above and fix the program. Focus on:\n"
        "1. Understanding why the program failed validation\n"
        "2. Identifying the root cause from the error messages\n"
        "3. Implementing a fix that addresses the issue\n\n"
        "IMPORTANT: Make the program correct first. Performance improvements "
        "can come later."
    )

    def _intervention_sys_base(
        self,
        audit_id: "str | None",
        task_block: "str | None" = None,
    ) -> str:
        """Build native system context; custom tasks stay in the user message."""

        if self.task_sys_msg is not None:
            # A task-specific problem/task description is native context and
            # is preserved VERBATIM.
            base = self.task_sys_msg
        else:
            # Never the BASE_SYSTEM_MSG fallback: its "tasked with improving
            # ... suggest improvements" wording is an ordinary action
            # objective. Keep only the neutral native role sentence.
            if not BASE_SYSTEM_MSG.startswith("You are an expert software engineer"):
                raise RuntimeError(
                    "native BASE_SYSTEM_MSG drifted: neutral role prefix not "
                    "found; refusing to assemble an intervention prompt"
                )
            base = self.INTERVENTION_NEUTRAL_ROLE
        if task_block is not None:
            if not isinstance(task_block, str) or not task_block.strip():
                raise ValueError("custom intervention task block must be non-empty")
            # Breadth tasks replace the native user-message objective below;
            # the system message retains only problem context and mechanics.
            return base

        raise ValueError('breadth intervention requires a concrete task block')

    def _intervention_mechanics(self, template: str, name: str) -> str:
        """Response mechanics of a native format template: everything from
        the exact ``INTERVENTION_MECHANICS_ANCHOR`` line on, with any improve
        clause neutralized. Drift-guarded, never keyword filtering."""
        if template.count(self.INTERVENTION_MECHANICS_ANCHOR) != 1:
            raise RuntimeError(
                f"native {name} format template drifted: mechanics anchor not "
                "found exactly once; refusing to assemble an intervention prompt"
            )
        mechanics = template[template.index(self.INTERVENTION_MECHANICS_ANCHOR) + 1:]
        return mechanics.replace(
            self.INTERVENTION_IMPROVE_MECHANIC,
            self.INTERVENTION_NEUTRAL_MECHANIC,
            1,
        )

    def _intervention_history(self, archive_inspirations, top_k_inspirations) -> str:
        """Native inspiration/search context, exactly as ordinary ``sample``
        builds it from the natively sampled inspirations."""
        sorted_inspirations = self.context_builder.build_context(
            archive_inspirations, top_k_inspirations
        )
        if len(sorted_inspirations) > 0:
            return construct_eval_history_msg(
                sorted_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
            )
        return ""

    def _intervention_text_feedback(self, program: Program) -> str:
        if self.use_text_feedback and program.text_feedback:
            return "\n" + format_text_feedback_section(program.text_feedback)
        return ""

    def _strip_section(self, template: str, section: str, name: str, replace: str = "") -> str:
        """Exact-literal template surgery with a drift guard."""
        if template.count(section) != 1:
            raise RuntimeError(
                f"native {name} template drifted: expected section not found "
                "exactly once; refusing to assemble an intervention prompt"
            )
        return template.replace(section, replace, 1)

    def sample_intervention_task(
        self,
        parent: Program,
        archive_inspirations: List[Program],
        top_k_inspirations: List[Program],
        *,
        patch_type: str,
        audit_id: "str | None" = None,
        task_block: "str | None" = None,
    ) -> Tuple[str, str, str]:
        """Build the one-shot reflection-and-act prompt for a NATIVELY chosen
        patch mode (trigger-only, unique-primary-objective contract).

        The caller supplies the NATIVELY sampled parent/inspirations and the
        patch mode drawn by :meth:`select_patch_type` (or native fix mode via
        :meth:`sample_intervention_fix_task`) — nothing here chooses or
        re-chooses a mode, parent, or direction. The canonical shared task
        (``meta_audit.INTERVENTION_TASK_PROMPT``, verbatim, exactly once) is
        the ONLY high-level action objective: the selected mode's ordinary
        improve/rewrite/crossover objective is stripped from BOTH native
        slots (format template and user "# Task" section) by exact
        drift-guarded literals, while the mode's response mechanics and the
        native context (problem statement, parent code/metrics/feedback,
        inspiration history, cross component) are preserved. ``full`` mode
        keeps the native format-variant behavior: one ``np.random.randint``
        draw per call, exactly like ordinary ``sample``.
        """
        patch_type = str(patch_type)
        text_feedback_section = self._intervention_text_feedback(parent)
        perf = perf_str(parent.combined_score, parent.public_metrics)
        task_replacement = (
            f"# Task\n\n{task_block}\n\n" if task_block is not None else ""
        )
        # Task text is a format value, never a template: preserve literal braces.

        if patch_type == "diff":
            if not DIFF_SYS_FORMAT.startswith(self.INTERVENTION_MECHANICS_ANCHOR):
                raise RuntimeError(
                    "native diff format template drifted: expected pure "
                    "mechanics; refusing to assemble an intervention prompt"
                )
            mechanics = DIFF_SYS_FORMAT
            iter_template = self._strip_section(
                DIFF_ITER_MSG,
                self.INTERVENTION_DIFF_INTRO_OBJECTIVE,
                "diff iteration",
                self.INTERVENTION_DIFF_INTRO_FACTUAL,
            )
            iter_template = self._strip_section(
                iter_template,
                self.INTERVENTION_DIFF_OBJECTIVE_SECTION,
                "diff iteration",
                "{intervention_task}",
            )
            iter_msg = iter_template.format(
                intervention_task=task_replacement,
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf,
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "full":
            # Native full-format variant behavior: sampled once per attempt,
            # exactly like ordinary ``sample``.
            variant = FULL_SYS_FORMATS[np.random.randint(0, len(FULL_SYS_FORMATS))]
            mechanics = "\n" + self._intervention_mechanics(variant, "full")
            iter_template = self._strip_section(
                FULL_ITER_MSG,
                self.INTERVENTION_ITER_INTRO_OBJECTIVE,
                "full iteration",
                self.INTERVENTION_ITER_INTRO_FACTUAL,
            )
            iter_template = self._strip_section(
                iter_template,
                self.INTERVENTION_ITER_OBJECTIVE_SECTION,
                "full iteration",
                "{intervention_task}",
            )
            iter_template = self._strip_section(
                iter_template,
                self.INTERVENTION_IMPROVE_MECHANIC,
                "full iteration",
                self.INTERVENTION_NEUTRAL_MECHANIC,
            )
            iter_msg = iter_template.format(
                intervention_task=task_replacement,
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf,
                text_feedback_section=text_feedback_section,
            )
        elif patch_type == "cross":
            mechanics = "\n" + self._intervention_mechanics(CROSS_SYS_FORMAT, "cross")
            iter_template = self._strip_section(
                CROSS_ITER_MSG,
                self.INTERVENTION_ITER_INTRO_OBJECTIVE,
                "cross iteration",
                self.INTERVENTION_ITER_INTRO_FACTUAL,
            )
            iter_template = self._strip_section(
                iter_template,
                self.INTERVENTION_CROSS_OBJECTIVE_SECTION,
                "cross iteration",
                "{intervention_task}",
            )
            iter_template = self._strip_section(
                iter_template,
                self.INTERVENTION_IMPROVE_MECHANIC,
                "cross iteration",
                self.INTERVENTION_NEUTRAL_MECHANIC,
            )
            iter_msg = iter_template.format(
                intervention_task=task_replacement,
                language=self.language,
                code_content=parent.code,
                performance_metrics=perf,
                text_feedback_section=text_feedback_section,
            )
            # Native cross component (its inspiration draw is the native one).
            iter_msg += "\n\n" + get_cross_component(
                archive_inspirations,
                top_k_inspirations,
                language=self.language,
            )
        else:
            raise ValueError(
                f"sample_intervention_task supports diff/full/cross, got "
                f"{patch_type!r} (fix mode uses sample_intervention_fix_task)"
            )

        sys_msg = self._intervention_sys_base(audit_id, task_block) + "\n" + mechanics
        eval_history_msg = self._intervention_history(
            archive_inspirations, top_k_inspirations
        )
        return (
            sys_msg,
            eval_history_msg + "\n" + iter_msg if eval_history_msg else iter_msg,
            patch_type,
        )

    def sample_intervention_fix_task(
        self,
        incorrect_program: Program,
        ancestor_inspirations: Optional[List[Program]] = None,
        *,
        audit_id: "str | None" = None,
        task_block: "str | None" = None,
    ) -> Tuple[str, str, str]:
        """Fix-mode counterpart of :meth:`sample_intervention_task`.

        Mirrors native :meth:`sample_fix` — ancestor history, incorrect
        program, validation-failure statement, and error output are preserved
        as factual context, the fix response mechanics stay, and the ordinary
        fix ACTION OBJECTIVE ("Analyze the error output ... correct first")
        is replaced by the canonical shared task as the sole objective.
        """
        mechanics = self._intervention_mechanics(FIX_SYS_FORMAT, "fix").format(
            language=self.language
        )
        sys_msg = self._intervention_sys_base(audit_id, task_block) + "\n" + mechanics

        if ancestor_inspirations and len(ancestor_inspirations) > 0:
            eval_history_msg = construct_eval_history_msg(
                ancestor_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
                correct=False,
            )
        else:
            eval_history_msg = ""

        metadata = incorrect_program.metadata or {}
        iter_template = self._strip_section(
            FIX_ITER_MSG,
            self.INTERVENTION_FIX_OBJECTIVE_SECTION,
            "fix iteration",
            "{intervention_task}",
        )
        iter_msg = iter_template.format(
            intervention_task=f"# Task\n\n{task_block}" if task_block is not None else "",
            language=self.language,
            code_content=incorrect_program.code,
            text_feedback_section=self._intervention_text_feedback(incorrect_program),
            error_output_section=format_error_output_section(
                stdout_log=metadata.get("stdout_log", ""),
                stderr_log=metadata.get("stderr_log", ""),
            ),
        )
        return (
            sys_msg,
            eval_history_msg + "\n" + iter_msg if eval_history_msg else iter_msg,
            "fix",
        )

    def sample_fix(
        self,
        incorrect_program: Program,
        ancestor_inspirations: Optional[List[Program]] = None,
    ) -> Tuple[str, str, str]:
        """
        Generate prompts for fixing an incorrect program.

        This is used when no correct parent exists in the database,
        and we need to fix an incorrect program using its error output.

        Args:
            incorrect_program: The incorrect program to fix
            ancestor_inspirations: Programs from the ancestry of the program
                (sorted chronologically, oldest first)
            meta_recommendations: Optional recommendations from meta summarizer

        Returns:
            Tuple of (system_message, user_message, patch_type="fix")
        """
        if self.task_sys_msg is None:
            sys_msg = BASE_SYSTEM_MSG
        else:
            sys_msg = self.task_sys_msg

        sys_msg += FIX_SYS_FORMAT.format(language=self.language)

        # Build eval history from ancestor inspirations (already chronological)
        if ancestor_inspirations and len(ancestor_inspirations) > 0:
            eval_history_msg = construct_eval_history_msg(
                ancestor_inspirations,
                language=self.language,
                include_text_feedback=self.use_text_feedback,
                correct=False,
            )
        else:
            eval_history_msg = ""

        # Format text feedback section
        text_feedback_section = ""
        if self.use_text_feedback and incorrect_program.text_feedback:
            text_feedback_section = "\n" + format_text_feedback_section(
                incorrect_program.text_feedback
            )

        # Extract stdout/stderr from metadata if available
        metadata = incorrect_program.metadata or {}
        stdout_log = metadata.get("stdout_log", "")
        stderr_log = metadata.get("stderr_log", "")

        error_output_section = format_error_output_section(
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

        iter_msg = FIX_ITER_MSG.format(
            language=self.language,
            code_content=incorrect_program.code,
            text_feedback_section=text_feedback_section,
            error_output_section=error_output_section,
        )

        patch_type = "fix"
        logger.info(
            f"Generated FIX prompt for incorrect program "
            f"(Gen: {incorrect_program.generation}, "
            f"Score: {incorrect_program.combined_score or 0.0:.4f}, "
            f"Ancestors: {len(ancestor_inspirations or [])})"
        )

        return (
            sys_msg,
            eval_history_msg + "\n" + iter_msg if eval_history_msg else iter_msg,
            patch_type,
        )
