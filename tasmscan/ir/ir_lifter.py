"""
IR Lifter - Transforms AnalysisFacts to TVMModule

This module implements the lifting pass that converts the low-level
AnalysisFacts representation to the higher-level TASIR representation.

The lifting process:
1. Lift individual instructions using opcode_mappings and stack_effects
2. Build TVMBasicBlocks from facts.BasicBlock
3. Convert facts.Continuation to ContinuationDescriptor
4. Assemble TVMFunctions by grouping blocks by context
5. Produce a complete TVMModule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..analyzer.facts import (
    AnalysisFacts,
    BasicBlock,
    Continuation,
    ControlFlowEdge,
    InstructionFact,
    StackState,
)
from ..analyzer.constants import MAIN_CONTEXT
from .tasir_types import (
    TVMModule,
    TVMFunction,
    TVMBasicBlock,
    TVMInstruction,
    ContinuationDescriptor,
    SaveList,
    InstructionKind,
    BlockEdge,
    FunctionSignature,
    StackLocation,
    RegisterLocation,
    TVMLocation,
    SemanticLabel,
)
from .ir_types import IRType
from .opcode_kind_mapper import get_instruction_kind_from_opcode
from .alias_registry import resolve_alias
from .stack_effects import get_stack_effect, StackEffect
from .opcode_mappings import OPCODE_TO_IR_TYPE, is_taint_source, is_sensitive_operation
from ..config import GUARD_OPCODES as CONFIG_GUARD_OPCODES


logger = logging.getLogger(__name__)


@dataclass
class LiftingContext:
    """Context maintained during lifting process."""

    instruction_cache: Dict[int, TVMInstruction] = field(default_factory=dict)
    block_cache: Dict[int, TVMBasicBlock] = field(default_factory=dict)
    continuation_cache: Dict[str, ContinuationDescriptor] = field(default_factory=dict)
    # Map from (instruction_index, arg_index) -> continuation IDs extracted by ProgramAnalyzer
    continuation_arg_map: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    # Map from source instruction index -> continuation IDs resolved by CFG edges
    cfg_continuation_map: Dict[int, List[str]] = field(default_factory=dict)
    # Map from context_id to list of block IDs
    context_blocks: Dict[str, List[int]] = field(default_factory=dict)
    # Map from instruction index -> StackState (for O(1) stack profile lookups)
    stack_state_index: Dict[int, StackState] = field(default_factory=dict)
    # Map from instruction index -> continuation context id
    instruction_context_map: Dict[int, Optional[str]] = field(default_factory=dict)
    # Set of processed instruction indices
    processed_instructions: Set[int] = field(default_factory=set)

    def clear(self) -> None:
        """Clear all caches."""
        self.instruction_cache.clear()
        self.block_cache.clear()
        self.continuation_cache.clear()
        self.continuation_arg_map.clear()
        self.cfg_continuation_map.clear()
        self.context_blocks.clear()
        self.stack_state_index.clear()
        self.instruction_context_map.clear()
        self.processed_instructions.clear()


class IRLifter:
    """
    Lifts AnalysisFacts to TVMModule.

    This class transforms the low-level analysis representation (facts)
    into the higher-level TASIR representation suitable for security analysis.
    """

    def __init__(self) -> None:
        self._ctx = LiftingContext()

    def lift(self, facts: AnalysisFacts) -> TVMModule:
        """
        Main entry point: lift AnalysisFacts to TVMModule.

        Args:
            facts: Analysis facts from ProgramAnalyzer

        Returns:
            Complete TVMModule representation
        """
        # Clear caches for fresh lift
        self._ctx.clear()

        try:
            # Build parent instruction argument -> continuation ID mapping first.
            # This preserves ProgramAnalyzer's extracted continuation IDs during lifting.
            self._ctx.continuation_arg_map = self._build_continuation_arg_map(facts)
            # Build source instruction -> continuation target mapping from CFG.
            self._ctx.cfg_continuation_map = self._build_cfg_continuation_map(facts)
            self._ctx.stack_state_index = {
                state.instruction_index: state for state in facts.stack_states
            }
            self._ctx.instruction_context_map = {
                inst.index: inst.continuation_id for inst in facts.instructions
            }

            # Step 1: Lift all instructions
            lifted_instructions = self._lift_all_instructions(facts)

            # Step 2: Lift continuations
            lifted_continuations = self._lift_continuations(facts)

            # Step 3: Build basic blocks
            lifted_blocks = self._build_blocks(facts, lifted_instructions)

            # Step 4: Assemble functions
            functions, global_blocks = self._assemble_functions(
                facts, lifted_blocks, lifted_continuations
            )

            # Step 5: Build call graph
            call_graph = self._build_call_graph(facts)

            module = TVMModule(
                functions=functions,
                global_blocks=global_blocks,
                call_graph=call_graph,
                cross_function_continuations=lifted_continuations,
                analysis_metadata={
                    "source": "ir_lifter",
                    "instruction_count": len(facts.instructions),
                    "block_count": len(facts.basic_blocks),
                    "continuation_count": len(lifted_continuations),
                },
            )
            self._propagate_analysis_metadata(module, facts)
            module.analysis_metadata["uncertain_branches"] = facts.metadata.get(
                "uncertain_branches", {}
            )

            # Step 6: Assign semantic labels
            self._assign_semantic_labels(module)

            return module
        except Exception as e:
            logger.error(f"Failed to lift AnalysisFacts to TVMModule: {e}")
            raise

    @staticmethod
    def _propagate_analysis_metadata(module: TVMModule, facts: AnalysisFacts) -> None:
        """Propagate completeness-critical metadata from facts into module metadata.

        This keeps `IRBuilder.build_tasir(...) -> DataFlowAnalyzer.analyze_tasir(...)`
        behavior correct even when callers do not attach private `_facts`.
        """
        facts_meta = facts.metadata if isinstance(getattr(facts, "metadata", None), dict) else {}
        if not facts_meta:
            return

        if facts_meta.get("analysis_incomplete"):
            module.analysis_metadata["analysis_incomplete"] = True

        reasons = facts_meta.get("analysis_incomplete_reasons", [])
        if isinstance(reasons, list):
            normalized_reasons = [reason for reason in reasons if isinstance(reason, str) and reason]
            if normalized_reasons:
                existing = module.analysis_metadata.get("analysis_incomplete_reasons", [])
                module.analysis_metadata["analysis_incomplete_reasons"] = list(
                    dict.fromkeys(list(existing) + normalized_reasons)
                )

        if facts_meta.get("continuation_extraction_failed"):
            module.analysis_metadata["continuation_extraction_failed"] = True
            if "continuation_extraction_error" in facts_meta:
                module.analysis_metadata["continuation_extraction_error"] = facts_meta.get(
                    "continuation_extraction_error"
                )

    def _lift_all_instructions(
        self, facts: AnalysisFacts
    ) -> Dict[int, TVMInstruction]:
        """Lift all instructions from AnalysisFacts."""
        result: Dict[int, TVMInstruction] = {}

        for inst_fact in facts.instructions:
            try:
                tvm_inst = self._lift_instruction(inst_fact)
                result[inst_fact.index] = tvm_inst
                self._ctx.instruction_cache[inst_fact.index] = tvm_inst
                self._ctx.processed_instructions.add(inst_fact.index)
            except Exception as e:
                logger.warning(
                    f"Failed to lift instruction at index {inst_fact.index} "
                    f"(opcode={inst_fact.opcode}): {e}"
                )
                # Create a minimal placeholder instruction
                result[inst_fact.index] = self._create_placeholder_instruction(
                    inst_fact
                )

        return result

    def _lift_instruction(self, inst_fact: InstructionFact) -> TVMInstruction:
        """Lift a single instruction."""
        opcode = inst_fact.opcode

        # Get instruction kind from mapper
        kind = get_instruction_kind_from_opcode(opcode)

        # Get stack effect from database
        stack_effect = get_stack_effect(opcode)

        # Parse inputs and outputs based on stack effect
        inputs, outputs = self._parse_locations(opcode, inst_fact, stack_effect)

        # Extract immediates and continuation references
        immediates, cont_refs = self._extract_operands(inst_fact)

        # Determine security attributes
        ir_type = OPCODE_TO_IR_TYPE.get(opcode)
        if ir_type is None:
            alias_opcode = resolve_alias(opcode)
            if alias_opcode != opcode:
                ir_type = OPCODE_TO_IR_TYPE.get(alias_opcode)
        is_source = is_taint_source(ir_type) if ir_type else False
        is_sensitive = is_sensitive_operation(ir_type) if ir_type else False
        taint_transfer = self._determine_taint_transfer(opcode, is_source, is_sensitive)

        return TVMInstruction(
            index=inst_fact.index,
            kind=kind,
            opcode=opcode,
            inputs=inputs,
            outputs=outputs,
            immediates=immediates,
            continuation_refs=cont_refs,
            stack_effect=stack_effect,
            taint_transfer=taint_transfer,
            is_sensitive=is_sensitive,
            is_taint_source=is_source,
            original_args=list(inst_fact.arguments) if inst_fact.arguments else [],
        )

    def _create_placeholder_instruction(
        self, inst_fact: InstructionFact
    ) -> TVMInstruction:
        """Create a placeholder instruction for failed lifts."""
        return TVMInstruction(
            index=inst_fact.index,
            kind=InstructionKind.UNKNOWN,
            opcode=inst_fact.opcode,
            inputs=[],
            outputs=[],
            immediates=[],
            continuation_refs=[],
            stack_effect=None,
            taint_transfer="propagate",
            is_sensitive=False,
            is_taint_source=False,
            original_args=list(inst_fact.arguments) if inst_fact.arguments else [],
        )

    def _parse_locations(
        self,
        opcode: str,
        inst_fact: InstructionFact,
        stack_effect: Optional[StackEffect],
    ) -> Tuple[List[TVMLocation], List[TVMLocation]]:
        """Parse input and output locations from stack effect."""
        inputs: List[TVMLocation] = []
        outputs: List[TVMLocation] = []

        if stack_effect:
            # Stack inputs (read from stack)
            for i in range(stack_effect.inputs):
                inputs.append(StackLocation(depth=i))

            # Stack outputs (written to stack)
            for i in range(stack_effect.outputs):
                outputs.append(StackLocation(depth=i))

        # Handle register operations
        upper = opcode.upper()
        if upper == "PUSHCTR":
            # Read from control register
            reg_idx = self._extract_register_index(inst_fact)
            if reg_idx is not None:
                inputs.append(RegisterLocation(index=reg_idx))
        elif upper == "POPCTR":
            # Write to control register
            reg_idx = self._extract_register_index(inst_fact)
            if reg_idx is not None:
                outputs.append(RegisterLocation(index=reg_idx))
        elif upper == "PUSHCTRX":
            # Dynamic register read - index is on stack
            pass  # Register is determined at runtime
        elif upper == "POPCTRX":
            # Dynamic register write - index is on stack
            pass  # Register is determined at runtime

        return inputs, outputs

    def _extract_register_index(self, inst_fact: InstructionFact) -> Optional[int]:
        """Extract control register index from instruction arguments."""
        if not inst_fact.arguments:
            return None

        arg = inst_fact.arguments[0]
        if hasattr(arg, "value"):
            return arg.value
        elif isinstance(arg, int):
            return arg

        return None

    def _extract_operands(
        self, inst_fact: InstructionFact
    ) -> Tuple[List[Any], List[str]]:
        """Extract immediates and continuation references from instruction."""
        immediates: List[Any] = []
        cont_refs: List[str] = []

        for cont_id in self._ctx.cfg_continuation_map.get(inst_fact.index, []):
            if cont_id not in cont_refs:
                cont_refs.append(cont_id)

        if not inst_fact.arguments:
            return immediates, cont_refs

        for arg_idx, arg in enumerate(inst_fact.arguments):
            # Preferred source: ProgramAnalyzer continuation extraction.
            mapped_refs = self._ctx.continuation_arg_map.get((inst_fact.index, arg_idx), [])
            for cont_id in mapped_refs:
                if cont_id not in cont_refs:
                    cont_refs.append(cont_id)

            arg_value = getattr(arg, "value", arg)

            # Check for continuation reference
            if hasattr(arg_value, "block_id"):
                cont_refs.append(f"cont_{arg_value.block_id}")
            elif hasattr(arg_value, "continuation_id"):
                cont_refs.append(str(arg_value.continuation_id))
            # Check for instruction objects with nested continuations
            elif hasattr(arg_value, "instructions") and not mapped_refs:
                # This is an inline continuation
                cont_id = f"inline_{inst_fact.index}_{arg_idx}"
                cont_refs.append(cont_id)
            # Check for cell references
            elif hasattr(arg_value, "cell_ref"):
                cont_refs.append(f"cellref_{id(arg_value)}")
            # Extract value
            elif hasattr(arg, "value"):
                immediates.append(arg_value)
            elif isinstance(arg, (int, str)):
                immediates.append(arg)
            # Handle other argument types
            else:
                # Try to extract useful information
                try:
                    immediates.append(repr(arg))
                except Exception:
                    pass

        return immediates, cont_refs

    @staticmethod
    def _build_cfg_continuation_map(
        facts: AnalysisFacts
    ) -> Dict[int, List[str]]:
        """Build source instruction -> continuation IDs map from resolved CFG edges."""
        cont_map: Dict[int, List[str]] = {}
        index_to_fact = {fact.index: fact for fact in facts.instructions}

        for edge in facts.cfg_edges:
            if edge.target is None:
                continue
            if edge.kind not in {"call_cont", "jump"}:
                continue
            target_fact = index_to_fact.get(edge.target)
            if not target_fact or target_fact.continuation_id is None:
                continue
            refs = cont_map.setdefault(edge.source, [])
            if target_fact.continuation_id not in refs:
                refs.append(target_fact.continuation_id)

        return cont_map

    def _build_continuation_arg_map(
        self, facts: AnalysisFacts
    ) -> Dict[Tuple[int, int], List[str]]:
        """Build mapping from parent instruction argument positions to continuation IDs."""
        continuation_arg_map: Dict[Tuple[int, int], List[str]] = {}
        continuations = facts.metadata.get("continuations", {})
        if not continuations:
            return continuation_arg_map

        # Build (context_id, local_index) -> global instruction index map.
        context_local_to_global: Dict[Tuple[str, int], int] = {}
        facts_by_context: Dict[str, List[InstructionFact]] = {}
        for fact in sorted(facts.instructions, key=lambda inst: inst.index):
            ctx = fact.continuation_id or MAIN_CONTEXT
            facts_by_context.setdefault(ctx, []).append(fact)
        for ctx, ctx_facts in facts_by_context.items():
            for local_idx, fact in enumerate(ctx_facts):
                context_local_to_global[(ctx, local_idx)] = fact.index

        for cont_id, cont in continuations.items():
            parent_context: Optional[str] = None
            parent_local_index: Optional[int] = None
            parent_arg_index: Optional[int] = None

            if isinstance(cont, Continuation):
                parent_context = cont.parent_context
                parent_local_index = cont.parent_local_index
                parent_arg_index = cont.parent_arg_index
            elif isinstance(cont, dict):
                parent_context = cont.get("parent_context")
                parent_local_index = cont.get("parent_local_index")
                parent_arg_index = cont.get("parent_arg_index")

            if parent_context is None:
                parent_context = MAIN_CONTEXT
            if not isinstance(parent_local_index, int) or parent_local_index < 0:
                continue

            arg_idx = parent_arg_index if isinstance(parent_arg_index, int) and parent_arg_index >= 0 else 0
            parent_global_idx = context_local_to_global.get((parent_context, parent_local_index))
            if parent_global_idx is None:
                continue

            key = (parent_global_idx, arg_idx)
            refs = continuation_arg_map.setdefault(key, [])
            if cont_id not in refs:
                refs.append(cont_id)

        return continuation_arg_map

    def _determine_taint_transfer(
        self, opcode: str, is_source: bool, is_sensitive: bool
    ) -> str:
        """Determine taint transfer semantics for an instruction."""
        if is_source:
            return "source"
        if is_sensitive:
            return "sink"

        upper = opcode.upper()

        # Guards/checks sanitize values
        sanitizing_opcodes = {
            "THROWIF",
            "THROWIFNOT",
            "THROW",
            "THROWANY",
            "THROWARGIF",
            "THROWARGIFNOT",
            "IFRET",
            "IFNOTRET",
            "IFRETALT",
            "IFNOTRETALT",
            "CHKSIGNU",
            "CHKSIGNS",
            "P256_CHKSIGNU",
            "P256_CHKSIGNS",
        }

        if upper in sanitizing_opcodes:
            return "sanitize"

        # Comparison operations can act as guards
        comparison_opcodes = {
            "EQUAL",
            "EQINT",
            "LESS",
            "LESSINT",
            "GREATER",
            "GTINT",
            "LEQ",
            "GEQ",
            "NEQ",
            "NEQINT",
            "CMP",
            "ISZERO",
            "ISNAN",
        }

        if upper in comparison_opcodes:
            return "check"

        return "propagate"

    def _lift_continuations(
        self, facts: AnalysisFacts
    ) -> Dict[str, ContinuationDescriptor]:
        """Lift all continuations to ContinuationDescriptor."""
        result: Dict[str, ContinuationDescriptor] = {}

        # Get continuations from metadata
        continuations = facts.metadata.get("continuations", {})

        for cont_id, cont in continuations.items():
            if isinstance(cont, Continuation):
                descriptor = self._lift_continuation(cont)
                result[cont_id] = descriptor
                self._ctx.continuation_cache[cont_id] = descriptor
            elif isinstance(cont, dict):
                # Handle dictionary representation
                descriptor = self._lift_continuation_dict(cont_id, cont)
                result[cont_id] = descriptor
                self._ctx.continuation_cache[cont_id] = descriptor

        return result

    def _lift_continuation(self, cont: Continuation) -> ContinuationDescriptor:
        """Lift a Continuation object to ContinuationDescriptor."""
        return ContinuationDescriptor(
            code_ref=cont.id,
            savelist=SaveList(),  # Will be populated during analysis
            nargs=self._infer_nargs(cont),
            captured_stack_depth=None,
            cont_type=cont.kind,
            gas_limit=None,
            is_return_cont=False,
        )

    def _lift_continuation_dict(
        self, cont_id: str, cont_dict: Dict[str, Any]
    ) -> ContinuationDescriptor:
        """Lift a continuation dictionary to ContinuationDescriptor."""
        return ContinuationDescriptor(
            code_ref=cont_dict.get("id", cont_id),
            savelist=SaveList(),
            nargs=cont_dict.get("nargs", -1),
            captured_stack_depth=cont_dict.get("captured_stack_depth"),
            cont_type=cont_dict.get("kind", "ordinary"),
            gas_limit=cont_dict.get("gas_limit"),
            is_return_cont=cont_dict.get("is_return_cont", False),
        )

    def _infer_nargs(self, cont: Continuation) -> int:
        """
        Infer expected argument count for continuation.

        Returns -1 for varargs (unknown argument count).
        """
        # Method continuations may have specific argument counts
        if cont.kind == "method" and cont.method_id is not None:
            # recv_internal typically expects 4 arguments
            if cont.method_id == 0:
                return 4
            # recv_external typically expects 1 argument
            elif cont.method_id == -1:
                return 1

        # Default to -1 (varargs)
        return -1

    def _build_blocks(
        self,
        facts: AnalysisFacts,
        lifted_instructions: Dict[int, TVMInstruction],
    ) -> Dict[int, TVMBasicBlock]:
        """Build TVMBasicBlocks from facts.BasicBlock."""
        result: Dict[int, TVMBasicBlock] = {}
        instr_to_block = self._build_instruction_to_block_map(facts.basic_blocks)
        block_context_by_id = {block.id: block.context for block in facts.basic_blocks}
        incoming_by_block, outgoing_by_block = self._index_cfg_edges_by_block(
            facts.cfg_edges, instr_to_block
        )

        for block in facts.basic_blocks:
            try:
                tvm_block = self._build_single_block(
                    block,
                    facts,
                    lifted_instructions,
                    instr_to_block,
                    block_context_by_id,
                    incoming_by_block.get(block.id, []),
                    outgoing_by_block.get(block.id, []),
                )
                result[block.id] = tvm_block
                self._ctx.block_cache[block.id] = tvm_block

                # Track blocks by context
                context_id = block.context
                if context_id not in self._ctx.context_blocks:
                    self._ctx.context_blocks[context_id] = []
                self._ctx.context_blocks[context_id].append(block.id)

            except Exception as e:
                logger.warning(f"Failed to build block {block.id}: {e}")

        return result

    def _build_single_block(
        self,
        block: BasicBlock,
        facts: AnalysisFacts,
        lifted_instructions: Dict[int, TVMInstruction],
        instr_to_block: Dict[int, int],
        block_context_by_id: Dict[int, str],
        incoming_edges: List[Tuple[int, ControlFlowEdge]],
        outgoing_edges: List[Tuple[int, ControlFlowEdge]],
    ) -> TVMBasicBlock:
        """Build a single TVMBasicBlock."""
        # Get instructions for this block
        block_instructions = [
            lifted_instructions[idx]
            for idx in block.instruction_indices
            if idx in lifted_instructions
        ]

        # Build predecessor and successor edges
        predecessors = self._build_predecessor_edges(
            block,
            incoming_edges,
            block_context_by_id=block_context_by_id,
        )
        successors = self._build_successor_edges(
            block,
            outgoing_edges,
            block_context_by_id=block_context_by_id,
        )

        # Get stack profile from facts if available.
        entry_height, exit_height, stack_metadata = self._get_stack_profile(block)

        return TVMBasicBlock(
            id=block.id,
            context_id=block.context,
            instructions=block_instructions,
            predecessors=predecessors,
            successors=successors,
            entry_stack_height=entry_height,
            exit_stack_height=exit_height,
            has_unknown_successor=block.has_unknown_successor,
            phi_nodes=[],
            metadata=stack_metadata,
        )

    def _build_predecessor_edges(
        self,
        block: BasicBlock,
        incoming_edges: List[Tuple[int, ControlFlowEdge]],
        block_context_by_id: Dict[int, str],
    ) -> List[BlockEdge]:
        """Build predecessor edges for a block."""
        edges: List[BlockEdge] = []
        seen: Set[Tuple[int, int, str]] = set()

        for source_block_id, cfg_edge in incoming_edges:
            dedup_key = (source_block_id, block.id, cfg_edge.kind)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            target_context = block_context_by_id.get(block.id)

            edges.append(
                BlockEdge(
                    source_block=source_block_id,
                    target_block=block.id,
                    edge_kind=cfg_edge.kind,
                    condition=self._extract_edge_condition(cfg_edge),
                    continuation_ref=self._extract_continuation_ref(
                        cfg_edge,
                        target_context=target_context,
                    ),
                    restored_registers=None,
                )
            )

        return edges

    def _build_successor_edges(
        self,
        block: BasicBlock,
        outgoing_edges: List[Tuple[int, ControlFlowEdge]],
        block_context_by_id: Dict[int, str],
    ) -> List[BlockEdge]:
        """Build successor edges for a block."""
        edges: List[BlockEdge] = []
        processed_successors: Set[Tuple[int, str]] = set()
        known_successors: Set[int] = set()

        for target_block_id, cfg_edge in outgoing_edges:
            dedup_key = (target_block_id, cfg_edge.kind)
            if dedup_key in processed_successors:
                continue
            processed_successors.add(dedup_key)
            known_successors.add(target_block_id)
            target_context = block_context_by_id.get(target_block_id)

            edges.append(
                BlockEdge(
                    source_block=block.id,
                    target_block=target_block_id,
                    edge_kind=cfg_edge.kind,
                    condition=self._extract_edge_condition(cfg_edge),
                    continuation_ref=self._extract_continuation_ref(
                        cfg_edge,
                        target_context=target_context,
                    ),
                    restored_registers=None,
                )
            )

        for succ_id in block.successors:
            if succ_id in known_successors:
                continue
            dedup_key = (succ_id, "fallthrough")
            if dedup_key in processed_successors:
                continue
            processed_successors.add(dedup_key)
            edges.append(
                BlockEdge(
                    source_block=block.id,
                    target_block=succ_id,
                    edge_kind="fallthrough",
                    condition=None,
                    continuation_ref=None,
                    restored_registers=None,
                )
            )

        return edges

    @staticmethod
    def _index_cfg_edges_by_block(
        cfg_edges: List[ControlFlowEdge],
        instr_to_block: Dict[int, int],
    ) -> Tuple[Dict[int, List[Tuple[int, ControlFlowEdge]]], Dict[int, List[Tuple[int, ControlFlowEdge]]]]:
        """
        Build incoming/outgoing CFG edge indices keyed by block ID.

        This avoids repeated O(B*E) scans when constructing predecessor/successor
        block edges for each basic block.
        """
        incoming_by_block: Dict[int, List[Tuple[int, ControlFlowEdge]]] = {}
        outgoing_by_block: Dict[int, List[Tuple[int, ControlFlowEdge]]] = {}

        for cfg_edge in cfg_edges:
            if cfg_edge.target is None:
                continue

            source_block_id = instr_to_block.get(cfg_edge.source)
            target_block_id = instr_to_block.get(cfg_edge.target)
            if source_block_id is None or target_block_id is None:
                continue

            incoming_by_block.setdefault(target_block_id, []).append((source_block_id, cfg_edge))
            outgoing_by_block.setdefault(source_block_id, []).append((target_block_id, cfg_edge))

        return incoming_by_block, outgoing_by_block

    @staticmethod
    def _build_instruction_to_block_map(
        basic_blocks: List[BasicBlock],
    ) -> Dict[int, int]:
        """Build instruction-index -> block-id mapping."""
        instr_to_block: Dict[int, int] = {}
        for block in basic_blocks:
            for idx in block.instruction_indices:
                instr_to_block[idx] = block.id
        return instr_to_block

    def _extract_edge_condition(self, cfg_edge: ControlFlowEdge) -> Optional[str]:
        """Extract condition string from CFG edge metadata."""
        if cfg_edge.metadata and "condition" in cfg_edge.metadata:
            return str(cfg_edge.metadata["condition"])
        return None

    def _extract_continuation_ref(
        self,
        cfg_edge: ControlFlowEdge,
        target_context: Optional[str] = None,
    ) -> Optional[str]:
        """Extract continuation reference from CFG edge metadata/context."""
        if cfg_edge.metadata:
            if "continuation" in cfg_edge.metadata:
                return str(cfg_edge.metadata["continuation"])
            if "cont_id" in cfg_edge.metadata:
                return str(cfg_edge.metadata["cont_id"])
            candidate_targets = cfg_edge.metadata.get("candidate_targets")
            if (
                isinstance(candidate_targets, list)
                and len(candidate_targets) == 1
                and isinstance(candidate_targets[0], str)
            ):
                return candidate_targets[0]
        if target_context and target_context != MAIN_CONTEXT:
            return target_context
        return None

    def _get_stack_profile(
        self,
        block: BasicBlock,
    ) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
        """Get point/range stack profile for block entry and exit."""
        if not block.instruction_indices:
            return None, None, {}

        first_idx = block.instruction_indices[0]
        last_idx = block.instruction_indices[-1]
        first_state = self._ctx.stack_state_index.get(first_idx)
        last_state = self._ctx.stack_state_index.get(last_idx)

        entry_height = first_state.height_before if first_state else None
        exit_height = last_state.height_after if last_state else None

        if first_state is None and last_state is None:
            return entry_height, exit_height, {}

        entry_min = None
        entry_max = None
        entry_unknown = False
        if first_state is not None:
            entry_min = (
                first_state.height_min
                if first_state.height_min is not None
                else first_state.height_before
            )
            if first_state.height_max is not None:
                entry_max = first_state.height_max
            else:
                entry_max = None if first_state.unknown else first_state.height_before
            entry_unknown = bool(first_state.unknown)

        exit_min = None
        exit_max = None
        exit_unknown = False
        if last_state is not None:
            exit_min = (
                last_state.height_min
                if last_state.height_min is not None
                else last_state.height_after
            )
            if last_state.height_max is not None:
                exit_max = last_state.height_max
            else:
                exit_max = None if last_state.unknown else last_state.height_after
            exit_unknown = bool(last_state.unknown or last_state.height_after is None)

        metadata: Dict[str, Any] = {
            "stack_profile": {
                "entry_min": entry_min,
                "entry_max": entry_max,
                "entry_unknown": entry_unknown,
                "exit_min": exit_min,
                "exit_max": exit_max,
                "exit_unknown": exit_unknown,
            }
        }

        return entry_height, exit_height, metadata

    def _assemble_functions(
        self,
        facts: AnalysisFacts,
        lifted_blocks: Dict[int, TVMBasicBlock],
        lifted_continuations: Dict[str, ContinuationDescriptor],
    ) -> Tuple[Dict[int, TVMFunction], List[TVMBasicBlock]]:
        """Assemble TVMFunctions from blocks."""
        functions: Dict[int, TVMFunction] = {}
        global_blocks: List[TVMBasicBlock] = []

        # Get method IDs from entry points
        method_ids = self._extract_method_ids(facts)

        # Track which blocks are assigned to functions
        assigned_blocks: Set[int] = set()

        # Create functions for each method
        for method_id, entry_info in method_ids.items():
            context_id = entry_info.get("context", "main")
            block_ids = self._ctx.context_blocks.get(context_id, [])
            blocks_for_method = {
                bid: lifted_blocks[bid]
                for bid in block_ids
                if bid in lifted_blocks
            }

            if not blocks_for_method:
                continue

            # Find entry block
            entry_block_id = entry_info.get("block_id")
            if entry_block_id is None or entry_block_id not in blocks_for_method:
                # Use the block with the smallest ID as entry
                entry_block_id = min(blocks_for_method.keys())

            # Get continuations for this context
            method_continuations = {
                cid: desc
                for cid, desc in lifted_continuations.items()
                if self._continuation_belongs_to_context(cid, context_id, facts)
            }

            # Create function signature
            signature = self._infer_function_signature(method_id, entry_info)

            # Create function
            func = TVMFunction(
                method_id=method_id,
                name=entry_info.get("name"),
                signature=signature,
                entry_block_id=entry_block_id,
                blocks=blocks_for_method,
                continuations=method_continuations,
                is_recv_internal=(method_id == 0),
                is_recv_external=(method_id == -1),
                metadata={
                    "context": context_id,
                    "entry_info": entry_info,
                },
            )

            functions[method_id] = func
            assigned_blocks.update(blocks_for_method.keys())

        # Collect global blocks (not assigned to any function)
        for block_id, block in lifted_blocks.items():
            if block_id not in assigned_blocks:
                global_blocks.append(block)

        return functions, global_blocks

    def _extract_method_ids(
        self, facts: AnalysisFacts
    ) -> Dict[int, Dict[str, Any]]:
        """Extract method IDs from entry points."""
        method_ids: Dict[int, Dict[str, Any]] = {}

        for entry in facts.entry_points:
            method_id = entry.get("method_id")
            if method_id is not None:
                method_ids[method_id] = entry

        # Default recv_internal if no entry points
        if not method_ids:
            # Try to infer entry points from continuations
            continuations = facts.metadata.get("continuations", {})
            for cont_id, cont in continuations.items():
                if isinstance(cont, Continuation) and cont.kind == "method":
                    mid = cont.method_id if cont.method_id is not None else 0
                    method_ids[mid] = {
                        "method_id": mid,
                        "context": cont.id,
                        "name": f"method_{mid}",
                    }

            # Still no methods, create default
            if not method_ids:
                method_ids[0] = {
                    "method_id": 0,
                    "context": "main",
                    "name": "recv_internal",
                }

        return method_ids

    def _continuation_belongs_to_context(
        self, cont_id: str, context_id: str, facts: AnalysisFacts
    ) -> bool:
        """Check if a continuation belongs to a given context."""
        # Exact context match (e.g. method context itself).
        if cont_id == context_id:
            return True

        # Walk continuation parent chain; substring checks (e.g. cont_1 in cont_10)
        # are unsound and cause cross-function contamination.
        continuations = facts.metadata.get("continuations", {})
        seen: Set[str] = set()
        current_id = cont_id
        while current_id and current_id not in seen:
            seen.add(current_id)
            cont = continuations.get(current_id)
            if cont is None:
                return False

            parent_context: Optional[str]
            if isinstance(cont, Continuation):
                parent_context = cont.parent_context
            elif isinstance(cont, dict):
                parent_context = cont.get("parent_context")
            else:
                return False

            if parent_context == context_id:
                return True
            if parent_context in (None, "main"):
                return False
            current_id = parent_context

        return False

    def _infer_function_signature(
        self, method_id: int, entry_info: Dict[str, Any]
    ) -> FunctionSignature:
        """Infer function signature based on method ID and entry info."""
        # Default signature
        signature = FunctionSignature()

        # recv_internal typically has specific signature
        if method_id == 0:
            signature.min_stack_inputs = 4
            signature.max_stack_inputs = 4
            # balance, msg_value, msg_cell, body_slice

        # recv_external typically has specific signature
        elif method_id == -1:
            signature.min_stack_inputs = 1
            signature.max_stack_inputs = 1
            # in_msg slice

        # Check entry info for additional signature hints
        if "signature" in entry_info:
            sig_info = entry_info["signature"]
            if isinstance(sig_info, dict):
                signature.min_stack_inputs = sig_info.get(
                    "min_inputs", signature.min_stack_inputs
                )
                signature.max_stack_inputs = sig_info.get("max_inputs")
                signature.min_stack_outputs = sig_info.get("min_outputs", 0)
                signature.max_stack_outputs = sig_info.get("max_outputs")

        return signature

    def _build_call_graph(
        self, facts: AnalysisFacts
    ) -> List[Tuple[int, int]]:
        """Build call graph from facts."""
        edges: List[Tuple[int, int]] = []

        for edge in facts.call_graph_edges:
            try:
                # Parse caller and callee from edge
                caller = self._parse_method_id(edge.caller)
                callee = self._parse_method_id(edge.callee)
                if caller is None and callee is not None:
                    caller = 0

                if caller is not None and callee is not None:
                    edges.append((caller, callee))
            except (ValueError, AttributeError) as e:
                logger.debug(f"Skipping malformed call graph edge: {e}")

        return edges

    def _parse_method_id(self, identifier: str) -> Optional[int]:
        """Parse method ID from a string identifier."""
        if not identifier:
            return None
        identifier = identifier.strip()

        # Direct integer string
        try:
            return int(identifier)
        except ValueError:
            pass

        # Format: "method:123" or "cont:123"
        if ":" in identifier:
            prefix, suffix = identifier.split(":", 1)
            if prefix in {"method", "cont"}:
                try:
                    return int(suffix)
                except ValueError:
                    pass

        # Format: "method_N" or "cont_N"
        if "_" in identifier:
            parts = identifier.rsplit("_", 1)
            try:
                return int(parts[-1])
            except ValueError:
                pass

        # Format: "main" -> 0
        if identifier == "main":
            return 0

        return None

    def _assign_semantic_labels(self, module: TVMModule) -> None:
        """Assign semantic labels to instructions based on opcode patterns."""
        for inst in module.all_instructions():
            labels: List[SemanticLabel] = []

            # Message sending
            if inst.kind == InstructionKind.SEND_MESSAGE:
                labels.append(SemanticLabel.MESSAGE_SEND)
                # TOKEN_TRANSFER requires further analysis of value (e.g., preceding STGRAMS)
                # to distinguish zero-value notifications from actual value transfers

            # Gas accept
            if inst.kind == InstructionKind.ACCEPT:
                labels.append(SemanticLabel.GAS_ACCEPT)

            # Authentication (signature verification) - include BLS and ECRECOVER
            upper = inst.opcode.upper()
            if upper in ("CHKSIGNU", "CHKSIGNS", "P256_CHKSIGNU", "P256_CHKSIGNS",
                         "BLS_VERIFY", "BLS_AGGREGATEVERIFY",
                         "BLS_FASTAGGREGATEVERIFY", "ECRECOVER"):
                labels.append(SemanticLabel.SIGNATURE_VERIFY)
                labels.append(SemanticLabel.AUTH_CHECK)

            # Hash compute - expand to cover all HASHEXT variants
            if upper in ("SHA256U", "HASHCU", "HASHSU", "HASHBU") or upper.startswith("HASHEXT"):
                labels.append(SemanticLabel.HASH_COMPUTE)

            # Guard pattern (align with analyzer event map)
            if upper in CONFIG_GUARD_OPCODES:
                labels.append(SemanticLabel.GUARD)

            # Storage read (c4 access)
            if upper in ("PUSHCTR", "PUSH_CTR"):
                args = inst.original_args or []
                if any(getattr(a, 'value', None) == 4 for a in args) or any(str(a) == "4" for a in args):
                    labels.append(SemanticLabel.STORAGE_READ)
            elif upper == "PUSHROOT":
                labels.append(SemanticLabel.STORAGE_READ)

            # Storage write (c4 access)
            if upper in ("POPCTR", "POP_CTR"):
                args = inst.original_args or []
                if any(getattr(a, 'value', None) == 4 for a in args) or any(str(a) == "4" for a in args):
                    labels.append(SemanticLabel.STORAGE_WRITE)
            elif upper == "POPROOT":
                labels.append(SemanticLabel.STORAGE_WRITE)

            # Message parsing
            if upper in ("LDMSGADDR", "LDMSGADDRQ", "LDGRAMS",
                         "PARSEMSGADDR", "PARSEMSGADDRQ",
                         "LDSTDADDR", "LDSTDADDRQ",
                         "REWRITESTDADDR", "REWRITESTDADDRQ",
                         "INMSG_SRC", "INMSG_VALUE", "INMSG_FWDFEE"):
                labels.append(SemanticLabel.MESSAGE_PARSE)

            # Bounced check
            if upper == "INMSG_BOUNCED":
                labels.append(SemanticLabel.BOUNCED_CHECK)
            # Bounced check pattern (LDU 4 or PLDU 4 followed by bit extraction)
            elif upper in ("LDU", "PLDU") and "4" in str(inst.original_args):
                labels.append(SemanticLabel.BOUNCED_CHECK)

            # Random generation (actual RNG output)
            if upper in ("RANDU256", "RAND"):
                labels.append(SemanticLabel.RANDOM_GENERATE)
            # Random seed (seed mixing/setting - includes RANDOMIZE)
            if upper in ("SETRAND", "ADDRAND", "RANDOMIZE"):
                labels.append(SemanticLabel.RANDOM_SEED)

            # Reserve balance
            if upper in ("RAWRESERVE", "RAWRESERVEX"):
                labels.append(SemanticLabel.RESERVE_BALANCE)

            # Dict dispatch - cover all dictionary-based dispatch patterns
            if ("DICT" in upper and ("JMP" in upper or "EXEC" in upper or "GETJMP" in upper)):
                labels.append(SemanticLabel.DICT_DISPATCH)
            elif upper == "JMPDICT":
                labels.append(SemanticLabel.DICT_DISPATCH)

            # Exception handler
            if upper in ("TRY", "TRYARGS"):
                labels.append(SemanticLabel.EXCEPTION_HANDLER)

            # Quiet arithmetic (QADD, QSUB, QMUL, QDIV, QFITS, etc.)
            ir_type = OPCODE_TO_IR_TYPE.get(inst.opcode)
            if ir_type in (IRType.ARITHMETIC_QUIET, IRType.QUIET_ADD,
                           IRType.QUIET_MUL, IRType.QUIET_FIT):
                labels.append(SemanticLabel.QUIET_ARITHMETIC)

            if not labels:
                labels.append(SemanticLabel.UNKNOWN)

            inst.semantic_labels = labels
