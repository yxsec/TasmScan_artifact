"""
IR Builder - Converts AnalysisFacts to TVMModule representation.

Provides build_tasir() API producing TVMModule with SaveList support
for precise cross-continuation data flow analysis.
"""
from ..analyzer.facts import AnalysisFacts
from .ir_lifter import IRLifter
from .tasir_types import TVMModule
from .solver_ir import SolverIRLowering, SolverModule

try:
    from .continuation_linker import ContinuationLinker
    _HAS_CONTINUATION_LINKER = True
except ImportError:
    _HAS_CONTINUATION_LINKER = False


class IRBuilder:
    """
    Builds TVMModule from AnalysisFacts.

    Features:
    - Produces TVMModule with SaveList for cross-continuation analysis
    - Links continuation references for complete control flow
    """

    def build_tasir(
        self,
        facts: AnalysisFacts,
        link_continuations: bool = True,
        skip_savelists: bool = False,
    ) -> TVMModule:
        """
        Build complete TASIR representation.

        This is the recommended API that produces a fully-linked
        TVMModule with SaveList information for precise cross-continuation
        data flow analysis.

        Args:
            facts: Analysis facts from ProgramAnalyzer
            link_continuations: Whether to run continuation linking pass
                (requires continuation_linker module)

        Returns:
            Complete TVMModule representation

        Raises:
            ImportError: If link_continuations=True but continuation_linker
                module is not available

        Example:
            >>> builder = IRBuilder()
            >>> module = builder.build_tasir(facts)
            >>> print(f"Module has {len(module.functions)} functions")
        """
        # Step 1: Lift to raw TASIR
        lifter = IRLifter()
        module = lifter.lift(facts)

        # Step 2: Link continuations (optional but recommended)
        if link_continuations:
            if not _HAS_CONTINUATION_LINKER:
                raise ImportError(
                    "continuation_linker module is not available. "
                    "Set link_continuations=False or install the module."
                )
            linker = ContinuationLinker(skip_savelists=skip_savelists)
            module = linker.link(module)

        # Keep source AnalysisFacts attached so adapter-based consumers can
        # retain richer metadata without requiring caller-side monkey patching.
        module._facts = facts

        return module

    def lower_tasir_to_solver_ir(self, module: TVMModule) -> SolverModule:
        """Lower an existing TASIR module into Solver IR."""
        lowering = SolverIRLowering()
        return lowering.lower(module)

    def build_solver_ir(
        self,
        facts: AnalysisFacts,
        link_continuations: bool = True,
    ) -> SolverModule:
        """Build TASIR first, then lower to Solver IR."""
        module = self.build_tasir(facts, link_continuations=link_continuations)
        return self.lower_tasir_to_solver_ir(module)
