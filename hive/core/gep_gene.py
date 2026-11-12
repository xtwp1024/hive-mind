# -*- coding: utf-8 -*-
"""
GEP Gene Expression Programming - Tree-Structured Gene Evolution

This module implements True GEP (Gene Expression Programming) with tree-structured
genes replacing the flat float-vector DNA. Each gene is an expression tree that can
represent complex decision logic and algorithms.

GEP Architecture:
- Gene: A single expression tree (subtree)
- Chromosome: Multiple genes linked by a connecting function (AND/OR)
- Genome: The complete evolutionary unit with multiple chromosomes

Key Concepts:
- Terminals: Leaf nodes (task_type, metric_name, threshold, action)
- Functions: Internal nodes (IF, AND, OR, NOT, GT, LT, EQ, ADD, SUB, MUL, DIV)
- Each gene represents a strategy sub-component
- Chromosomes combine multiple genes into complete strategies
"""

import copy
import logging
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

logger = logging.getLogger("Hive.GEPGene")


# ============================================================================
# Function and Terminal Definitions
# ============================================================================


class FunctionType(str, Enum):
    """Function node types for expression trees."""

    # Conditional
    IF = "IF"  # if(condition, then, else)

    # Logical
    AND = "AND"  # and(left, right)
    OR = "OR"  # or(left, right)
    NOT = "NOT"  # not(value)

    # Comparison
    GT = "GT"  # greater than
    LT = "LT"  # less than
    EQ = "EQ"  # equal

    # Arithmetic
    ADD = "ADD"  # addition
    SUB = "SUB"  # subtraction
    MUL = "MUL"  # multiplication
    DIV = "DIV"  # division (protected)

    # Aggregation
    MAX = "MAX"  # max of two values
    MIN = "MIN"  # min of two values


class TerminalType(str, Enum):
    """Terminal (leaf) node types for expression trees."""

    # Task context
    TASK_TYPE = "TASK_TYPE"  # Type of current task
    TASK_COMPLEXITY = "TASK_COMPLEXITY"

    # Metrics
    METRIC_SUCCESS_RATE = "METRIC_SUCCESS_RATE"
    METRIC_ERROR_COUNT = "METRIC_ERROR_COUNT"
    METRIC_AVG_TIME = "METRIC_AVG_TIME"
    METRIC_MEMORY_USAGE = "METRIC_MEMORY_USAGE"
    METRIC_CPU_USAGE = "METRIC_CPU_USAGE"

    # Thresholds (numeric constants)
    THRESHOLD = "THRESHOLD"  # Numeric constant

    # Actions (output values)
    ACTION_PRIORITY = "ACTION_PRIORITY"
    ACTION_STRATEGY = "ACTION_STRATEGY"
    ACTION_RESOURCE_BUDGET = "ACTION_RESOURCE_BUDGET"


# Function arity (number of children)
FUNCTION_ARITY: dict[FunctionType, int] = {
    FunctionType.IF: 3,  # condition, then, else
    FunctionType.AND: 2,
    FunctionType.OR: 2,
    FunctionType.NOT: 1,
    FunctionType.GT: 2,
    FunctionType.LT: 2,
    FunctionType.EQ: 2,
    FunctionType.ADD: 2,
    FunctionType.SUB: 2,
    FunctionType.MUL: 2,
    FunctionType.DIV: 2,  # protected division
    FunctionType.MAX: 2,
    FunctionType.MIN: 2,
}

# All available functions
ALL_FUNCTIONS = list(FunctionType)

# Terminal generators - returns available terminal types for a category
TERMINAL_CATEGORIES = {
    "metrics": [
        TerminalType.METRIC_SUCCESS_RATE,
        TerminalType.METRIC_ERROR_COUNT,
        TerminalType.METRIC_AVG_TIME,
        TerminalType.METRIC_MEMORY_USAGE,
        TerminalType.METRIC_CPU_USAGE,
    ],
    "tasks": [
        TerminalType.TASK_TYPE,
        TerminalType.TASK_COMPLEXITY,
    ],
    "actions": [
        TerminalType.ACTION_PRIORITY,
        TerminalType.ACTION_STRATEGY,
        TerminalType.ACTION_RESOURCE_BUDGET,
    ],
    "constants": [
        TerminalType.THRESHOLD,
    ],
}


# ============================================================================
# Gene Node Structure
# ============================================================================


@dataclass
class GeneNode:
    """
    A single node in a gene expression tree.

    Can be either a Function node (internal) or Terminal node (leaf).
    """

    is_function: bool
    function_type: Optional[FunctionType] = None
    terminal_type: Optional[TerminalType] = None
    value: Any = None  # For THRESHOLD constants
    children: list["GeneNode"] = field(default_factory=list)

    def __post_init__(self):
        if self.is_function and self.function_type is None:
            raise ValueError("Function node requires function_type")
        if not self.is_function and self.terminal_type is None:
            raise ValueError("Terminal node requires terminal_type")

    @property
    def arity(self) -> int:
        """Number of children this node expects."""
        if not self.is_function:
            return 0
        return FUNCTION_ARITY.get(self.function_type, 0)

    def get_depth(self) -> int:
        """Calculate depth of this node (leaf = 0)."""
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children)

    def __repr__(self) -> str:
        if self.is_function:
            return f"Func({self.function_type.value})"
        if self.terminal_type == TerminalType.THRESHOLD:
            return f"Term({self.terminal_type.value}, {self.value})"
        return f"Term({self.terminal_type.value})"

    def to_readable(self) -> str:
        """Convert node to readable expression string."""
        if not self.is_function:
            if self.terminal_type == TerminalType.THRESHOLD:
                return str(self.value)
            return f"${self.terminal_type.value}"

        func = self.function_type.value
        if func == "NOT":
            return f"(NOT {self.children[0].to_readable()})"
        if func == "IF":
            return f"(IF {self.children[0].to_readable()} THEN {self.children[1].to_readable()} ELSE {self.children[2].to_readable()})"

        # Binary functions
        if len(self.children) == 2:
            left = self.children[0].to_readable()
            right = self.children[1].to_readable()
            return f"({left} {func} {right})"
        elif len(self.children) == 1:
            return f"({func} {self.children[0].to_readable()})"
        return f"({func})"


# ============================================================================
# GEP Gene (Single Expression Tree)
# ============================================================================


@dataclass
class GEPGene:
    """
    A single GEP gene represented as an expression tree.

    Each gene represents a strategy sub-component that can be combined
    with other genes to form a complete chromosome.

    Attributes:
        root: Root node of the expression tree
        max_depth: Maximum allowed tree depth (default 10)
        id: Unique identifier for this gene
        fitness: Current fitness score (0.0-1.0)
    """

    root: GeneNode
    max_depth: int = 10
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fitness: float = 0.0

    def __post_init__(self):
        if not self.validate_structure():
            raise ValueError(f"Invalid gene structure: depth exceeds {self.max_depth}")

    def validate_structure(self) -> bool:
        """Ensure gene structure is valid (depth within limits, arity correct)."""
        if self.root.get_depth() > self.max_depth:
            return False
        return self._validate_node(self.root)

    def _validate_node(self, node: GeneNode) -> bool:
        """Recursively validate a node and its children."""
        if node.is_function:
            # Check arity
            expected_arity = node.arity
            if len(node.children) != expected_arity:
                logger.warning(
                    f"Node {node.function_type} has wrong arity: expected {expected_arity}, got {len(node.children)}"
                )
                return False
            # Recursively validate children
            for child in node.children:
                if not self._validate_node(child):
                    return False
        return True

    def evaluate(self, context: dict[str, Any]) -> Any:
        """
        Evaluate this gene against a task context.

        Args:
            context: Dictionary with metric values and task info

        Returns:
            Evaluated result (typically boolean for conditions, numeric for actions)
        """
        return self._evaluate_node(self.root, context)

    def _evaluate_node(self, node: GeneNode, context: dict[str, Any]) -> Any:
        """Recursively evaluate a node."""
        if not node.is_function:
            return self._get_terminal_value(node, context)

        func = node.function_type
        children_values = [self._evaluate_node(child, context) for child in node.children]

        if func == FunctionType.IF:
            condition, then_val, else_val = children_values
            return then_val if condition else else_val

        elif func == FunctionType.AND:
            return children_values[0] and children_values[1]

        elif func == FunctionType.OR:
            return children_values[0] or children_values[1]

        elif func == FunctionType.NOT:
            return not children_values[0]

        elif func == FunctionType.GT:
            try:
                return float(children_values[0]) > float(children_values[1])
            except (TypeError, ValueError):
                return False

        elif func == FunctionType.LT:
            try:
                return float(children_values[0]) < float(children_values[1])
            except (TypeError, ValueError):
                return False

        elif func == FunctionType.EQ:
            try:
                return float(children_values[0]) == float(children_values[1])
            except (TypeError, ValueError):
                return children_values[0] == children_values[1]

        elif func == FunctionType.ADD:
            try:
                return float(children_values[0]) + float(children_values[1])
            except (TypeError, ValueError):
                return 0.0

        elif func == FunctionType.SUB:
            try:
                return float(children_values[0]) - float(children_values[1])
            except (TypeError, ValueError):
                return 0.0

        elif func == FunctionType.MUL:
            try:
                return float(children_values[0]) * float(children_values[1])
            except (TypeError, ValueError):
                return 0.0

        elif func == FunctionType.DIV:
            # Protected division
            try:
                divisor = float(children_values[1])
                if abs(divisor) < 1e-10:
                    return float(children_values[0])  # Return numerator if divisor is near zero
                return float(children_values[0]) / divisor
            except (TypeError, ValueError):
                return 0.0

        elif func == FunctionType.MAX:
            try:
                return max(float(children_values[0]), float(children_values[1]))
            except (TypeError, ValueError):
                return 0.0

        elif func == FunctionType.MIN:
            try:
                return min(float(children_values[0]), float(children_values[1]))
            except (TypeError, ValueError):
                return 0.0

        raise ValueError(f"Unknown function type: {func}")

    def _get_terminal_value(self, node: GeneNode, context: dict[str, Any]) -> Any:
        """Get the value of a terminal node from context."""
        term = node.terminal_type

        if term == TerminalType.THRESHOLD:
            return node.value

        elif term == TerminalType.TASK_TYPE:
            return context.get("task_type", "unknown")

        elif term == TerminalType.TASK_COMPLEXITY:
            return context.get("task_complexity", 0.5)

        elif term == TerminalType.METRIC_SUCCESS_RATE:
            return context.get("success_rate", 0.5)

        elif term == TerminalType.METRIC_ERROR_COUNT:
            return context.get("error_count", 0)

        elif term == TerminalType.METRIC_AVG_TIME:
            return context.get("avg_time", 10.0)

        elif term == TerminalType.METRIC_MEMORY_USAGE:
            return context.get("memory_usage", 0.5)

        elif term == TerminalType.METRIC_CPU_USAGE:
            return context.get("cpu_usage", 0.5)

        elif term == TerminalType.ACTION_PRIORITY:
            return context.get("action_priority", 0.5)

        elif term == TerminalType.ACTION_STRATEGY:
            return context.get("action_strategy", "default")

        elif term == TerminalType.ACTION_RESOURCE_BUDGET:
            return context.get("resource_budget", 1.0)

        return 0.5

    def clone(self) -> "GEPGene":
        """Create a deep copy of this gene."""
        new_root = self._clone_node(self.root)
        return GEPGene(
            root=new_root, max_depth=self.max_depth, id=str(uuid.uuid4())[:8], fitness=0.0
        )

    def _clone_node(self, node: GeneNode) -> GeneNode:
        """Recursively clone a node."""
        new_children = [self._clone_node(child) for child in node.children]
        return GeneNode(
            is_function=node.is_function,
            function_type=node.function_type,
            terminal_type=node.terminal_type,
            value=copy.deepcopy(node.value),
            children=new_children,
        )

    def __repr__(self) -> str:
        return f"GEPGene(id={self.id}, depth={self.root.get_depth()}, fitness={self.fitness:.3f}, expr={self.root.to_readable()})"


# ============================================================================
# GEP Chromosome (Collection of Genes)
# ============================================================================


@dataclass
class GEPChromosome:
    """
    A GEP chromosome containing multiple genes linked by a connecting function.

    The chromosome represents a complete strategy where:
    - Each gene is an independent expression tree
    - Genes are combined using the linking_function (AND/OR)
    - Only the linking function's output is used as the chromosome output

    Attributes:
        genes: List of genes in this chromosome
        linking_function: Function to combine gene outputs (AND or OR)
        max_genes: Maximum genes per chromosome (default 5)
        id: Unique identifier
        fitness: Current fitness score
    """

    genes: list[GEPGene] = field(default_factory=list)
    linking_function: FunctionType = FunctionType.AND
    max_genes: int = 5
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fitness: float = 0.0

    def __post_init__(self):
        if len(self.genes) > self.max_genes:
            raise ValueError(f"Too many genes: {len(self.genes)} > {self.max_genes}")
        if self.linking_function not in [FunctionType.AND, FunctionType.OR]:
            raise ValueError("Linking function must be AND or OR")

    def add_gene(self, gene: GEPGene) -> bool:
        """Add a gene if within max_genes limit."""
        if len(self.genes) >= self.max_genes:
            return False
        self.genes.append(gene)
        return True

    def evaluate(self, context: dict[str, Any]) -> Any:
        """
        Evaluate all genes and combine with linking function.

        Args:
            context: Task context for evaluation

        Returns:
            Combined result from linking function
        """
        if not self.genes:
            return True

        results = [gene.evaluate(context) for gene in self.genes]

        if self.linking_function == FunctionType.AND:
            return all(results)
        elif self.linking_function == FunctionType.OR:
            return any(results)

        return results[0] if results else True

    def get_gene_expressions(self) -> list[str]:
        """Get readable expressions for all genes."""
        return [gene.root.to_readable() for gene in self.genes]

    def clone(self) -> "GEPChromosome":
        """Create a deep copy of this chromosome."""
        return GEPChromosome(
            genes=[gene.clone() for gene in self.genes],
            linking_function=self.linking_function,
            max_genes=self.max_genes,
            id=str(uuid.uuid4())[:8],
            fitness=0.0,
        )

    def __repr__(self) -> str:
        gene_summaries = [g.id for g in self.genes]
        return f"GEPChromosome(id={self.id}, genes={len(self.genes)}, link={self.linking_function.value}, fitness={self.fitness:.3f})"


# ============================================================================
# Genetic Operators
# ============================================================================


class GEPOperators:
    """
    Genetic operators for GEP evolution.

    Includes crossover, mutation, and validation operators
    that preserve gene validity.
    """

    def __init__(
        self,
        max_gene_depth: int = 10,
        max_genes_per_chromosome: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        self.max_gene_depth = max_gene_depth
        self.max_genes_per_chromosome = max_genes_per_chromosome
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def point_mutation(self, gene: GEPGene, rate: Optional[float] = None) -> GEPGene:
        """
        Point mutation: randomly mutate individual nodes in the tree.

        Mutation types:
        - Function node: change to another function with same arity
        - Terminal node: change to another terminal or generate new threshold value

        Args:
            gene: Gene to mutate
            rate: Mutation rate (probability per node)

        Returns:
            Mutated gene (may be same if no mutation occurred)
        """
        mutate_rate = rate if rate is not None else self.mutation_rate
        mutated_root = self._mutate_node(gene.root, mutate_rate)

        new_gene = GEPGene(
            root=mutated_root, max_depth=gene.max_depth, id=str(uuid.uuid4())[:8], fitness=0.0
        )
        return new_gene

    def _create_random_node(self, is_function: bool, arity: int) -> GeneNode:
        """Create a random node for mutation."""
        if is_function:
            # Create a random function node with children
            func_type = random.choice(ALL_FUNCTIONS)
            children = []
            for _ in range(FUNCTION_ARITY[func_type]):
                # For mutation, create terminals as children to avoid deep recursion
                child = GeneNode(
                    is_function=False, terminal_type=TerminalType.THRESHOLD, value=random.random()
                )
                children.append(child)
            return GeneNode(is_function=True, function_type=func_type, children=children)
        else:
            # Create a random terminal node
            terminal_type = random.choice(list(TerminalType))
            value = None
            if terminal_type == TerminalType.THRESHOLD:
                value = random.random()
            return GeneNode(is_function=False, terminal_type=terminal_type, value=value)

    def _mutate_node(self, node: GeneNode, rate: float) -> GeneNode:
        """Recursively apply point mutation to a node and its children."""
        # Decide if this node mutates
        if random.random() < rate:
            return self._create_random_node(node.is_function, node.arity if node.is_function else 0)

        # Otherwise, recursively mutate children
        if node.children:
            new_children = [self._mutate_node(child, rate) for child in node.children]
            return GeneNode(
                is_function=node.is_function,
                function_type=node.function_type,
                terminal_type=node.terminal_type,
                value=copy.deepcopy(node.value),
                children=new_children,
            )

        return node

    def subtree_crossover(self, gene1: GEPGene, gene2: GEPGene) -> tuple[GEPGene, GEPGene]:
        """
        Swap random subtrees between two genes.

        Selects a random subtree from each gene and swaps them.
        Preserves validity by ensuring depth constraints are met.

        Args:
            gene1: First gene
            gene2: Second gene

        Returns:
            Tuple of (mutated_gene1, mutated_gene2)
        """
        # Find all subtrees in each gene
        subtrees1 = self._find_all_subtrees(gene1.root)
        subtrees2 = self._find_all_subtrees(gene2.root)

        if not subtrees1 or not subtrees2:
            return gene1.clone(), gene2.clone()

        # Randomly select subtrees to swap
        subtree1 = random.choice(subtrees1)
        subtree2 = random.choice(subtrees2)

        # Create new genes with swapped subtrees
        new_root1 = self._swap_subtree(gene1.root, subtree1[0], subtree2[1], subtree2[0])
        new_root2 = self._swap_subtree(gene2.root, subtree2[0], subtree1[1], subtree1[0])

        # Validate and return
        try:
            new_gene1 = GEPGene(root=new_root1, max_depth=gene1.max_depth)
            new_gene2 = GEPGene(root=new_root2, max_depth=gene2.max_depth)
            return new_gene1, new_gene2
        except ValueError:
            # Swap resulted in invalid gene, return clones
            return gene1.clone(), gene2.clone()

    def _find_all_subtrees(self, node: GeneNode) -> list[tuple[GeneNode, GeneNode, str]]:
        """
        Find all possible subtrees in a gene tree.

        Returns:
            List of (parent, subtree_root, position) tuples
        """
        results = []

        def traverse(n: GeneNode, parent: Optional[GeneNode] = None, pos: str = ""):
            if n.children:
                for i, child in enumerate(n.children):
                    results.append((n, child, f"{pos}{i}"))
                    traverse(child, n, f"{pos}{i}")

        traverse(node)
        return results

    def _swap_subtree(
        self, root: GeneNode, old_subtree_parent: GeneNode, new_subtree: GeneNode, position: str
    ) -> GeneNode:
        """Replace a subtree at a given position with a new subtree."""
        root_copy = self._deep_copy(root)

        # Navigate to parent and replace child
        def replace_at_path(node: GeneNode, path: str) -> GeneNode:
            if not path:
                return self._deep_copy(new_subtree)

            idx = int(path[0])
            new_children = list(node.children)
            new_children[idx] = replace_at_path(node.children[idx], path[1:])
            return GeneNode(
                is_function=node.is_function,
                function_type=node.function_type,
                terminal_type=node.terminal_type,
                value=copy.deepcopy(node.value),
                children=new_children,
            )

        # Find and replace
        def find_and_replace(node: GeneNode) -> GeneNode:
            if id(node) == id(old_subtree_parent):
                return replace_at_path(node, position)

            if node.children:
                new_children = [find_and_replace(child) for child in node.children]
                return GeneNode(
                    is_function=node.is_function,
                    function_type=node.function_type,
                    terminal_type=node.terminal_type,
                    value=copy.deepcopy(node.value),
                    children=new_children,
                )
            return node

        return find_and_replace(root_copy)

    def _deep_copy(self, node: GeneNode) -> GeneNode:
        """Create a deep copy of a node tree."""
        return GeneNode(
            is_function=node.is_function,
            function_type=node.function_type,
            terminal_type=node.terminal_type,
            value=copy.deepcopy(node.value),
            children=[self._deep_copy(child) for child in node.children],
        )

    def gene_crossover(
        self, chrom1: GEPChromosome, chrom2: GEPChromosome
    ) -> tuple[GEPChromosome, GEPChromosome]:
        """
        Chromosome-level crossover: swap entire genes between chromosomes.

        Randomly selects genes from each chromosome and swaps them.

        Args:
            chrom1: First chromosome
            chrom2: Second chromosome

        Returns:
            Tuple of (new_chrom1, new_chrom2)
        """
        if not chrom1.genes or not chrom2.genes:
            return chrom1.clone(), chrom2.clone()

        # Select random genes to swap
        idx1 = random.randint(0, len(chrom1.genes) - 1)
        idx2 = random.randint(0, len(chrom2.genes) - 1)

        # Create copies
        new_genes1 = list(chrom1.genes)
        new_genes2 = list(chrom2.genes)

        # Swap
        new_genes1[idx1], new_genes2[idx2] = new_genes2[idx2].clone(), new_genes1[idx1].clone()

        new_chrom1 = GEPChromosome(
            genes=new_genes1, linking_function=chrom1.linking_function, max_genes=chrom1.max_genes
        )
        new_chrom2 = GEPChromosome(
            genes=new_genes2, linking_function=chrom2.linking_function, max_genes=chrom2.max_genes
        )

        return new_chrom1, new_chrom2

    def validate_gene(self, gene: GEPGene) -> bool:
        """
        Validate a gene for fitness evaluation.

        Checks:
        - Tree depth within limits
        - All nodes have correct arity
        - No infinite loops in evaluation

        Args:
            gene: Gene to validate

        Returns:
            True if valid, False otherwise
        """
        if gene.root.get_depth() > gene.max_depth:
            return False

        return self._check_node_validity(gene.root)

    def _check_node_validity(self, node: GeneNode) -> bool:
        """Recursively check node validity."""
        if node.is_function:
            if len(node.children) != node.arity:
                return False
            for child in node.children:
                if not self._check_node_validity(child):
                    return False
        return True


# ============================================================================
# Gene Factory - Random Gene Generation
# ============================================================================


class GEPGeneFactory:
    """
    Factory for creating random GEP genes and chromosomes.

    Used to generate initial population for evolution.
    """

    # Terminals that return numeric values (safe for comparisons/arithmetic)
    NUMERIC_TERMINALS = [
        TerminalType.METRIC_SUCCESS_RATE,
        TerminalType.METRIC_ERROR_COUNT,
        TerminalType.METRIC_AVG_TIME,
        TerminalType.METRIC_MEMORY_USAGE,
        TerminalType.METRIC_CPU_USAGE,
        TerminalType.THRESHOLD,
        TerminalType.TASK_COMPLEXITY,
    ]

    # All terminals
    ALL_TERMINALS = list(TerminalType)

    def __init__(self, max_depth: int = 10, max_genes: int = 5, function_ratio: float = 0.3):
        """
        Initialize factory.

        Args:
            max_depth: Maximum tree depth
            max_genes: Maximum genes per chromosome
            function_ratio: Ratio of function nodes to terminals (0.0-1.0)
        """
        self.max_depth = max_depth
        self.max_genes = max_genes
        self.function_ratio = function_ratio

    def create_random_gene(self, depth: Optional[int] = None) -> GEPGene:
        """
        Create a randomly structured gene.

        Args:
            depth: Target depth (uses max_depth if None)

        Returns:
            Random GEPGene
        """
        target_depth = depth if depth is not None else self.max_depth

        # Start with full tree generation (all positions filled)
        root = self._generate_tree(target_depth, 0, parent_function=None)

        return GEPGene(root=root, max_depth=self.max_depth)

    def _generate_tree(
        self, max_depth: int, current_depth: int, parent_function: Optional[FunctionType] = None
    ) -> GeneNode:
        """Generate a random tree with given depth constraints."""
        is_root = current_depth == 0

        # At root or deeper than function_ratio, prefer functions
        if current_depth < max_depth and (is_root or random.random() < self.function_ratio):
            # Generate function node
            func_type = random.choice(ALL_FUNCTIONS)
            arity = FUNCTION_ARITY[func_type]

            children = []
            for _ in range(arity):
                # For comparison/arithmetic functions, use numeric terminals
                # For IF/AND/OR, can use mixed
                child = self._generate_tree(max_depth, current_depth + 1, func_type)
                children.append(child)

            return GeneNode(is_function=True, function_type=func_type, children=children)
        else:
            # Generate terminal node
            return self._create_random_terminal(parent_function)

    def _create_random_terminal(self, parent_function: Optional[FunctionType] = None) -> GeneNode:
        """Create a random terminal node.

        For comparison and arithmetic functions, only use numeric terminals.
        """
        # Determine which terminals are valid for this parent
        if parent_function in [
            FunctionType.GT,
            FunctionType.LT,
            FunctionType.EQ,
            FunctionType.ADD,
            FunctionType.SUB,
            FunctionType.MUL,
            FunctionType.DIV,
            FunctionType.MAX,
            FunctionType.MIN,
        ]:
            # Only use numeric terminals for comparisons and arithmetic
            terminal_type = random.choice(self.NUMERIC_TERMINALS)
        else:
            # Can use any terminal
            terminal_type = random.choice(self.ALL_TERMINALS)

        value = None
        if terminal_type == TerminalType.THRESHOLD:
            # Generate random threshold value (0.0-1.0)
            value = random.random()

        return GeneNode(is_function=False, terminal_type=terminal_type, value=value)

    def create_random_chromosome(self, num_genes: Optional[int] = None) -> GEPChromosome:
        """
        Create a random chromosome with multiple genes.

        Args:
            num_genes: Number of genes (uses random if None, max 1-max_genes)

        Returns:
            Random GEPChromosome
        """
        if num_genes is None:
            num_genes = random.randint(1, self.max_genes)

        genes = [self.create_random_gene() for _ in range(num_genes)]

        linking_func = random.choice([FunctionType.AND, FunctionType.OR])

        return GEPChromosome(genes=genes, linking_function=linking_func, max_genes=self.max_genes)

    def create_homogeneous_chromosome(
        self, num_genes: int, gene_template: GEPGene
    ) -> GEPChromosome:
        """
        Create a chromosome with copies of the same gene.

        Args:
            num_genes: Number of genes
            gene_template: Gene to replicate

        Returns:
            Chromosome with identical genes
        """
        genes = [gene_template.clone() for _ in range(num_genes)]

        return GEPChromosome(
            genes=genes, linking_function=FunctionType.AND, max_genes=self.max_genes
        )


# ============================================================================
# Fitness Evaluation
# ============================================================================


class GEPFitnessEvaluator:
    """
    Evaluates GEP genes and chromosomes against task contexts.

    Provides fitness scoring based on:
    - Decision accuracy (for conditional genes)
    - Action quality (for action genes)
    - Task completion success
    """

    def __init__(self):
        pass

    def evaluate_gene(
        self, gene: GEPGene, context: dict[str, Any], expected_result: Any = None
    ) -> float:
        """
        Evaluate a single gene's fitness on a task context.

        Args:
            gene: Gene to evaluate
            context: Task context with metrics
            expected_result: Expected output (if known)

        Returns:
            Fitness score (0.0-1.0)
        """
        try:
            result = gene.evaluate(context)

            if expected_result is not None:
                # Supervised evaluation
                if isinstance(result, bool) and isinstance(expected_result, bool):
                    return 1.0 if result == expected_result else 0.0
                if isinstance(result, (int, float)) and isinstance(expected_result, (int, float)):
                    # Proximity score for numeric results
                    error = abs(result - expected_result)
                    return max(0.0, 1.0 - error)

            # Unsupervised: score based on decision confidence
            # Higher variance in decisions might indicate overfitting
            if isinstance(result, bool):
                return 0.7 if result else 0.3
            if isinstance(result, (int, float)):
                # Normalize to 0-1 range
                return max(0.0, min(1.0, result))

            return 0.5

        except Exception as e:
            logger.warning(f"Gene evaluation error: {e}")
            return 0.0

    def evaluate_chromosome(
        self,
        chromosome: GEPChromosome,
        contexts: list[dict[str, Any]],
        expected_results: Optional[list[Any]] = None,
    ) -> float:
        """
        Evaluate a chromosome across multiple task contexts.

        Args:
            chromosome: Chromosome to evaluate
            contexts: List of task contexts
            expected_results: Expected outputs (if available)

        Returns:
            Average fitness score (0.0-1.0)
        """
        if not contexts:
            return 0.5

        total_fitness = 0.0
        results = expected_results or [None] * len(contexts)

        for ctx, expected in zip(contexts, results):
            gene_fitnesses = []
            for gene in chromosome.genes:
                fitness = self.evaluate_gene(gene, ctx, expected)
                gene_fitnesses.append(fitness)

            # Combine gene fitnesses (average)
            if gene_fitnesses:
                total_fitness += sum(gene_fitnesses) / len(gene_fitnesses)

        avg_fitness = total_fitness / len(contexts)

        # Update chromosome fitness
        chromosome.fitness = avg_fitness
        for gene in chromosome.genes:
            gene.fitness = avg_fitness

        return avg_fitness

    def select_best(self, chromosomes: list[GEPChromosome], count: int = 1) -> list[GEPChromosome]:
        """
        Select top N chromosomes by fitness.

        Args:
            chromosomes: List of chromosomes
            count: Number to select

        Returns:
            Top N chromosomes sorted by fitness
        """
        sorted_chroms = sorted(chromosomes, key=lambda c: c.fitness, reverse=True)
        return sorted_chroms[:count]


# ============================================================================
# Serialization Utilities
# ============================================================================


def gene_to_dict(gene: GEPGene) -> dict:
    """Serialize a gene to dictionary."""
    return {
        "id": gene.id,
        "fitness": gene.fitness,
        "max_depth": gene.max_depth,
        "root": _node_to_dict(gene.root),
    }


def dict_to_gene(data: dict) -> GEPGene:
    """Deserialize a gene from dictionary."""
    return GEPGene(
        id=data["id"],
        fitness=data.get("fitness", 0.0),
        max_depth=data.get("max_depth", 10),
        root=_dict_to_node(data["root"]),
    )


def _node_to_dict(node: GeneNode) -> dict:
    """Serialize a node to dictionary."""
    return {
        "is_function": node.is_function,
        "function_type": node.function_type.value if node.function_type else None,
        "terminal_type": node.terminal_type.value if node.terminal_type else None,
        "value": node.value,
        "children": [_node_to_dict(child) for child in node.children],
    }


def _dict_to_node(data: dict) -> GeneNode:
    """Deserialize a node from dictionary."""
    return GeneNode(
        is_function=data["is_function"],
        function_type=FunctionType(data["function_type"]) if data.get("function_type") else None,
        terminal_type=TerminalType(data["terminal_type"]) if data.get("terminal_type") else None,
        value=data.get("value"),
        children=[_dict_to_node(child) for child in data.get("children", [])],
    )


def chromosome_to_dict(chrom: GEPChromosome) -> dict:
    """Serialize a chromosome to dictionary."""
    return {
        "id": chrom.id,
        "fitness": chrom.fitness,
        "linking_function": chrom.linking_function.value,
        "max_genes": chrom.max_genes,
        "genes": [gene_to_dict(g) for g in chrom.genes],
    }


def dict_to_chromosome(data: dict) -> GEPChromosome:
    """Deserialize a chromosome from dictionary."""
    return GEPChromosome(
        id=data["id"],
        fitness=data.get("fitness", 0.0),
        linking_function=FunctionType(data["linking_function"]),
        max_genes=data.get("max_genes", 5),
        genes=[dict_to_gene(g) for g in data["genes"]],
    )
