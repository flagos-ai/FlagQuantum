# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/repomix/repomix-driver.py.

"""Pack selected native component sources and rendered Markdown for development."""

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from tools.development import process
from tools.development.repository import get_git_root

# Order follows the twenty original selections, with split native owners kept together.
SOURCE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "integer-addition",
        (
            "flagquantum/drawer/addition_symbols.py",
            "flagquantum/simulation/classical_addition.py",
            "flagquantum/compiler/registers/arithmetic/constant_addition_component.py",
            "flagquantum/compiler/registers/arithmetic/addition_width.py",
            "flagquantum/compiler/registers/arithmetic/addition_component.py",
            "flagquantum/compiler/registers/arithmetic/out_of_place_addition.py",
            "flagquantum/compiler/registers/arithmetic/constant_addition_width.py",
            "flagquantum/compiler/registers/arithmetic/out_of_place_add_resource_model.py",
            "flagquantum/compiler/registers/arithmetic/addition_structure.py",
            "flagquantum/compiler/registers/arithmetic/out_of_place_addition_width.py",
            "flagquantum/compiler/registers/arithmetic/constant_add_resource_model.py",
            "flagquantum/compiler/registers/arithmetic/arithmetic_resource_model.py",
            "flagquantum/compiler/registers/arithmetic/addition.py",
            "flagquantum/compiler/registers/arithmetic/out_of_place_addition_component.py",
            "flagquantum/compiler/registers/resources/symbolic_resource_model.py",
            "flagquantum/compiler/registers/arithmetic/out_of_place_addition_structure.py",
            "flagquantum/compiler/registers/arithmetic/physical_addition.py",
            "flagquantum/compiler/registers/arithmetic/inplace_addition.py",
            "flagquantum/compiler/registers/arithmetic/physical_inplace_addition.py",
        ),
    ),
    (
        "integer-and-fixed-point-multiplication",
        (
            "flagquantum/compiler/registers/fixed_point/reciprocal.py",
            "flagquantum/compiler/registers/fixed_point/reciprocal_newton.py",
            "flagquantum/compiler/registers/fixed_point/reciprocal_newton_component.py",
            "flagquantum/compiler/registers/fixed_point/fixed_point_square.py",
            "flagquantum/compiler/registers/fixed_point/fixed_point_scaling.py",
            "flagquantum/compiler/registers/fixed_point/reciprocal_component.py",
            "flagquantum/compiler/registers/fixed_point/reciprocal_seed_component.py",
            "flagquantum/compiler/registers/fixed_point/reciprocal_seed.py",
            "flagquantum/simulation/classical_product_accumulation.py",
            "flagquantum/simulation/classical_square.py",
            "flagquantum/compiler/registers/arithmetic/integer_product.py",
            "flagquantum/compiler/registers/arithmetic/product_accumulation.py",
            "flagquantum/compiler/registers/arithmetic/arithmetic_resource_model.py",
            "flagquantum/compiler/registers/arithmetic/integer_square.py",
            "flagquantum/compiler/registers/arithmetic/product_accumulation_component.py",
            "flagquantum/compiler/registers/arithmetic/integer_product_component.py",
            "flagquantum/compiler/registers/arithmetic/integer_square_component.py",
            "flagquantum/compiler/registers/arithmetic/sum_of_squares_component.py",
            "flagquantum/compiler/registers/fixed_point/fixed_point_scaling_component.py",
            "flagquantum/compiler/registers/fixed_point/fixed_point_product_component.py",
            "flagquantum/compiler/registers/fixed_point/fixed_point_square_component.py",
        ),
    ),
    (
        "integer-comparison",
        (
            "flagquantum/simulation/classical_half_comparison.py",
            "flagquantum/compiler/registers/comparison/_linear_words.py",
            "flagquantum/compiler/registers/comparison/controlled_comparison.py",
            "flagquantum/compiler/registers/comparison/less_than.py",
            "flagquantum/compiler/registers/comparison/ordering.py",
            "flagquantum/compiler/registers/comparison/comparison_tree.py",
            "flagquantum/compiler/registers/comparison/ordering_component.py",
            "flagquantum/compiler/registers/comparison/half_comparison.py",
            "flagquantum/compiler/registers/comparison/half_ordering.py",
            "flagquantum/compiler/registers/comparison/equality.py",
            "flagquantum/compiler/registers/comparison/controlled_component.py",
            "flagquantum/compiler/registers/comparison/comparison_resource_model.py",
            "flagquantum/compiler/registers/comparison/half_component.py",
            "flagquantum/compiler/registers/comparison/linear_comparison.py",
            "flagquantum/compiler/registers/comparison/equality_component.py",
            "flagquantum/compiler/registers/comparison/tree_component.py",
            "flagquantum/compiler/registers/comparison/constant_component.py",
            "flagquantum/compiler/registers/comparison/linear_component.py",
            "flagquantum/compiler/registers/comparison/less_equal.py",
            "flagquantum/simulation/classical_comparison.py",
            "flagquantum/simulation/classical_greater_comparison.py",
        ),
    ),
    (
        "hadamard",
        (
            "flagquantum/drawer/hadamard_symbols.py",
            "flagquantum/compiler/registers/control/multi_control.py",
            "flagquantum/compiler/registers/primitives/controlled_hadamard.py",
            "flagquantum/core/operator_schema.py",
        ),
    ),
    (
        "toffoli",
        (
            "flagquantum/drawer/controlled_not_symbols.py",
            "flagquantum/compiler/registers/primitives/controlled_not.py",
        ),
    ),
    (
        "workspace-allocation",
        (
            "flagquantum/compiler/registers/layout/workspace.py",
            "flagquantum/compiler/registers/layout/component_workspace.py",
            "flagquantum/core/ir/program/model.py",
        ),
    ),
    (
        "register-join",
        (
            "flagquantum/drawer/partition_symbols.py",
            "flagquantum/compiler/registers/layout/partition.py",
            "flagquantum/compiler/registers/layout/component_partition.py",
            "flagquantum/core/ir/program/model.py",
        ),
    ),
    (
        "qrom",
        (
            "flagquantum/drawer/qrom_symbols.py",
            "flagquantum/compiler/registers/resources/symbolic_resource_model.py",
            "flagquantum/compiler/registers/lookup/source_qrom.py",
            "flagquantum/compiler/registers/lookup/qrom.py",
            "flagquantum/compiler/registers/lookup/qrom_table.py",
            "flagquantum/compiler/registers/lookup/qrom_resource_model.py",
        ),
    ),
    (
        "uniform-preparation",
        (
            "flagquantum/compiler/registers/preparation/uniform_preparation.py",
            "flagquantum/compiler/registers/preparation/uniform_preparation_resource_model.py",
        ),
    ),
    (
        "alias-preparation",
        (
            "flagquantum/algorithms/primitives/alias_preparation.py",
            "flagquantum/algorithms/primitives/symbolic_alias_preparation.py",
            "flagquantum/algorithms/primitives/alias_discretization.py",
            "flagquantum/algorithms/primitives/alias_sampling.py",
            "flagquantum/algorithms/primitives/factor_alias_preparation.py",
        ),
    ),
    (
        "textbook-qft",
        (
            "flagquantum/algorithms/primitives/qft.py",
            "flagquantum/algorithms/primitives/qft_resource_model.py",
        ),
    ),
    (
        "hamming-weight-phasing",
        (
            "flagquantum/compiler/registers/phase/hamming_weight_phasing.py",
            "flagquantum/compiler/registers/phase/phase_gradient.py",
            "flagquantum/compiler/registers/bits/hamming_weight.py",
        ),
    ),
    (
        "unitary-block-encoding",
        ("flagquantum/algorithms/primitives/unitary_block_encoding.py",),
    ),
    (
        "linear-combination-block-encoding",
        (
            "flagquantum/algorithms/primitives/linear_combination_preparation.py",
            "flagquantum/algorithms/primitives/linear_combination_select.py",
            "flagquantum/algorithms/primitives/linear_combination_block_encoding.py",
            "flagquantum/algorithms/primitives/linear_combination.py",
        ),
    ),
    (
        "product-block-encoding",
        ("flagquantum/algorithms/primitives/product_block_encoding.py",),
    ),
    (
        "qubitization-walk",
        (
            "flagquantum/algorithms/primitives/double_factorization_walk.py",
            "flagquantum/algorithms/primitives/qubitization_walk.py",
            "flagquantum/algorithms/primitives/prepared_state_reflection.py",
        ),
    ),
    (
        "textbook-phase-estimation",
        (
            "flagquantum/algorithms/primitives/textbook_phase_estimation.py",
            "flagquantum/algorithms/primitives/phase_estimation_window.py",
            "flagquantum/algorithms/primitives/_block_binding.py",
            "flagquantum/algorithms/primitives/black_box_prepare.py",
        ),
    ),
    (
        "thc-preparation",
        (
            "flagquantum/algorithms/primitives/thc_chemistry_data.py",
            "flagquantum/algorithms/primitives/thc_chemistry_prepare.py",
            "flagquantum/algorithms/primitives/thc_uniform_preparation.py",
        ),
    ),
    (
        "hubbard-select",
        (
            "flagquantum/drawer/hubbard_symbols.py",
            "flagquantum/algorithms/primitives/hubbard_qubitization_resource_model.py",
            "flagquantum/algorithms/primitives/hubbard_majorana.py",
            "flagquantum/algorithms/primitives/hubbard_spin_up.py",
            "flagquantum/algorithms/primitives/hubbard_select.py",
        ),
    ),
    (
        "conjunction",
        (
            "flagquantum/drawer/conjunction_symbols.py",
            "flagquantum/compiler/registers/primitives/and_compute.py",
            "flagquantum/compiler/registers/control/multi_and_width.py",
            "flagquantum/compiler/registers/control/conjunction_program.py",
            "flagquantum/compiler/registers/control/multi_control.py",
            "flagquantum/compiler/registers/arithmetic/addition.py",
            "flagquantum/simulation/classical_and.py",
            "flagquantum/simulation/tensor_network/conjunction.py",
        ),
    ),
)


def source_paths() -> tuple[str, ...]:
    """Return each selected native path once, preserving group order."""
    return tuple(dict.fromkeys(path for _, paths in SOURCE_GROUPS for path in paths))


def pack_context(
    root: Path, *, output: Path | None = None, executable: str = "repomix"
) -> int:
    """Run the installed packer over native sources and all docs Markdown.

    Relative output paths are interpreted from the checkout root. Render reference
    pages and notebook Markdown before packing. Existing/stale Markdown remains
    part of the input, preserving the source selection rule.
    """
    root = root.resolve(strict=True)
    paths = source_paths()
    for path in paths:
        if not (root / path).is_file():
            raise FileNotFoundError(f"Selected context source is missing: {path}")
    if output is None:
        output = (
            Path("tools/development/context")
            / f"flagquantum-context-{date.today().isoformat()}.md"
        )
    return int(
        process.run(
            [
                executable,
                "--no-gitignore",
                "--include",
                ",".join((*paths, "docs/**/*.md")),
                "--output",
                str(output),
            ],
            cwd=root,
            check=False,
            timeout=1200,
            log_run_to_stderr=False,
        ).returncode
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--repomix", default="repomix", help="Installed Repomix executable"
    )
    arguments = parser.parse_args(argv)
    raise SystemExit(
        pack_context(
            get_git_root(), output=arguments.output, executable=arguments.repomix
        )
    )


if __name__ == "__main__":
    main()
