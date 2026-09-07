#!/usr/bin/env python3
"""
Cross-validation: MANUAL_STACK_EFFECTS vs cp0.json

Detects:
1. VALUE MISMATCH: manual and spec disagree on inputs/outputs
2. PRECISION LOSS: manual (non-dynamic) overrides spec's dynamic interval
3. MISSING IN SPEC: manual entries with no cp0.json counterpart (informational)

Usage:
    python scripts/validate_stack_effects.py
"""
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_cp0_effects():
    """Parse cp0.json and extract stack effects per mnemonic."""
    spec_path = PROJECT_ROOT / "tasmscan" / "spec" / "cp0.json"
    if not spec_path.exists():
        print(f"ERROR: cp0.json not found at {spec_path}")
        sys.exit(1)

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    effects = {}

    def count_stack(stack_list):
        """Count min/max stack entries, handling conditional/array types."""
        min_count = 0
        max_count = 0
        is_dynamic = False

        for entry in (stack_list or []):
            if not isinstance(entry, dict):
                min_count += 1
                max_count += 1
                continue

            entry_type = entry.get("type")
            if entry_type == "conditional":
                branch_counts = []
                for case in entry.get("match", []):
                    case_stack = case.get("stack", [])
                    c_min, c_max, _ = count_stack(case_stack)
                    branch_counts.append((c_min, c_max))
                if entry.get("else"):
                    c_min, c_max, _ = count_stack(entry["else"])
                    branch_counts.append((c_min, c_max))
                if branch_counts:
                    min_count += min(c[0] for c in branch_counts)
                    max_count += max(c[1] for c in branch_counts)
                    is_dynamic = True
                else:
                    min_count += 0
                    max_count += 0
                    is_dynamic = True
            elif entry_type == "array":
                is_dynamic = True
                # Can't determine fixed count
            else:
                min_count += 1
                max_count += 1

        return min_count, max_count, is_dynamic

    for instr in spec.get("instructions", []):
        mnemonic = instr.get("mnemonic")
        if not mnemonic:
            continue
        value_flow = instr.get("value_flow", {})
        if not value_flow:
            continue

        inputs_stack = value_flow.get("inputs", {}).get("stack", [])
        outputs_stack = value_flow.get("outputs", {}).get("stack", [])

        in_min, in_max, in_dyn = count_stack(inputs_stack)
        out_min, out_max, out_dyn = count_stack(outputs_stack)

        effects[mnemonic] = {
            "inputs": in_min,
            "max_inputs": in_max,
            "outputs": out_min,
            "max_outputs": out_max,
            "is_dynamic": in_dyn or out_dyn or (in_min != in_max) or (out_min != out_max),
        }

    # Also process aliases
    for alias in spec.get("aliases", []):
        mnemonic = alias.get("mnemonic")
        alias_of = alias.get("alias_of")
        if mnemonic and alias_of and alias_of in effects and mnemonic not in effects:
            effects[mnemonic] = dict(effects[alias_of])

    return effects


def load_manual_effects():
    """Load MANUAL_STACK_EFFECTS from the stack_effects module."""
    from tasmscan.ir.stack_effects import StackEffectDatabase
    return dict(StackEffectDatabase.MANUAL_STACK_EFFECTS)


def validate():
    """Run cross-validation and print report."""
    cp0 = load_cp0_effects()
    manual = load_manual_effects()

    value_mismatches = []
    precision_losses = []
    missing_in_spec = []
    consistent = []

    for mnemonic, (inputs, outputs, in_names, out_names, is_dynamic) in manual.items():
        spec = cp0.get(mnemonic)
        if spec is None:
            missing_in_spec.append(mnemonic)
            continue

        # Check value mismatch (compare minimum inputs/outputs)
        spec_inputs = spec["inputs"]
        spec_outputs = spec["outputs"]

        if inputs != spec_inputs or outputs != spec_outputs:
            value_mismatches.append({
                "mnemonic": mnemonic,
                "manual": (inputs, outputs),
                "spec": (spec_inputs, spec_outputs),
                "spec_dynamic": spec["is_dynamic"],
            })
        elif spec["is_dynamic"] and not is_dynamic:
            precision_losses.append({
                "mnemonic": mnemonic,
                "manual_outputs": outputs,
                "spec_min_outputs": spec["outputs"],
                "spec_max_outputs": spec["max_outputs"],
            })
        else:
            consistent.append(mnemonic)

    # Print report
    print("=" * 70)
    print("MANUAL_STACK_EFFECTS vs cp0.json Cross-Validation Report")
    print("=" * 70)

    if value_mismatches:
        print(f"\n{'!'*3} VALUE MISMATCHES ({len(value_mismatches)}):")
        for m in sorted(value_mismatches, key=lambda x: x["mnemonic"]):
            print(f"  {m['mnemonic']:30s} manual={m['manual']}  spec={m['spec']}  spec_dynamic={m['spec_dynamic']}")
    else:
        print("\n✓ No value mismatches found.")

    if precision_losses:
        print(f"\n{'!'*3} PRECISION LOSSES ({len(precision_losses)}):")
        print("  (Manual is non-dynamic but spec has conditional/array outputs)")
        for p in sorted(precision_losses, key=lambda x: x["mnemonic"]):
            print(f"  {p['mnemonic']:30s} manual_out={p['manual_outputs']}  "
                  f"spec_range=[{p['spec_min_outputs']},{p['spec_max_outputs']}]")
    else:
        print("\n✓ No precision losses found.")

    print(f"\n--- Summary ---")
    print(f"  Consistent:       {len(consistent)}")
    print(f"  Value mismatches: {len(value_mismatches)}")
    print(f"  Precision losses: {len(precision_losses)}")
    print(f"  Missing in spec:  {len(missing_in_spec)}")
    print(f"  Total manual:     {len(manual)}")
    print(f"  Total cp0:        {len(cp0)}")

    if value_mismatches or precision_losses:
        print(f"\n⚠ Issues found! Review the mismatches above.")
        return 1
    else:
        print(f"\n✓ All validations passed.")
        return 0


if __name__ == "__main__":
    sys.exit(validate())
