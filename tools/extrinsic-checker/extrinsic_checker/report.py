"""Per-camera verdict structure, printing, and overall aggregation."""

from dataclasses import dataclass, field


@dataclass
class Verdict:
    camera: str
    method: str            # "depth" | "projection"
    passed: bool
    metrics: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)


def print_report(verdicts):
    """Print each verdict; return True iff all passed."""
    overall = True
    for v in verdicts:
        overall = overall and v.passed
        status = "PASS" if v.passed else "FAIL"
        print(f"[{status}] {v.camera} ({v.method})")
        for k, val in v.metrics.items():
            print(f"    {k}: {val}")
        for a in v.artifacts:
            print(f"    artifact: {a}")
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return overall
