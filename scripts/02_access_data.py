#!/usr/bin/env python3
"""Example: Programmatic access to evaluation data"""

from verifiers_monitor import MonitorData

data = MonitorData()

# =============================================================================
# Session Information
# =============================================================================

sessions = data.list_sessions()
session = next(
    (s for s in sessions if s.env_id and s.num_examples > 0),
    sessions[0] if sessions else None,
)

if not session:
    print("No sessions found. Run an evaluation first.")
    exit(1)

print(f"Session: {session.model_name} on {session.env_id}")
print(f"  Started: {session.started_at}")
print(f"  Examples: {session.num_examples}")
print(f"  Status: {session.status}")

# =============================================================================
# Failure Analysis
# =============================================================================

print(f"\n📊 Failure Analysis:")
failures = data.get_failed_examples(session.session_id, threshold=0.5)
print(f"  Found {len(failures)} failing examples\n")

for ex in failures[:3]:
    print(f"  Example {ex.example_number}:")
    print(
        f"    Rewards: best={ex.max_reward:.2f}, avg={ex.mean_reward:.2f}, std={ex.std_reward:.2f}"
    )

    # Show prompt preview using convenience property
    rollout = ex.rollouts[0]
    if rollout.prompt_messages:
        user_msg = next(
            (m["content"] for m in rollout.prompt_messages if m.get("role") == "user"),
            None,
        )
        if user_msg:
            preview = user_msg[:80] + "..." if len(user_msg) > 80 else user_msg
            print(f"    Prompt: {preview}")
    print()

# =============================================================================
# Variance Analysis
# =============================================================================

print(f"📈 Variance Analysis:")
examples = data.get_examples(session.session_id)
unstable = [ex for ex in examples if ex.is_unstable(threshold=0.3)]
print(f"  Total examples: {len(examples)}")
print(f"  Unstable (std > 0.3): {len(unstable)}\n")

for ex in unstable[:3]:
    print(f"  Example {ex.example_number}:")
    print(f"    Std dev: {ex.std_reward:.3f}")
    print(f"    Rewards: {[r.reward for r in ex.rollouts]}")
    # Show best vs worst
    best = ex.get_best_rollout()
    worst = ex.get_worst_rollout()
    print(
        f"    Range: {best.reward:.2f} (rollout #{best.rollout_number}) → {worst.reward:.2f} (rollout #{worst.rollout_number})"
    )
    print()

# High-variance examples
variance = df.groupby("example_number")["reward"].std()
high_variance = variance[variance > 0.3]
print(f"\n⚠️  Found {len(high_variance)} high-variance examples (std > 0.3)")
if len(high_variance) > 0:
    print(f"  Example numbers: {sorted(high_variance.index.tolist())}")
