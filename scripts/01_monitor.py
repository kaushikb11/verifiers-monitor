#!/usr/bin/env python3
"""Example: Real-time monitoring during LLM evaluation"""

import verifiers as vf
from openai import OpenAI

from verifiers_monitor import monitor

# Load environment and add monitoring
env = vf.load_environment("math-python")
env = monitor(env)  # Dashboard launches at http://localhost:8080

# Run evaluation - watch live progress in the dashboard
client = OpenAI()  # Requires OPENAI_API_KEY in environment
results = env.evaluate(
    client=client, model="gpt-5-mini", num_examples=30, rollouts_per_example=3
)
