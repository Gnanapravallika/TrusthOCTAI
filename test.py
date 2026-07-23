"""Wrapper test script calling orchestrator in evaluation mode."""

import os
import sys

# Inject eval_only into args dynamically
sys.argv.append("--mode")
sys.argv.append("eval_only")

from main import main

if __name__ == "__main__":
    main()
