"""Wrapper train script pointing directly to the orchestrator entrypoint."""

import os
import sys

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from main import main

if __name__ == "__main__":
    main()
