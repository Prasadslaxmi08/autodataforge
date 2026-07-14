"""AutoDataForge — agentic AI vision dataset engineering platform.

Package layout mirrors System Design §2 (dependencies point downward only):

    L4  api · cli · sdk
    L3  agents
    L2  ingest · engine · curation · quality · snapshot · export
    L1  models · store · jobs
    L0  core · config
"""

__version__ = "0.1.0"
