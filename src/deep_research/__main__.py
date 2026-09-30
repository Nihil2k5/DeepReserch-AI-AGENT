"""``python -m deep_research`` — start the API with uvicorn."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "deep_research.api.app:app",
        host=os.environ.get("DR_HOST", "0.0.0.0"),
        port=int(os.environ.get("DR_PORT", "8100")),
        reload=os.environ.get("DR_RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
