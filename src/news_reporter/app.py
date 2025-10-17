from __future__ import annotations
import asyncio, logging
from .config import Settings
from .workflows.workflow_factory import run_sequential_goal

async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = Settings.load()

    goal = "prepare a news script for John on latest news for the world?"
    print("Goal:", goal)  # keep print

    result = await run_sequential_goal(cfg, goal)

    print("\n=== RESULT ===\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
