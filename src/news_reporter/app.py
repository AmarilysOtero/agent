from __future__ import annotations
import asyncio
import logging
import sys

# === Load .env early ===
from dotenv import load_dotenv
load_dotenv()

# === Project imports ===
from .config import Settings
from .workflows.workflow_factory import run_sequential_goal   # adjust if your file name differs

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Starting app...")

    # ✅ Correct call (Option B)
    cfg = Settings.from_env()
    logger.info("Loaded configuration: %s", cfg.redacted_dict())

    # Optional: allow CLI goal argument
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Summarize the latest AI news"
    logger.info("Goal: %s", goal)

    result = await run_sequential_goal(cfg, goal)
    print("\n=== Final Output ===\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
