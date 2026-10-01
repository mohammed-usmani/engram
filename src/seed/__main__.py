import asyncio
import logging

from src.config import settings
from src.database import AsyncSessionLocal
from src.seed.loader import run_seed


async def main() -> None:
    logging.basicConfig(level=settings.log_level)
    async with AsyncSessionLocal() as session:
        summary = await run_seed(session, settings.data_dir)
        await session.commit()
        print(summary)


if __name__ == "__main__":
    asyncio.run(main())
