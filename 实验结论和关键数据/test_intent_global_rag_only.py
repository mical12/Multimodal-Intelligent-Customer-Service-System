import asyncio

from model import intent_agent
from test_intent import main


async def disabled_summary_llm(*args, **kwargs):
    return None


intent_agent._match_manual_with_llm = disabled_summary_llm


if __name__ == "__main__":
    asyncio.run(main())
