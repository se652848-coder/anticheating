from typing import Optional
from app.ai.anti_cheating_ai import AntiCheatingAI
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ai_instance: Optional[AntiCheatingAI] = None


async def initialize_dependencies():
    global _ai_instance
    logger.info("Loading AI models (this may take a moment)...")
    _ai_instance = AntiCheatingAI()
    logger.info("AI models loaded successfully.")


def get_ai() -> Optional[AntiCheatingAI]:
    return _ai_instance
