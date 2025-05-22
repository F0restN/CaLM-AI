from classes.ChatSession import BaseChatMessage


def format_conversation(chat_history: list[BaseChatMessage]) -> str:
    """Format converation into clear and concise conversation list."""
    conversation = [
        f"{msg.role.upper()}: {msg.content}" for msg in chat_history
    ]
    return "\n".join(conversation)


def format_conversation_pipeline(chat_history: list[BaseChatMessage]) -> str:
    """Format conversation into clear and concise conversation list.

    Args:
        chat_history: list[BaseChatMessage]

    Returns:
        str: formatted conversation

    """
    conversation = [
        f"{msg.role.upper()}: {msg.content}" for msg in chat_history
    ]
    return "\n".join(conversation)
