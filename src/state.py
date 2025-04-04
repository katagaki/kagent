from typing import TypedDict, Annotated

from langgraph.graph import add_messages


class BaseState(TypedDict):
    messages: Annotated[list, add_messages]
    last_error: str
