import traceback
from asyncio import gather
from typing import Literal, Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool


class BaseNode:
    user_stream: Any  # KStreamer
    tools_by_name: dict[str, BaseTool]

    def __init__(
            self,
            user_stream: Any | None = None,
            tools: list[BaseTool] | None = None
    ):
        self.user_stream = user_stream
        if tools:
            for tool in tools:
                try:
                    tool.user_stream = user_stream
                except Exception:
                    pass
            self.tools_by_name = {
                tool.name: tool for tool in tools
            }
        else:
            self.tools_by_name = {}

    async def _pre_call(self, state):
        pass

    async def _post_call(self, output: dict):
        pass

    def __call__(self, state) -> dict:
        raise NotImplementedError("This method must be implemented in a subclass.")

    # noinspection PyMethodMayBeStatic
    def last_message(self, state) -> BaseMessage:
        if messages := state.get("messages", []):
            return messages[-1]
        else:
            raise ValueError("No messages found in state.")

    async def invoke_tools(
            self,
            state,
            supported_tools: list[BaseTool],
            invoke_concurrently: bool = False
    ) -> list[tuple[ToolCall, ToolMessage]]:
        tools: list[tuple[ToolCall, BaseTool]] = []
        last_message: BaseMessage = self.last_message(state)
        if isinstance(last_message, AIMessage):
            for tool_call in last_message.tool_calls:
                if tool_name := tool_call.get("name"):
                    if tool_name not in self.tools_by_name.keys():
                        raise ValueError(f"{tool_name} tool is not registered to {self.__class__.__name__}.")
                    if tool_name in [tool.name for tool in supported_tools]:
                        tool_name: str = tool_call["name"]
                        tool: BaseTool = self.tools_by_name[tool_name]
                        try:
                            tool.user_stream = self.user_stream
                        except Exception:
                            pass
                        tools.append((tool_call, tool))
        else:
            raise ValueError(f"{self.__class__.__name__} called, but last message was not an AIMessage.")

        all_tool_results: list[tuple[ToolCall, ToolMessage]] = []

        async def invoke_tool(tool_call, tool) -> tuple[ToolCall, ToolMessage]:
            tool_result = await tool.ainvoke(
                {
                    **tool_call["args"],
                    "state": state
                }
            )
            return tool_call, tool_result

        if invoke_concurrently:
            tasks = [invoke_tool(tool_call, tool) for tool_call, tool in tools]
            all_tool_results = await gather(*tasks)

        else:
            for tool_call, tool in tools:
                tool_result = await invoke_tool(tool_call, tool)
                all_tool_results.append(tool_result)

        return all_tool_results

    def error_output(self, e: Exception, message: str, message_type: Literal["ai", "tool"]) -> dict:
        print(f"!!! Error in node {self.__class__.__name__}: {e}\nStack trace:\n{traceback.format_exc()}")
        match message_type:
            case "ai":
                return_message = AIMessage(
                    id=str(uuid4()),
                    content=f"⚠️ {message}"
                )
                return_message.response_metadata["is_part_of_history"] = True
            case "tool":
                return_message = ToolMessage(
                    id=str(uuid4()),
                    content=f"⚠️ {message}"
                )
                return_message.response_metadata["is_part_of_history"] = True
            case _:
                return_message = None
        return {
            "messages": [return_message] if return_message else [],
            "last_error": message
        }
