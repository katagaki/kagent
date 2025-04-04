import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from .base import BaseNode


class BaseToolNode(BaseNode):
    should_invoke_concurrently: bool

    def __init__(
            self,
            user_stream,
            tools: BaseTool | list[BaseTool],
            output_key: str = None,
            should_invoke_concurrently: bool = False
    ):
        if not isinstance(tools, list):
            tools = [tools]
        super().__init__(user_stream, tools)
        self.tools = tools
        self.output_key = output_key
        self.should_invoke_concurrently = should_invoke_concurrently

    async def __call__(self, state) -> dict:
        try:
            await self._pre_call(state)

            output: dict = {
                "messages": [],
                **({self.output_key: []} if self.output_key else {})
            }

            for tool_call, tool_result in await self.invoke_tools(
                    state,
                    self.tools,
                    invoke_concurrently=self.should_invoke_concurrently
            ):
                output["messages"].append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
                if self.output_key:
                    output[self.output_key].extend(tool_result)

            await self._post_call(output)
            return output

        except Exception as e:
            return self.error_output(e, "An error occurred while running a tool.", message_type="tool")
