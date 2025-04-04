from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from openai import BadRequestError

from .base import BaseNode


class BaseLLMNode(BaseNode):
    llm: Runnable
    system_message: str | None
    prompt: str | None
    is_part_of_history: bool
    should_stream_response: bool

    def __init__(
            self,
            llm: Runnable,
            user_stream: Any | None = None,
            tools: list[BaseTool] | None = None,
            system_message: str | None = None,
            prompt: str | None = None,
            is_part_of_history: bool = True,
            should_stream_response: bool = False,
            additional_configs: dict = None
    ):
        super().__init__(user_stream)
        llm = llm.with_config({
            "callbacks": [self.user_stream] if self.user_stream and should_stream_response else [],
            **(additional_configs if additional_configs else {})
        })
        if tools:
            self.llm = llm.bind_tools(tools)
        else:
            self.llm = llm
        self.system_message = system_message
        self.prompt = prompt
        self.is_part_of_history = is_part_of_history
        self.should_stream_response = should_stream_response

    async def __call__(self, state) -> dict:
        try:
            await self._pre_call(state)

            llm_output: AIMessage = await self.llm.ainvoke(
                input=[
                    *([SystemMessage(content=self.system_message)] if self.system_message else []),
                    *state["messages"],
                    AIMessage(
                        id="BaseLLMNode-Private-NextRequest",
                        content=self.prompt
                    )
                ]
            )
            
            llm_output.response_metadata["is_part_of_history"] = self.is_part_of_history

            await self._post_call({
                "messages": [llm_output]
            })
            return {
                "messages": [llm_output]
            }

        except BadRequestError as e:
            return self.error_output(e, "Could not generate answer.", message_type="ai")
        except Exception as e:
            return self.error_output(e, "An unexpected error occurred.", message_type="ai")
