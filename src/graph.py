import os
import traceback
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from langgraph.utils.config import RunnableConfig
from kagent.nodes import BaseLLMNode, BaseNode, BaseToolNode
from kagent.streamer import KStreamer
from socketio import AsyncServer


class KGraph:
    sio: AsyncServer
    config: RunnableConfig
    graph: CompiledStateGraph | None
    user_id: str
    thread_id: str
    user_stream: KStreamer
    nodes: dict[str, BaseNode] | None
    edges: dict[str, str] | None
    conditional_edges: list[tuple[str, callable, list[str] | str]] | None
    interrupt_at: list[str] | None
    checkpointer: BaseCheckpointSaver | None
    tools: list[BaseTool]
    state_type: type
    should_interrupt: bool

    def __init__(
            self,
            sio: AsyncServer,
            user_id: str,
            thread_id: str,
            state_type: type,
            nodes: dict[str, BaseNode | BaseToolNode | BaseLLMNode] = None,
            edges: dict[str, str] = None,
            conditional_edges: list[tuple[str, callable, list[str] | str]] = None,
            interrupt_at: list[str] = None,
            checkpointer: BaseCheckpointSaver | None = None,
            tools: list[BaseTool] | None = None,
            recursion_limit: int = 10,
            should_interrupt: bool = False,
            should_skip_build: bool = False,
            should_output_image: bool = False
    ):
        self.sio = sio
        self.config = {
            "configurable": {
                "thread_id": f"{user_id}.{thread_id}",
                "k_thread_id": thread_id,
                "k_user_id": user_id,
            },
            "recursion_limit": recursion_limit
        }
        self.graph = None
        self.user_id = user_id
        self.thread_id = thread_id
        self.user_stream = KStreamer(sio, user_id, thread_id)
        self.state_type = state_type
        self.nodes = nodes if nodes else {}
        self.edges = edges if edges else {}
        self.conditional_edges = conditional_edges if conditional_edges else []
        self.interrupt_at = interrupt_at if interrupt_at else []
        self.checkpointer = checkpointer
        if tools is None:
            self.tools = []
            self.tools_by_name = {}
        else:
            self.tools = tools
            self.tools_by_name = {tool.name: tool for tool in tools}
        self.should_interrupt = should_interrupt
        if not should_skip_build and checkpointer is not None:
            self.build(checkpointer)
        if should_output_image:
            self.save_image()

    def build(self, checkpointer: BaseCheckpointSaver):
        graph_builder: StateGraph = StateGraph(self.state_type)
        for node_name, node in self.nodes.items():
            graph_builder.add_node(node_name, node)

        for from_node, to_node in self.edges.items():
            graph_builder.add_edge(from_node, to_node)

        for from_node, path, to_nodes in self.conditional_edges:
            if not isinstance(to_nodes, list):
                to_nodes = [to_nodes]
            if callable(path):
                graph_builder.add_conditional_edges(
                    from_node,
                    path,
                    {node_name: node_name for node_name in to_nodes}
                )
            else:
                raise ValueError(
                    f"Path for {from_node} to {to_nodes} must be a callable function, was handed {type(path)} instead.")

        self.graph = graph_builder.compile(
            checkpointer=checkpointer,
            interrupt_before=self.interrupt_at if self.should_interrupt else []
        )

    def save_image(self, file_path: str = "./graph.png"):
        try:
            folder_path: str = os.path.dirname(file_path)
            if graph := self.graph:
                graph_image: bytes = graph.get_graph().draw_mermaid_png()
                os.makedirs(folder_path, exist_ok=True)
                with open(file_path, "wb") as graph_image_file:
                    graph_image_file.write(graph_image)
            else:
                raise RuntimeError("Graph not yet built.")
        except Exception as e:
            print(f"!!! Error saving graph image: {e}\n{traceback.format_exc()}")

    async def stream_state_updates(self, new_query: str):
        if not self.graph:
            raise RuntimeError("Graph not yet built.")
        current_state: StateSnapshot = await self.graph.aget_state(self.config)
        inputs: dict = {"messages": [
            *current_state.values.get("messages", []),
            {"type": "user", "content": new_query}
        ]}
        await self.user_stream.start()
        async for state in self.graph.astream(inputs, self.config, stream_mode="values"):
            yield state
        await self.user_stream.stop()

    async def message_history(
            self,
            included_message_types: list[Literal["system"] | Literal["human"] | Literal["ai"] | Literal["tool"]] | None = None
    ) -> list[dict]:
        if not self.graph:
            raise RuntimeError("Graph not yet built.")
        if not included_message_types:
            included_message_types = ["human", "ai", "tool"]

        messages: list[dict] = []
        async for state in self.graph.aget_state_history(self.config):
            state_messages: list[BaseMessage] = state.values.get("messages", [])
            if len(state_messages) > 0:
                last_message: BaseMessage = state_messages[-1]
                current_message: dict | None = KGraph.message_object_to_dict(last_message, state)

                if current_message and current_message["type"] in included_message_types:
                    if len(messages) > 0:
                        if messages[-1].get("id") == current_message["id"]:
                            messages[-1] = current_message
                        else:
                            messages.append(current_message)
                    else:
                        messages.append(current_message)

        return messages

    async def current_state(self) -> StateSnapshot:
        if not self.graph:
            raise RuntimeError("Graph not yet built.")
        state: StateSnapshot = await self.graph.aget_state(self.config)
        return state

    @staticmethod
    async def all_thread_ids(
            checkpointer: BaseCheckpointSaver,
            user_id: str
    ) -> list[str]:
        threads: list[tuple[str, str]] = []
        async for checkpoint in checkpointer.alist(
                config=None,
                filter={
                    "k_user_id": user_id
                }
        ):
            threads.append((
                checkpoint.checkpoint["ts"],
                checkpoint.metadata["k_thread_id"]
            ))
        sorted_threads: list[tuple[str, str]] = sorted(threads, key=lambda x: x[0])
        thread_ids: list[str] = [thread_id for _, thread_id in sorted_threads]
        unique_thread_ids: list[str] = list(dict.fromkeys(thread_ids))
        return list(unique_thread_ids)

    @staticmethod
    async def thread_previews(
            checkpointer: BaseCheckpointSaver,
            user_id: str,
            thread_ids: list[str]
    ) -> list[dict]:
        threads: list[dict] = []
        for thread_id in thread_ids:
            checkpoint: Checkpoint = await checkpointer.aget({
                "configurable": {
                    "thread_id": f"{user_id}.{thread_id}"
                }
            })
            checkpoint_messages: list[BaseMessage] = checkpoint["channel_values"]["messages"]
            checkpoint_human_messages: list[HumanMessage] = [
                message for message in checkpoint_messages if isinstance(message, HumanMessage)
            ]
            if checkpoint_human_messages:
                threads.append({
                    "id": thread_id,
                    "query": checkpoint_human_messages[0].content
                })
            else:
                threads.append({
                    "id": thread_id,
                    "query": None
                })
        return threads

    @staticmethod
    async def thread_history(
            checkpointer: BaseCheckpointSaver,
            user_id: str,
            thread_id: str,
            included_message_types: list[Literal["system"] | Literal["human"] | Literal["ai"] | Literal["tool"]] | None = None
    ) -> list[dict]:
        if not included_message_types:
            included_message_types = ["human", "ai", "tool"]

        checkpoint: Checkpoint = await checkpointer.aget({
            "configurable": {
                "thread_id": f"{user_id}.{thread_id}"
            }
        })
        if checkpoint:
            checkpoint_state = checkpoint["channel_values"]
            checkpoint_messages: list[BaseMessage] = checkpoint_state["messages"]
            messages: list[dict] = []
            for checkpoint_message in checkpoint_messages:
                message: dict | None = KGraph.message_object_to_dict(checkpoint_message, checkpoint["channel_values"])
                if message and message["type"] in included_message_types:
                    messages.append(message)
            return messages
        else:
            return []

    @staticmethod
    def message_object_to_dict(message: BaseMessage, state) -> dict:
        if isinstance(message, AIMessage):
            if message.content and message.response_metadata.get("is_part_of_history", False):
                return {
                    "id": message.id,
                    "type": "ai",
                    "content": message.content,
                    "sources": state.values.get("sources", [])
                }
        elif isinstance(message, HumanMessage):
            if message.content:
                return {
                    "id": message.id,
                    "type": "human",
                    "content": message.content
                }
        elif isinstance(message, SystemMessage):
            if message.content:
                return {
                    "id": message.id,
                    "type": "system",
                    "content": message.content
                }
        elif isinstance(message, ToolMessage):
            return {
                "id": message.id,
                "type": "tool",
                "content": message.content
            }
        return None
