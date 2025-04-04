from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from socketio import AsyncServer


class KStreamer(BaseCallbackHandler):
    sio: AsyncServer | None
    user_id: str
    thread_id: str
    message_id: str
    generated_answer: str | dict
    is_working: bool
    is_generating: bool
    concurrent_progress_title: str
    concurrent_items: list[tuple[str, str]]
    sources: list[dict]
    state: Literal["stopped", "started", "generating", "concurrentlyGenerating", "finished"]

    def __init__(
        self,
        sio: AsyncServer | None = None,
        user_id: str = "default",
        thread_id: str = "0"
    ):
        self.sio = sio
        self.user_id = user_id
        self.thread_id = thread_id
        self.message_id = str(uuid4())
        self.generated_answer = ""
        self.concurrent_progress_title = ""
        self.concurrent_items = []
        self.sources = []
        self.state = "stopped"

    async def emit_current_data(self):
        if self.sio:
            if self.state == "concurrentlyGenerating":
                answer_to_send = f"## {self.concurrent_progress_title}"
                for item in self.concurrent_items:
                    answer_to_send += f"\n- {item[0]}: {item[1]}"
            else:
                answer_to_send = self.generated_answer

            await self.sio.emit(
                event="newMessage",
                data={
                    "threadId": self.thread_id,
                    "state": self.state if self.state != "concurrentlyGenerating" else "generating",
                    "message": {
                        "id": self.message_id,
                        "type": "ai",
                        "sources": self.sources,
                        "content": answer_to_send
                    }
                },
                room=self.user_id
            )

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        self.state = "started"
        self.generated_answer = ""
        await self.emit_current_data()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> Any:
        await self.emit_current_data()

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs
    ):
        self.state = "generating"
        self.generated_answer = f"{self.generated_answer}{token}"
        await self.emit_current_data()

    async def start(self):
        self.state = "started"
        await self.emit_current_data()

    async def stop(self):
        self.state = "stopped"
        await self.emit_current_data()

    async def start_concurrent_progress(self, title: str):
        self.state = "concurrentlyGenerating"
        self.concurrent_progress_title = title
        await self.emit_current_data()

    async def stop_concurrent_progress(self):
        self.state = "generating"
        self.concurrent_progress_title = ""
        self.concurrent_items = []
        await self.emit_current_data()

    async def update_concurrent_item(self, item_name: str, progress_text: str):
        is_one_or_more_items_updated: bool = False
        for i, item in enumerate(self.concurrent_items):
            if item[0] == item_name:
                self.concurrent_items[i] = (item_name, progress_text)
                is_one_or_more_items_updated = True
        if not is_one_or_more_items_updated:
            self.concurrent_items.append((item_name, progress_text))
        await self.emit_current_data()

    async def remove_concurrent_item(self, item_name: str):
        for item in self.concurrent_items:
            if item[0] == item_name:
                self.concurrent_items.remove(item)
        await self.emit_current_data()

    async def overwrite_answer(self, text: str | dict):
        self.generated_answer = text
        await self.emit_current_data()

    async def append_source(self, title: str, url: str):
        self.sources.append({"title": title, "url": url})
        await self.emit_current_data()

    async def append_source_dict(self, source_dict: dict):
        self.sources.append(source_dict)
        await self.emit_current_data()

    async def append_sources(self, sources: list):
        self.sources.extend(sources)
        await self.emit_current_data()

    async def clear_all_sources(self):
        self.sources = []
        await self.emit_current_data()
