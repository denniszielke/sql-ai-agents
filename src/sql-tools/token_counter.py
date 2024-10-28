from typing import Any
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs.llm_result import LLMResult

class TokenCounterCallback(BaseCallbackHandler):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        for result in response.flatten():
            generation = result.generations[0][0]
            if isinstance(generation.message, AIMessage):
                self.completion_tokens += generation.message.usage_metadata.get("output_tokens", 0)
                self.prompt_tokens += generation.message.usage_metadata.get("input_tokens", 0)
        self.total_tokens = self.completion_tokens + self.prompt_tokens
        return