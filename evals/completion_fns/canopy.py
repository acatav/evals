from typing import Any, Union

from canopy.chat_engine.query_generator import LastMessageQueryGenerator
from canopy.tokenizer.tokenizer import Tokenizer
from canopy.knowledge_base.knowledge_base import KnowledgeBase
from canopy.context_engine.context_engine import ContextEngine
from canopy.chat_engine.chat_engine import ChatEngine
from canopy.models.api_models import ChatResponse
from canopy.models.data_models import Messages, Role, MessageBase, Query
from canopy.knowledge_base.reranker.cohere import CohereReranker

from evals import CompletionFn, CompletionResult


class CanopyCompletionResult(CompletionResult):
    def __init__(self, response: ChatResponse) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.choices[0].message.content]


class CanopyCompletionFn(CompletionFn):

    def __init__(
            self,
            index_name: str,
            top_k: int,
            system_prompt: str,
            top_n: int = 5,
            reranker: bool = False,
            **_kwargs: Any):
        Tokenizer.initialize()

        self.index_name = index_name
        self.top_k = int(top_k)

        if reranker == "cohere":
            self.reranker = CohereReranker(top_n=int(top_n))
        elif reranker == "none":
            self.reranker = None
        else:
            raise ValueError(f"Unknown reranker {reranker}")

        query_generator = LastMessageQueryGenerator()
        self.kb = KnowledgeBase(index_name, default_top_k=self.top_k, reranker=self.reranker)
        self.kb.connect()
        self.context_engine = ContextEngine(self.kb)
        self.chat_engine = ChatEngine(self.context_engine,
                                      system_prompt=system_prompt,
                                      query_builder=query_generator)

    def __call__(self, prompt: Union[str, list[dict]], **kwargs: Any) -> CanopyCompletionResult:
        messages = self.parse_prompt_to_messages(prompt)
        response = self.chat_engine.chat(messages)
        return CanopyCompletionResult(response)

    @staticmethod
    def parse_prompt_to_messages(data: list[dict]) -> Messages:
        def create_message(data: dict) -> MessageBase:
            role_name = data.get("role")
            content = data.get("content")
            role = Role[role_name.upper()]  # Convert string to Role enum
            return MessageBase(role=role, content=content)

        return [create_message(item) for item in data]
