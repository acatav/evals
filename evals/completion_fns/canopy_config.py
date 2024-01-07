import yaml
from pathlib import Path
from typing import Any, Union

from evals import CompletionFn, CompletionResult

from canopy.tokenizer.tokenizer import Tokenizer
from canopy.chat_engine.chat_engine import ChatEngine
from canopy.models.api_models import ChatResponse
from canopy.models.data_models import Messages, Role, MessageBase


class CanopyCompletionResult(CompletionResult):
    def __init__(self, response: ChatResponse) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.choices[0].message.content]


def recursive_dict_update(orig_dict: dict, new_dict: dict) -> dict:
    res_dict = orig_dict.copy()
    for key, value in new_dict.items():
        if key in res_dict and isinstance(res_dict[key], dict) and isinstance(value, dict):
            recursive_dict_update(res_dict[key], value)
        else:
            res_dict[key] = value
    return res_dict


class CanopyConfigCompletionFn(CompletionFn):

    def __init__(
            self,
            config: dict[str, Any],
            **_kwargs: Any):
        base_config = self._load_default_config()
        config = recursive_dict_update(base_config, config)

        tokenizer_config = config.get("tokenizer", {})
        Tokenizer.initialize_from_config(tokenizer_config)

        chat_engine_config = config["chat_engine"]
        self.chat_engine = ChatEngine.from_config(chat_engine_config)

    def __call__(self,
                 prompt: Union[str, list[dict]],
                 **kwargs: Any) -> CanopyCompletionResult:

        response = self.chat_engine.chat(self.parse_prompt_to_messages(prompt))
        return CanopyCompletionResult(response)

    @staticmethod
    def parse_prompt_to_messages(data: list[dict]) -> Messages:
        def create_message(data: dict) -> MessageBase:
            role_name = data.get("role")
            content = data.get("content")
            role = Role[role_name.upper()]  # Convert string to Role enum
            return MessageBase(role=role, content=content)

        return [create_message(item) for item in data]

    @staticmethod
    def _load_default_config() -> dict[str, Any]:
        current_path = Path(__file__).resolve()

        yaml_file_path = current_path.parent.parent / 'resources' / 'canopy_default_config.yaml'

        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        return config
