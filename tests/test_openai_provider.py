from unittest import mock

import pytest
from openai.openai_object import OpenAIObject

from openai_provider import OpenAIProvider


@pytest.fixture()
def openai_chat_completion_create_positive_response() -> OpenAIObject:
    class OpenAIChoiceStub:
        @property
        def message(self) -> dict:
            return {"role": "assistant", "content": '{"response": 1}'}

    class OpenAIObjectStub(OpenAIObject):
        @property
        def choices(self) -> list[OpenAIChoiceStub]:
            return [
                OpenAIChoiceStub(),
            ]

        def __repr__(self):
            return "OpenAIObjectStub"

        def __str__(self):
            return self.__repr__()

    return OpenAIObjectStub()


@pytest.mark.unit
def test__query_openai(openai_chat_completion_create_positive_response: OpenAIObject):
    openai_provider = OpenAIProvider()
    format_directive = (
        'Respond in the JSON format: {{"response": sentiment_classification}}.'
    )
    input_text = "test_message the party was nice"
    prompt = f"{format_directive}\nMessage: {input_text}\nSentiment (0, 1):"
    with mock.patch(
        "openai.ChatCompletion.create",
        return_value=openai_chat_completion_create_positive_response,
    ):
        ret = openai_provider._query_openai(prompt=prompt)
        assert ret["response"] == "1"
