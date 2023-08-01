from __future__ import annotations

import re

from typing import Generator, Any, TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from openai.openai_object import OpenAIObject


class OpenAIProvider:
    def __init__(self, api_key: str = "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") -> None:
        self._api_key: str = api_key

    @staticmethod
    def get_sentiment(self, input_text: str) -> dict:
        """
        Create a prompt based on input_text and use it to query OpenAI API to return sentiment values (1 or 0) -- good/bad opinion about the product.
        """
        format_directive = "Respond in the JSON format: {{'response': sentiment_classification}}."

        description = """\
The following message will represent opinions about products by customers who own the given product.  While usually the comments are complete sentences, opinions \
are commonly expressed using just a single word (e.g., "good" or "bad") or a couple of words.  Alternatively, they may emphasize an attribute of the product.

If the opinion message contains praise, appreciation, gratitude, and other favorable positive remarks, then the sentiment is clearly good (output is 1).  However, \
if the mood in the text is negative, not advising other users to get this product, or complaining, even displaying anger, then the sentiment is poor (output is 0).
"""

        annotation = "Each message begins with the product name and ends on the customer feedback.\n"

        task = "The goal is to determine whether or not the opinion is favorable."

        # prompt: str = f"{description}\n{annotation}\n{task}\n{format_directive}"
        prompt: str = f"{description}\n{task}\n{format_directive}"

        prompt = f"{prompt}\nMessage: {input_text}\nSentiment (0, 1):"

        return self._query_openai(prompt=prompt)

    def _query_openai(self, prompt: str) -> dict:
        """
        Given a prompt, return sentiment values (0 or 1) -- opinion about the product (positive/negative) -- in the JSON format (the outputs will have values "0" or "1").
        """
        self._ensure_openai_api_key_is_set()

        try:
            response: Generator[list | OpenAIObject | dict, Any, None] | list | OpenAIObject | dict = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=40,
                n=1,
                stop=None,
                temperature=0.5,
            )
            response_text: str = response.choices[0].message["content"].strip()
            sentiment_match: re.Match[str] | None = re.search("0|1", response_text)
            sentiment: str | None
            if sentiment_match:
                sentiment = sentiment_match.group(0)
            else:
                sentiment = None
            # Add input_text back in for the result

            return {"prompt": prompt, "response": sentiment}
        except (openai.error.ServiceUnavailableError, AttributeError, ValueError) as e:
            print(f'Error "{e}" was encountered for prompt: "{prompt}"')
            return {"text": prompt, "response": "0"}

    def _ensure_openai_api_key_is_set(self) -> None:
        if not openai.api_key:
            openai.api_key = self._api_key
