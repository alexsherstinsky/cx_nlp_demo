from __future__ import annotations

import os
import logging
import re
from typing import TYPE_CHECKING, Any, Generator

import openai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    from openai.openai_object import OpenAIObject


class OpenAIProvider:
    """
    The OpenAIProvider class is a wrapper around the OpenAI ChatCompletion.create() method and processing its output.
    It initializes the openai library with the api_key and provide a public get_sentiment() method to handle text input.
    """

    def __init__(self, api_key: str = os.environ.get("OPENAI_API_KEY")) -> None:
        self._api_key: str = api_key

    def get_sentiment(self, input_text: str) -> dict:
        """
        Create a prompt based on input_text and use it to query OpenAI API to return sentiment values ("0" or "1") -- good/bad opinion about the product.

        Args:
            input_text: Input text to be incorporated into the classification instruction prompt to be passed to OpenAI.

        Returns:
            Dictionary containing a copy of the prompt and the sentiment ("0": "negative" or "1": "positive") response.
        """
        format_directive = (
            "Respond in the JSON format: {{'response': sentiment_classification}}."
        )

        description = """\
The following message will represent opinions about products by customers who own the given product.  While usually the comments are complete sentences, opinions \
are commonly expressed using just a single word (e.g., "good" or "bad") or a couple of words.  Alternatively, they may emphasize an attribute of the product.

If the opinion message contains praise, appreciation, gratitude, and other favorable positive remarks, then the sentiment is clearly good (output is 1).  However, \
if the mood in the text is negative, not advising other users to get this product, or complaining, even displaying anger, then the sentiment is poor (output is 0).
"""

        # TODO: <Alex>ALEX -- saving the next line for a later use.</Alex>
        # annotation = "Each message begins with the product name and ends on the customer feedback.\n"
        # TODO: <Alex>ALEX</Alex>

        task = "The goal is to determine whether or not the opinion is favorable."

        # prompt: str = f"{description}\n{annotation}\n{task}\n{format_directive}"
        prompt: str = f"{description}\n{task}\n{format_directive}"

        prompt = f"{prompt}\nMessage: {input_text}\nSentiment (0, 1):"

        return self._query_openai(prompt=prompt)

    def _query_openai(self, prompt: str) -> dict:
        """
        Given a prompt, return sentiment values -- opinion about the product (positive/negative) -- in the JSON format (the outputs will have values "0" or "1").
        """
        self._ensure_openai_api_key_is_set()

        try:
            response: Generator[
                list | OpenAIObject | dict, Any, None
            ] | list | OpenAIObject | dict = openai.ChatCompletion.create(
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
