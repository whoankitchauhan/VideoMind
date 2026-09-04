import os
import time

import httpx
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI


load_dotenv()

_rate_limit_until = 0.0


def get_llm(temperature: float = 0.2, max_tokens: int = 1200):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing. Add it to your .env file."
        )

    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def invoke_llm(chain, payload, operation: str, fallback: str | None = None) -> str:
    global _rate_limit_until

    now = time.monotonic()
    if fallback is not None and now < _rate_limit_until:
        print(
            f"Skipping {operation} because the Mistral API is still "
            "rate-limited. Continuing with a fallback response."
        )
        return fallback

    try:
        return chain.invoke(payload)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code != 429:
            raise

        retry_after = exc.response.headers.get("retry-after")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 20

        print(
            f"Mistral rate limit hit while {operation}. "
            f"Waiting {wait_seconds} seconds before retrying..."
        )
        time.sleep(wait_seconds)

        try:
            return chain.invoke(payload)
        except httpx.HTTPStatusError as retry_exc:
            if retry_exc.response.status_code != 429:
                raise

            _rate_limit_until = time.monotonic() + wait_seconds

            if fallback is None:
                raise RuntimeError(
                    "Mistral API rate limit was reached after one retry. "
                    "Wait for your quota window to reset, then run the analysis again."
                ) from retry_exc

            print(
                f"Mistral is still rate-limited while {operation}. "
                "Continuing with a fallback response."
            )
            return fallback
