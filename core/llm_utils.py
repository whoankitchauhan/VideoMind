import time

import httpx


_rate_limit_until = 0.0


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
            if retry_exc.response.status_code != 429 or fallback is None:
                raise

            _rate_limit_until = time.monotonic() + wait_seconds

            print(
                f"Mistral is still rate-limited while {operation}. "
                "Continuing with a fallback response."
            )
            return fallback
