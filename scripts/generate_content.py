import os
from pathlib import Path
import random

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv

load_dotenv()

def _build_prompt() -> str:
    # random prompts, pick from the list of prompts
    prompts = [
        "Generate a neutral article about the topic: {topic}",
        "Write a neutral article about the topic: {topic}",
        "Write a news on {topic}, SCMP style",
    ]
    return random.choice(prompts)


def _openrouter_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _build_agent(model_id: str, api_key: str, extra_headers: dict[str, str]) -> Agent:
    model = OpenAIChat(
        id=model_id,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        extra_headers=extra_headers,
    )
    return Agent(
        model=model,
        instructions=(
            "You are a helpful writing assistant. "
            "Follow the requested format and length closely."
        ),
        markdown=True,
    )


def _prompt_value(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _ensure_data_dir() -> Path:
    # data directory is at project root, not in scripts/
    data_dir = Path(__file__).resolve().parent.parent / "data" / "ai"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _next_index(prefix: str, data_dir: Path) -> int:
    max_index = 0
    for path in data_dir.glob(f"{prefix}[0-9][0-9][0-9].txt"):
        suffix = path.stem[len(prefix) :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _save_result_to_data(result: str, data_dir: Path) -> Path:
    prefix = "mimo"
    next_index = _next_index(prefix, data_dir)
    filename = f"{prefix}{next_index:03d}.txt"
    path = data_dir / filename
    content = f"{result}\n"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")

    extra_headers = _openrouter_headers()
    MODEL = "xiaomi/mimo-v2-flash:free"
    agent = _build_agent(MODEL, api_key, extra_headers)
    data_dir = _ensure_data_dir()

    while True:
        topic = _prompt_value("Topic", "AI impact on finance")
        prompt = _build_prompt().format(topic=topic)

        print("Generating content from OpenRouter model...")
        print(f"Model: {MODEL}")
        result = agent.run(prompt).content
        print("\nResult:")
        print(result)

        saved_path = _save_result_to_data(result, data_dir)
        print(f"\nSaved to: {saved_path}")

        again = _prompt_value("Generate another", "y").lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
