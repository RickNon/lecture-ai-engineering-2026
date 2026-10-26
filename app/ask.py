"""第1回: 素のLLMと会話するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールは持たないが、同じ Chat セッションで会話履歴を保つ呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

def answer_question(
    question: str, chat, temperature: float = 0.0
) -> types.GenerateContentResponse:
    response = chat.send_message(
        question,
        config=types.GenerateContentConfig(
            candidate_count=1, temperature=temperature
        ),
    )
    return response

def ask(question: str, chat, temperature: float = 0.0) -> str:
    started_at = time.perf_counter()
    response = answer_question(question, chat, temperature)
    elapsed_seconds = time.perf_counter() - started_at

    print(f"[usage] 応答時間: {elapsed_seconds:.2f}秒")
    usage = response.usage_metadata
    if usage is not None:
        print(f"[usage] 入力トークン数: {usage.prompt_token_count}")
        print(f"[usage] 出力トークン数: {usage.candidates_token_count}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに複数回質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)

    try:
        print(f"[質問] {args.question}\n")
        print(ask(args.question, chat, args.temperature))

        while True:
            question = input("\n質問（終了するには exit）> ").strip()
            if question.lower() in {"exit", "quit", "終了"}:
                break
            if not question:
                continue
            print(f"\n[質問] {question}\n")
            print(ask(question, chat, args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
