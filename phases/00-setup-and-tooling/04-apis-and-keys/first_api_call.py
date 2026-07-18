# Lesson 04 — apis-and-keys · first OpenAI API call
# Load OPENAI_API_KEY from the environment (never hardcode).
# One call. Print reply + token counts.
import json
import os
import sys
import urllib.request

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def make_first_api_call_via_sdk():
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you tell me a joke?"},
        ],
    )
    return response


def make_first_api_call_via_raw_request():
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": "gpt-4o-mini",
                "input": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! Can you tell me a joke?"},
                ],
            }
        ).encode("utf-8"),
    )

    json_res = urllib.request.urlopen(request).read().decode("utf-8")
    return json.loads(json_res)


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set — add it to .env", file=sys.stderr)
        sys.exit(1)

    print("Making first API call via SDK...")
    response = make_first_api_call_via_sdk()
    print("Response:", response.choices[0].message.content)
    print("Token usage:", response.usage)
    print("=========")
    print("Making first api call via raw request...")
    raw_response = make_first_api_call_via_raw_request()
    print("Raw response:", raw_response["output"][0]["content"][0]["text"])
    print("Raw token usage:", raw_response["usage"])
