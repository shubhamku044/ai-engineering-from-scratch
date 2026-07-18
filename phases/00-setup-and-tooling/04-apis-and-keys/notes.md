# Lesson 04 — apis-and-keys · notes

> Drafted from how I explained each concept while building. Mine to revise.

## What this produced
`first_api_call.py` — my first LLM API call, done two ways:
- **SDK** (`openai`) via **Chat Completions**: `client.chat.completions.create(...)`,
  reply at `response.choices[0].message.content`, tokens at `response.usage`.
- **Raw HTTP** via the **Responses API** (`/v1/responses`, `urllib`): reply at
  `raw["output"][0]["content"][0]["text"]`, tokens at `raw["usage"]`.

Same idea, different response shapes — which is exactly why you *read* the response
object instead of assuming its structure. Even the token fields differ:
Chat Completions says `prompt_tokens`/`completion_tokens`; Responses says
`input_tokens`/`output_tokens`.

## Key safety (the point of the lesson)
Never hardcode the API key. If it's in the code, anyone who sees the code can drain my
quota / bill. So the key lives in `.env`, loaded with `load_dotenv()` + `os.getenv`,
and `.env` is in `.gitignore` so git never tracks it. `.env` beats a shell `export`
because it's per-project and portable — a teammate copies `.env.example`, fills their
own key, done. (Also: never `print()` the key — my first draft did, and that leaks it
to scrollback / logs / screenshots.)

## system vs user messages
The `system` message defines the assistant's behaviour, rules, and context. The `user`
message is the actual input/query I'm sending.

## ⚠️ To internalise (mentor taught this — I haven't explained it back yet)
The model's reply comes back with role **`assistant`**. The API is **stateless** — for a
multi-turn chat you append the assistant reply into the `messages` list and resend the
*whole growing history* each call: `[system, user, assistant, user, assistant, ...]`.
"The conversation is a list you keep resending" — the basis of chat memory / agent loops.

## Robustness done right
- The key-not-set guard prints a clear reason to `stderr` (`"OPENAI_API_KEY not set —
  add it to .env"`) and then `sys.exit(1)` — no raw traceback, and the user knows *why*.
- `.env.example` documents the required var (placeholder value) so a teammate knows what
  to set without ever seeing my real key.

## Definition of done (protocol check)
- [x] Wrote first_api_call.py from a blank file, no copying
- [x] Can explain key safety, system vs user, and token usage out loud
- [x] Real call returns a reply + token counts; missing-key guard fires
