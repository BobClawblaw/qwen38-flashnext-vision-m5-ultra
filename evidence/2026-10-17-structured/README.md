# Structured output on engine 8e4ba6db (upstream PR #986's head), 2026-10-17

The recipe harness ([BobClawblaw/recipe-harness](https://github.com/BobClawblaw/recipe-harness)) on this Mac (M5 Ultra
256 GB), served by this recipe's build of TensorFold `8e4ba6db` (binary sha256 `d6b9a33d238b...`, `meta.json`):

    python3 -m harness run qwen38-flashnext-vision-m5 --plan plan.toml

- `structured.json`, `structured.summary.json`, `run.log`, `meta.json`: the `structured` check
  (`harness/checks/server.py` at recipe-harness `0baf46f`).
  - Six grammar kinds (`json_object`, `json_schema`, `guided_json`, `guided_regex`, `guided_choice`,
    `guided_grammar`), each greedy, sampled (temperature 0.8, top-k 20, top-p 0.95, seed 3) and with thinking.
  - Each case keeps its request, reply, finish, usage and seconds. Greedy and sampled cases are sent again with
    `"draft": false` (`plain_text`, `plain_seconds`, `plain_usage`).
  - Result: 18/18 obey, drafted equal to undrafted 12/12, a regex that cannot compile answered with a 400.
- `structured_speed.*`: the `structured_speed` check on the same build (run `20261017-093813`). Two long JSON
  replies with the grammar off and on, on one stream, undrafted, two grammar streams at once, and a plain stream beside
  a grammar one. Each streamed reply keeps its usage, time to its first token and seconds.
