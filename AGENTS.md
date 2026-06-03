# AGENTS.md

Guidance for future Codex agents working in ThesisBoard.

## Commands

- Install dependencies: `python -m pip install -r requirements.txt`
- Run app: `streamlit run app.py`
- Run tests: `python -m pytest`

## Working Rules

- Preserve local user data; do not delete or overwrite local databases without explicit user approval.
- Do not commit `.env`, `*.db`, `data/*.db`, `.streamlit/secrets.toml`, cache files, local paths, personal notes, API keys, or brokerage/trading account data.
- Keep paid market-data providers behind adapter boundaries. Do not require paid API keys in V1.
- Do not add brokerage execution, brokerage credentials, or direct buy/sell advice.
- Prefer focused commits with conventional messages such as `feat: add watchlist manager`.
- Add or update tests when changing database, scoring, related asset, or theme aggregation behavior.

## GitHub workflow

- Never push directly to main.
- Every task must use a feature branch.
- Every completed task must be committed and pushed to GitHub.
- Open a PR for every completed task.
- Use draft PRs for incomplete, experimental, or local-review work.
- PR descriptions must include summary, tests run, known limitations, and next steps.
- Do not commit databases, secrets, caches, local paths, personal notes, API keys, or brokerage/trading account data.
- Before pushing, run:
  - `git status`
  - `python -m pytest`, or the project venv equivalent if global python is unavailable
  - any relevant demo script
- Do not merge PRs automatically. User review is required before merge.
- If rebasing is necessary, use `--force-with-lease`, never plain `--force`.

## Testing Expectations

- Database CRUD should be covered by tests.
- Score classification and risk penalty behavior should be covered by tests.
- Related asset mappings and theme aggregation should be covered by tests.
