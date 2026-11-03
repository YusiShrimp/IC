# Modern Programming Methods: coding challenge

Today you'll implement two kinds of cellular automata: the Lorenz 96 system and Conway's Game of Life.
[SPEC.md](SPEC.md) specifies what to implement; this README explains how the day runs.

AI tools can do everything in today's challenge with very little effort on your part. We know this. The point of today
is your learning: work through the problem yourself first, then see how that changes the way you use AI.
(Understanding your own code is also what next week's consolidation test asks of you.)

This is an **individual** and **unassessed** exercise.

| File | What it is |
|---|---|
| [SPEC.md](SPEC.md) | The specification: what to implement, and how it's checked |
| `automata.py` | Part 1: your base version |
| `automata_updated.py` | Part 2: your AI-updated version |
| [REFLECTION.md](REFLECTION.md) | A short reflection, in two parts |
| `tests/` | A few tests to start from: add your own |
| [slides.pdf](slides.pdf) | The introductory slides |

## Getting started

1. Clone your repository and open the folder in VS Code:
   ```bash
   git clone https://github.com/ese-ada-lovelace-2026/mpm-coding-challenge-<your-github-username>.git
   ```
2. Activate the module's environment, `conda activate mpm2026`, and select it as the interpreter in VS Code.
3. Check the tests run: `pytest tests/` (they'll fail until you've written the functions).

## The day

### Part 1: base version (`automata.py`)

Write the functions yourself, **without any AI tools**.

- **You may use:** the course notes, your own notes, Python's `help()` and `dir()`, and the official documentation
  for Python, NumPy, SciPy and `pytest`.
- **You may not use:** AI tools of any kind (chatbots, GitHub Copilot in VS Code, AI summaries in search results),
  Stack Overflow, or code from anywhere else. Turn Copilot off in VS Code for Part 1.

When you're ready to move on (a sensible point is around lunchtime, once Lorenz 96 and at least the basic Game of
Life work), fill in Part 1 of [REFLECTION.md](REFLECTION.md), then commit and push.

### Part 2: AI-updated version (`automata_updated.py`)

Start from a copy of your base version:

```bash
cp automata.py automata_updated.py
git add automata_updated.py
git commit -m "Start the AI-updated version"
git push
```

Then use AI tools to understand and improve your code: speed, clarity, structure, tests, and anything you didn't
finish in Part 1. Leave `automata.py` as it is from here on.

- The same tests must still pass (`pytest -U tests/`). AI tools sometimes change things they shouldn't, such as names,
  the order of arguments, or the sign in the Lorenz 96 update (see SPEC.md).
- As you go, think about how understanding the problem first changed what you asked for, and how you judged the
  answers.

### Reflection

Before 17:00, fill in Part 2 of [REFLECTION.md](REFLECTION.md), then commit and push.

## Running the tests and checks

From the top folder of your repository:

```bash
pytest tests/                               # base version
pytest -U tests/                            # AI-updated version
flake8 automata.py automata_updated.py      # code style (Lecture 4)
```

## Committing and pushing

- Commit small and often, with clear messages, and push regularly (Lecture 1).
- Push to `main`: today, don't use other branches.
- Push everything (both files and REFLECTION.md) by **17:00**: feedback is based on what you've pushed by then.

## What happens next

After 17:00, both of your files are run through a larger set of hidden tests (see SPEC.md, *How your code is
checked*), and you'll get automatically generated feedback as an issue in this repository, by Monday.

## Rules

- Work on your own. You're welcome to discuss ideas with others, but don't share your code or let others see it.
- Keep your code in `automata.py` and `automata_updated.py`, and your tests in `tests/`. Scratch notebooks are fine,
  but don't commit them.
