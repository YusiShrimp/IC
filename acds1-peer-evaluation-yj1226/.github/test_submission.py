import csv
import io
import pathlib
import subprocess

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).parents[1].absolute()
POINTS_FILE = REPO / "points.csv"

# Must match the placeholder written by points_content() in release.ipynb.
PLACEHOLDER = "Write your comment here and ensure it is in quotes."
MIN_COMMENT_LENGTH = 60


def is_comment(sh):
    """Rows whose name is 'Comment', in any capitalisation."""
    return sh["name"].str.lower() == "comment"


def parse(text):
    """Parse points.csv text.

    Returns (table, problems): a table of stripped strings with columns name and
    points, and a list of format problems (empty if the file is fine).
    """
    # Every non-blank line must have exactly two values: name,points or Comment,"...".
    rows = csv.reader(io.StringIO(text), skipinitialspace=True)
    bad_lines = [
        f"line {i} has {len(row)} value{'' if len(row) == 1 else 's'}"
        for i, row in enumerate(rows, 1)
        if row and len(row) != 2
    ]
    if bad_lines:
        return None, [
            "Every line of points.csv must have exactly two values separated by one "
            "comma (name,points), and the comment must be in double quotes, e.g. "
            'Comment,"your comment, which can contain commas". Check: '
            + "; ".join(bad_lines)
            + "."
        ]

    sh = pd.read_csv(
        io.StringIO(text),
        names=["name", "points"],
        dtype=str,
        keep_default_na=False,
        skipinitialspace=True,
    ).apply(lambda col: col.str.strip())

    problems = []

    n_comments = is_comment(sh).sum()
    if n_comments != 1:
        problems.append(
            f"points.csv must have exactly one Comment row, at the bottom of the file "
            f"(found {n_comments}). Do not remove it, even if you have no comment to make."
        )

    people = sh[~is_comment(sh)]
    not_integers = people[~people["points"].str.fullmatch(r"\d+")]
    if not not_integers.empty:
        problems.append(
            "Points must be whole numbers between 0 and 40. Check the points for: "
            + ", ".join(not_integers["name"])
        )

    return sh, problems


def original_points_text():
    """Return points.csv as it was when first added to the repository."""
    log = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", "points.csv"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
    )
    commits = log.stdout.split()
    if log.returncode != 0 or not commits:
        pytest.fail(
            "Could not find the original points.csv in the git history. "
            "Please contact the group project lead."
        )
    show = subprocess.run(
        ["git", "show", f"{commits[-1]}:points.csv"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8-sig", check=True,
    )
    return show.stdout


# Parse once. If the format is wrong, test_file_format fails and the rest are skipped.
if POINTS_FILE.is_file():
    TABLE, PROBLEMS = parse(POINTS_FILE.read_text(encoding="utf-8-sig"))
else:
    TABLE, PROBLEMS = None, [
        "points.csv is not in the repository. Do not rename, move or delete it."
    ]


@pytest.fixture
def table():
    if PROBLEMS:
        pytest.skip("points.csv has format problems; see test_file_format.")
    return TABLE


@pytest.fixture
def points(table):
    people = table[~is_comment(table)].copy()
    people["points"] = people["points"].astype(int)
    return people


@pytest.fixture
def comment(table):
    return table.loc[is_comment(table), "points"].iloc[0]


def test_file_format():
    assert not PROBLEMS, "\n".join(PROBLEMS)


def test_names_unchanged(points):
    original, _ = parse(original_points_text())
    original = original.loc[~is_comment(original), "name"]
    current = points["name"]

    removed = sorted(set(original) - set(current))
    added = sorted(set(current) - set(original))
    duplicated = sorted(current[current.duplicated()].unique())

    problems = []
    if removed:
        problems.append("Missing or changed: " + ", ".join(removed))
    if added:
        problems.append("Added or changed: " + ", ".join(added))
    if duplicated:
        problems.append("Listed more than once: " + ", ".join(duplicated))

    assert not problems, (
        "points.csv must list exactly the group mates it started with. Do not add "
        "yourself, add or remove rows, or change anyone's name; only change the "
        "numbers and/or the comment row.\n" + "\n".join(problems)
    )


def test_points_range(points):
    out_of_range = points[~points["points"].between(0, 40)]
    assert out_of_range.empty, "Points must be between 0 and 40. Check: " + ", ".join(
        f"{row.name} ({row.points})" for row in out_of_range.itertuples()
    )


def test_points_sum(points):
    n = len(points)
    total = points["points"].sum()
    assert total == n * 20, (
        f"Your points add up to {total}, but they must add up to {n * 20} "
        f"(20 x the {n} group mates listed in points.csv)."
    )


def test_comment(points, comment):
    extreme = points[~points["points"].between(6, 34)]
    if extreme.empty:
        return

    given = ", ".join(f"{row.name} ({row.points})" for row in extreme.itertuples())
    assert comment != PLACEHOLDER and len(comment) > MIN_COMMENT_LENGTH, (
        f"You gave 5 points or fewer, or more than 34 points, to: {given}. "
        "Please explain why in the Comment row at the bottom of points.csv, replacing "
        f"the placeholder text with your own comment (more than {MIN_COMMENT_LENGTH} "
        'characters, in double quotes, e.g. Comment,"your comment").'
    )
