#!/usr/bin/env python3
"""Fetch MorningKay's public GitHub statistics and update both profile SVGs.

Repository discovery uses paginated GraphQL connections. Commit and line-change
totals use GitHub's contributor statistics endpoint and only accept entries
whose ``author.login`` matches the account, so the result does not depend on a
single commit email address.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_LOGIN = "MorningKay"
GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"
API_VERSION = "2026-03-10"
SVG_PATHS = (Path("dark_mode.svg"), Path("light_mode.svg"))
STAT_IDS = {
    "repos": "stat-repos",
    "contributed": "stat-contributed",
    "stars": "stat-stars",
    "commits": "stat-commits",
    "followers": "stat-followers",
    "loc": "stat-loc",
    "additions": "stat-additions",
    "deletions": "stat-deletions",
}

OWNED_REPOSITORIES_QUERY = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    repositories(
      first: 100
      after: $cursor
      ownerAffiliations: [OWNER]
      isFork: false
      privacy: PUBLIC
    ) {
      nodes { nameWithOwner stargazerCount }
      pageInfo { hasNextPage endCursor }
    }
    followers { totalCount }
  }
}
"""

CONTRIBUTED_REPOSITORIES_QUERY = """
query($login: String!, $cursor: String) {
  user(login: $login) {
    repositoriesContributedTo(
      first: 100
      after: $cursor
      contributionTypes: [COMMIT]
      includeUserRepositories: false
      privacy: PUBLIC
    ) {
      nodes { nameWithOwner }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""


class GitHubAPIError(RuntimeError):
    """An API response cannot safely be used to update published statistics."""


def read_token() -> str:
    """Use Actions' token first, then a local gh login without printing it."""
    for variable in ("GITHUB_TOKEN", "GH_TOKEN"):
        if token := os.getenv(variable):
            return token.strip()

    if shutil.which("gh"):
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    raise SystemExit(
        "GitHub authentication is required. Set GITHUB_TOKEN/GH_TOKEN or run "
        "`gh auth login` before refreshing stats locally."
    )


def request_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "MorningKay-profile-stats",
        "X-GitHub-Api-Version": API_VERSION,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method="POST" if body else "GET")

    try:
        with urlopen(request, timeout=45) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except HTTPError as error:
        raw = error.read()
        try:
            response_body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            response_body = raw.decode(errors="replace")
        return error.code, response_body
    except URLError as error:
        raise GitHubAPIError(f"GitHub request failed for {url}: {error.reason}") from error


def graphql(token: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    status, payload = request_json(
        GRAPHQL_URL,
        token,
        {"query": query, "variables": variables},
    )
    if status != 200:
        raise GitHubAPIError(f"GraphQL returned HTTP {status}: {payload}")
    if not isinstance(payload, dict):
        raise GitHubAPIError("GraphQL returned a non-object response")
    if payload.get("errors"):
        raise GitHubAPIError(f"GraphQL errors: {payload['errors']}")
    if not isinstance(payload.get("data"), dict):
        raise GitHubAPIError("GraphQL response did not contain data")
    return payload["data"]


def fetch_owned_repositories(token: str, login: str) -> tuple[list[dict[str, Any]], int]:
    repositories: list[dict[str, Any]] = []
    followers: int | None = None
    cursor: str | None = None

    while True:
        data = graphql(token, OWNED_REPOSITORIES_QUERY, {"login": login, "cursor": cursor})
        user = data.get("user")
        if not isinstance(user, dict):
            raise GitHubAPIError(f"GitHub user not found: {login}")
        connection = user.get("repositories")
        if not isinstance(connection, dict):
            raise GitHubAPIError("owned repository connection is missing")
        repositories.extend(connection.get("nodes") or [])
        followers = int(user["followers"]["totalCount"])
        page = connection.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        if not cursor:
            raise GitHubAPIError("owned repository pagination has no end cursor")

    assert followers is not None
    return repositories, followers


def fetch_contributed_repositories(token: str, login: str) -> list[dict[str, Any]]:
    repositories: list[dict[str, Any]] = []
    cursor: str | None = None

    while True:
        data = graphql(
            token,
            CONTRIBUTED_REPOSITORIES_QUERY,
            {"login": login, "cursor": cursor},
        )
        user = data.get("user")
        if not isinstance(user, dict):
            raise GitHubAPIError(f"GitHub user not found: {login}")
        connection = user.get("repositoriesContributedTo")
        if not isinstance(connection, dict):
            raise GitHubAPIError("contributed repository connection is missing")
        repositories.extend(connection.get("nodes") or [])
        page = connection.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
        if not cursor:
            raise GitHubAPIError("contributed repository pagination has no end cursor")

    # The API is expected to exclude owned repositories, but de-duplicate defensively.
    return list({repository["nameWithOwner"]: repository for repository in repositories}.values())


def contributor_activity(
    token: str,
    login: str,
    name_with_owner: str,
    max_retries: int,
    initial_delay: float,
) -> tuple[int, int, int] | None:
    """Return commits/additions/deletions for login, or None when absent."""
    endpoint = f"{REST_URL}/repos/{quote(name_with_owner, safe='/')}/stats/contributors"

    for attempt in range(max_retries + 1):
        status, payload = request_json(endpoint, token)
        if status == 202:
            if attempt == max_retries:
                raise GitHubAPIError(
                    f"contributor statistics for {name_with_owner} remained HTTP 202 "
                    f"after {max_retries + 1} attempts"
                )
            delay = min(initial_delay * (2**attempt), 30)
            print(
                f"{name_with_owner}: statistics are being built; retrying in {delay:g}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            continue
        if status in (204, 404):
            print(f"{name_with_owner}: no contributor statistics (HTTP {status}); skipped", file=sys.stderr)
            return None
        if status != 200:
            raise GitHubAPIError(
                f"contributor statistics for {name_with_owner} returned HTTP {status}: {payload}"
            )
        if not isinstance(payload, list):
            raise GitHubAPIError(
                f"contributor statistics for {name_with_owner} returned a non-list response"
            )

        for contributor in payload:
            author = contributor.get("author") or {}
            if str(author.get("login", "")).casefold() != login.casefold():
                continue
            weeks = contributor.get("weeks") or []
            commits = int(contributor.get("total") or 0)
            additions = sum(int(week.get("a") or 0) for week in weeks)
            deletions = sum(int(week.get("d") or 0) for week in weeks)
            return commits, additions, deletions

        print(f"{name_with_owner}: no contributor entry for {login}; skipped", file=sys.stderr)
        return None

    raise AssertionError("retry loop ended unexpectedly")


def fetch_stats(token: str, login: str) -> dict[str, int]:
    owned, followers = fetch_owned_repositories(token, login)
    contributed = fetch_contributed_repositories(token, login)
    related_names = sorted(
        {repository["nameWithOwner"] for repository in owned + contributed},
        key=str.casefold,
    )

    max_retries = int(os.getenv("GITHUB_STATS_MAX_RETRIES", "5"))
    initial_delay = float(os.getenv("GITHUB_STATS_RETRY_DELAY", "2"))
    commits = additions = deletions = 0
    counted_repositories = 0
    for position, name_with_owner in enumerate(related_names, start=1):
        print(
            f"[{position}/{len(related_names)}] fetching contributor statistics: {name_with_owner}",
            file=sys.stderr,
        )
        activity = contributor_activity(
            token,
            login,
            name_with_owner,
            max_retries,
            initial_delay,
        )
        if activity is None:
            continue
        repository_commits, repository_additions, repository_deletions = activity
        commits += repository_commits
        additions += repository_additions
        deletions += repository_deletions
        counted_repositories += 1

    print(
        f"aggregated contributor activity from {counted_repositories}/{len(related_names)} repositories",
        file=sys.stderr,
    )
    return {
        "repos": len(owned),
        "contributed": len(contributed),
        "stars": sum(int(repository.get("stargazerCount") or 0) for repository in owned),
        "commits": commits,
        "followers": followers,
        "loc": additions - deletions,
        "additions": additions,
        "deletions": deletions,
    }


def rendered_values(stats: dict[str, int]) -> dict[str, str]:
    return {
        "repos": f"{stats['repos']:,}",
        "contributed": f"{stats['contributed']:,}",
        "stars": f"{stats['stars']:,}",
        "commits": f"{stats['commits']:,}",
        "followers": f"{stats['followers']:,}",
        "loc": f"{stats['loc']:,}",
        "additions": f"{stats['additions']:,}++",
        "deletions": f"{stats['deletions']:,}--",
    }


def updated_svg(path: Path, values: dict[str, str]) -> str:
    source = path.read_text(encoding="utf-8")
    updated = source
    for stat_name, element_id in STAT_IDS.items():
        pattern = rf'(<tspan\b[^>]*\bid="{re.escape(element_id)}"[^>]*>)[^<]*(</tspan>)'
        updated, replacements = re.subn(
            pattern,
            lambda match, value=values[stat_name]: f"{match.group(1)}{value}{match.group(2)}",
            updated,
        )
        if replacements != 1:
            raise SystemExit(
                f"expected exactly one {element_id} element in {path}, found {replacements}"
            )
    return updated


def update_svgs(values: dict[str, str], dry_run: bool) -> list[Path]:
    pending = {path: updated_svg(path, values) for path in SVG_PATHS}
    changed = [path for path, content in pending.items() if content != path.read_text(encoding="utf-8")]
    if not dry_run:
        for path in changed:
            path.write_text(pending[path], encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login",
        default=os.getenv("GITHUB_REPOSITORY_OWNER", DEFAULT_LOGIN),
        help="GitHub login to aggregate (default: repository owner or MorningKay)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and validate values without writing SVG files",
    )
    args = parser.parse_args()

    token = read_token()
    stats = fetch_stats(token, args.login)
    changed = update_svgs(rendered_values(stats), args.dry_run)
    print(
        json.dumps(
            {
                "login": args.login,
                "stats": stats,
                "changed": [str(path) for path in changed],
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
