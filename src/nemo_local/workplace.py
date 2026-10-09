"""Allowlisted, in-memory simulation; no shell, arbitrary Python, or live APIs."""

import json
from functools import lru_cache

import jsonschema

from ._vendor.workplace_assistant.utils import get_tools

TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
STATE_FIELDS = {
    "email": "_emails",
    "calendar": "_calendar_events",
    "analytics": "_plots_data",
    "project_management": "_project_tasks",
    "customer_relationship_manager": "_crm_data",
}


def new_session():
    return get_tools(TOOLKITS)


def schemas(session):
    return {s["name"]: s for s in session["schemas"]}


def execute(session, name, arguments):
    catalog = schemas(session)
    if name not in catalog or name not in session["functions"]:
        raise ValueError(f"Unknown simulated tool: {name}")
    jsonschema.validate(arguments, catalog[name]["parameters"])
    return session["functions"][name](**arguments)


def snapshot(session):
    frames = {}
    for name, attr in STATE_FIELDS.items():
        df = getattr(session["containers"][name], attr).copy(deep=True)
        for col in df.columns:
            if col not in ["status", "list_name", "board"]:
                df[col] = df[col].str.lower()
        frames[name] = df
    return frames


def replay(actions):
    from .grading import strict_json

    session = new_session()
    for action in actions:
        arguments = action["arguments"]
        if isinstance(arguments, str):
            arguments = strict_json(arguments)
        execute(session, action["name"], arguments)
    return snapshot(session)


@lru_cache(maxsize=2048)
def expected_snapshot(actions_json):
    return replay(json.loads(actions_json))


def same_state(left, right):
    return all(left[k].equals(right[k]) for k in STATE_FIELDS)
