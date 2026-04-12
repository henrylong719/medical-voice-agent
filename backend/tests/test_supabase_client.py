from __future__ import annotations

import warnings

from app import supabase_client


def test_create_supabase_client_avoids_postgrest_deprecation_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        client = supabase_client._create_supabase_client()
        client.table("patients")

    messages = [str(warning.message) for warning in caught]

    assert not any(
        "The 'timeout' parameter is deprecated." in message
        for message in messages
    )
    assert not any(
        "The 'verify' parameter is deprecated." in message
        for message in messages
    )
