"""Tests for the per-model PTU flat-cost daily rollup."""

import types
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm.proxy.spend_tracking.ptu_flat_cost_rollup as ptu_rollup
from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.proxy.spend_tracking.ptu_flat_cost_rollup import (
    PTUModel,
    _active_hours_on_day,
    _compute_daily_flat_cost,
    _parse_ptu_model,
    run_ptu_flat_cost_rollup,
    run_scheduled_ptu_rollup,
)

DAY = date(2026, 7, 30)


def _model_row(model_id="m1", model_name="gpt-4o-mini-ptu", model_info=None):
    row = MagicMock()
    row.model_id = model_id
    row.model_name = model_name
    row.model_info = model_info
    return row


def _model(**overrides):
    base = dict(model_id="m", model_name="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    base.update(overrides)
    return PTUModel(**base)


def test_full_day_when_no_window():
    # 5 PTU * $2.00/hr * 24h = $240
    assert _compute_daily_flat_cost(_model(), DAY) == pytest.approx(240.0)


def test_window_opening_at_2300_charges_one_hour():
    m = _model(effective_from=datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(1.0)
    # 5 * 2.0 * 1 = 10
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(10.0)


def test_window_closing_at_0600_charges_six_hours():
    m = _model(effective_to=datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(6.0)
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(60.0)


def test_window_fully_covering_day_charges_24h():
    m = _model(
        effective_from=datetime(2026, 7, 1, tzinfo=timezone.utc),
        effective_to=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert _active_hours_on_day(m, DAY) == pytest.approx(24.0)


def test_window_before_day_charges_zero():
    m = _model(effective_to=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0
    assert _compute_daily_flat_cost(m, DAY) == 0.0


def test_window_after_day_charges_zero():
    m = _model(effective_from=datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0


def test_naive_effective_from_is_treated_as_utc():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T23:00:00",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(1.0)


def test_effective_from_with_z_suffix_parses():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 1,
                "cost_per_ptu_per_hour": 1.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T18:00:00Z",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(6.0)


@pytest.mark.parametrize(
    "model_info",
    [
        None,
        {},
        {"ptu_count": 5},
        {"cost_per_ptu_per_hour": 2.0},
        {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0},  # missing team_id
        {"ptu_count": 0, "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
        {"ptu_count": 5, "cost_per_ptu_per_hour": -1.0, "team_id": "t"},
        {"ptu_count": "not-int", "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
    ],
)
def test_parse_ptu_model_rejects_invalid(model_info):
    assert _parse_ptu_model(_model_row(model_info=model_info)) is None


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Keep the upsert retry backoff out of the test runtime."""
    monkeypatch.setattr(ptu_rollup, "_UPSERT_RETRY_BACKOFF_SECONDS", 0)


def _sentinel_row(row_id, team_id, model):
    row = MagicMock()
    row.id = row_id
    row.team_id = team_id
    row.model = model
    return row


def _prisma_with_models(rows, existing_sentinel_rows=()):
    prisma = MagicMock()
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=rows)
    daily = MagicMock()
    daily.find_many = AsyncMock(return_value=list(existing_sentinel_rows))
    daily.upsert = AsyncMock()
    daily.delete_many = AsyncMock()
    prisma.db = types.SimpleNamespace(litellm_proxymodeltable=model_table, litellm_dailyteamspend=daily)
    return prisma, daily


@pytest.mark.asyncio
async def test_rollup_writes_sentinel_row_with_hourly_cost():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1
    created = table.upsert.await_args.kwargs["data"]["create"]
    assert created["api_key"] == PTU_SENTINEL_API_KEY
    assert created["ptu_flat_cost"] == pytest.approx(240.0)
    assert created["team_id"] == "team_x"
    assert created["model"] == "gpt-4o-mini-ptu"


@pytest.mark.asyncio
async def test_rollup_prunes_stale_row_when_config_is_gone():
    prisma, table = _prisma_with_models(
        [_model_row(model_info={"team_id": "team_x"})],
        existing_sentinel_rows=[_sentinel_row("stale-1", "team_x", "gpt-4o-mini-ptu")],
    )

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 0
    table.upsert.assert_not_awaited()
    table.delete_many.assert_awaited_once()
    where = table.delete_many.await_args.kwargs["where"]
    assert where["date"] == DAY.isoformat()
    assert where["api_key"] == PTU_SENTINEL_API_KEY
    # the row is garbage because this run did not refresh it, not because of a key list
    assert "lt" in where["updated_at"]


@pytest.mark.asyncio
async def test_rollup_writes_current_row_before_pruning_and_keeps_it():
    prisma, table = _prisma_with_models(
        [_model_row(model_id="ptu", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})],
        existing_sentinel_rows=[
            _sentinel_row("live", "team_x", "gpt-4o-mini-ptu"),
            _sentinel_row("stale", "team_x", "removed-model"),
        ],
    )

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 1
    table.upsert.assert_awaited_once()
    # the upsert lands before the cutoff is applied, so the refreshed row is out of reach
    upsert_order = table.method_calls.index(("upsert", (), table.upsert.call_args.kwargs))
    assert upsert_order < [c[0] for c in table.method_calls].index("delete_many")


@pytest.mark.asyncio
async def test_rollup_sums_same_name_deployments_into_one_row():
    rows = [
        _model_row(model_id="dep-b", model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "team_x"}),
        _model_row(model_id="dep-a", model_info={"ptu_count": 3, "cost_per_ptu_per_hour": 1.0, "team_id": "team_x"}),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 1
    table.upsert.assert_awaited_once()
    created = table.upsert.await_args.kwargs["data"]["create"]
    assert created["ptu_flat_cost"] == pytest.approx(120.0)
    assert created["ptu_source_model_id"] == "dep-a,dep-b"


@pytest.mark.asyncio
async def test_rollup_skips_zero_active_hours():
    rows = [
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "team_x",
                "ptu_effective_from": "2026-08-01T00:00:00Z",
            }
        )
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 0
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollup_skips_models_without_ptu_config():
    rows = [
        _model_row(model_id="plain", model_info={"team_id": "team_x"}),
        _model_row(model_id="ptu", model_info={"ptu_count": 3, "cost_per_ptu_per_hour": 1.0, "team_id": "team_y"}),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1


def test_parse_ptu_model_accepts_json_string_model_info():
    # Some query paths deliver model_info as a JSON string, not a dict.
    import json as _json

    raw = _json.dumps({"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})
    parsed = _parse_ptu_model(_model_row(model_info=raw))
    assert parsed is not None
    assert parsed.ptu_count == 5 and parsed.team_id == "team_x"


def test_parse_ptu_model_rejects_unparseable_string():
    assert _parse_ptu_model(_model_row(model_info="not-json")) is None


def test_parse_ptu_model_accepts_datetime_object_effective_from():
    # model_info can carry a real datetime object, not just an ISO string.
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "t",
                "ptu_effective_from": datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc),
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bounds",
    [
        {"ptu_effective_from": "not-a-date"},
        {"ptu_effective_to": 12345},
        {"ptu_effective_from": "not-a-date", "ptu_effective_to": 12345},
    ],
)
def test_parse_ptu_model_rejects_malformed_effective_dates(bounds):
    # Treating an unparseable bound as "no bound" would widen the window to the whole
    # day and overcharge, so the deployment is skipped until the config is fixed.
    parsed = _parse_ptu_model(
        _model_row(model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "t", **bounds})
    )
    assert parsed is None


def test_parse_ptu_model_rejects_an_inverted_window():
    # An end at or before the start can only mean a broken config; charging it as an
    # open-ended window would bill a full day.
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 2,
                "cost_per_ptu_per_hour": 1.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-31T12:00:00Z",
                "ptu_effective_to": "2026-07-31T06:00:00Z",
            }
        )
    )
    assert parsed is None


@pytest.mark.asyncio
async def test_rollup_returns_empty_when_prisma_client_is_none():
    result = await run_ptu_flat_cost_rollup(None, target_date=DAY)
    assert result.models_processed == 0
    assert result.rows_written == 0
    assert result.day == DAY


@pytest.mark.asyncio
async def test_rollup_continues_after_a_failed_upsert():
    rows = [
        _model_row(model_id="a", model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"}),
        _model_row(model_id="b", model_info={"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "team_id": "team_b"}),
    ]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    # both models exhausted their retries, and the batch still ran to completion
    assert result.models_processed == 2
    assert result.rows_written == 0
    assert result.rows_failed == 2
    assert table.upsert.await_count == 2 * ptu_rollup._UPSERT_ATTEMPTS


@pytest.mark.asyncio
async def test_rollup_retries_a_transient_upsert_failure_and_succeeds():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=[RuntimeError("connection reset"), None])

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    # the retry writes the day's charge, so nothing is left for a manual rerun
    assert result.rows_written == 1
    assert result.rows_failed == 0
    assert table.upsert.await_count == 2


def _pod_lock(acquired):
    """A lock manager that acquires (or not) and, by default, still owns the lease."""
    lock = MagicMock()
    lock.pod_id = "this-pod"
    lock.redis_cache = MagicMock()
    lock.redis_cache.async_get_cache = AsyncMock(return_value="this-pod")
    lock.get_redis_lock_key = MagicMock(return_value="lock-key")
    lock.acquire_lock = AsyncMock(return_value=acquired)
    lock.release_lock = AsyncMock()
    return lock


@pytest.mark.asyncio
async def test_scheduled_rollup_skips_the_run_when_another_pod_holds_the_lock():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=False)

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # the losing pod must not write or prune, or it could delete the winner's fresh rows
    assert result is None
    assert table.upsert.await_count == 0
    assert table.delete_many.await_count == 0
    lock.release_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduled_rollup_runs_and_releases_the_lock_when_it_wins():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=True)

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    assert result is not None and result.rows_written == 1
    lock.acquire_lock.assert_awaited_once()
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_rollup_releases_the_lock_even_when_the_run_raises():
    prisma, table = _prisma_with_models([])
    prisma.db.litellm_proxymodeltable.find_many = AsyncMock(side_effect=RuntimeError("db down"))
    lock = _pod_lock(acquired=True)

    with pytest.raises(RuntimeError):
        await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # a stuck lock would block every later run until its TTL expires
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_rollup_runs_unguarded_without_a_redis_backed_lock():
    rows = [_model_row(model_info={"ptu_count": 1, "cost_per_ptu_per_hour": 1.0, "team_id": "team_a"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=True)
    lock.redis_cache = None

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    # single-writer deployments have no lock to take, and must still reconcile the day
    assert result is not None and result.rows_written == 1
    lock.acquire_lock.assert_not_awaited()

    assert await run_scheduled_ptu_rollup(prisma, target_date=DAY) is not None


@pytest.mark.asyncio
async def test_rollup_skips_the_prune_when_a_replacement_write_failed():
    # The deployment was renamed, so the old sentinel row is stale only once its
    # replacement lands. Pruning against the intended charges after a failed write
    # would delete the old row and leave the team with no charge at all.
    prisma, table = _prisma_with_models(
        [
            _model_row(
                model_name="renamed-ptu", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
            )
        ],
        existing_sentinel_rows=[_sentinel_row("previous", "t", "old-name-ptu")],
    )
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_failed == 1
    table.delete_many.assert_not_awaited()


class _FakeSentinelTable:
    """In-memory LiteLLM_DailyTeamSpend that honours the sentinel key and prune predicate."""

    def __init__(self, upsert_gate=None):
        self.rows = {}
        self._upsert_gate = upsert_gate

    async def upsert(self, where, data):
        if self._upsert_gate is not None:
            await self._upsert_gate.wait()
        key = where["team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"]
        row_key = (key["team_id"], key["date"], key["api_key"], key["model"])
        self.rows[row_key] = {
            "ptu_flat_cost": data["create"]["ptu_flat_cost"],
            "updated_at": datetime.now(timezone.utc),
        }

    async def delete_many(self, where):
        cutoff = where["updated_at"]["lt"]
        doomed = [
            k
            for k, v in self.rows.items()
            if k[1] == where["date"] and k[2] == where["api_key"] and v["updated_at"] < cutoff
        ]
        for k in doomed:
            del self.rows[k]

    async def find_many(self, where=None):
        return []


def _prisma_for(model_rows, daily_table):
    prisma = MagicMock()
    model_table = MagicMock()
    model_table.find_many = AsyncMock(return_value=model_rows)
    prisma.db = types.SimpleNamespace(litellm_proxymodeltable=model_table, litellm_dailyteamspend=daily_table)
    return prisma


@pytest.mark.asyncio
async def test_an_older_run_cannot_delete_a_newer_runs_row():
    """The race the absolute predicate exists for: an admin renames a PTU model while two
    pods are mid-rollup, so each pod prices a different model name. The pod that started
    first must not be able to delete the charge the second pod just wrote."""
    import asyncio

    gate = asyncio.Event()
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}

    # pod A read the config before the rename and is stalled inside its upsert
    slow_table = _FakeSentinelTable(upsert_gate=gate)
    slow_table.rows = table.rows
    pod_a = asyncio.create_task(
        run_ptu_flat_cost_rollup(
            _prisma_for([_model_row(model_name="old-name", model_info=ptu)], slow_table), target_date=DAY
        )
    )
    await asyncio.sleep(0)  # let pod A capture run_started and reach the gate

    # pod B read the config after the rename and completes first
    await run_ptu_flat_cost_rollup(
        _prisma_for([_model_row(model_name="new-name", model_info=ptu)], table), target_date=DAY
    )
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "new-name") in table.rows

    gate.set()
    await pod_a

    # pod A's cutoff predates every row written during the race, so its delete reaches none
    assert table.rows[("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "new-name")]["ptu_flat_cost"] == pytest.approx(480.0)


@pytest.mark.asyncio
async def test_a_later_clean_run_clears_the_row_the_race_left_behind():
    """The race can leave the renamed-away charge in place for a day; the next run, seeing
    only the current config, must sweep it."""
    table = _FakeSentinelTable()
    ptu = {"ptu_count": 10, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
    stale_key = ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "old-name")
    table.rows[stale_key] = {"ptu_flat_cost": 480.0, "updated_at": datetime(2020, 1, 1, tzinfo=timezone.utc)}

    await run_ptu_flat_cost_rollup(
        _prisma_for([_model_row(model_name="new-name", model_info=ptu)], table), target_date=DAY
    )

    assert stale_key not in table.rows
    assert ("t", DAY.isoformat(), PTU_SENTINEL_API_KEY, "new-name") in table.rows


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_when_a_team_charge_never_landed():
    """A failed charge is a silent underbill: the team shows no PTU cost for the date and
    the next cron run moves on to the next day. It has to reach an operator."""
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.rows_failed == 1
    alert.assert_awaited_once()
    message = alert.await_args.args[0]
    assert DAY.isoformat() in message
    assert "rerun" in message


@pytest.mark.asyncio
async def test_scheduled_rollup_stays_quiet_when_every_charge_landed():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    alert = AsyncMock()

    result = await run_scheduled_ptu_rollup(prisma, target_date=DAY, alert=alert)

    assert result.rows_failed == 0
    alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_broken_alert_channel_does_not_fail_the_rollup():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))

    result = await run_scheduled_ptu_rollup(
        prisma, target_date=DAY, alert=AsyncMock(side_effect=RuntimeError("slack down"))
    )

    # losing the alert must not also lose the run's result or leave the lock held
    assert result.rows_failed == 1


@pytest.mark.asyncio
async def test_scheduled_rollup_alerts_from_under_the_pod_lock_too():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    table.upsert = AsyncMock(side_effect=RuntimeError("db down"))
    lock = _pod_lock(acquired=True)
    alert = AsyncMock()

    await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY, alert=alert)

    alert.assert_awaited_once()
    lock.release_lock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lock_read",
    [
        pytest.param(AsyncMock(side_effect=RuntimeError("redis down")), id="redis-unreachable"),
        pytest.param(AsyncMock(return_value=None), id="lock-key-missing"),
    ],
)
async def test_scheduled_rollup_runs_the_day_when_the_lock_is_unavailable_but_unheld(lock_read):
    """acquire_lock reports contention and a Redis outage identically. Treating both as
    "someone else has it" would skip the day on every pod at once, losing every team's
    charge for that date; the reconcile is safe to run twice, so the day wins."""
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"})]
    prisma, table = _prisma_with_models(rows)
    lock = _pod_lock(acquired=False)
    lock.redis_cache.async_get_cache = lock_read

    result = await run_scheduled_ptu_rollup(prisma, pod_lock_manager=lock, target_date=DAY)

    assert result is not None and result.rows_written == 1
    table.upsert.assert_awaited_once()


def _team_scoped_row(public_name, model_id="m1", team_id="team_x", **ptu):
    """A deployment as POST /model/new actually stores it: synthetic routing name in
    model_name, the operator's chosen name in model_info.team_public_model_name."""
    info = {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": team_id, **ptu}
    info["team_public_model_name"] = public_name
    return _model_row(model_id=model_id, model_name=f"model_name_{team_id}_{model_id}-uuid", model_info=info)


def test_parse_ptu_model_keys_on_the_public_name_not_the_routing_key():
    # PTU requires a team_id, so every PTU deployment carries the synthetic model_name.
    # Keying the charge on it files the cost under a UUID no usage view can resolve.
    parsed = _parse_ptu_model(_team_scoped_row("gpt-4o"))
    assert parsed is not None
    assert parsed.model_name == "gpt-4o"


def test_parse_ptu_model_falls_back_to_model_name_without_a_public_name():
    parsed = _parse_ptu_model(
        _model_row(
            model_name="plain-deployment", model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "t"}
        )
    )
    assert parsed is not None
    assert parsed.model_name == "plain-deployment"


@pytest.mark.parametrize("bad_public_name", ["", None, 123, {"nested": "value"}])
def test_parse_ptu_model_ignores_an_unusable_public_name(bad_public_name):
    row = _model_row(
        model_name="routing-key",
        model_info={
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "team_id": "t",
            "team_public_model_name": bad_public_name,
        },
    )
    parsed = _parse_ptu_model(row)
    assert parsed is not None
    assert parsed.model_name == "routing-key"


@pytest.mark.asyncio
async def test_two_deployments_of_one_public_name_collapse_into_a_single_row():
    """Two deployments of the same public name in a team get different synthetic
    model_names, so before the public-name lookup this collapse could never fire and the
    team got two sentinel rows instead of one summed charge."""
    rows = [
        _team_scoped_row("gpt-4o", model_id="dep-b", ptu_count=2, cost_per_ptu_per_hour=1.0),
        _team_scoped_row("gpt-4o", model_id="dep-a", ptu_count=3, cost_per_ptu_per_hour=1.0),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.rows_written == 1
    created = table.upsert.await_args.kwargs["data"]["create"]
    assert created["model"] == "gpt-4o"
    assert created["ptu_flat_cost"] == pytest.approx(120.0)
    assert created["ptu_source_model_id"] == "dep-a,dep-b"
