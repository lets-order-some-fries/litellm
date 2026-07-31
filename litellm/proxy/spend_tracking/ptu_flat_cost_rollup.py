"""
Daily rollup for per-model PTU (provisioned throughput) flat cost.

v1 reads PTU config straight off the model deployment
(``LiteLLM_ProxyModelTable.model_info``): a deployment carrying ``ptu_count``
and ``cost_per_ptu_per_hour`` accrues flat cost of
``ptu_count * cost_per_ptu_per_hour * active_hours`` for a given UTC day, where
``active_hours`` is the overlap between the day and the optional
``[ptu_effective_from, ptu_effective_to)`` window (a window opening at 23:00
charges one hour that day). The amount is written to ``LiteLLM_DailyTeamSpend``
under a sentinel api_key so the rows are distinguishable from per-request rows
and share the existing unique constraint.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from itertools import groupby
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    PTU_ROLLUP_JOB_ID,
    PTU_ROLLUP_LOCK_TTL_SECONDS,
    PTU_SENTINEL_API_KEY,
)

if TYPE_CHECKING:
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient

_HOURS_PER_DAY = 24
_UPSERT_ATTEMPTS = 3
_UPSERT_RETRY_BACKOFF_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class RollupResult:
    day: date
    models_processed: int
    rows_written: int
    rows_failed: int = 0


@dataclass(frozen=True, slots=True)
class PTUModel:
    """A model deployment carrying valid manual PTU config."""

    model_id: str
    model_name: str
    team_id: str
    ptu_count: int
    cost_per_ptu_per_hour: float
    effective_from: datetime | None = None
    effective_to: datetime | None = None


def _parse_utc_datetime(value: object) -> datetime | None:
    """Parse a model_info datetime (ISO string or datetime) into a UTC-aware datetime, else None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _public_model_name(row: object, model_info: Mapping[str, object]) -> str:
    """The name an operator recognises for this deployment.

    Creating a team-scoped deployment rewrites model_name to a synthetic routing key
    (``model_name_<team_id>_<uuid4>``) and keeps the chosen name in
    ``model_info.team_public_model_name``. PTU config is only accepted alongside a
    team_id, so every PTU deployment carries that synthetic name; keying the sentinel
    row on it would file each charge under a UUID that no usage view can resolve and
    that never lines up with the same model's request rows.
    """
    public_name = model_info.get("team_public_model_name")
    if isinstance(public_name, str) and public_name:
        return public_name
    return str(getattr(row, "model_name", "") or "")


def _parse_ptu_model(row: object) -> PTUModel | None:
    """Return a PTUModel when the deployment carries valid manual PTU config, else None.

    Valid means model_info has a positive ptu_count, a non-negative
    cost_per_ptu_per_hour, and a team_id (1 model -> 1 team).
    """
    model_info = getattr(row, "model_info", None)
    if isinstance(model_info, str):
        try:
            model_info = json.loads(model_info)
        except (TypeError, ValueError):
            return None
    if not isinstance(model_info, dict):
        return None
    ptu_count = model_info.get("ptu_count")
    cost_per_hour = model_info.get("cost_per_ptu_per_hour")
    team_id = model_info.get("team_id")
    if ptu_count is None or cost_per_hour is None or not team_id:
        return None
    try:
        ptu_count_int = int(ptu_count)
        cost_per_hour_float = float(cost_per_hour)
    except (TypeError, ValueError):
        return None
    if ptu_count_int <= 0 or cost_per_hour_float < 0:
        return None
    raw_from = model_info.get("ptu_effective_from")
    raw_to = model_info.get("ptu_effective_to")
    effective_from = _parse_utc_datetime(raw_from)
    effective_to = _parse_utc_datetime(raw_to)
    # A present-but-unparseable bound would read as "no bound" and silently widen the
    # window to the whole day, so the deployment is skipped until the config is fixed
    if (raw_from is not None and effective_from is None) or (raw_to is not None and effective_to is None):
        return None
    if effective_from is not None and effective_to is not None and effective_to <= effective_from:
        return None
    return PTUModel(
        model_id=str(getattr(row, "model_id", "") or ""),
        model_name=_public_model_name(row, model_info),
        team_id=str(team_id),
        ptu_count=ptu_count_int,
        cost_per_ptu_per_hour=cost_per_hour_float,
        effective_from=effective_from,
        effective_to=effective_to,
    )


def _active_hours_on_day(model: PTUModel, day: date) -> float:
    """Hours the model's PTU window overlaps ``day`` (UTC), clamped to [0, 24]."""
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    start = max(day_start, model.effective_from) if model.effective_from else day_start
    end = min(day_end, model.effective_to) if model.effective_to else day_end
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 3600.0


def _compute_daily_flat_cost(model: PTUModel, day: date) -> float:
    """Flat cost for ``day``: ptu_count * cost_per_ptu_per_hour * active_hours."""
    return float(model.ptu_count) * model.cost_per_ptu_per_hour * _active_hours_on_day(model, day)


@dataclass(frozen=True, slots=True)
class _PTUCharge:
    """One sentinel row's worth of flat cost for a (team, model name) on a day."""

    team_id: str
    model_name: str
    flat_cost: float
    source_model_id: str


def _aggregate_charges(ptu_models: tuple[PTUModel, ...], day: date) -> tuple[_PTUCharge, ...]:
    """Sum flat cost per (team_id, model name) so that multiple deployments sharing a
    model name in one team collapse into a single sentinel row rather than overwriting
    each other on the (team, date, model) unique key. Zero-cost groups are dropped."""
    priced = sorted(
        ((m.team_id, m.model_name, m.model_id, _compute_daily_flat_cost(m, day)) for m in ptu_models),
        key=lambda r: (r[0], r[1]),
    )
    grouped = (
        (team_id, model_name, tuple(rows))
        for (team_id, model_name), rows in groupby(priced, key=lambda r: (r[0], r[1]))
    )
    return tuple(
        _PTUCharge(
            team_id=team_id,
            model_name=model_name,
            flat_cost=sum(r[3] for r in rows),
            source_model_id=",".join(sorted(r[2] for r in rows)),
        )
        for (team_id, model_name, rows) in grouped
        if sum(r[3] for r in rows) > 0
    )


async def _upsert_ptu_daily_row(
    prisma_client: "PrismaClient",
    *,
    team_id: str,
    model: str,
    date_str: str,
    source_model_id: str,
    flat_cost: float,
) -> None:
    """Idempotent upsert of a sentinel-api_key row on LiteLLM_DailyTeamSpend."""
    where = {  # mutable-ok: prisma upsert filter payload
        "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint": {  # mutable-ok: prisma composite-key filter
            "team_id": team_id,
            "date": date_str,
            "api_key": PTU_SENTINEL_API_KEY,
            "model": model,
            "custom_llm_provider": "",
            "mcp_namespaced_tool_name": "",
            "endpoint": "",
        }
    }
    now = datetime.now(timezone.utc)
    await prisma_client.db.litellm_dailyteamspend.upsert(
        where=where,
        data={  # mutable-ok: prisma upsert data payload
            "create": {  # mutable-ok: prisma create payload
                "team_id": team_id,
                "date": date_str,
                "api_key": PTU_SENTINEL_API_KEY,
                "model": model,
                "custom_llm_provider": "",
                "mcp_namespaced_tool_name": "",
                "endpoint": "",
                "ptu_flat_cost": flat_cost,
                "ptu_source_model_id": source_model_id,
            },
            "update": {  # mutable-ok: prisma update payload
                "ptu_flat_cost": flat_cost,
                "ptu_source_model_id": source_model_id,
                "updated_at": now,
            },
        },
    )


async def _upsert_charge_with_retry(
    prisma_client: "PrismaClient",
    *,
    charge: _PTUCharge,
    date_str: str,
) -> bool:
    """Write one charge, retrying transient failures. Returns False once attempts are spent.

    The upsert is idempotent on the sentinel unique key, so a retry can only rewrite the
    same amount for the same day. Retrying in-run matters because the scheduled job moves
    on to the next date: a write lost here is a day of PTU cost that no later run replays.
    """
    for attempt in range(1, _UPSERT_ATTEMPTS + 1):
        try:
            await _upsert_ptu_daily_row(
                prisma_client,
                team_id=charge.team_id,
                model=charge.model_name,
                date_str=date_str,
                source_model_id=charge.source_model_id,
                flat_cost=charge.flat_cost,
            )
            return True
        except Exception as exc:  # noqa: BLE001  # one bad row must not stop the batch
            if attempt < _UPSERT_ATTEMPTS:
                verbose_proxy_logger.warning(
                    "PTU rollup: upsert attempt %d/%d failed for team=%s model=%s day=%s: %s",
                    attempt,
                    _UPSERT_ATTEMPTS,
                    charge.team_id,
                    charge.model_name,
                    date_str,
                    exc,
                )
                await asyncio.sleep(_UPSERT_RETRY_BACKOFF_SECONDS * attempt)
                continue
            verbose_proxy_logger.error(
                "PTU rollup: upsert failed after %d attempts for team=%s model=%s day=%s "
                "(rerun the rollup for that date to recover): %s",
                _UPSERT_ATTEMPTS,
                charge.team_id,
                charge.model_name,
                date_str,
                exc,
            )
    return False


async def run_ptu_flat_cost_rollup(
    prisma_client: "PrismaClient",
    target_date: date | None = None,
) -> RollupResult:
    """Rollup one UTC day of flat PTU cost across all PTU-configured model deployments.

    Defaults to yesterday UTC. Authoritative for the day: it upserts the current charges
    first, then deletes the day's sentinel rows this run did not refresh, so a
    since-removed, invalidated, or now-out-of-window deployment leaves no stale charge.

    The prune predicate is ``updated_at < run_started`` rather than "not in the charge
    set I computed", which matters under concurrency: whether a row is garbage becomes a
    property of the row instead of one run's in-memory config snapshot, so a run can
    never delete a row a concurrent run just wrote. It is still skipped when any charge
    failed to write, since a row whose replacement never landed would look unrefreshed.
    """
    day = target_date or (datetime.now(timezone.utc).date() - timedelta(days=1))

    if prisma_client is None:
        verbose_proxy_logger.warning("PTU rollup: prisma_client is None, skipping")
        return RollupResult(day=day, models_processed=0, rows_written=0)

    date_str = day.isoformat()
    run_started = datetime.now(timezone.utc)

    rows = await prisma_client.db.litellm_proxymodeltable.find_many()
    ptu_models = tuple(parsed for parsed in (_parse_ptu_model(row) for row in rows) if parsed is not None)
    charges = _aggregate_charges(ptu_models, day)

    rows_written = 0
    for charge in charges:
        if await _upsert_charge_with_retry(prisma_client, charge=charge, date_str=date_str):
            rows_written += 1
    rows_failed = len(charges) - rows_written

    if rows_failed:
        # A charge that never landed leaves its row looking unrefreshed, so the prune
        # would delete the very row the failed write was meant to replace
        verbose_proxy_logger.warning(
            "PTU rollup: %d charge(s) failed for %s, skipping the prune so a row whose "
            "replacement did not land is not deleted; rerun that date to reconcile",
            rows_failed,
            date_str,
        )
    else:
        await _prune_unrefreshed_sentinel_rows(prisma_client, date_str=date_str, run_started=run_started)

    verbose_proxy_logger.info(
        "PTU rollup for %s: %d PTU models processed, %d rows written, %d rows failed",
        date_str,
        len(ptu_models),
        rows_written,
        rows_failed,
    )
    return RollupResult(
        day=day,
        models_processed=len(ptu_models),
        rows_written=rows_written,
        rows_failed=rows_failed,
    )


async def run_scheduled_ptu_rollup(
    prisma_client: "PrismaClient",
    pod_lock_manager: "PodLockManager | None" = None,
    target_date: date | None = None,
    alert: Callable[[str], Awaitable[None]] | None = None,
) -> RollupResult | None:
    """Run the daily rollup under a cross-pod lock so only one proxy reconciles a day.

    Every proxy process schedules this cron, and the read-charge-prune sequence is not
    atomic: two pods reading different config snapshots can have the loser's prune delete
    a row the winner just wrote. Returns None when another pod holds the lock, since that
    pod is doing the work. A deployment without a Redis-backed lock manager runs
    unguarded, as ``SpendLogCleanup`` does, and so does a run that cannot reach Redis at
    all: the lock exists to avoid duplicate work, so no lock problem may cost a day.

    The lease is a fixed TTL with no renewal, so a long scan can outlive it. That costs
    duplicate work rather than correctness: the upserts are idempotent on the sentinel
    key and the prune reads only the row's own timestamp, so a second pod arriving
    mid-run cannot corrupt the day.
    """
    if pod_lock_manager is None or pod_lock_manager.redis_cache is None:
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert)

    if not await pod_lock_manager.acquire_lock(cronjob_id=PTU_ROLLUP_JOB_ID, ttl=PTU_ROLLUP_LOCK_TTL_SECONDS):
        if await _lock_is_held(pod_lock_manager):
            verbose_proxy_logger.info("PTU rollup: another pod holds the rollup lock, skipping this run")
            return None
        # acquire_lock reports contention and a Redis outage the same way, so an
        # unreachable Redis would otherwise skip the day on every pod at once. The
        # reconcile is safe to run concurrently, so losing the lock costs duplicate
        # work; losing the day costs a team's charges
        verbose_proxy_logger.warning(
            "PTU rollup: could not take the rollup lock and no other pod holds it, "
            "running unguarded rather than skipping the day"
        )
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert)

    try:
        return await _run_and_alert(prisma_client, target_date=target_date, alert=alert)
    finally:
        await pod_lock_manager.release_lock(cronjob_id=PTU_ROLLUP_JOB_ID)


async def _lock_is_held(pod_lock_manager: "PodLockManager") -> bool:
    """True only when the rollup lock is readable and someone is holding it.

    A Redis that cannot be read is reported as "not held" so the caller runs the day
    rather than skipping it; the cost of being wrong here is a duplicate reconcile.
    """
    try:
        lock_key = pod_lock_manager.get_redis_lock_key(PTU_ROLLUP_JOB_ID)
        return bool(await pod_lock_manager.redis_cache.async_get_cache(lock_key))
    except Exception as exc:  # noqa: BLE001  # an unreadable lock must not skip the day
        verbose_proxy_logger.warning("PTU rollup: could not read the rollup lock: %s", exc)
        return False


async def _run_and_alert(
    prisma_client: "PrismaClient",
    *,
    target_date: date | None,
    alert: "Callable[[str], Awaitable[None]] | None",
) -> RollupResult:
    """Run the rollup and raise an operator alert if any team's charge did not land.

    A charge that exhausts its retries leaves that team showing no PTU cost for the date,
    and the scheduled job moves on to the next day rather than replaying it. That is a
    silent underbill unless someone is reading proxy logs, so it is escalated to whatever
    alerting the deployment has configured.
    """
    result = await run_ptu_flat_cost_rollup(prisma_client, target_date=target_date)
    if result.rows_failed and alert is not None:
        try:
            await alert(
                f"PTU flat-cost rollup for {result.day.isoformat()}: {result.rows_failed} of "
                f"{result.rows_written + result.rows_failed} team charges failed to write. Those teams show no PTU "
                f"cost for that date until the rollup is rerun for it."
            )
        except Exception as exc:  # noqa: BLE001  # a broken alert channel must not fail the rollup
            verbose_proxy_logger.error("PTU rollup: could not deliver the failed-charge alert: %s", exc)
    return result


async def _prune_unrefreshed_sentinel_rows(
    prisma_client: "PrismaClient",
    *,
    date_str: str,
    run_started: datetime,
) -> None:
    """Delete the day's PTU sentinel rows this run did not refresh.

    Every charge the run wrote bumps ``updated_at`` past ``run_started``, so anything
    left below that mark is a (team, model) the current config no longer prices. The
    predicate reads only the row, never the caller's config snapshot, which is what
    makes it safe to run twice, out of order, or beside another pod: a row written
    after this run began is out of reach of its delete. Mirrors the retention predicate
    ``SpendLogCleanup`` deletes by."""
    await prisma_client.db.litellm_dailyteamspend.delete_many(
        where={  # mutable-ok: prisma delete filter
            "date": date_str,
            "api_key": PTU_SENTINEL_API_KEY,
            "updated_at": {"lt": run_started},  # mutable-ok: prisma comparison filter
        }
    )


__all__ = (
    "PTU_ROLLUP_JOB_ID",
    "PTU_SENTINEL_API_KEY",
    "PTUModel",
    "RollupResult",
    "run_ptu_flat_cost_rollup",
    "run_scheduled_ptu_rollup",
)
