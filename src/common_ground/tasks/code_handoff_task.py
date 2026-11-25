"""Task 4: Code Handoff (Asymmetric PR Review & Deploy Gate) — Author vs Release Gatekeeper."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from common_ground.tasks.base import Role, RoundFeedback, TaskEnv, TaskScore


@dataclass(frozen=True)
class PatchOption:
    option_id: str
    description: str
    kind: str  # "winner" | "trap" | "unfixed" | "partial"
    failure_feedback_a: str
    failure_feedback_b: str
    trap_invariant: str | None = None
    pass_evidence: str = ""
    pre_shock_failure_a: str = ""
    pre_shock_failure_b: str = ""


@dataclass(frozen=True)
class PatchSlot:
    slot_key: str
    title: str
    options: tuple[PatchOption, PatchOption, PatchOption, PatchOption]

    def option_map(self) -> dict[str, PatchOption]:
        return {o.option_id: o for o in self.options}

    def winner_id(self) -> str:
        for o in self.options:
            if o.kind == "winner":
                return o.option_id
        raise RuntimeError(f"Slot {self.slot_key} missing winner option")

    def trap_ids(self) -> list[str]:
        return [o.option_id for o in self.options if o.kind == "trap"]

    def partial_ids(self) -> list[str]:
        return [o.option_id for o in self.options if o.kind == "partial"]


@dataclass(frozen=True)
class CodeHandoffArchetype:
    archetype_id: str
    pr_id: str
    ticket_id: str
    service_name: str
    pr_title: str
    code_excerpts_a: str
    failing_trace_a: str
    pr_comments_a: str
    checks_a: dict[str, tuple[str, str, str, str]]  # CHECK-k -> (name, pre_r4_out, post_r4_out, slot_key)
    invariants_b: tuple[tuple[str, str], ...]  # (INV-k, description)
    slots: tuple[PatchSlot, PatchSlot, PatchSlot, PatchSlot]
    shock_slot: str
    shock_round: int
    shock_alert_a: str

    def slot_map(self) -> dict[str, PatchSlot]:
        return {s.slot_key: s for s in self.slots}


def _build_archetypes(seed: int) -> list[CodeHandoffArchetype]:
    rng = random.Random(50_000 + seed * 3571)
    pr_base = rng.randint(4100, 8900)
    tkt_base = rng.randint(210, 890)
    env_tag = f"svc-{rng.randint(100, 999)}"

    # =========================================================================
    # Archetype 1: settlement_ledger_pr
    # Latin-square winners: (2, 4, 1, 3) -> MIG-2, WIRE-4, IDEM-1, CFG-3
    # Shock slot: runtime
    #   R1-R3 (10 replicas, floor >=380 workers, cap <=400): CFG-4 (10*40=400) PASSES; CFG-3 (10*30=300 < 380) FAILS
    #   R4-R6 (HPA 12 replicas, floor >=350, cap <=400): CFG-4 (12*40=480 > 400) FAILS; CFG-3 (12*30=360) PASSES
    # =========================================================================
    settlement = CodeHandoffArchetype(
        archetype_id="settlement_ledger_pr",
        pr_id=f"PR-{pr_base}",
        ticket_id=f"PAY-{tkt_base}",
        service_name=f"settlement-{env_tag}",
        pr_title="Multi-currency partial refunds & settlement idempotency",
        code_excerpts_a=(
            "1) migrations/04_idem.sql (settlement_entries has 142M rows, hot write path):\n"
            "```sql\n"
            "BEGIN;\n"
            "ALTER TABLE settlement_entries\n"
            "  DROP COLUMN legacy_ext_ref,\n"
            "  ADD COLUMN currency_minor_unit SMALLINT NOT NULL DEFAULT 2,\n"
            "  ALTER COLUMN amount_minor TYPE BIGINT;\n"
            "DROP INDEX IF EXISTS idx_settlement_idem;\n"
            "CREATE UNIQUE INDEX idx_settlement_tenant_idem\n"
            "  ON settlement_entries (tenant_id, idempotency_key) WHERE status <> 'VOIDED';\n"
            "COMMIT;\n"
            "```\n"
            "2) src/wire_codec.py:\n"
            "```python\n"
            "def compute_fx_minor(amount_minor: int, src_mu: int, dst_mu: int, fx_rate: str) -> int:\n"
            "    return int(float(amount_minor) * float(fx_rate) * (10 ** (dst_mu - src_mu)))\n"
            "def encode_event(e: dict) -> bytes:\n"
            '    return json.dumps({"schema_ver": "v3", "entry_id": e["entry_id"],\n'
            '        "amount_minor": e["amount_minor"], "currency": e["currency"],\n'
            '        "currency_minor_unit": e["currency_minor_unit"],\n'
            '        "settled_at": datetime.fromtimestamp(e["settled_at_ms"]/1000, tz=timezone.utc).isoformat()}).encode()\n'
            "```\n"
            "3) src/processor.py:\n"
            "```python\n"
            "def apply_refund(self, tenant_id, charge_id, req_minor, dst_ccy, caller_idem_key):\n"
            "    with self.db.connection() as conn, conn.transaction():\n"
            "        bal = conn.execute('SELECT balance_minor, minor_unit FROM merchant_balances WHERE tenant_id=%s FOR UPDATE', (tenant_id,)).fetchone()\n"
            "        ch = conn.execute('SELECT amount_minor, currency, currency_minor_unit FROM settlement_entries WHERE entry_id=%s', (charge_id,)).fetchone()\n"
            "        n_prev = conn.execute(\"SELECT count(*) FROM settlement_entries WHERE parent_charge_id=%s AND op_type='REFUND'\", (charge_id,)).fetchone()[0]\n"
            "        idem_key = sha256(f'{tenant_id}:{charge_id}:{n_prev}'.encode()).hexdigest()[:32]\n"
            "        debit = req_minor\n"
            "        if dst_ccy != ch['currency']:\n"
            "            q = self.fx.fetch_live_quote(ch['currency'], dst_ccy)\n"
            "            debit = compute_fx_minor(req_minor, ch['currency_minor_unit'], q['dst_mu'], q['rate'])\n"
            "        entry = conn.execute('INSERT INTO settlement_entries (tenant_id, parent_charge_id, op_type, idempotency_key, amount_minor, currency, currency_minor_unit) '\n"
            "                             \"VALUES (%s,%s,'REFUND',%s,%s,%s,%s) RETURNING *\", (tenant_id, charge_id, idem_key, debit, dst_ccy, bal['minor_unit'])).fetchone()\n"
            "        conn.execute('UPDATE merchant_balances SET balance_minor=balance_minor-%s WHERE tenant_id=%s', (debit, tenant_id))\n"
            "```\n"
            "4) deploy/worker.yaml:\n"
            "```yaml\n"
            "replicas: 10\n"
            "DB_POOL_MAX_CONN: 40\n"
            "WORKER_CONCURRENCY: 64\n"
            "FX_QUOTE_TIMEOUT_MS: 2500\n"
            "STATEMENT_TIMEOUT_MS: 30000\n"
            "WIRE_SCHEMA_VERSION: v3\n"
            "DUAL_WRITE_LEGACY_FIELDS: false\n"
            "```"
        ),
        failing_trace_a=(
            "- test_timeout_retry_idempotent: double refund on retry with same X-Idempotency-Key (n_prev=0 then n_prev=1; 2 rows inserted).\n"
            "- test_fx_jpy_usd_rounding: EUR(2)->JPY(0) got 15429 expected 15430 (HALF_EVEN); stored currency_minor_unit=2 (USD balance) instead of 0 (JPY).\n"
            "- test_burst_50_fx_refunds: PoolTimeout after 3.0s; p99=2840ms (40/40 DB conns held in FX HTTP calls); baseline 10-pod throughput floor requires >=380 worker slots.\n"
            "- test_v214_consumer_contract: payout-router@v2.14 failed: missing integer 'amount_cents' and non-int 'settled_at'."
        ),
        pr_comments_a=(
            "- @marta-db: 'Can we run ALTER TABLE overnight, or is expand-contract + keeping legacy_ext_ref mandatory?'\n"
            "- @devon-pay: 'Drop amount_cents—consumers probably upgraded. Bump DB_POOL_MAX_CONN to 120 so FX calls stop starving the pool.'\n"
            "- @ravi-sre: 'Check PgBouncer cap before changing pool size; keep >=380 cluster workers at 10 pods and check why bal[minor_unit] is stored on FX refunds.'"
        ),
        checks_a={
            "CHECK-1": (
                "inspect_table_and_locks",
                "[REPO BASELINE FIXTURE] settlement_entries has 142M rows (4,800 writes/s), zero maintenance windows; v2.14 pods actively query legacy_ext_ref. Schema linter requirements: expand-contract via nullable amount_minor_v2 BIGINT + currency_minor_unit DEFAULT 2 with lock_timeout=1500ms, throttled backfill, CREATE UNIQUE INDEX CONCURRENTLY on (tenant_id, op_type, idempotency_key) WHERE status <> 'VOIDED', and zero plain DROP INDEX or inline ALTER COLUMN TYPE.",
                "[REPO BASELINE FIXTURE] settlement_entries has 142M rows (4,800 writes/s), zero maintenance windows; v2.14 pods still read legacy_ext_ref. Required DDL spec unchanged: lock_timeout=1500ms, expand-contract nullable amount_minor_v2 BIGINT + currency_minor_unit, CREATE/DROP INDEX CONCURRENTLY scoped to (tenant_id, op_type, idempotency_key) WHERE status <> 'VOIDED', and legacy_ext_ref retained.",
                "migration",
            ),
            "CHECK-2": (
                "inspect_wire_consumers",
                "[REPO BASELINE FIXTURE] Active consumer census: payout-router@v2.14 and recon-batch@v2.14 remain live for 14 more days and require integer amount_cents + epoch-ms settled_at dual-written alongside v3 fields. FX test fixture expects exact Decimal ROUND_HALF_EVEN (EUR->JPY = 15430) and destination currency minor_unit=0 persisted rather than merchant balance minor_unit=2.",
                "[REPO BASELINE FIXTURE] Consumer contract requirements unchanged: payout-router@v2.14, recon-batch@v2.14, and ledger-v3 require dual-written integer amount_cents + epoch-ms settled_at, exact Decimal ROUND_HALF_EVEN (15430 JPY), and destination currency minor_unit=0.",
                "wire",
            ),
            "CHECK-3": (
                "inspect_idem_traffic",
                "[REPO BASELINE FIXTURE] Production gateway traffic profile: 504 retries reuse identical (tenant_id, op_type, X-Idempotency-Key), while distinct partial refunds on one charge send distinct X-Idempotency-Key values; concurrent retries hit multiple pods within 15ms. Spec requires atomic DB INSERT ... ON CONFLICT (tenant_id, op_type, idempotency_key) WHERE status <> 'VOIDED' DO NOTHING returning the original row without re-debiting.",
                "[REPO BASELINE FIXTURE] Replay spec unchanged: 50-way concurrent duplicate retries must yield 0 double debits via atomic DB ON CONFLICT on (tenant_id, op_type, caller_idem_key) while allowing multiple distinct partial refunds on the same charge.",
                "idempotency",
            ),
            "CHECK-4": (
                "inspect_pgbouncer_budget",
                "[REPO BASELINE FIXTURE] Pre-soak topology runs 10 fixed replicas against a hard 400-connection PgBouncer cluster budget (max 40/pod) with a baseline peak throughput floor of >=380 worker slots (38+/pod at 10 replicas). Reference benchmark with FX HTTP moved before DB tx (400ms cached quote) and statement_timeout=1500ms achieves p99=92ms (<250ms SLO).",
                "[REPO BASELINE FIXTURE] Post-soak HPA topology: cluster scaled to 12 replicas against the same fixed 400-connection PgBouncer cap (max 33.3/pod) with updated throughput floor >=350 slots. Reference benchmark at 30 conns/pod and 30 workers/pod (12*30=360 <= 400) with FX before DB tx achieves p99=88ms (<250ms SLO) with 40 conns headroom.",
                "runtime",
            ),
        },
        invariants_b=(
            (
                "INV-1",
                "Zero-downtime DDL policy on live hot-write ledger tables: forbid blocking table rewrites, non-concurrent index builds, or dropping columns still read by N-1 pods; enforce lock_timeout <= 2000ms and scope unique indexes to match active write/correlation traffic.",
            ),
            (
                "INV-2",
                "N-1 wire contract & financial precision SLO: keep all legacy fields expected by active prior-version consumers alongside v3 fields during rollout, eliminate float FX rounding drift, and record minor-unit metadata matching the settled transaction currency.",
            ),
            (
                "INV-3",
                "Concurrent retry safety & partial-refund semantics: duplicate retries must never double-debit merchant balances under multi-pod concurrency, while distinct legitimate partial refunds against the same charge must still succeed.",
            ),
            (
                "INV-4",
                "Latency & cluster PgBouncer cap: keep refund p99 < 250ms (never hold DB transactions across external HTTP calls), enforce STATEMENT_TIMEOUT_MS <= 2000 and WORKER_CONCURRENCY <= DB_POOL_MAX_CONN, stay <= 400 total cluster connections across active replicas, and meet the active throughput floor.",
            ),
        ),
        slots=(
            PatchSlot(
                slot_key="migration",
                title="Database Migration (MIG-1..MIG-4)",
                options=(
                    PatchOption(
                        option_id="MIG-1",
                        description="Run single-transaction ALTER TABLE dropping legacy_ext_ref, adding currency_minor_unit DEFAULT 2, rewriting amount_minor inline to BIGINT, and rebuilding idx_settlement_tenant_idem on (tenant_id, op_type, idempotency_key) inside the same migration block.",
                        kind="trap",
                        trap_invariant="INV-1",
                        failure_feedback_a="Migration staging gate FAILED: ACCESS EXCLUSIVE table rewrite locked 142M-row table and broke v2.14 reads on legacy_ext_ref.",
                        failure_feedback_b="INV-1 VIOLATED by MIG-1: inline ALTER TYPE + DROP COLUMN + non-concurrent index caused table lock and v2.14 pod crashes.",
                    ),
                    PatchOption(
                        option_id="MIG-2",
                        description="Expand-contract: retain legacy_ext_ref for a later cleanup PR, add nullable amount_minor_v2 BIGINT + currency_minor_unit DEFAULT 2 with lock_timeout=1500ms, backfill in throttled batches, and build CREATE UNIQUE INDEX CONCURRENTLY on (tenant_id, op_type, idempotency_key) WHERE status <> 'VOIDED'.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="DDL verified: zero ACCESS EXCLUSIVE rewrites, lock_timeout=1500ms, CREATE/DROP INDEX CONCURRENTLY only, legacy_ext_ref retained.",
                    ),
                    PatchOption(
                        option_id="MIG-3",
                        description="Keep legacy_ext_ref untouched and defer schema widening and new index creation to a follow-up quarterly window, leaving INT4 amount_minor, missing currency_minor_unit, and legacy global idx_settlement_idem active with lock_timeout=1500ms.",
                        kind="unfixed",
                        failure_feedback_a="Migration test FAILED: currency_minor_unit column missing and INT4 amount_minor overflow/scoped index defect unresolved.",
                        failure_feedback_b="Slot 'migration' (MIG-3) FAILED CI: zero-downtime safe, but required columns and tenant/op-scoped unique index are still missing.",
                    ),
                    PatchOption(
                        option_id="MIG-4",
                        description="Expand-contract: keep legacy_ext_ref, add nullable amount_minor_v2 BIGINT + currency_minor_unit DEFAULT 2 with lock_timeout=1500ms, run throttled batched backfill, and build CREATE UNIQUE INDEX CONCURRENTLY on (tenant_id, idempotency_key) WHERE status <> 'VOIDED' omitting op_type.",
                        kind="partial",
                        failure_feedback_a="Migration test FAILED: unique index on (tenant_id, idempotency_key) collides when CHARGE and REFUND share a correlation key.",
                        failure_feedback_b="Slot 'migration' (MIG-4) FAILED CI: omitting op_type from the unique index violates (tenant_id, op_type, idempotency_key) scoping.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="wire",
                title="Wire Codec & FX Rounding (WIRE-1..WIRE-4)",
                options=(
                    PatchOption(
                        option_id="WIRE-1",
                        description="Dual-write legacy integer amount_cents and epoch-ms settled_at alongside v3 fields (WIRE_SCHEMA_VERSION=v2_compat, DUAL_WRITE_LEGACY_FIELDS=true), while keeping existing float compute_fx_minor() and storing bal['minor_unit'] on cross-currency rows.",
                        kind="unfixed",
                        failure_feedback_a="Wire/FX test FAILED: test_fx_jpy_usd_rounding still off by 1 JPY (float truncation) and wrong currency_minor_unit.",
                        failure_feedback_b="Slot 'wire' (WIRE-1) FAILED CI: v2.14 schema passes, but FX rounding drift and currency_minor_unit bug remain.",
                    ),
                    PatchOption(
                        option_id="WIRE-2",
                        description="Emit clean v3-only payload (amount_minor + ISO-8601 settled_at with WIRE_SCHEMA_VERSION=v3, DUAL_WRITE_LEGACY_FIELDS=false, dropping amount_cents), rewrite compute_fx_minor() using Decimal ROUND_HALF_EVEN, and persist destination currency's minor_unit.",
                        kind="trap",
                        trap_invariant="INV-2",
                        failure_feedback_a="Wire contract gate FAILED: payout-router@v2.14 and recon-batch@v2.14 crashed on missing amount_cents and ISO settled_at.",
                        failure_feedback_b="INV-2 VIOLATED by WIRE-2: dropping legacy amount_cents and epoch-ms settled_at broke active v2.14 consumers.",
                    ),
                    PatchOption(
                        option_id="WIRE-3",
                        description="Dual-write legacy integer amount_cents and epoch-ms settled_at alongside v3 fields (WIRE_SCHEMA_VERSION=v2_compat, DUAL_WRITE_LEGACY_FIELDS=true), rewrite compute_fx_minor() with Decimal ROUND_HALF_EVEN, while continuing to persist bal['minor_unit'] in settlement_entries.",
                        kind="partial",
                        failure_feedback_a="FX test FAILED: rounding passes, but cross-currency JPY refunds still record USD minor_unit=2 instead of 0.",
                        failure_feedback_b="Slot 'wire' (WIRE-3) FAILED CI: Decimal rounding passes, but destination currency_minor_unit is still overwritten by merchant balance minor_unit.",
                    ),
                    PatchOption(
                        option_id="WIRE-4",
                        description="Dual-write legacy integer amount_cents and epoch-ms settled_at alongside v3 fields (WIRE_SCHEMA_VERSION=v2_compat, DUAL_WRITE_LEGACY_FIELDS=true), compute FX via Decimal ROUND_HALF_EVEN, and persist the destination currency's minor_unit field.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="v2.14+v3 wire contracts verified (amount_cents int + epoch-ms settled_at), Decimal ROUND_HALF_EVEN=15430 JPY, dst minor_unit=0.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="idempotency",
                title="Refund Idempotency (IDEM-1..IDEM-4)",
                options=(
                    PatchOption(
                        option_id="IDEM-1",
                        description="Key by (tenant_id, 'REFUND', caller_idem_key) using atomic INSERT ... ON CONFLICT (tenant_id, op_type, idempotency_key) WHERE status <> 'VOIDED' DO NOTHING, returning the original refund without re-debiting.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="50-way concurrent replay test verified: 0 duplicate debits, original refund returned, distinct partial refunds permitted.",
                    ),
                    PatchOption(
                        option_id="IDEM-2",
                        description="Keep derived sha256(tenant_id:charge_id:n_prev) idempotency key generation inside apply_refund(), wrap the INSERT in a Python UniqueViolation handler, and return the matched row if a hash collision occurs.",
                        kind="unfixed",
                        failure_feedback_a="Idempotency test FAILED: test_timeout_retry_idempotent still double-refunded because retry computed incremented n_prev.",
                        failure_feedback_b="Slot 'idempotency' (IDEM-2) FAILED CI: timeout retries still generate a new key and double-debit merchant balance.",
                    ),
                    PatchOption(
                        option_id="IDEM-3",
                        description="Key strictly by (tenant_id, 'REFUND', caller_idem_key) from X-Idempotency-Key and perform a read-only SELECT pre-check for an existing non-VOIDED row before opening the DB write transaction, without DB ON CONFLICT enforcement.",
                        kind="partial",
                        failure_feedback_a="Concurrency replay test FAILED: parallel identical retries bypassed non-atomic SELECT pre-check and double-debited.",
                        failure_feedback_b="Slot 'idempotency' (IDEM-3) FAILED CI: non-atomic SELECT pre-check allowed concurrent duplicate retries to race.",
                    ),
                    PatchOption(
                        option_id="IDEM-4",
                        description="Key strictly by (tenant_id, charge_id) using atomic INSERT ... ON CONFLICT DO NOTHING and ignore caller X-Idempotency-Key so each charge accepts at most one refund entry across all retry and partial-refund calls.",
                        kind="trap",
                        trap_invariant="INV-3",
                        failure_feedback_a="Idempotency gate FAILED: legitimate second partial refund on the same charge was silently rejected as a duplicate.",
                        failure_feedback_b="INV-3 VIOLATED by IDEM-4: keying only on (tenant_id, charge_id) breaks multi-partial-refund caller-scoped semantics.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="runtime",
                title="Worker Concurrency & Pool Config (CFG-1..CFG-4)",
                options=(
                    PatchOption(
                        option_id="CFG-1",
                        description="Keep fx.fetch_live_quote() inside the DB transaction so balances stay locked during quote retrieval, and bump DB_POOL_MAX_CONN=120 with WORKER_CONCURRENCY=64 and STATEMENT_TIMEOUT_MS=5000 to absorb FX bursts.",
                        kind="trap",
                        trap_invariant="INV-4",
                        failure_feedback_a="Runtime gate FAILED: 10*120=1200 conns crashed cluster PgBouncer (400 cap) and p99 stayed > 2000ms.",
                        failure_feedback_b="INV-4 VIOLATED by CFG-1: DB_POOL_MAX_CONN=120 breached the 400-conn PgBouncer cap and kept FX HTTP inside DB tx.",
                    ),
                    PatchOption(
                        option_id="CFG-2",
                        description="Keep fx.fetch_live_quote() inside the DB transaction with a 400ms HTTP timeout while setting DB_POOL_MAX_CONN=30, WORKER_CONCURRENCY=30, and STATEMENT_TIMEOUT_MS=1500 across worker pods to stay under PgBouncer limits.",
                        kind="unfixed",
                        failure_feedback_a="Burst load test FAILED: p99=2410ms (> 250ms SLO) with 30/30 DB conns still stalled waiting on FX HTTP quotes.",
                        failure_feedback_b="Slot 'runtime' (CFG-2) FAILED CI: connection cap respected, but holding DB tx across FX HTTP calls broke the < 250ms p99 SLO.",
                    ),
                    PatchOption(
                        option_id="CFG-3",
                        description="Fetch/cache FX quote (400ms timeout) strictly BEFORE opening the DB transaction across all worker pods; configure DB_POOL_MAX_CONN=30, WORKER_CONCURRENCY=30, and STATEMENT_TIMEOUT_MS=1500.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="12-replica soak verified: 12*30=360 <= 400 PgBouncer conns (40 headroom), FX outside tx, p99=88ms (<250ms SLO), 0 timeouts.",
                        pre_shock_failure_a="Pre-soak throughput check FAILED: at baseline 10 replicas, 10*30=300 worker slots is below the 380-slot peak throughput floor.",
                        pre_shock_failure_b="Slot 'runtime' (CFG-3) FAILED pre-soak throughput CI: at 10 baseline replicas, 10*30=300 worker slots is below the required 380-slot floor.",
                    ),
                    PatchOption(
                        option_id="CFG-4",
                        description="Fetch/cache FX quote (400ms timeout) strictly BEFORE opening the DB transaction; configure DB_POOL_MAX_CONN=40, WORKER_CONCURRENCY=40, and STATEMENT_TIMEOUT_MS=1500 across worker pods.",
                        kind="partial",
                        failure_feedback_a="Round-4 autoscaler soak FAILED: at 12 replicas, 12*40=480 DB conns exceeded the 400-conn cluster PgBouncer cap!",
                        failure_feedback_b="Slot 'runtime' (CFG-4) FAILED Round-4 soak: 12 burst replicas * 40 conns = 480 > 400 cluster PgBouncer limit.",
                        pass_evidence="Pre-soak 10-replica check passed (10*40=400 conns, >=380 floor, p99=92ms).",
                    ),
                ),
            ),
        ),
        shock_slot="runtime",
        shock_round=4,
        shock_alert_a=(
            "STAGING SOAK UPDATE (Round 4 — PRIVATE TO YOU): Ops just enabled HPA burst autoscaling from 10 to 12 "
            "replicas for this release (updating the throughput floor to >=350 cluster worker slots), while PgBouncer's "
            "cluster connection budget remains hard-capped at 400 connections (400 / 12 = 33.3 max conns/pod). "
            "At 12 replicas, 40 conns/pod (12*40=480 > 400) now FAILS PgBouncer soak; runtime MUST switch to 30 "
            "conns/pod & 30 workers/pod (12*30=360 <= 400, soak p99=88ms < 250ms)."
        ),
    )

    # =========================================================================
    # Archetype 2: auth_oidc_rotation_pr
    # Latin-square winners: (3, 1, 4, 2) -> JWKS-3, JWT-1, REV-4, CFG-2
    # Shock slot: jwt_compat
    #   R1-R3 (iOS v4.8 + audit-gw@v2 requiring 'usr:<uuid>' sub): JWT-4 PASSES; JWT-1 FAILS audit-gw@v2 check
    #   R4-R6 (Android v4.7 UUID crash + audit-gw@v2 waiver to raw UUID): JWT-4 FAILS; JWT-1 PASSES
    # =========================================================================
    auth_oidc = CodeHandoffArchetype(
        archetype_id="auth_oidc_rotation_pr",
        pr_id=f"PR-{pr_base + 11}",
        ticket_id=f"SEC-{tkt_base + 7}",
        service_name=f"auth-oidc-{env_tag}",
        pr_title="Zero-downtime JWKS key rotation & refresh-token replay protection",
        code_excerpts_a=(
            "1) src/jwks_rotator.py:\n"
            "```python\n"
            "def rotate_key(store, new_key):\n"
            "    store.set_jwks([new_key.public_jwk()], cache_control='public, max-age=86400')\n"
            "    store.set_active_signer(new_key.kid)  # immediately drops old kid k-2026-10\n"
            "```\n"
            "2) src/jwt_issuer.py:\n"
            "```python\n"
            "def issue_access_token(user, key):\n"
            "    claims = {'sub': f'usr:{user.uuid}', 'entitlements_v2': user.entitlements,\n"
            "              'iat': now_s(), 'exp': now_s() + 1800}\n"
            "    return jwt.encode(claims, key.private_pem, algorithm='EdDSA', headers={'kid': key.kid})\n"
            "```\n"
            "3) src/refresh_handler.py:\n"
            "```python\n"
            "def rotate_refresh(db, token_hash):\n"
            "    row = db.execute('SELECT used, family_id, user_id FROM refresh_tokens WHERE token_hash=%s', (token_hash,)).fetchone()\n"
            "    if not row:\n"
            "        raise AuthError('unknown')\n"
            "    if row['used']:\n"
            "        raise AuthError('reused')  # leaves rest of family_id active\n"
            "    pair = mint_pair(row['user_id'], row['family_id'])\n"
            "    db.execute('UPDATE refresh_tokens SET used=true WHERE token_hash=%s', (token_hash,))\n"
            "    return pair\n"
            "```\n"
            "4) deploy/token-svc.yaml:\n"
            "```yaml\n"
            "replicas: 8\n"
            "REDIS_POOL_MAX: 200\n"
            "JWKS_FETCH_ON_EVERY_VERIFY: true\n"
            "REQUEST_TIMEOUT_MS: 5000\n"
            "ACCESS_TOKEN_TTL_SEC: 1800\n"
            "```"
        ),
        failing_trace_a=(
            "- test_active_token_during_rotation: 401 InvalidSignature (unknown kid k-2026-10) on valid 10m-old tokens right after rotation.\n"
            "- test_mobile_ios_v4_claims: iOS v4.8 SDK crashed with UnsupportedAlg('EdDSA') and KeyError('roles'); audit-gw@v2 currently expects 'usr:<uuid>' sub.\n"
            "- test_concurrent_refresh_race_and_replay: 2 parallel refreshes on same token BOTH returned 200 OK; replay left family_id sessions alive.\n"
            "- test_verify_burst_p99: p99=410ms (SLO < 60ms) due to live JWKS fetches; auth-redis returned ERR max number of clients reached."
        ),
        pr_comments_a=(
            "- @liam-sec: 'Switch everyone to EdDSA immediately and evict k-2026-10 on rotation so old keys cannot linger.'\n"
            "- @priya-mobile: 'iOS v4.8 is still 14% of traffic and needs RS256 + flat roles; audit-gw@v2 currently expects usr:<uuid> sub (Android soak still queued).'\n"
            "- @chen-infra: 'Why is token-svc opening 200 Redis conns per pod? Verify auth-redis total client budget with the gatekeeper.'"
        ),
        checks_a={
            "CHECK-1": (
                "inspect_token_ttl_and_jwks_cache",
                "[REPO BASELINE FIXTURE] Live access token TTL is 1800s (30m) and edge gateways enforce a 900s (15m) max-stale JWKS fallback cache (45m combined validity window). Rotation fixture requires max-age<=300s, >=15m pre-publish of k-2026-11 before signing, and >=45m retention of k-2026-10 (e.g., 20m pre-publish, 60m retention) to avoid 401s across the overlap window.",
                "[REPO BASELINE FIXTURE] Rotation timing requirements unchanged: 30m token TTL + 15m edge fallback requires max-age<=300s, >=15m pre-publish of k-2026-11, and >=45m retention of k-2026-10.",
                "jwks",
            ),
            "CHECK-2": (
                "inspect_legacy_mobile_sdks",
                "[REPO BASELINE FIXTURE] Pre-soak contract suite: iOS v4.8 (14% traffic) requires RS256 + flat 'roles' array (tolerates 'usr:<uuid>' sub), and audit-gw@v2 currently enforces 'usr:<uuid>' prefixed sub. Android v4.7 soak fixture is queued for Round 4.",
                "[REPO BASELINE FIXTURE] Post-soak contract suite: Android v4.7 (8% traffic) crashed on 'usr:<uuid>' sub (strict UUID parser); Security deployed audit-gw@v2 shim accepting raw unprefixed UUID sub + additive sub_type='user'. All clients (iOS v4.8, Android v4.7, audit-gw@v2) now require RS256 + flat 'roles' + raw UUID sub + sub_type='user'.",
                "jwt_compat",
            ),
            "CHECK-3": (
                "inspect_refresh_race_traces",
                "[REPO BASELINE FIXTURE] Production mobile retry traces show duplicate refreshes hitting different pods within 25ms, bypassing non-locking SELECT and pod-local locks. Spec requires atomic DB conditional UPDATE refresh_tokens SET used=true WHERE token_hash=%s AND used=false RETURNING family_id, plus atomic revocation of the entire family_id on replay.",
                "[REPO BASELINE FIXTURE] Multi-pod concurrency spec unchanged: must prevent double-mints across all 8 pods via DB-level atomic consumption and revoke the full family_id returning 401 on stolen-token replay.",
                "revocation",
            ),
            "CHECK-4": (
                "inspect_auth_redis_budget",
                "[REPO BASELINE FIXTURE] Shared auth-redis allocates 480 total client connections to token-svc across 8 replicas (max 60 conns/pod). Reference load benchmark with 60s background JWKS cache and REDIS_POOL_MAX=48 (8*48=384 <= 480) achieves verify p99=14ms (<60ms SLO) with 0 live per-verify JWKS fetches.",
                "[REPO BASELINE FIXTURE] Redis/verify budget unchanged: 480 total auth-redis conns across 8 pods (max 60/pod), background JWKS caching required for verify p99 < 60ms (reference benchmark 14ms at REDIS_POOL_MAX=48).",
                "runtime",
            ),
        },
        invariants_b=(
            (
                "INV-1",
                "Zero-downtime signing-key lifecycle: edge caches must refresh within <= 300s, newly rotated keys must be pre-published >= 15m before signing switches over, and retiring keys must remain published across the full token TTL plus edge stale-cache fallback window (>= 45m).",
            ),
            (
                "INV-2",
                "N-1 mobile & gateway token compatibility: keep signing algorithm and required claim formats compatible with every active legacy mobile SDK and gateway verifier in staging contract suites; additive claims are permitted, breaking removals are forbidden.",
            ),
            (
                "INV-3",
                "Multi-pod single-use refresh security: concurrent refresh calls across replicas must never double-mint token pairs, and any replay of an already-consumed refresh token must immediately revoke the entire compromised session family.",
            ),
            (
                "INV-4",
                "Auth verify latency & shared cache cap: maintain verify p99 < 60ms (never perform synchronous live JWKS network fetches per verify request) and stay within the 480-connection cluster auth-redis budget across active pods.",
            ),
        ),
        slots=(
            PatchSlot(
                slot_key="jwks",
                title="JWKS Rotation Lifecycle (JWKS-1..JWKS-4)",
                options=(
                    PatchOption(
                        option_id="JWKS-1",
                        description="Replace the published JWKS endpoint with only k-2026-11 immediately, switch the active signer atomically while dropping k-2026-10 in one step, and retain Cache-Control max-age=86400 on edge gateways.",
                        kind="trap",
                        trap_invariant="INV-1",
                        failure_feedback_a="JWKS staging gate FAILED: immediate eviction of k-2026-10 invalidated 100% of active access tokens.",
                        failure_feedback_b="INV-1 VIOLATED by JWKS-1: immediate key eviction and 24h JWKS caching broke in-flight tokens and gateways.",
                    ),
                    PatchOption(
                        option_id="JWKS-2",
                        description="Keep retiring key k-2026-10 published alongside k-2026-11 for 60m after cutover, but leave Cache-Control max-age=86400 unchanged and begin signing new tokens with k-2026-11 immediately without a pre-publish lead window.",
                        kind="unfixed",
                        failure_feedback_a="JWKS rotation test FAILED: gateways cached stale JWKS (max-age=86400) and rejected newly signed k-2026-11 tokens.",
                        failure_feedback_b="Slot 'jwks' (JWKS-2) FAILED CI: missing pre-publish window and 24h cache header broke new-key verification.",
                    ),
                    PatchOption(
                        option_id="JWKS-3",
                        description="Set Cache-Control max-age=300 on JWKS responses, pre-publish k-2026-11 alongside k-2026-10 for 20m before switching the active signer, and retain k-2026-10 in JWKS for 60m before final key removal.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Soak trace verified: max-age=300, 20m pre-publish, 60m retention; both k-2026-10 and k-2026-11 RS256 tokens validated with 0 401s.",
                    ),
                    PatchOption(
                        option_id="JWKS-4",
                        description="Set Cache-Control max-age=300 on the JWKS endpoint, pre-publish k-2026-11 alongside k-2026-10 for 20m before activating the new signer, and then evict k-2026-10 after a 10m post-cutover grace period.",
                        kind="partial",
                        failure_feedback_a="JWKS rotation test FAILED: evicting k-2026-10 after 10m still rejected valid 15–30m-old access tokens.",
                        failure_feedback_b="Slot 'jwks' (JWKS-4) FAILED CI: 10m retention is shorter than the 30m access-token TTL + 15m gateway cache.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="jwt_compat",
                title="JWT Algorithm & Claim Compatibility (JWT-1..JWT-4)",
                options=(
                    PatchOption(
                        option_id="JWT-1",
                        description="Keep RS256 signing across all mobile clients, emit flat legacy 'roles' alongside additive 'entitlements_v2' and 'sub_type': 'user', and format 'sub' as the raw unprefixed UUID string for strict UUID parsers.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Post-soak mobile+gateway verified: iOS v4.8, Android v4.7 (raw UUID sub), and audit-gw@v2 (sub_type='user') all 100% green on RS256.",
                        pre_shock_failure_a="Pre-soak contract check FAILED: audit-gw@v2 currently rejects unprefixed UUID 'sub' (expects 'usr:<uuid>' prefix).",
                        pre_shock_failure_b="Slot 'jwt_compat' (JWT-1) FAILED pre-soak CI: audit-gw@v2 currently requires 'usr:<uuid>' prefixed 'sub'.",
                    ),
                    PatchOption(
                        option_id="JWT-2",
                        description="Switch all newly issued access tokens unconditionally to EdDSA signatures, format 'sub' as prefixed 'usr:<uuid>', and remove the legacy flat 'roles' claim in favor of 'entitlements_v2'.",
                        kind="trap",
                        trap_invariant="INV-2",
                        failure_feedback_a="JWT compat gate FAILED: iOS v4.8 and Android v4.7 rejected EdDSA tokens and crashed on missing 'roles'.",
                        failure_feedback_b="INV-2 VIOLATED by JWT-2: unconditional EdDSA + dropping 'roles' broke live legacy mobile SDKs.",
                    ),
                    PatchOption(
                        option_id="JWT-3",
                        description="Keep RS256 signing and format 'sub' as raw unprefixed UUID with additive 'sub_type': 'user', while retiring the legacy flat 'roles' array to emit only the newer 'entitlements_v2' structure.",
                        kind="unfixed",
                        failure_feedback_a="JWT compat test FAILED: test_mobile_ios_v4_claims still threw KeyError('roles').",
                        failure_feedback_b="Slot 'jwt_compat' (JWT-3) FAILED CI: removing flat 'roles' array still breaks iOS v4.8 claim parsing.",
                    ),
                    PatchOption(
                        option_id="JWT-4",
                        description="Keep RS256 signing, emit both the legacy flat 'roles' array and additive 'entitlements_v2' structure, and preserve 'sub' formatted as prefixed 'usr:<uuid>' for gateway parsing.",
                        kind="partial",
                        failure_feedback_a="Round-4 Android soak FAILED: Android v4.7 threw IllegalArgumentException parsing prefixed 'sub' ('usr:<uuid>') as UUID!",
                        failure_feedback_b="Slot 'jwt_compat' (JWT-4) FAILED Round-4 soak: prefixing 'sub' with 'usr:' broke Android v4.7 strict UUID parsing.",
                        pass_evidence="Pre-soak iOS v4.8 + audit-gw@v2 checks passed (RS256, flat roles, usr:<uuid> sub).",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="revocation",
                title="Refresh Token Atomicity & Replay Guard (REV-1..REV-4)",
                options=(
                    PatchOption(
                        option_id="REV-1",
                        description="Keep separate SELECT and autocommit UPDATE used=true queries inside rotate_refresh(), guarding the handler with a per-pod Python asyncio.Lock keyed by token_hash while returning 401 on duplicate calls.",
                        kind="unfixed",
                        failure_feedback_a="Refresh race test FAILED: per-pod lock did not synchronize across 8 replicas; parallel retries still double-minted.",
                        failure_feedback_b="Slot 'revocation' (REV-1) FAILED CI: in-process lock failed across 8 replicas and replay did not revoke family_id.",
                    ),
                    PatchOption(
                        option_id="REV-2",
                        description="Disable single-use refresh-token rotation across pods so valid refresh tokens remain reusable until natural expiration, eliminating parallel-retry races without mutating rows or revoking family_id.",
                        kind="trap",
                        trap_invariant="INV-3",
                        failure_feedback_a="Security gate FAILED: reusable refresh tokens disabled stolen-token replay detection entirely.",
                        failure_feedback_b="INV-3 VIOLATED by REV-2: disabling single-use refresh rotation violates replay-protection invariant.",
                    ),
                    PatchOption(
                        option_id="REV-3",
                        description="Consume tokens via atomic UPDATE refresh_tokens SET used=true WHERE token_hash=%s AND used=false RETURNING family_id, user_id in one DB tx; if 0 rows match an existing hash, return 401 without revoking sibling tokens in family_id.",
                        kind="partial",
                        failure_feedback_a="Replay test FAILED: atomic race fixed, but replaying a stolen token left active sibling tokens in family_id valid.",
                        failure_feedback_b="Slot 'revocation' (REV-3) FAILED CI: reuse returned 401 but did not revoke the compromised family_id.",
                    ),
                    PatchOption(
                        option_id="REV-4",
                        description="Run atomic UPDATE refresh_tokens SET used=true WHERE token_hash=%s AND used=false RETURNING family_id, user_id in one DB tx; if 0 rows match an existing hash, revoke all tokens in family_id and return 401.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="8-pod concurrency + replay test verified: 0 double-mints, full family_id revoked atomically on reuse.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="runtime",
                title="Verify Caching & Redis Pool Config (CFG-1..CFG-4)",
                options=(
                    PatchOption(
                        option_id="CFG-1",
                        description="Raise REDIS_POOL_MAX=250 per pod across all 8 replicas, keep JWKS_FETCH_ON_EVERY_VERIFY=true to guarantee fresh key lookup on every request, and leave REQUEST_TIMEOUT_MS=5000.",
                        kind="trap",
                        trap_invariant="INV-4",
                        failure_feedback_a="Runtime gate FAILED: 8*250=2000 conns crashed auth-redis (480 cap) and p99 stayed > 400ms.",
                        failure_feedback_b="INV-4 VIOLATED by CFG-1: 2000 total Redis conns exceeded the 480 cap and live JWKS fetch broke < 60ms p99.",
                    ),
                    PatchOption(
                        option_id="CFG-2",
                        description="Set JWKS_FETCH_ON_EVERY_VERIFY=false (60s background in-memory JWKS cache refresh), REDIS_POOL_MAX=48 per pod across 8 pods (384 <= 480 cluster cap), and REQUEST_TIMEOUT_MS=500.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Load soak verified: 0 per-request JWKS fetches, verify p99=14ms (<60ms SLO), 384/480 auth-redis conns.",
                    ),
                    PatchOption(
                        option_id="CFG-3",
                        description="Set REDIS_POOL_MAX=48 per pod (384 total <= 480 cluster cap) and tighten REQUEST_TIMEOUT_MS=500, while keeping JWKS_FETCH_ON_EVERY_VERIFY=true for live key verification on incoming tokens.",
                        kind="unfixed",
                        failure_feedback_a="Latency test FAILED: Redis pool passed, but per-request JWKS HTTPS fetches kept p99 at 340ms (> 60ms SLO).",
                        failure_feedback_b="Slot 'runtime' (CFG-3) FAILED CI: Redis cap respected, but live per-verify JWKS fetches still violated < 60ms p99.",
                    ),
                    PatchOption(
                        option_id="CFG-4",
                        description="Set JWKS_FETCH_ON_EVERY_VERIFY=false (60s background in-memory JWKS refresh) and tighten REQUEST_TIMEOUT_MS=500, while sizing REDIS_POOL_MAX=80 per pod across 8 replicas for burst headroom.",
                        kind="partial",
                        failure_feedback_a="Pool load test FAILED: 8*80=640 Redis connections exceeded auth-redis 480-connection cap.",
                        failure_feedback_b="Slot 'runtime' (CFG-4) FAILED CI: 8 pods * 80 conns = 640 exceeds the 480 auth-redis connection cap.",
                    ),
                ),
            ),
        ),
        shock_slot="jwt_compat",
        shock_round=4,
        shock_alert_a=(
            "STAGING SOAK UPDATE (Round 4 — PRIVATE TO YOU): Canary replay against Android v4.7 mobile fixtures "
            "(8% live traffic) crashed with IllegalArgumentException parsing prefixed 'sub' ('usr:<uuid>') as UUID! "
            "Security just deployed an emergency audit-gw@v2 shim accepting raw unprefixed UUID 'sub' paired with "
            "additive 'sub_type': 'user', lifting the 'usr:' prefix requirement. Prefixed 'usr:<uuid>' now FAILS "
            "mobile soak; JWT claims MUST switch to raw unprefixed UUID 'sub' (keeping RS256 + flat 'roles' + entitlements_v2)."
        ),
    )

    # =========================================================================
    # Archetype 3: search_index_sharding_pr
    # Latin-square winners: (4, 2, 3, 1) -> IDX-4, API-2, UPS-3, CFG-1
    # Shock slot: upsert
    #   R1-R3 (storefront direct GET /_doc/{sku_id} check): UPS-4 (bare sku_id) PASSES; UPS-3 FAILS direct lookup
    #   R4-R6 (multi-tenant SKU-100 collision + query-api proxy live): UPS-4 FAILS; UPS-3 PASSES
    # =========================================================================
    search_shard = CodeHandoffArchetype(
        archetype_id="search_index_sharding_pr",
        pr_id=f"PR-{pr_base + 23}",
        ticket_id=f"SRCH-{tkt_base + 14}",
        service_name=f"search-idx-{env_tag}",
        pr_title="Zero-downtime 16-shard index cutover & bulk catalog upsert",
        code_excerpts_a=(
            "1) scripts/reshard_index.py (catalog_v3 has 18M docs, 45m backfill):\n"
            "```python\n"
            "def reshard(es, db):\n"
            "    es.indices.delete(index='catalog_v3')\n"
            "    es.indices.create(index='catalog_v3', body={'settings': {'number_of_shards': 16}})\n"
            "    backfill_all_docs(es, db, target='catalog_v3')\n"
            "```\n"
            "2) src/query_adapter.py:\n"
            "```python\n"
            "def search_catalog(req, es):\n"
            "    if 'category_ids' not in req.params:\n"
            "        raise BadRequest('category_ids[] required')  # ignores legacy cat_id\n"
            "    res = es.search(index='catalog_read', body=build_query(req.params['category_ids']))\n"
            "    return {'total': {'value': res['hits']['total'], 'relation': 'eq'},\n"
            "            'items': [{'id': h['_id'], 'score': str(h['_score'])} for h in res['hits']['hits']]}\n"
            "```\n"
            "3) src/bulk_upserter.py:\n"
            "```python\n"
            "def build_bulk_actions(events):\n"
            "    return [{'_op_type': 'index', '_index': 'catalog_read',\n"
            "             '_id': str(uuid.uuid4()), '_source': ev['doc']} for ev in events]\n"
            "```\n"
            "4) deploy/indexer.yaml:\n"
            "```yaml\n"
            "replicas: 6\n"
            "BULK_BATCH_DOCS: 25000\n"
            "AVG_DOC_KB: 12\n"
            "INFLIGHT_BATCHES_PER_POD: 4\n"
            "POD_MEM_LIMIT_MB: 2048\n"
            "```"
        ),
        failing_trace_a=(
            "- test_zero_downtime_reshard: query-api returned 404 IndexNotFound and 0 hits during 45m rebuild.\n"
            "- test_storefront_v1_response: storefront-web@v1 failed (missing integer hits_total, string score) and mobile-search cat_id=19 got 400.\n"
            "- test_kafka_redelivery_and_ooo_price: Kafka redelivery indexed 2 duplicate docs for SKU-901; stale version_ts=100 overwrote newer version_ts=105; storefront currently looks up /_doc/{sku_id} directly.\n"
            "- test_indexer_memory_soak: pod OOMKilled (exit 137, RSS 2,310 MB > 2,048 MB) at BULK_BATCH_DOCS=25000, INFLIGHT=4."
        ),
        pr_comments_a=(
            "- @sam-search: 'Just drop catalog_v3 and rebuild faster with BULK_BATCH_DOCS=50000 so it finishes quickly.'\n"
            "- @elena-web: 'storefront-web@v1 sends cat_id, reads integer hits_total + numeric score, and currently fetches /_doc/{sku_id} directly.'\n"
            "- @omar-data: 'catalog.updates has Kafka redeliveries AND out-of-order regional updates; enforce external version_ts ordering (multi-tenant soak runs at Round 4).'"
        ),
        checks_a={
            "CHECK-1": (
                "inspect_alias_topology",
                "[REPO BASELINE FIXTURE] query-api reads from alias catalog_read -> catalog_v3 (18M docs, 45m backfill, currently 4 shards). Reshard cutover spec requires creating 16-shard catalog_v4, enabling live dual-writes during throttled backfill to reach 18M/18M doc parity, atomically swapping catalog_read, and keeping catalog_v3 warm 24h.",
                "[REPO BASELINE FIXTURE] Alias cutover requirements unchanged: 16-shard catalog_v4, live dual-writes + throttled backfill to 18M/18M parity, atomic catalog_read alias swap, and catalog_v3 retained warm 24h.",
                "index",
            ),
            "CHECK-2": (
                "inspect_query_traffic",
                "[REPO BASELINE FIXTURE] Active caller telemetry: storefront-web@v1 (62% traffic) sends cat_id and parses integer hits_total + numeric float score; web-v2 (38%) sends category_ids[] and reads total.value. Contract fixture requires both v1 and v2 inputs/outputs simultaneously.",
                "[REPO BASELINE FIXTURE] Search API contract requirements unchanged: must accept both cat_id and category_ids[], and return integer hits_total + numeric float score alongside additive v2 total object.",
                "api_compat",
            ),
            "CHECK-3": (
                "inspect_kafka_catalog_stream",
                "[REPO BASELINE FIXTURE] catalog.updates carries 1.8% cross-pod redeliveries and 0.4% out-of-order writes with monotonic epoch-ms version_ts. Pre-soak storefront test currently looks up /_doc/{sku_id} directly (multi-tenant soak fixture runs at Round 4).",
                "[REPO BASELINE FIXTURE] Post-soak multi-tenant trace: tenant_alpha and tenant_beta both updated SKU-100, colliding on bare _id=sku_id. Web team deployed query-api lookup proxy translating /_doc/{sku_id} -> {tenant_id}:{sku_id}. Upserts now require tenant-scoped _id=f'{tenant_id}:{sku_id}' with version_type='external' (version=version_ts, ignoring 409).",
                "upsert",
            ),
            "CHECK-4": (
                "inspect_indexer_memory_profile",
                "[REPO BASELINE FIXTURE] Indexer profiling shows 3x working RAM multiplier (12 KB raw * 3 = 36 KB/doc inflight) + 350 MB base RSS inside a 2,048 MB container limit; cluster bulk endpoint rejects raw HTTP payloads > 50 MB (~4,100 docs at 12 KB). Reference benchmark at BULK_BATCH_DOCS=2000 & INFLIGHT=2 uses 24 MB raw payload and 494 MB peak RSS.",
                "[REPO BASELINE FIXTURE] Memory/bulk limits unchanged: <= 50 MB raw bulk HTTP payload and < 1,600 MB peak RSS (reference benchmark: BULK_BATCH_DOCS=2000, INFLIGHT=2 -> 24 MB payload, 494 MB RSS).",
                "runtime",
            ),
        },
        invariants_b=(
            (
                "INV-1",
                "Zero-downtime index topology cutover: never delete the live serving index or drop live catalog mutations during backfill; build the replacement index at the required shard count, maintain full write parity until atomic alias cutover, and keep the old index warm 24h for rollback.",
            ),
            (
                "INV-2",
                "N-1 search API contract: maintain full request-parameter and response-field type compatibility for all active v1 storefront and mobile clients alongside additive v2 response fields.",
            ),
            (
                "INV-3",
                "Idempotent, monotonic catalog indexing: document IDs must prevent duplicates across multi-pod redeliveries while meeting active lookup and tenant-isolation requirements, and stale out-of-order updates must never overwrite newer records.",
            ),
            (
                "INV-4",
                "Indexer pod memory & bulk payload limits: keep peak pod RSS < 1,600 MB (350 MB base + BULK_BATCH_DOCS * 36 KB * INFLIGHT_BATCHES) inside the 2,048 MB limit, and keep raw bulk HTTP request payloads <= 50 MB.",
            ),
        ),
        slots=(
            PatchSlot(
                slot_key="index",
                title="Index Resharding Cutover (IDX-1..IDX-4)",
                options=(
                    PatchOption(
                        option_id="IDX-1",
                        description="Delete live index catalog_v3 in place, recreate catalog_v3 immediately with 16 primary shards and matching mappings, and run a full Postgres backfill directly into the recreated index without alias indirection.",
                        kind="trap",
                        trap_invariant="INV-1",
                        failure_feedback_a="Index gate FAILED: deleting catalog_v3 caused 404 IndexNotFound and 45m search outage.",
                        failure_feedback_b="INV-1 VIOLATED by IDX-1: deleting live catalog_v3 caused immediate production search downtime.",
                    ),
                    PatchOption(
                        option_id="IDX-2",
                        description="Create catalog_v4 keeping the existing 4-shard setting unchanged, enable live dual-writes alongside throttled Postgres backfill to 18M parity, atomically swap catalog_read alias, and retain catalog_v3 warm 24h.",
                        kind="unfixed",
                        failure_feedback_a="Reshard test FAILED: catalog_v4 still has 4 shards instead of the required 16 shards.",
                        failure_feedback_b="Slot 'index' (IDX-2) FAILED CI: zero-downtime swap worked, but shard count remained 4 instead of 16.",
                    ),
                    PatchOption(
                        option_id="IDX-3",
                        description="Create catalog_v4 with 16 shards, backfill all 18M rows from a Postgres snapshot without enabling live dual-writes during the 45m window, atomically swap catalog_read alias, and retain catalog_v3 warm 24h.",
                        kind="partial",
                        failure_feedback_a="Parity check FAILED: omitting live dual-writes lost 45 minutes of concurrent catalog updates during backfill.",
                        failure_feedback_b="Slot 'index' (IDX-3) FAILED CI: missing live dual-write dropped updates written during the 45m backfill.",
                    ),
                    PatchOption(
                        option_id="IDX-4",
                        description="Create new 16-shard catalog_v4, enable live dual-writes to catalog_v3 and catalog_v4 alongside throttled backfill, verify 18M doc parity, atomically swap catalog_read alias, and retain catalog_v3 warm for 24h rollback.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="16-shard cutover verified: 18M/18M doc parity with live dual-write, 0 search 404s, catalog_v3 warm 24h.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="api_compat",
                title="Query & Response Schema Compatibility (API-1..API-4)",
                options=(
                    PatchOption(
                        option_id="API-1",
                        description="Enforce strict v2 query parsing requiring category_ids[] only (rejecting legacy cat_id with 400) and emit the v2-only total object with string-serialized score fields, dropping legacy hits_total.",
                        kind="trap",
                        trap_invariant="INV-2",
                        failure_feedback_a="API contract gate FAILED: storefront-web@v1 and mobile-search broke on missing hits_total and rejected cat_id.",
                        failure_feedback_b="INV-2 VIOLATED by API-1: breaking cat_id, hits_total, and numeric score crashed v1 storefront clients.",
                    ),
                    PatchOption(
                        option_id="API-2",
                        description="Accept both legacy cat_id (normalized to category_ids) and category_ids[] query parameters, and return integer hits_total + numeric float score alongside the additive v2 total response object.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="v1+v2 search contracts verified: cat_id & category_ids[] accepted, int hits_total + float score + v2 total emitted.",
                    ),
                    PatchOption(
                        option_id="API-3",
                        description="Emit integer hits_total and numeric float score alongside the additive v2 total object for storefront-web@v1 compatibility, while still rejecting legacy cat_id query parameters with 400 Bad Request.",
                        kind="unfixed",
                        failure_feedback_a="API test FAILED: mobile-search requests sending cat_id=19 still received 400 Bad Request.",
                        failure_feedback_b="Slot 'api_compat' (API-3) FAILED CI: response fields pass, but legacy cat_id query param is still rejected.",
                    ),
                    PatchOption(
                        option_id="API-4",
                        description="Accept both legacy cat_id (normalized to category_ids) and category_ids[] inputs and emit integer hits_total alongside additive v2 total, while keeping each item's score serialized as a string.",
                        kind="partial",
                        failure_feedback_a="API test FAILED: storefront-web@v1 numeric relevance sort failed on string score.",
                        failure_feedback_b="Slot 'api_compat' (API-4) FAILED CI: score must remain numeric float for storefront-web@v1 compatibility.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="upsert",
                title="Bulk Upsert Dedup & Ordering (UPS-1..UPS-4)",
                options=(
                    PatchOption(
                        option_id="UPS-1",
                        description="Keep random uuid4() document _id generation in build_bulk_actions() and suppress recent Kafka redeliveries using a per-pod in-memory LRU cache keyed on (sku_id, version_ts).",
                        kind="unfixed",
                        failure_feedback_a="Dedup test FAILED: per-pod LRU missed cross-pod Kafka redeliveries and created duplicate SKU-901 docs.",
                        failure_feedback_b="Slot 'upsert' (UPS-1) FAILED CI: random _id + pod-local LRU still produces duplicate documents across pods.",
                    ),
                    PatchOption(
                        option_id="UPS-2",
                        description="Use tenant-scoped deterministic _id=f'{tenant_id}:{sku_id}' across all indexer pods with unconditional index overwrite (no external version_ts guard) so every incoming Kafka event immediately replaces the doc.",
                        kind="trap",
                        trap_invariant="INV-3",
                        failure_feedback_a="Ordering gate FAILED: out-of-order stale event (version_ts=100) silently overwrote live price (version_ts=105).",
                        failure_feedback_b="INV-3 VIOLATED by UPS-2: omitting external version_ts check allowed stale events to corrupt catalog prices.",
                    ),
                    PatchOption(
                        option_id="UPS-3",
                        description="Use tenant-scoped deterministic _id=f'{tenant_id}:{sku_id}' across all indexer pods with version_type='external' (version=version_ts), cleanly ignoring 409 VersionConflict on stale or duplicate Kafka writes.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Multi-tenant soak verified: 0 duplicates, 0 stale overwrites (409 ignored), 0 cross-tenant SKU-100 collisions.",
                        pre_shock_failure_a="Pre-soak lookup test FAILED: storefront direct GET /_doc/{sku_id} returned 404 when _id was prefixed with tenant_id.",
                        pre_shock_failure_b="Slot 'upsert' (UPS-3) FAILED pre-soak CI: storefront direct GET /_doc/{sku_id} lookup currently expects bare sku_id.",
                    ),
                    PatchOption(
                        option_id="UPS-4",
                        description="Use bare deterministic _id=sku_id (omitting tenant_id prefix) with version_type='external' (version=version_ts), cleanly ignoring 409 VersionConflict on stale or duplicate writes.",
                        kind="partial",
                        failure_feedback_a="Round-4 multi-tenant soak FAILED: tenant_beta SKU-100 overwrote tenant_alpha SKU-100 because _id lacked tenant_id!",
                        failure_feedback_b="Slot 'upsert' (UPS-4) FAILED Round-4 soak: bare _id=sku_id caused cross-tenant document overwrite on shared SKU-100.",
                        pass_evidence="Pre-soak single-tenant redelivery/OOO + direct /_doc/{sku_id} lookup passed.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="runtime",
                title="Bulk Batching & Pod Memory Config (CFG-1..CFG-4)",
                options=(
                    PatchOption(
                        option_id="CFG-1",
                        description="Set BULK_BATCH_DOCS=2000 and INFLIGHT_BATCHES_PER_POD=2 (24 MB raw payload; 494 MB peak RSS across 6 pods).",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Memory soak verified: 24 MB raw payload (<=50 MB), 494 MB peak RSS (<1,600 MB), 0 OOMKills.",
                    ),
                    PatchOption(
                        option_id="CFG-2",
                        description="Raise BULK_BATCH_DOCS=50000 with INFLIGHT_BATCHES_PER_POD=4 across all 6 replicas to maximize backfill throughput without throttling batch payload size.",
                        kind="trap",
                        trap_invariant="INV-4",
                        failure_feedback_a="Memory gate FAILED: 600 MB raw payload rejected and pod OOMKilled immediately (~7.5 GB RSS > 2,048 MB).",
                        failure_feedback_b="INV-4 VIOLATED by CFG-2: 50k batch * 4 inflight breached both 50 MB bulk payload cap and 2,048 MB pod memory limit.",
                    ),
                    PatchOption(
                        option_id="CFG-3",
                        description="Keep BULK_BATCH_DOCS=25000 unchanged across indexer pods and cut INFLIGHT_BATCHES_PER_POD from 4 down to 2 to halve concurrent bulk buffers.",
                        kind="unfixed",
                        failure_feedback_a="Memory soak FAILED: 300 MB raw payload (> 50 MB) and ~2,150 MB RSS still OOMKilled the 2,048 MB pod.",
                        failure_feedback_b="Slot 'runtime' (CFG-3) FAILED CI: 25k batch * 2 inflight still exceeded 50 MB bulk cap and 2,048 MB pod memory limit.",
                    ),
                    PatchOption(
                        option_id="CFG-4",
                        description="Set BULK_BATCH_DOCS=6000 and INFLIGHT_BATCHES_PER_POD=2 across all 6 replicas (estimated 782 MB peak pod RSS and 72 MB raw bulk HTTP payload).",
                        kind="partial",
                        failure_feedback_a="Bulk API test FAILED: HTTP 413 Payload Too Large (6000 * 12 KB = 72 MB > 50 MB cluster bulk limit).",
                        failure_feedback_b="Slot 'runtime' (CFG-4) FAILED CI: pod memory fits, but 72 MB raw payload exceeds the 50 MB cluster bulk HTTP cap.",
                    ),
                ),
            ),
        ),
        shock_slot="upsert",
        shock_round=4,
        shock_alert_a=(
            "STAGING SOAK UPDATE (Round 4 — PRIVATE TO YOU): Multi-tenant soak replayed concurrent updates from "
            "tenant_alpha and tenant_beta who BOTH own SKU-100, causing silent cross-tenant overwrites on bare "
            "_id=sku_id! Web team just enabled the query-api proxy translating /_doc/{sku_id} -> {tenant_id}:{sku_id}, "
            "lifting the bare-_id requirement. Bare _id=sku_id now FAILS multi-tenant soak; upsert _id MUST switch "
            "to tenant-scoped f'{tenant_id}:{sku_id}' with version_type='external' (version=version_ts)."
        ),
    )

    # =========================================================================
    # Archetype 4: webhook_billing_outbox_pr
    # Latin-square winners: (1, 3, 2, 4) -> OUT-1, SIG-3, DED-2, CFG-4
    # Shock slot: migration
    #   R1-R3 (1M-row staging + Finance DELIVERED range audit check): OUT-4 (full concurrent index) PASSES; OUT-1 FAILS finance audit
    #   R4-R6 (95M-row 98.2%-DELIVERED soak + Finance moved to replica): OUT-4 FAILS (8.4GB/1180ms); OUT-1 PASSES (140MB/4ms)
    # =========================================================================
    webhook_outbox = CodeHandoffArchetype(
        archetype_id="webhook_billing_outbox_pr",
        pr_id=f"PR-{pr_base + 37}",
        ticket_id=f"BILL-{tkt_base + 29}",
        service_name=f"hook-outbox-{env_tag}",
        pr_title="Transactional billing webhook outbox & dual-secret HMAC rotation",
        code_excerpts_a=(
            "1) migrations/09_outbox.sql (invoice_events has 95M rows, 98.2% DELIVERED):\n"
            "```sql\n"
            "BEGIN;\n"
            "ALTER TABLE invoice_events\n"
            "  DROP COLUMN legacy_hook_url,\n"
            "  ADD COLUMN delivery_status TEXT NOT NULL DEFAULT 'PENDING';\n"
            "CREATE INDEX idx_outbox_pending ON invoice_events (delivery_status, created_at);\n"
            "COMMIT;\n"
            "```\n"
            "2) src/signer.py:\n"
            "```python\n"
            "def sign_and_send(url, payload: dict, new_secret: bytes, ts: int):\n"
            "    sig = hmac_sha256(new_secret, f'{ts}.{json.dumps(payload)}'.encode())\n"
            "    return http.post(url, json=payload, headers={'X-Hook-Signature': f't={ts},v2={sig}'})\n"
            "```\n"
            "3) src/dispatcher.py:\n"
            "```python\n"
            "def poll_and_dispatch(db):\n"
            "    rows = db.execute(\"SELECT id, event_id, url, payload FROM invoice_events WHERE delivery_status='PENDING' ORDER BY created_at LIMIT 100\").fetchall()\n"
            "    for r in rows:\n"
            "        send_webhook(r['url'], r['payload'], headers={'X-Event-Id': str(uuid.uuid4())})\n"
            "        db.execute(\"UPDATE invoice_events SET delivery_status='DELIVERED' WHERE id=%s\", (r['id'],))\n"
            "```\n"
            "4) deploy/dispatcher.yaml:\n"
            "```yaml\n"
            "replicas: 8\n"
            "WORKERS_PER_POD: 40\n"
            "PER_TENANT_MAX_INFLIGHT: 0\n"
            "RETRY_BACKOFF_MODE: fixed_100ms\n"
            "HONOR_RETRY_AFTER: false\n"
            "HTTP_TIMEOUT_MS: 15000\n"
            "```"
        ),
        failing_trace_a=(
            "- test_outbox_poll_and_legacy_read: v1 worker crashed (UndefinedColumn legacy_hook_url); non-concurrent CREATE INDEX blocked writes; finance audit currently queries DELIVERED+created_at on primary.\n"
            "- test_partner_v1_sig_verify: partners on old secret failed 401 (missing v1=); v2 receivers failed HMAC due to body re-serialization mismatch.\n"
            "- test_multi_worker_dedup: 8 pods dispatched the same invoice event 4x with 4 different X-Event-Id headers.\n"
            "- test_tenant_429_storm: slow tenant returning 429 Retry-After:5 triggered 320 concurrent fixed-100ms retries, starving other tenants."
        ),
        pr_comments_a=(
            "- @nina-sec: 'Drop v1= from X-Hook-Signature right away—emitting both v1= and v2= during rotation looks cluttered.'\n"
            "- @tomas-db: 'Build the index CONCURRENTLY and keep legacy_hook_url; Finance currently needs DELIVERED rows indexed too (95M-row bloat soak runs at Round 4).'\n"
            "- @kai-sre: '8*40=320 unlimited workers hammered one tenant with 100ms fixed retries—check outbound NAT socket and per-tenant caps with the gatekeeper.'"
        ),
        checks_a={
            "CHECK-1": (
                "inspect_outbox_table_profile",
                "[REPO BASELINE FIXTURE] invoice_events has 95M prod rows (98.2% DELIVERED, 1.8% PENDING), 3,200 writes/s, zero maintenance windows; v1 workers actively read legacy_hook_url. Pre-soak 1M-row staging runs Finance's DELIVERED+created_at audit query on primary (95M-row bloat soak runs at Round 4).",
                "[REPO BASELINE FIXTURE] Post-soak 95M-row benchmark: full index bloats to 8.4 GB (poll p99=1,180ms > 100ms SLO), whereas partial CREATE INDEX CONCURRENTLY WHERE delivery_status='PENDING' stays 140 MB (poll p99=4ms), with lock_timeout=1500ms and legacy_hook_url kept; Finance DELIVERED audits were rerouted to read replica.",
                "migration",
            ),
            "CHECK-2": (
                "inspect_receiver_sig_audit",
                "[REPO BASELINE FIXTURE] 64% of enterprise webhook receivers still verify v1= (old secret) during the 7-day rotation overlap and validate HMAC over exact raw wire bytes f'{ts}.{raw_body}'. Golden receiver contract requires single-serialized raw_body bytes signed with t=<ts> and both v1= and v2=.",
                "[REPO BASELINE FIXTURE] Receiver signature requirements unchanged: single-serialized raw_body bytes bound to timestamp t=<ts> with both v1= and v2= HMAC signatures.",
                "wire",
            ),
            "CHECK-3": (
                "inspect_dispatcher_claims",
                "[REPO BASELINE FIXTURE] 8 dispatcher pods poll simultaneously; downstream receivers deduplicate strictly on X-Event-Id. Concurrency spec requires atomic DB claiming (FOR UPDATE SKIP LOCKED + IN_FLIGHT lease) without pre-marking DELIVERED, and reusing persistent outbox.event_id as X-Event-Id on every retry.",
                "[REPO BASELINE FIXTURE] Multi-pod dispatch requirements unchanged: FOR UPDATE SKIP LOCKED + IN_FLIGHT lease + persistent outbox.event_id on every attempt.",
                "dedup",
            ),
            "CHECK-4": (
                "inspect_egress_nat_limits",
                "[REPO BASELINE FIXTURE] Outbound NAT proxy enforces 160 max concurrent webhook sockets across 8 pods (max 20 workers/pod); enterprise SLAs cap per-tenant in-flight deliveries at <=10. Reference benchmark at WORKERS_PER_POD=16 (128/160 sockets) & PER_TENANT_MAX_INFLIGHT=8 with exp_jitter + Retry-After honoring keeps 0 NAT drops.",
                "[REPO BASELINE FIXTURE] Egress NAT & backpressure limits unchanged: <=160 cluster sockets across 8 pods (<=20/pod), <=10 per-tenant inflight, exp_jitter backoff, and Retry-After honored.",
                "runtime",
            ),
        },
        invariants_b=(
            (
                "INV-1",
                "Zero-downtime outbox schema & poll latency SLO (p99 < 100ms): never drop columns still read by v1 workers, never build blocking non-concurrent indexes on the live events table, enforce lock_timeout <= 2000ms, and match concurrent index scope to active primary-DB query and bloat constraints.",
            ),
            (
                "INV-2",
                "Dual-secret webhook signing during 7-day rotation: sign the exact transmitted wire body bytes bound to timestamp replay metadata, emitting valid signatures for both retiring and new secrets so active v1 and v2 receivers verify cleanly.",
            ),
            (
                "INV-3",
                "Exactly-once worker claim & retry deduplication: multi-pod dispatchers must claim rows atomically without concurrent duplicates or dropping failed sends, and every retry must carry the event's stable persistent identifier.",
            ),
            (
                "INV-4",
                "Outbound NAT egress & multi-tenant backpressure: stay within 160 total sockets across active pods, cap per-tenant in-flight deliveries at <= 10, apply jittered exponential backoff on failures, and honor downstream Retry-After headers.",
            ),
        ),
        slots=(
            PatchSlot(
                slot_key="migration",
                title="Outbox Schema & Poll Index (OUT-1..OUT-4)",
                options=(
                    PatchOption(
                        option_id="OUT-1",
                        description="Retain legacy_hook_url, add delivery_status TEXT NOT NULL DEFAULT 'PENDING' (lock_timeout=1500ms), and build partial CREATE INDEX CONCURRENTLY idx_outbox_pending ON invoice_events (created_at) WHERE delivery_status='PENDING'.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="95M-row soak verified: partial CONCURRENTLY index 140 MB, poll p99=4ms (<100ms SLO), lock_timeout=1500ms, legacy_hook_url retained.",
                        pre_shock_failure_a="Pre-soak audit check FAILED: Finance's primary DELIVERED+created_at audit query cannot use a PENDING-only partial index.",
                        pre_shock_failure_b="Slot 'migration' (OUT-1) FAILED pre-soak CI: Finance's DELIVERED+created_at audit query currently requires DELIVERED rows indexed on primary.",
                    ),
                    PatchOption(
                        option_id="OUT-2",
                        description="Run single-transaction ALTER TABLE dropping legacy_hook_url, adding delivery_status TEXT NOT NULL DEFAULT 'PENDING', and executing a blocking non-concurrent CREATE INDEX on (delivery_status, created_at) inside the same tx block.",
                        kind="trap",
                        trap_invariant="INV-1",
                        failure_feedback_a="Migration gate FAILED: non-concurrent index locked 95M-row table and dropping legacy_hook_url crashed v1 workers.",
                        failure_feedback_b="INV-1 VIOLATED by OUT-2: DROP COLUMN + non-concurrent index locked table and broke active v1 workers.",
                    ),
                    PatchOption(
                        option_id="OUT-3",
                        description="Retain legacy_hook_url and add delivery_status TEXT NOT NULL DEFAULT 'PENDING' with lock_timeout=1500ms, skipping index creation completely so polling relies on sequential scans over invoice_events.",
                        kind="unfixed",
                        failure_feedback_a="Outbox poll test FAILED: sequential scan on 95M-row table took 4,800ms (> 100ms SLO).",
                        failure_feedback_b="Slot 'migration' (OUT-3) FAILED CI: lock-safe, but missing poll index caused full sequential scans.",
                    ),
                    PatchOption(
                        option_id="OUT-4",
                        description="Retain legacy_hook_url, add delivery_status TEXT NOT NULL DEFAULT 'PENDING' with lock_timeout=1500ms, and build a FULL non-partial CREATE INDEX CONCURRENTLY on (delivery_status, created_at) outside a tx block.",
                        kind="partial",
                        failure_feedback_a="Round-4 95M-row soak FAILED: full non-partial index bloated to 8.4 GB (98.2% DELIVERED) and poll p99 spiked to 1,180ms (> 100ms)!",
                        failure_feedback_b="Slot 'migration' (OUT-4) FAILED Round-4 soak: full non-partial index bloated on 98% DELIVERED rows (p99=1,180ms > 100ms).",
                        pass_evidence="Pre-soak 1M-row staging + Finance DELIVERED audit query passed (CONCURRENTLY, legacy_hook_url retained).",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="wire",
                title="Webhook Dual-HMAC Wire Signing (SIG-1..SIG-4)",
                options=(
                    PatchOption(
                        option_id="SIG-1",
                        description="Serialize raw_body=json.dumps(payload, separators=(',',':')).encode() once, sign f'{ts}.'+raw_body with only the new secret, transmit raw_body bytes, and send X-Hook-Signature: t=<ts>,v2=<new_hmac> dropping v1= immediately.",
                        kind="trap",
                        trap_invariant="INV-2",
                        failure_feedback_a="Signature gate FAILED: 64% of enterprise receivers on old secret rejected webhooks with 401.",
                        failure_feedback_b="INV-2 VIOLATED by SIG-1: dropping v1= during the 7-day rotation window broke 64% of partner receivers.",
                    ),
                    PatchOption(
                        option_id="SIG-2",
                        description="Emit X-Hook-Signature: t=<ts>,v1=<old_hmac>,v2=<new_hmac> using both secrets over f'{ts}.{json.dumps(payload)}', while letting http.post(url, json=payload) re-serialize the request body separately on the wire.",
                        kind="unfixed",
                        failure_feedback_a="Signature test FAILED: body re-serialization differed in whitespace/key order from signed string, causing HMAC mismatch.",
                        failure_feedback_b="Slot 'wire' (SIG-2) FAILED CI: HMAC was not computed over the exact raw wire bytes sent in the HTTP body.",
                    ),
                    PatchOption(
                        option_id="SIG-3",
                        description="Serialize raw_body=json.dumps(payload, separators=(',',':')).encode() once, sign f'{ts}.'+raw_body with both old (v1) and new (v2) secrets, send X-Hook-Signature: t=<ts>,v1=...,v2=..., and POST the exact raw_body bytes.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Dual-HMAC suite verified: single-serialized raw_body + t=<ts>,v1=...,v2=... passes 100% of v1 and v2 receivers.",
                    ),
                    PatchOption(
                        option_id="SIG-4",
                        description="Serialize raw_body=json.dumps(payload, separators=(',',':')).encode() once, POST exact raw_body bytes, and sign raw_body alone with both old (v1) and new (v2) secrets while omitting timestamp t=<ts> from the signed input.",
                        kind="partial",
                        failure_feedback_a="Signature test FAILED: receivers rejected webhooks because signed payload omitted replay-protection timestamp t=<ts>.",
                        failure_feedback_b="Slot 'wire' (SIG-4) FAILED CI: timestamp t=<ts> must be bound into the HMAC signed input.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="dedup",
                title="Worker Claiming & Retry Idempotency (DED-1..DED-4)",
                options=(
                    PatchOption(
                        option_id="DED-1",
                        description="Keep plain non-locking SELECT ... WHERE delivery_status='PENDING' ORDER BY created_at LIMIT 100 across all 8 pods, changing only the X-Event-Id header to reuse persistent outbox.event_id and marking DELIVERED after HTTP 200.",
                        kind="unfixed",
                        failure_feedback_a="Multi-worker test FAILED: 8 pods still claimed the same PENDING rows simultaneously and delivered 4x duplicates.",
                        failure_feedback_b="Slot 'dedup' (DED-1) FAILED CI: missing FOR UPDATE SKIP LOCKED still caused 4x duplicate dispatches across pods.",
                    ),
                    PatchOption(
                        option_id="DED-2",
                        description="Claim rows atomically via SELECT ... WHERE delivery_status='PENDING' ORDER BY created_at LIMIT 50 FOR UPDATE SKIP LOCKED updating to IN_FLIGHT with lease expiry in one tx, and send persistent outbox.event_id as X-Event-Id on every attempt.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="8-pod dispatch test verified: FOR UPDATE SKIP LOCKED + IN_FLIGHT lease + persistent X-Event-Id, 0 duplicates, 0 lost sends.",
                    ),
                    PatchOption(
                        option_id="DED-3",
                        description="Claim rows atomically via SELECT ... LIMIT 50 FOR UPDATE SKIP LOCKED updating to IN_FLIGHT with lease expiry in one tx, while continuing to generate a fresh uuid4() X-Event-Id header on each delivery and retry attempt.",
                        kind="partial",
                        failure_feedback_a="Retry dedup test FAILED: timeout retries sent new uuid4() X-Event-Id headers, causing duplicate invoice processing at receivers.",
                        failure_feedback_b="Slot 'dedup' (DED-3) FAILED CI: retries must reuse persistent outbox.event_id instead of generating fresh uuid4().",
                    ),
                    PatchOption(
                        option_id="DED-4",
                        description="Claim rows via blocking SELECT ... LIMIT 50 FOR UPDATE (without SKIP LOCKED), mark rows DELIVERED before issuing the outbound HTTP POST, and send persistent outbox.event_id as X-Event-Id.",
                        kind="trap",
                        trap_invariant="INV-3",
                        failure_feedback_a="Dispatch gate FAILED: failed HTTP deliveries were permanently lost because rows were marked DELIVERED before send.",
                        failure_feedback_b="INV-3 VIOLATED by DED-4: pre-marking DELIVERED silently dropped failed webhooks and blocking FOR UPDATE serialized pods.",
                    ),
                ),
            ),
            PatchSlot(
                slot_key="runtime",
                title="Dispatcher Egress & Backpressure Config (CFG-1..CFG-4)",
                options=(
                    PatchOption(
                        option_id="CFG-1",
                        description="Keep WORKERS_PER_POD=40 across 8 pods (320 concurrent sockets), PER_TENANT_MAX_INFLIGHT=0 (unbounded), RETRY_BACKOFF_MODE=fixed_100ms, HONOR_RETRY_AFTER=false, and HTTP_TIMEOUT_MS=15000.",
                        kind="trap",
                        trap_invariant="INV-4",
                        failure_feedback_a="Egress gate FAILED: 320 outbound sockets breached 160 NAT cap and 100ms retry storm starved all tenants.",
                        failure_feedback_b="INV-4 VIOLATED by CFG-1: 320 sockets exceeded 160 cap and unlimited fixed-100ms retries violated tenant isolation.",
                    ),
                    PatchOption(
                        option_id="CFG-2",
                        description="Set WORKERS_PER_POD=16 across 8 pods (128 <= 160 NAT cap), PER_TENANT_MAX_INFLIGHT=8, and HTTP_TIMEOUT_MS=5000, while keeping RETRY_BACKOFF_MODE=fixed_100ms and HONOR_RETRY_AFTER=false.",
                        kind="unfixed",
                        failure_feedback_a="Backpressure test FAILED: ignoring 429 Retry-After with fixed 100ms retries still triggered a tenant retry storm.",
                        failure_feedback_b="Slot 'runtime' (CFG-2) FAILED CI: socket cap respected, but fixed 100ms retries ignored 429 Retry-After.",
                    ),
                    PatchOption(
                        option_id="CFG-3",
                        description="Set WORKERS_PER_POD=25 across 8 pods (200 concurrent sockets), PER_TENANT_MAX_INFLIGHT=8, RETRY_BACKOFF_MODE=exp_jitter, HONOR_RETRY_AFTER=true, and HTTP_TIMEOUT_MS=5000 for rapid queue drain.",
                        kind="partial",
                        failure_feedback_a="Egress soak FAILED: 8 pods * 25 workers = 200 concurrent sockets exceeded the 160-socket NAT egress cap.",
                        failure_feedback_b="Slot 'runtime' (CFG-3) FAILED CI: 8 * 25 = 200 outbound sockets exceeds the 160 cluster NAT cap.",
                    ),
                    PatchOption(
                        option_id="CFG-4",
                        description="Set WORKERS_PER_POD=16 across 8 pods (128 <= 160 NAT cap), PER_TENANT_MAX_INFLIGHT=8, RETRY_BACKOFF_MODE=exp_jitter, HONOR_RETRY_AFTER=true, and HTTP_TIMEOUT_MS=5000 for surge headroom.",
                        kind="winner",
                        failure_feedback_a="",
                        failure_feedback_b="",
                        pass_evidence="Egress soak verified: 128/160 NAT sockets, <=8 per-tenant inflight, exp_jitter + Retry-After honored, 0 starvation.",
                    ),
                ),
            ),
        ),
        shock_slot="migration",
        shock_round=4,
        shock_alert_a=(
            "STAGING SOAK UPDATE (Round 4 — PRIVATE TO YOU): Full-scale 95M-row soak (98.2% DELIVERED, 1.8% PENDING) "
            "showed a full non-partial concurrent index on (delivery_status, created_at) bloats to 8.4 GB and degrades "
            "outbox poll p99 to 1,180ms (> 100ms SLO)! Finance agreed to route DELIVERED range audits to the read "
            "replica warehouse, lifting the primary full-index requirement. Full non-partial index now FAILS 95M-row "
            "soak; migration MUST switch to a PARTIAL concurrent index WHERE delivery_status = 'PENDING' (140 MB, p99=4ms)."
        ),
    )

    return [settlement, auth_oidc, search_shard, webhook_outbox]


class CodeHandoffEnv(TaskEnv):
    """Task 4: 6-round asymmetric PR code-review & production deploy-gate handoff."""

    task_name = "code_handoff"
    total_rounds = 6

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        archetypes = _build_archetypes(seed)
        self.archetype: CodeHandoffArchetype = archetypes[(seed - 1) % len(archetypes)]
        self.slot_keys: tuple[str, ...] = tuple(s.slot_key for s in self.archetype.slots)
        self.staged: dict[str, str | None] = {k: None for k in self.slot_keys}
        self.checks_run: list[str] = []
        self.distinct_traps_triggered: set[tuple[str, str]] = set()
        self.distinct_wrong_options: set[tuple[str, str]] = set()
        self._pre_soak_valid_staged: set[tuple[str, str]] = set()
        self.pre_soak_ready_round: int | None = None
        self.ci_ready_round: int | None = None
        self.released: bool = False
        self.released_round: int | None = None
        self._released_staged: dict[str, str | None] | None = None
        self.final_gate: str = "HOLD"
        self.shipped_clean_round: int | None = None
        self.round_history: list[dict[str, Any]] = []

    def _slot_passes(self, slot_key: str, round_num: int) -> bool:
        opt_id = self.staged.get(slot_key)
        if not opt_id:
            return False
        slot = self.archetype.slot_map()[slot_key]
        opt = slot.option_map().get(opt_id)
        if opt is None:
            return False
        if slot_key == self.archetype.shock_slot and round_num < self.archetype.shock_round:
            # Before Round-4 soak shock, ONLY the pre-soak option ("partial") satisfies
            # the active pre-soak constraint; the post-shock winner fails pre-soak CI until Round 4.
            return opt.kind == "partial"
        return opt.kind == "winner"

    def system_briefing(self, role: Role, my_handle: str, partner_handle: str) -> str:
        ar = self.archetype
        if role == "A":
            check_lines = "\n".join(
                f"- `{cid}`: {cname}" for cid, (cname, _, _, _) in ar.checks_a.items()
            )
            return (
                f"You are PR author and code investigator `{my_handle}` working with release "
                f"gatekeeper `{partner_handle}` on `{ar.pr_id}` ({ar.ticket_id} / `{ar.service_name}`: "
                f"{ar.pr_title}).\n\n"
                f"You are the ONLY teammate with access to the 4 modified files, failing staging "
                f"test trace, PR review comments, and read-only repo baseline checks (`CHECK-1`..`CHECK-4`). "
                f"`{partner_handle}` CANNOT see these files or check outputs; `{partner_handle}` holds "
                f"the production deploy invariants and applies fixes by selecting predefined patch "
                f"options for the 4 subsystems ({', '.join(self.slot_keys)}) rather than pasting raw "
                f"diffs. Describe exact defects, root causes, and required behaviors/parameters "
                f"rather than rewriting full source files.\n"
                f"VERIFICATION SCOPE: `CHECK-1`..`CHECK-4` (repo baseline fixtures), per-round CI "
                f"staging results, and the Round-{ar.shock_round} staging soak ARE the complete "
                f"verification suite for this PR (no external ticket or artifact upload exists). "
                f"CI flags invariant traps immediately but reports only aggregate pass counts for "
                f"failing non-trap slots. Scoring rewards fast 4/4 pre-soak readiness (Rounds 1–3) "
                f"and fast post-soak 4/4 green CI + clean release (`SHIP` or `SHIP_IF_GREEN`, Rounds "
                f"{ar.shock_round}–{self.total_rounds}), while penalizing traps and wrong-option guessing.\n\n"
                f"MODIFIED FILES:\n{ar.code_excerpts_a}\n\n"
                f"FAILING STAGING TEST TRACE:\n{ar.failing_trace_a}\n\n"
                f"CONFLICTING PR REVIEW COMMENTS:\n{ar.pr_comments_a}\n\n"
                f"READ-ONLY REPO BASELINE CHECKS (run at most ONE per round via ACTION):\n{check_lines}\n\n"
                f"You have {self.total_rounds} rounds. Coordinate with `{partner_handle}` so all 4 "
                f"subsystems pass staging/soak checks and ship safely.\n"
                f"End EVERY response with a single final line:\n"
                f'ACTION: {{"check": "CHECK-1"}}  (use CHECK-1..CHECK-4 or null)'
            )

        inv_lines = "\n".join(f"- **{iid}**: {desc}" for iid, desc in ar.invariants_b)
        slot_blocks: list[str] = []
        for slot in ar.slots:
            opt_lines = "\n".join(f"  - `{o.option_id}`: {o.description}" for o in slot.options)
            slot_blocks.append(f"### Slot `{slot.slot_key}` — {slot.title}\n{opt_lines}")
        slots_text = "\n\n".join(slot_blocks)
        schema_example = ", ".join(f'"{k}": null' for k in self.slot_keys)

        return (
            f"You are release gatekeeper `{my_handle}` reviewing `{ar.pr_id}` ({ar.ticket_id} / "
            f"`{ar.service_name}`: {ar.pr_title}) with PR author `{partner_handle}`.\n\n"
            f"You CANNOT see `{partner_handle}`'s source code, failing test traces, PR comments, or "
            f"repo check outputs. You hold the 4 production deploy invariants and control the 4 "
            f"patch staging slots below.\n\n"
            f"PRODUCTION DEPLOY INVARIANTS (PRIVATE TO YOU):\n{inv_lines}\n\n"
            f"CANDIDATE PATCH OPTIONS (select 1 option ID per slot):\n{slots_text}\n\n"
            f"DEPLOY GATE & VERIFICATION RULES:\n"
            f"- You have {self.total_rounds} rounds. Stage options in `patches` as you confirm which "
            f"code bugs they fix and verify they satisfy your invariants.\n"
            f"- Any slot set to `null` (or omitted) keeps its previously staged option.\n"
            f"- CI flags invariant-violating trap options immediately (with a safety penalty), but "
            f"reports only aggregate passing counts for failing non-trap slots without naming which "
            f"non-trap slot failed. Staging wrong distractor options also incurs a trial-and-error penalty.\n"
            f"- VERIFICATION SCOPE: `{partner_handle}`'s repo baseline check findings, per-round CI "
            f"staging feedback, and the Round-{ar.shock_round} staging soak ARE the complete "
            f"verification suite for this PR—there is no external ticket, raw DDL upload, or "
            f"out-of-band approval channel. Each passing patch option applies and verifies its full "
            f"implementation automatically in staging CI.\n"
            f"- Scoring rewards fast 4/4 pre-soak readiness (Rounds 1–3) AND fast post-soak 4/4 "
            f"green CI followed by clean release approval (`\"gate\": \"SHIP\"` or `\"SHIP_IF_GREEN\"` "
            f"by Round {self.total_rounds}). Post-soak velocity is scored from the first post-soak "
            f"round where 4/4 pass CI once cleanly released, so holding on Round {ar.shock_round} to "
            f"verify green CI and shipping on Round {ar.shock_round + 1} (or using `SHIP_IF_GREEN`) "
            f"earns full velocity credit. Once cleanly released post-soak, the release is latched.\n\n"
            f"End EVERY response with a single final line:\n"
            f'ACTION: {{"patches": {{{schema_example}}}, "gate": "HOLD"}}  (gate is "HOLD", "SHIP", or "SHIP_IF_GREEN")'
        )

    def turn_prompt(
        self,
        role: Role,
        round_num: int,
        partner_handle: str,
        partner_message: str | None,
        last_feedback: str | None,
    ) -> str:
        ar = self.archetype
        parts = [f"=== ROUND {round_num} of {self.total_rounds} ==="]
        if last_feedback:
            parts.append(f"CI / STAGING GATE FEEDBACK:\n{last_feedback}")
        if role == "A" and round_num == ar.shock_round:
            parts.append(ar.shock_alert_a)
        if partner_message is not None:
            parts.append(f"Teammate {partner_handle} said:\n{partner_message}")
        elif role == "A":
            parts.append(f"You open the PR handoff with {partner_handle}.")

        if round_num == self.total_rounds:
            parts.append("FINAL ROUND: Finalize all 4 patch slots and the release gate decision.")

        if role == "A":
            parts.append('End with `ACTION: {"check": "CHECK-X"}` (or `{"check": null}`).')
        else:
            schema_hint = ", ".join(f'"{k}": "<OPT-ID or null>"' for k in self.slot_keys)
            parts.append(
                f'End with `ACTION: {{"patches": {{{schema_hint}}}, "gate": "HOLD"|"SHIP"|"SHIP_IF_GREEN"}}`.'
            )
        return "\n\n".join(parts)

    def step_round(
        self,
        round_num: int,
        action_a: dict[str, Any] | None,
        action_b: dict[str, Any] | None,
        msg_a: str,
        msg_b: str,
    ) -> RoundFeedback:
        ar = self.archetype
        slot_map = ar.slot_map()
        fb_a_lines: list[str] = []
        fb_b_lines: list[str] = []

        # --- Role A read-only baseline check (strictly tagged as repo baseline fixture) ---
        check_req = (action_a or {}).get("check")
        if isinstance(check_req, str) and check_req.strip().upper() in ar.checks_a:
            cid = check_req.strip().upper()
            self.checks_run.append(cid)
            cname, pre_out, post_out, _target_slot = ar.checks_a[cid]
            cout = post_out if round_num >= ar.shock_round else pre_out
            fb_a_lines.append(f"CHECK RESULT [{cid} / {cname}]: {cout}")
        elif check_req is None:
            fb_a_lines.append("No check executed this round.")
        else:
            fb_a_lines.append(f"Unknown check `{check_req}` ignored.")

        raw_gate = (action_b or {}).get("gate")
        gate_req = str(raw_gate).strip().upper() if isinstance(raw_gate, str) else "HOLD"
        if gate_req not in ("HOLD", "SHIP", "SHIP_IF_GREEN"):
            gate_req = "HOLD"

        # If a clean post-soak release has ALREADY been latched, later HOLD / missing gate /
        # parse glitches CANNOT un-ship or mutate the released build.
        if self.released:
            staged_map_str = ", ".join(
                f"{k}={self.staged[k] or 'UNSTAGED'}" for k in self.slot_keys
            )
            latched_note = (
                f"RELEASE ALREADY SHIPPED GREEN (latched in Round {self.released_round}): "
                "All 4 subsystems passed post-soak CI with zero invariant violations; no further changes needed."
            )
            fb_a_lines.append(latched_note)
            fb_b_lines.append(f"Staged slots: [{staged_map_str}]")
            fb_b_lines.append(latched_note)
            self.round_history.append(
                {
                    "round": round_num,
                    "check": check_req,
                    "staged": dict(self.staged),
                    "gate_requested": gate_req,
                    "gate_effective": "SHIP",
                    "released": True,
                    "passing_slots": list(self.slot_keys),
                    "non_trap_fail_count": 0,
                    "active_traps": [],
                }
            )
            return RoundFeedback(
                round_num=round_num,
                feedback_a="\n".join(fb_a_lines),
                feedback_b="\n".join(fb_b_lines),
                metrics={
                    "staged_count": 4,
                    "passing_count": 4,
                    "non_trap_fail_count": 0,
                    "active_trap_count": 0,
                    "distinct_traps_triggered": len(self.distinct_traps_triggered),
                    "distinct_wrong_options": len(self.distinct_wrong_options),
                    "pre_soak_ready_round": self.pre_soak_ready_round,
                    "ci_ready_round": self.ci_ready_round,
                    "released": True,
                    "released_round": self.released_round,
                    "shipped_clean_round": self.shipped_clean_round,
                    "gate_requested": gate_req,
                    "gate": "SHIP",
                },
            )

        # --- Role B patch staging ---
        prev_staged = dict(self.staged)
        explicitly_staged_this_round: set[str] = set()
        raw_patches = (action_b or {}).get("patches")
        if isinstance(raw_patches, dict):
            for sk in self.slot_keys:
                val = raw_patches.get(sk)
                if val is None:
                    continue
                if isinstance(val, str):
                    norm = val.strip().upper()
                    if norm == "CLEAR":
                        self.staged[sk] = None
                    elif norm in slot_map[sk].option_map():
                        self.staged[sk] = norm
                        explicitly_staged_this_round.add(sk)

        # Evaluate every staged slot deterministically
        passing_slots: list[str] = []
        passing_evidence_lines: list[str] = []
        active_traps: list[tuple[str, str, str]] = []
        trap_issues_a: list[str] = []
        trap_issues_b: list[str] = []
        non_trap_fail_cnt = 0

        for sk in self.slot_keys:
            opt_id = self.staged[sk]
            if opt_id is None:
                continue
            opt = slot_map[sk].option_map()[opt_id]
            if opt.kind == "trap":
                self.distinct_traps_triggered.add((sk, opt.option_id))
                active_traps.append((sk, opt.option_id, opt.trap_invariant or "INV"))
                trap_issues_a.append(f"- `{sk}`: {opt.failure_feedback_a}")
                trap_issues_b.append(f"- `{sk}` (`{opt.option_id}`): {opt.failure_feedback_b}")
            elif self._slot_passes(sk, round_num):
                passing_slots.append(sk)
                if round_num < ar.shock_round:
                    self._pre_soak_valid_staged.add((sk, opt.option_id))
                if opt.pass_evidence:
                    passing_evidence_lines.append(f"- `{sk}`: {opt.pass_evidence}")
            else:
                non_trap_fail_cnt += 1
                # Count distinct genuine wrong distractor options actively staged.
                # Do NOT penalize:
                #   (a) a pre-soak partial option that was legitimately passing in R1-R3
                #       and is merely carried over when the Round-4 soak shock invalidates it;
                #   (b) staging the eventual post-soak winner during R1-R3.
                is_pre_soak_carryover = (
                    prev_staged.get(sk) == opt.option_id
                    and (sk, opt.option_id) in self._pre_soak_valid_staged
                )
                is_eventual_winner = opt.kind == "winner"
                if (
                    sk in explicitly_staged_this_round
                    and not is_pre_soak_carryover
                    and not is_eventual_winner
                ):
                    self.distinct_wrong_options.add((sk, opt.option_id))

        staged_cnt = sum(1 for v in self.staged.values() if v is not None)
        pass_cnt = len(passing_slots)
        all_four_passing = pass_cnt == 4 and not active_traps
        phase_lbl = "post-soak" if round_num >= ar.shock_round else "pre-soak"

        if round_num < ar.shock_round and all_four_passing and self.pre_soak_ready_round is None:
            self.pre_soak_ready_round = round_num

        if round_num >= ar.shock_round and all_four_passing and self.ci_ready_round is None:
            self.ci_ready_round = round_num

        # Enforce mandatory Round-4 soak gate before latching SHIP
        if gate_req in ("SHIP", "SHIP_IF_GREEN") and round_num < ar.shock_round:
            self.final_gate = "HOLD"
            if all_four_passing:
                gate_note_a = (
                    f"ALL 4/4 SLOTS PASSED PRE-SOAK CI — ready for mandatory Round-{ar.shock_round} "
                    "staging soak (early SHIP held until soak completes)."
                )
                gate_note_b = (
                    f"ALL 4/4 SLOTS PASSED PRE-SOAK CI — ready for mandatory Round-{ar.shock_round} "
                    "staging soak (SHIP HELD until Round-4 soak completes)."
                )
            else:
                gate_note_a = (
                    f"Gatekeeper requested {gate_req} early; held by mandatory Round-{ar.shock_round} "
                    f"staging soak gate ({staged_cnt}/4 staged, {pass_cnt}/4 passing pre-soak CI)."
                )
                gate_note_b = (
                    f"SHIP HELD: Mandatory staging soak gate completes at Round {ar.shock_round} "
                    f"({staged_cnt}/4 staged, {pass_cnt}/4 passing pre-soak CI)."
                )
        elif gate_req in ("SHIP", "SHIP_IF_GREEN") and all_four_passing:
            # Clean post-soak release -> LATCH permanently!
            self.final_gate = "SHIP"
            self.released = True
            self.released_round = round_num
            self.shipped_clean_round = round_num
            self._released_staged = dict(self.staged)
            gate_note_a = (
                "RELEASE SHIPPED GREEN: All 4 subsystems passed post-soak CI and production "
                "invariants with zero violations! (Release latched.)"
            )
            gate_note_b = (
                "RELEASE SHIPPED GREEN: All 4 slots passed post-soak CI with zero invariant "
                "violations! (Release latched.)"
            )
        elif gate_req == "SHIP":
            self.final_gate = "SHIP"
            gate_note_a = (
                f"Gatekeeper issued SHIP with incomplete/failing checks ({staged_cnt}/4 staged, "
                f"{pass_cnt}/4 passing post-soak CI, {len(active_traps)} active invariant violation(s))."
            )
            gate_note_b = (
                f"SHIP BLOCKED / INCOMPLETE: {staged_cnt}/4 staged, {pass_cnt}/4 passing post-soak CI, "
                f"{len(active_traps)} active invariant violation(s). Fix failing slots before final round."
            )
        elif gate_req == "SHIP_IF_GREEN":
            # Conditional ship safely holds when <4/4 passing or traps active
            self.final_gate = "HOLD"
            gate_note_a = (
                f"Release gate: HOLD (SHIP_IF_GREEN deferred — {staged_cnt}/4 staged, "
                f"{pass_cnt}/4 passing {phase_lbl} CI)."
            )
            gate_note_b = (
                f"Release gate: HOLD (SHIP_IF_GREEN deferred — {staged_cnt}/4 staged, "
                f"{pass_cnt}/4 passing {phase_lbl} CI)."
            )
        else:
            self.final_gate = "HOLD"
            if round_num >= ar.shock_round and all_four_passing:
                gate_note_a = (
                    "ALL 4/4 SLOTS PASSED POST-SOAK CI AND PRODUCTION INVARIANTS (VERIFICATION COMPLETE — "
                    "no additional external artifacts exist). Release is ready for gatekeeper to set gate='SHIP'."
                )
                gate_note_b = (
                    "ALL 4/4 SLOTS PASSED POST-SOAK CI AND PRODUCTION INVARIANTS (VERIFICATION COMPLETE — "
                    "no external sign-off or additional artifacts required). Set \"gate\": \"SHIP\" by Round "
                    f"{self.total_rounds} to complete the release."
                )
            elif round_num < ar.shock_round and all_four_passing:
                gate_note_a = (
                    f"ALL 4/4 SLOTS PASSED PRE-SOAK CI — ready for mandatory Round-{ar.shock_round} "
                    "staging soak."
                )
                gate_note_b = (
                    f"ALL 4/4 SLOTS PASSED PRE-SOAK CI — ready for mandatory Round-{ar.shock_round} "
                    "staging soak."
                )
            else:
                gate_note_a = f"Release gate: HOLD ({staged_cnt}/4 staged, {pass_cnt}/4 passing {phase_lbl} CI)."
                gate_note_b = f"Release gate: HOLD ({staged_cnt}/4 staged, {pass_cnt}/4 passing {phase_lbl} CI)."

        non_trap_summary: str | None = None
        if not all_four_passing and non_trap_fail_cnt > 0:
            non_trap_summary = (
                f"CI staging: {pass_cnt}/4 slots passing ({staged_cnt}/4 staged; "
                f"{non_trap_fail_cnt} staged non-trap slot(s) failed active {phase_lbl} checks — "
                "diagnose which slot conflicts with the codebase/invariants rather than guessing)."
            )

        staged_map_str = ", ".join(f"{k}={self.staged[k] or 'UNSTAGED'}" for k in self.slot_keys)
        fb_a_lines.append(gate_note_a)
        if all_four_passing and passing_evidence_lines:
            fb_a_lines.append("Verified passing staging telemetry:\n" + "\n".join(passing_evidence_lines))
        if non_trap_summary:
            fb_a_lines.append(non_trap_summary)
        if trap_issues_a:
            fb_a_lines.append("CRITICAL invariant violation(s):\n" + "\n".join(trap_issues_a))

        fb_b_lines.append(f"Staged slots: [{staged_map_str}]")
        fb_b_lines.append(gate_note_b)
        if all_four_passing and passing_evidence_lines:
            fb_b_lines.append("Verified passing staging telemetry:\n" + "\n".join(passing_evidence_lines))
        if non_trap_summary:
            fb_b_lines.append(non_trap_summary)
        if trap_issues_b:
            fb_b_lines.append("CRITICAL invariant violation(s):\n" + "\n".join(trap_issues_b))

        self.round_history.append(
            {
                "round": round_num,
                "check": check_req,
                "staged": dict(self.staged),
                "gate_requested": gate_req,
                "gate_effective": self.final_gate,
                "released": self.released,
                "passing_slots": list(passing_slots),
                "non_trap_fail_count": non_trap_fail_cnt,
                "active_traps": [t[1] for t in active_traps],
            }
        )

        return RoundFeedback(
            round_num=round_num,
            feedback_a="\n".join(fb_a_lines),
            feedback_b="\n".join(fb_b_lines),
            metrics={
                "staged_count": staged_cnt,
                "passing_count": pass_cnt,
                "non_trap_fail_count": non_trap_fail_cnt,
                "active_trap_count": len(active_traps),
                "distinct_traps_triggered": len(self.distinct_traps_triggered),
                "distinct_wrong_options": len(self.distinct_wrong_options),
                "pre_soak_ready_round": self.pre_soak_ready_round,
                "ci_ready_round": self.ci_ready_round,
                "released": self.released,
                "released_round": self.released_round,
                "shipped_clean_round": self.shipped_clean_round,
                "gate_requested": gate_req,
                "gate": self.final_gate,
            },
        )

    def final_score(self) -> TaskScore:
        slot_map = self.archetype.slot_map()
        effective_staged = self._released_staged if self._released_staged is not None else self.staged
        correct_slots = 0
        active_traps = 0
        slot_outcomes: dict[str, str] = {}

        for sk in self.slot_keys:
            opt_id = effective_staged[sk]
            if opt_id is None:
                slot_outcomes[sk] = "unstaged"
                continue
            opt = slot_map[sk].option_map()[opt_id]
            slot_outcomes[sk] = opt.kind
            if opt.kind == "winner":
                correct_slots += 1
            elif opt.kind == "trap":
                active_traps += 1

        # 1. Final post-soak slot accuracy (60% max)
        slot_accuracy_score = round(0.15 * correct_slots, 6)

        # 2. Pre-soak readiness velocity (20% max)
        pre_soak_velocity_bonus = {1: 0.20, 2: 0.14, 3: 0.08}.get(self.pre_soak_ready_round, 0.0)

        # 3. Clean post-soak release velocity (20% max)
        # Requires clean latched release (4/4 passing post-soak, 0 traps, SHIP/SHIP_IF_GREEN).
        # Scored from ci_ready_round so prudent R4 verify-then-R5 ship earns full velocity credit.
        # Shipping with <4/4 or active traps earns ZERO ship credit.
        ship_velocity_bonus = 0.0
        if (
            self.released
            and active_traps == 0
            and correct_slots == 4
            and self.ci_ready_round is not None
        ):
            ship_velocity_bonus = {4: 0.20, 5: 0.15, 6: 0.10}.get(self.ci_ready_round, 0.0)

        approval_lag = (
            self.released_round - self.ci_ready_round
            if self.released_round is not None and self.ci_ready_round is not None
            else None
        )

        # 4. Safety & trial-and-error penalties
        trap_penalty = round(0.08 * len(self.distinct_traps_triggered), 6)
        wrong_option_penalty = round(0.04 * len(self.distinct_wrong_options), 6)

        raw_score = (
            slot_accuracy_score
            + pre_soak_velocity_bonus
            + ship_velocity_bonus
            - trap_penalty
            - wrong_option_penalty
        )
        score = max(0.0, min(1.0, round(raw_score, 6)))

        return TaskScore(
            score=score,
            details={
                "archetype_id": self.archetype.archetype_id,
                "pr_id": self.archetype.pr_id,
                "correct_slots": correct_slots,
                "active_traps": active_traps,
                "pre_soak_ready_round": self.pre_soak_ready_round,
                "ci_ready_round": self.ci_ready_round,
                "released": self.released,
                "released_round": self.released_round,
                "approval_lag": approval_lag,
                "shipped_clean_round": self.shipped_clean_round,
                "distinct_traps_triggered": sorted(
                    f"{sk}:{oid}" for sk, oid in self.distinct_traps_triggered
                ),
                "distinct_wrong_options": sorted(
                    f"{sk}:{oid}" for sk, oid in self.distinct_wrong_options
                ),
                "final_gate": "SHIP" if self.released else self.final_gate,
                "staged_final": dict(effective_staged),
                "slot_outcomes": slot_outcomes,
                "checks_run": list(self.checks_run),
                "score_breakdown": {
                    "slot_accuracy": slot_accuracy_score,
                    "pre_soak_velocity": pre_soak_velocity_bonus,
                    "ship_velocity": ship_velocity_bonus,
                    "trap_penalty": trap_penalty,
                    "wrong_option_penalty": wrong_option_penalty,
                },
            },
        )

    def oracle_actions(self, round_num: int) -> tuple[dict[str, Any], dict[str, Any]]:
        ar = self.archetype
        slot_map = ar.slot_map()
        check_keys = list(ar.checks_a.keys())
        act_a: dict[str, Any] = {
            "check": check_keys[round_num - 1] if round_num <= len(check_keys) else None
        }
        winners = {s.slot_key: s.winner_id() for s in ar.slots}
        pre_shock_opt = slot_map[ar.shock_slot].partial_ids()[0]
        non_shock = [k for k in self.slot_keys if k != ar.shock_slot]

        if round_num == 1:
            # Stage all 4 pre-soak passing options immediately in Round 1 -> pre_soak_ready_round = 1
            patches_r1 = {k: winners[k] for k in non_shock}
            patches_r1[ar.shock_slot] = pre_shock_opt
            act_b = {"patches": patches_r1, "gate": "HOLD"}
        elif round_num < ar.shock_round:
            act_b = {"patches": {k: None for k in self.slot_keys}, "gate": "HOLD"}
        else:
            # Round 4+: Role A sees soak alert first and tells Role B -> 1-shot swap + clean SHIP in Round 4!
            act_b = {"patches": dict(winners), "gate": "SHIP"}
        return act_a, act_b
