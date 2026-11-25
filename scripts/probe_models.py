"""v2 Live OpenRouter endpoint, parameter, 3-judge panel, and DeepSeek rewriter probe (<$0.15 total).

Writes a fresh pinned manifest strictly to `data/v2/probe_manifest_v2.json` without touching v1 `data/probe_report.json`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from common_ground.config import (
    DECISION_PANEL_SPECS,
    DEFAULT_MODELS,
    OPENROUTER_BASE_URL,
    PROBE_MANIFEST_V2_PATH,
    load_api_key,
)
from common_ground.jev import OPENROUTER_DECISIONS_URL

console = Console()

PREFERRED_PROVIDERS: dict[str, list[str]] = {
    "astra62": ["OpenAI"],
    "astra60": ["OpenAI"],
    "opus55": ["Anthropic"],
    "neon4": ["Google AI Studio", "Google", "Google Vertex"],
    "kimi31": ["Moonshot AI", "Moonshot"],
    "muse": ["Meta"],
    "qwen4": ["Alibaba"],
    "deepseek": ["BaseTen", "Baseten"],
}


def pick_provider(model_key: str, endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not endpoints:
        return None
    prefs = PREFERRED_PROVIDERS.get(model_key, [])
    for pref in prefs:
        for ep in endpoints:
            pname = str(ep.get("provider_name") or ep.get("name") or "")
            tag = str(ep.get("tag") or "")
            if "highspeed" in pname.lower() or "batch" in pname.lower():
                continue
            if pname.lower() == pref.lower() or pref.lower() in pname.lower() or tag.lower() == pref.lower():
                return ep
    for ep in endpoints:
        pname = str(ep.get("provider_name") or ep.get("name") or "")
        if "highspeed" not in pname.lower() and "batch" not in pname.lower():
            return ep
    return endpoints[0]


async def run_probe(out_path: Path = PROBE_MANIFEST_V2_PATH) -> dict[str, Any]:
    api_key = load_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "Mighty Models, Terrible Teammates - v2 Probe",
    }

    report: dict[str, Any] = {
        "schema_version": "v2",
        "models_metadata": {},
        "endpoints": {},
        "live_tests": {},
        "verified_pins": {},
        "decision_panel_tests": {},
        "total_probe_cost_usd": 0.0,
    }

    async with httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers=headers,
        timeout=httpx.Timeout(120.0, connect=30.0),
    ) as client:
        console.print("[bold cyan]1. Fetching OpenRouter model catalog for v2 roster (7 models + DeepSeek rewriter)...[/bold cyan]")
        models_resp = await client.get("/models")
        models_resp.raise_for_status()
        catalog = {m["id"]: m for m in models_resp.json().get("data", [])}

        table = Table(title="OpenRouter v2 Live Metadata & Provider Pins")
        table.add_column("Key")
        table.add_column("Slug")
        table.add_column("Selected Provider")
        table.add_column("Pricing (in/out $/M)")
        table.add_column("All Providers")

        for key, spec in DEFAULT_MODELS.items():
            cat_entry = catalog.get(spec.slug)
            if cat_entry is None:
                matches = [mid for mid in catalog if spec.slug.split("/")[-1][:10] in mid]
                console.print(
                    f"[bold red]Slug NOT found in catalog: {spec.slug}[/bold red] (near: {matches[:8]})"
                )
                report["models_metadata"][key] = {"found": False, "near_matches": matches[:10]}
                continue

            report["models_metadata"][key] = {
                "found": True,
                "id": cat_entry.get("id"),
                "name": cat_entry.get("name"),
                "context_length": cat_entry.get("context_length"),
                "pricing": cat_entry.get("pricing"),
                "supported_parameters": cat_entry.get("supported_parameters"),
            }

            author, slug_tail = spec.slug.split("/", 1)
            ep_resp = await client.get(f"/models/{author}/{slug_tail}/endpoints")
            endpoints_list: list[dict[str, Any]] = []
            if ep_resp.status_code == 200:
                ep_data = ep_resp.json().get("data", {})
                endpoints_list = ep_data.get("endpoints", []) if isinstance(ep_data, dict) else ep_data
            report["endpoints"][key] = endpoints_list

            chosen = pick_provider(key, endpoints_list)
            chosen_name = (
                str(chosen.get("provider_name") or chosen.get("name") or "")
                if chosen
                else (spec.provider_only[0] if spec.provider_only else "")
            )
            pricing = (chosen or {}).get("pricing") or cat_entry.get("pricing") or {}
            p_in = float(pricing.get("prompt") or 0) * 1_000_000
            p_out = float(pricing.get("completion") or 0) * 1_000_000
            supported = (chosen or {}).get("supported_parameters") or cat_entry.get("supported_parameters") or []
            all_prov_names = [
                f"{e.get('provider_name', e.get('name'))}({e.get('quantization', '?')})"
                for e in endpoints_list
            ]

            table.add_row(
                key,
                spec.slug,
                chosen_name or "UNPINNED",
                f"${p_in:.3f} / ${p_out:.3f}",
                ", ".join(all_prov_names[:6]) or "n/a",
            )

            report["verified_pins"][key] = {
                "slug": spec.slug,
                "provider_only": [chosen_name] if chosen_name else list(spec.provider_only),
                "reasoning_effort": spec.reasoning_effort,
                "max_tokens": spec.max_tokens,
                "input_per_m": round(p_in, 4),
                "output_per_m": round(p_out, 4),
                "supported_parameters": supported,
                "quantization": (chosen or {}).get("quantization"),
                "aa_index_v5": spec.aa_index_v5,
            }

        console.print(table)

        console.print("\n[bold cyan]2. Running live 1-turn parameter & reasoning probe on all 8 chat models...[/bold cyan]")
        live_table = Table(title="Live Chat Model Probe Calls")
        live_table.add_column("Key")
        live_table.add_column("Status")
        live_table.add_column("Returned Model / Provider")
        live_table.add_column("Finish")
        live_table.add_column("Tokens (in/out/reason)")
        live_table.add_column("Cost ($)")

        total_cost = 0.0
        for key, pin in report["verified_pins"].items():
            effort = pin["reasoning_effort"]
            payload: dict[str, Any] = {
                "model": pin["slug"],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Reply with one short plain sentence confirming readiness, "
                            "then on the final line write exactly:\nACTION: {\"ready\": true}"
                        ),
                    }
                ],
                "max_tokens": pin["max_tokens"],
                "usage": {"include": True},
                "provider": {
                    "only": pin["provider_only"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            }
            if effort is not None:
                payload["reasoning"] = {"effort": effort}

            resp = await client.post("/chat/completions", json=payload)
            used_fallback_no_effort = False
            if resp.status_code >= 400 and "reasoning" in payload:
                err1 = resp.text[:300]
                payload_no_effort = {k: v for k, v in payload.items() if k != "reasoning"}
                resp2 = await client.post("/chat/completions", json=payload_no_effort)
                if resp2.status_code == 200:
                    resp = resp2
                    used_fallback_no_effort = True
                    pin["reasoning_effort"] = None
                    pin["effort_rejected_note"] = err1

            if resp.status_code != 200:
                report["live_tests"][key] = {
                    "ok": False,
                    "http_status": resp.status_code,
                    "error": resp.text[:600],
                }
                live_table.add_row(key, f"ERR {resp.status_code}", "-", "-", "-", "-")
                continue

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = str(msg.get("content") or "")
            usage = data.get("usage") or {}
            p_tok = int(usage.get("prompt_tokens") or 0)
            c_tok = int(usage.get("completion_tokens") or 0)
            raw_rtok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            r_tok = int(raw_rtok) if raw_rtok is not None else None
            cost = float(usage.get("cost") or 0.0)
            total_cost += cost

            pin["returned_model"] = data.get("model")
            pin["returned_provider"] = data.get("provider")
            report["live_tests"][key] = {
                "ok": True,
                "used_fallback_no_effort": used_fallback_no_effort,
                "returned_model": data.get("model"),
                "provider": data.get("provider"),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "reasoning_tokens": r_tok,
                "cost_usd": cost,
                "content_preview": content[:200],
            }
            status_str = "OK (no effort)" if used_fallback_no_effort else "OK"
            live_table.add_row(
                key,
                status_str,
                f"{data.get('model')} @ {data.get('provider')}",
                str(choice.get("finish_reason")),
                f"{p_tok}/{c_tok}/{r_tok}",
                f"{cost:.5f}",
            )

        console.print(live_table)

        console.print("\n[bold cyan]3. Probing 3-Model Vendor-Independent Decision Panel (/api/alpha/decisions)...[/bold cyan]")
        panel_table = Table(title="Live Decision Panel Probes")
        panel_table.add_column("Judge Key")
        panel_table.add_column("Status")
        panel_table.add_column("Returned Model / Provider")
        panel_table.add_column("skips_why")
        panel_table.add_column("warmth")
        panel_table.add_column("Cost ($)")

        probe_state = {
            "task_type": "triage",
            "round": 2,
            "role": "A",
            "previous_partner_message": "Can you confirm why CA-v1 must remain in the bundle?",
            "message": "Run CMD-3 now. No time to explain.",
        }
        probe_qs = {
            "skips_why": {
                "type": "noul",
                "instructions": "Does the message issue unexplained imperatives instead of explaining why?",
            },
            "warmth": {
                "type": "score",
                "instructions": "How warm and supportive is the tone?",
                "criteria": ["Cold", "Curt", "Neutral", "Polite", "Warm"],
            },
        }
        for jkey, jspec in DECISION_PANEL_SPECS.items():
            dresp = await client.post(
                OPENROUTER_DECISIONS_URL,
                json={"model": jspec["slug"], "state": probe_state, "questions": probe_qs},
            )
            if dresp.status_code != 200:
                report["decision_panel_tests"][jkey] = {
                    "ok": False,
                    "requested_slug": jspec["slug"],
                    "http_status": dresp.status_code,
                    "error": dresp.text[:500],
                }
                panel_table.add_row(jkey, f"ERR {dresp.status_code}", "-", "-", "-", "-")
                continue
            ddata = dresp.json()
            dcost = float((ddata.get("usage") or {}).get("cost") or 0.0)
            total_cost += dcost
            ans = ddata.get("answers") or {}
            sw = (ans.get("skips_why") or {}).get("noul")
            wm = (ans.get("warmth") or {}).get("score")
            report["decision_panel_tests"][jkey] = {
                "ok": True,
                "requested_slug": jspec["slug"],
                "returned_model": ddata.get("model"),
                "provider": ddata.get("provider"),
                "usage": ddata.get("usage"),
                "answers": ans,
                "cost_usd": dcost,
            }
            panel_table.add_row(
                jkey,
                "OK",
                f"{ddata.get('model')} @ {ddata.get('provider')}",
                f"{sw:.3f}" if isinstance(sw, (int, float)) else str(sw),
                f"{wm:.2f}" if isinstance(wm, (int, float)) else str(wm),
                f"{dcost:.6f}",
            )

        console.print(panel_table)
        report["total_probe_cost_usd"] = round(total_cost, 6)
        console.print(f"\n[bold green]Total v2 probe cost: ${total_cost:.5f}[/bold green]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"Saved v2 probe manifest to {out_path}")
    return report


if __name__ == "__main__":
    asyncio.run(run_probe())
