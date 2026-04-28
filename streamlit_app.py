"""
PharmaQuery Pro

Drug-gene interaction discovery across ChEMBL, DGIdb, Open Targets,
IUPHAR, and PubChem, with AI-assisted shortlist synthesis.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="PharmaQuery Pro",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    background:#f8f9fa;
    border-radius:8px;
    padding:.5rem 1rem;
    border:1px solid #e0e0e0;
}
.stDataFrame { border-radius:8px; }
</style>
""",
    unsafe_allow_html=True,
)


CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
DGIDB = "https://dgidb.org/api/graphql"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
IUPHAR = "https://www.guidetopharmacology.org/services"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 20

APP_ROOT = Path(__file__).resolve().parent
LOCAL_SETTINGS_PATH = APP_ROOT / ".streamlit" / "pharmaquery_state.json"
PUTER_BRIDGE_DIR = APP_ROOT / "components" / "puter_bridge"
PUTER_BRIDGE = components.declare_component("puter_bridge", path=str(PUTER_BRIDGE_DIR))

ITYPE_KEYWORDS = {
    "inhibitor": [
        "inhibitor",
        "antagonist",
        "blocker",
        "inverse agonist",
        "negative modulator",
        "suppressor",
        "inhibition",
        "channel blocker",
    ],
    "agonist": [
        "agonist",
        "activator",
        "positive modulator",
        "partial agonist",
        "full agonist",
        "superagonist",
        "potentiator",
        "activation",
    ],
    "modulator": [],
}

STATUS_ORDER = [
    "FDA Approved",
    "Phase 4",
    "Phase 3",
    "Phase 2",
    "Phase 1",
    "Preclinical",
    "Research Tool",
    "Unknown",
]

STATUS_COLORS = {
    "FDA Approved": "#2e7d32",
    "Phase 4": "#388e3c",
    "Phase 3": "#1565c0",
    "Phase 2": "#283593",
    "Phase 1": "#e65100",
    "Preclinical": "#880e4f",
    "Research Tool": "#616161",
    "Unknown": "#9e9e9e",
}

AI_PROVIDERS = {
    "Free - Puter.js - GPT-5 nano (browser auth)": {
        "id": "puter_js",
        "model": "gpt-5-nano",
        "key_placeholder": "",
        "key_help": "No developer key needed. Each user signs in with Puter in the browser.",
        "key_required": False,
        "supports_search": False,
        "free": True,
        "note": "Browser-side AI via Puter.js. Each user authenticates with Puter instead of pasting a developer key.",
    },
    "Free - Groq - Llama 3.3 70B": {
        "id": "oai_compat",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "key_placeholder": "gsk_...",
        "key_help": "Free API key at console.groq.com with no credit card needed",
        "key_required": True,
        "supports_search": False,
        "free": True,
        "note": "Fast inference with a generous free tier.",
    },
    "Free - Groq - Llama 3.1 8B": {
        "id": "oai_compat",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "key_placeholder": "gsk_...",
        "key_help": "Free API key at console.groq.com with no credit card needed",
        "key_required": True,
        "supports_search": False,
        "free": True,
        "note": "Fastest free hosted option in the list.",
    },
    "Free - Groq - Mixtral 8x7B": {
        "id": "oai_compat",
        "model": "mixtral-8x7b-32768",
        "base_url": "https://api.groq.com/openai/v1",
        "key_placeholder": "gsk_...",
        "key_help": "Free API key at console.groq.com with no credit card needed",
        "key_required": True,
        "supports_search": False,
        "free": True,
        "note": "Strong MoE model with a wide context window.",
    },
    "Free - Ollama - local": {
        "id": "ollama",
        "model": "",
        "base_url": "http://localhost:11434/v1",
        "key_placeholder": "(none)",
        "key_help": "Install Ollama from ollama.com, then run: ollama pull llama3",
        "key_required": False,
        "supports_search": False,
        "free": True,
        "note": "Runs entirely on your machine.",
    },
    "Free - OpenRouter - Llama 3 8B": {
        "id": "oai_compat",
        "model": "meta-llama/llama-3-8b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_placeholder": "sk-or-...",
        "key_help": "Free key at openrouter.ai for free models",
        "key_required": True,
        "supports_search": False,
        "free": True,
        "note": "Hosted Llama 3 through OpenRouter's free tier.",
    },
    "Free - OpenRouter - Mistral 7B": {
        "id": "oai_compat",
        "model": "mistralai/mistral-7b-instruct:free",
        "base_url": "https://openrouter.ai/api/v1",
        "key_placeholder": "sk-or-...",
        "key_help": "Free key at openrouter.ai for free models",
        "key_required": True,
        "supports_search": False,
        "free": True,
        "note": "Hosted Mistral 7B through OpenRouter's free tier.",
    },
    "Free - Gemini 2.5 Pro": {
        "id": "gemini",
        "model": "gemini-2.5-pro",
        "key_placeholder": "AIza...",
        "key_help": "Free key at aistudio.google.com with no credit card needed",
        "key_required": True,
        "supports_search": True,
        "free": True,
        "note": "Fast Gemini model with search grounding when available.",
    },
    "Free - Gemini 1.5 Flash": {
        "id": "gemini",
        "model": "gemini-1.5-flash",
        "key_placeholder": "AIza...",
        "key_help": "Free key at aistudio.google.com with no credit card needed",
        "key_required": True,
        "supports_search": True,
        "free": True,
        "note": "Reliable Gemini free-tier fallback.",
    },
    "Paid - OpenAI - GPT-5.4": {
        "id": "oai_compat",
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "key_placeholder": "sk-...",
        "key_help": "Paid key at platform.openai.com",
        "key_required": True,
        "supports_search": False,
        "free": False,
        "note": "OpenAI flagship model.",
    },
    "Paid - OpenAI - GPT-4o mini": {
        "id": "oai_compat",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "key_placeholder": "sk-...",
        "key_help": "Paid key at platform.openai.com",
        "key_required": True,
        "supports_search": False,
        "free": False,
        "note": "Cheaper OpenAI option.",
    },
    "Paid - Gemini 1.5 Pro": {
        "id": "gemini",
        "model": "gemini-1.5-pro",
        "key_placeholder": "AIza...",
        "key_help": "Paid or high-quota key at aistudio.google.com",
        "key_required": True,
        "supports_search": True,
        "free": False,
        "note": "Highest-quality Gemini option in this app.",
    },
}


@st.cache_data(ttl=3600, show_spinner=False)
def query_chembl(gene: str, itype: str) -> dict:
    out = {"target": None, "activities": [], "molecules": {}}
    try:
        r = requests.get(
            f"{CHEMBL}/target/search.json",
            params={"q": gene, "limit": 10},
            timeout=TIMEOUT,
        )
        targets = r.json().get("targets", [])
        target = (
            next(
                (
                    t
                    for t in targets
                    if t["target_type"] == "SINGLE PROTEIN"
                    and "Homo sapiens" in t.get("organism", "")
                ),
                None,
            )
            or next((t for t in targets if t["target_type"] == "SINGLE PROTEIN"), None)
            or (targets[0] if targets else None)
        )
        if not target:
            return out

        cid = target["target_chembl_id"]
        out["target"] = {
            "name": target.get("pref_name"),
            "chembl_id": cid,
            "organism": target.get("organism"),
        }

        r2 = requests.get(
            f"{CHEMBL}/activity.json",
            params={
                "target_chembl_id": cid,
                "standard_type__in": "IC50,Ki,EC50,Kd,pIC50",
                "standard_relation__in": "=,<,<=",
                "limit": 100,
                "order_by": "standard_value",
            },
            timeout=TIMEOUT,
        )
        acts = [a for a in r2.json().get("activities", []) if a.get("standard_value")]
        out["activities"] = [
            {
                "molecule": a.get("molecule_pref_name") or a.get("molecule_chembl_id"),
                "chembl_id": a.get("molecule_chembl_id"),
                "meas_type": a.get("standard_type"),
                "value": float(a["standard_value"]),
                "units": a.get("standard_units", "nM"),
            }
            for a in acts
        ]

        mol_ids = list({a["chembl_id"] for a in out["activities"] if a.get("chembl_id")})[:25]

        def _fetch_mol(mid: str):
            try:
                mr = requests.get(f"{CHEMBL}/molecule/{mid}.json", timeout=TIMEOUT)
                mol = mr.json()
                return mid, {"name": mol.get("pref_name"), "max_phase": mol.get("max_phase", 0)}
            except Exception:
                return mid, {}

        with ThreadPoolExecutor(max_workers=8) as ex:
            for mid, info in ex.map(_fetch_mol, mol_ids):
                out["molecules"][mid] = info
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def query_dgidb(gene: str, itype: str) -> list:
    try:
        query = """query($names:[String!]!){
            genes(names:$names){nodes{name interactions{
                drug{name approved conceptId}
                interactionScore interactionTypes{type directionality}
                sources{fullName}
            }}}
        }"""
        r = requests.post(
            DGIDB,
            json={"query": query, "variables": {"names": [gene.upper()]}},
            timeout=TIMEOUT,
        )
        nodes = r.json().get("data", {}).get("genes", {}).get("nodes", [])
        ixns = nodes[0].get("interactions", []) if nodes else []
        filters = ITYPE_KEYWORDS.get(itype, [])
        results = []
        for i in ixns:
            type_strs = [
                ((t.get("directionality") or "") + " " + (t.get("type") or "")).strip().lower()
                for t in (i.get("interactionTypes") or [])
            ]
            combined = " ".join(type_strs)
            if filters and not any(f in combined for f in filters):
                continue
            results.append(
                {
                    "drug": i.get("drug", {}).get("name"),
                    "approved": i.get("drug", {}).get("approved"),
                    "dgi_score": i.get("interactionScore") or 0,
                    "types": ", ".join(type_strs),
                    "n_sources": len(i.get("sources") or []),
                    "sources": [s.get("fullName") for s in (i.get("sources") or [])],
                }
            )
        return results
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def query_open_targets(gene: str) -> dict:
    out = {"target_id": None, "drugs": []}
    try:
        sq = """query($q:String!){search(queryString:$q,entityNames:["target"],
            page:{index:0,size:3}){hits{id name}}}"""
        r = requests.post(OT, json={"query": sq, "variables": {"q": gene}}, timeout=TIMEOUT)
        hits = r.json().get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return out

        tid = hits[0]["id"]
        out["target_id"] = tid

        dq = """query($id:String!){target(ensemblId:$id){
            approvedName knownDrugs{rows{
                drug{id name maximumClinicalTrialPhase isApproved}
                mechanismOfAction actionType phase status disease{name}
            }}
        }}"""
        r2 = requests.post(OT, json={"query": dq, "variables": {"id": tid}}, timeout=TIMEOUT)
        rows = r2.json().get("data", {}).get("target", {}).get("knownDrugs", {}).get("rows", [])
        for row in rows:
            drug = row.get("drug", {})
            out["drugs"].append(
                {
                    "name": drug.get("name"),
                    "ot_id": drug.get("id"),
                    "max_phase": drug.get("maximumClinicalTrialPhase"),
                    "approved": drug.get("isApproved"),
                    "mechanism": row.get("mechanismOfAction"),
                    "action_type": row.get("actionType"),
                    "indication": (row.get("disease") or {}).get("name"),
                }
            )
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def query_iuphar(gene: str, itype: str) -> list:
    results = []
    try:
        r = requests.get(
            f"{IUPHAR}/targets",
            params={"geneSymbol": gene.upper(), "species": "Human"},
            timeout=TIMEOUT,
        )
        targets = r.json() if r.ok and r.content else []
        if not targets:
            r2 = requests.get(
                f"{IUPHAR}/targets/search/{requests.utils.quote(gene)}",
                timeout=TIMEOUT,
            )
            targets = r2.json() if r2.ok and r2.content else []
        if not targets:
            return []

        tid = targets[0].get("targetId")
        r3 = requests.get(f"{IUPHAR}/interactions/target/{tid}", timeout=TIMEOUT)
        ixns = r3.json() if r3.ok and r3.content else []
        filters = ITYPE_KEYWORDS.get(itype, [])
        for i in ixns:
            itype_str = ((i.get("type") or "") + " " + (i.get("action") or "")).lower()
            if filters and not any(f in itype_str for f in filters):
                continue
            aff_type = i.get("affinityType") or ""
            aff_val = i.get("affinity")
            ic50_nm = None
            if aff_val and aff_type.startswith("p"):
                ic50_nm = (10 ** -float(aff_val)) * 1e9
            lid = i.get("ligandId")
            results.append(
                {
                    "drug": i.get("ligandName"),
                    "iuphar_id": lid,
                    "action": i.get("action") or i.get("type"),
                    "aff_type": aff_type,
                    "aff_value": aff_val,
                    "ic50_nm": ic50_nm,
                    "approved": i.get("ligandApproved"),
                    "url": (
                        f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lid}"
                        if lid
                        else None
                    ),
                }
            )
    except Exception:
        pass
    return results[:40]


@st.cache_data(ttl=3600, show_spinner=False)
def query_pubchem(gene: str) -> list:
    results = []
    try:
        r = requests.get(
            f"{PUBCHEM}/assay/target/genesymbol/{gene.upper()}/aids/JSON",
            timeout=TIMEOUT,
        )
        aids = r.json().get("IdentifierList", {}).get("AID", [])[:5]
        for aid in aids:
            r2 = requests.get(f"{PUBCHEM}/assay/aid/{aid}/summary/JSON", timeout=TIMEOUT)
            assay_info = r2.json().get("AssaySummaries", {}).get("AssaySummary", [{}])[0]
            name = assay_info.get("Name", "")
            r3 = requests.get(
                f"{PUBCHEM}/assay/aid/{aid}/cids/JSON",
                params={"cids_type": "active", "list_return": 10},
                timeout=TIMEOUT,
            )
            cids = r3.json().get("AssayLink", {}).get("CID", [])[:10]
            for cid in cids:
                results.append({"cid": cid, "assay": name, "aid": aid})
    except Exception:
        pass
    return results


def pic50_from_nm(ic50_nm: Optional[float]) -> Optional[float]:
    if not ic50_nm or ic50_nm <= 0:
        return None
    return -math.log10(ic50_nm * 1e-9)


def _potency_pts(ic50_nm):
    potency = pic50_from_nm(ic50_nm)
    if potency is None:
        return 4.0
    if potency >= 10:
        return 30
    if potency >= 9:
        return 27
    if potency >= 8:
        return 23
    if potency >= 7:
        return 18
    if potency >= 6:
        return 12
    if potency >= 5:
        return 7
    return 3


def _clinical_pts(status, max_phase=None):
    if max_phase is not None:
        mp = int(max_phase) if max_phase else 0
        return {4: 30, 3: 25, 2: 20, 1: 15, 0: 6}.get(mp, 5)
    if not status:
        return 4.0
    status = status.lower()
    for keys, pts in [
        (["fda approved", "approved", "marketed"], 30),
        (["phase 4", "phase iv"], 29),
        (["phase 3", "phase iii"], 25),
        (["phase 2/3"], 22),
        (["phase 2", "phase ii"], 20),
        (["phase 1/2"], 17),
        (["phase 1", "phase i"], 15),
        (["ind filed"], 12),
        (["preclinical"], 8),
        (["research tool", "research"], 4),
        (["withdrawn", "discontinued", "failed"], 2),
    ]:
        if any(k in status for k in keys):
            return float(pts)
    return 4.0


def _evidence_pts(n_assays, n_databases):
    return min(10.0, math.log1p(n_assays) * 2.3) + min(10.0, n_databases * 2.0)


def _selectivity_pts(selectivity):
    return {"high": 10, "moderate": 6, "low": 2}.get((selectivity or "").lower(), 4)


def compute_score(drug):
    potency = _potency_pts(drug.get("best_ic50_nm"))
    clinical = _clinical_pts(drug.get("clinical_status", ""), drug.get("max_phase"))
    evidence = _evidence_pts(int(drug.get("n_assays") or 1), int(drug.get("n_databases") or 1))
    selectivity = _selectivity_pts(drug.get("selectivity", ""))
    ai_bonus = float(drug.get("ai_bonus") or 5)
    raw = potency * 0.30 + clinical * 0.30 + evidence * 0.20 + selectivity * 0.10 + ai_bonus * 0.10
    return round(min(10.0, raw), 2)


def _norm(name: str) -> str:
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _load_local_settings() -> dict:
    if not LOCAL_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_local_settings(settings: dict) -> None:
    LOCAL_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SETTINGS_PATH.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")


def _update_local_settings(**updates) -> dict:
    settings = _load_local_settings()
    settings.update(updates)
    _save_local_settings(settings)
    return settings


def _save_provider_api_key(provider_name: str, api_key: str) -> dict:
    settings = _load_local_settings()
    settings.setdefault("api_keys", {})[provider_name] = api_key
    _save_local_settings(settings)
    return settings


def _remove_provider_api_key(provider_name: str) -> dict:
    settings = _load_local_settings()
    settings.setdefault("api_keys", {}).pop(provider_name, None)
    _save_local_settings(settings)
    return settings


def _get_api_widget_key(provider_name: str) -> str:
    return f"api_key_{_norm(provider_name)}"


def _default_provider_for_tier(tier_name: str) -> str:
    free_providers = [k for k, v in AI_PROVIDERS.items() if v["free"]]
    paid_providers = [k for k, v in AI_PROVIDERS.items() if not v["free"]]
    provider_list = free_providers if tier_name == "Free" else paid_providers
    return provider_list[0] if provider_list else next(iter(AI_PROVIDERS))


def _init_session_defaults(settings: dict) -> None:
    st.session_state.setdefault("tier_name", settings.get("tier_name", "Free"))
    st.session_state.setdefault(
        "provider_name",
        settings.get("provider_name", _default_provider_for_tier(st.session_state["tier_name"])),
    )
    st.session_state.setdefault("gene_input", settings.get("last_gene", ""))
    st.session_state.setdefault("itype_input", settings.get("last_itype", "inhibitor"))
    st.session_state.setdefault("ollama_model", settings.get("ollama_model", "llama3"))
    st.session_state.setdefault("puter_model", settings.get("puter_model", "gpt-5-nano"))
    st.session_state.setdefault("active_search", None)
    st.session_state.setdefault("pending_puter_request", None)
    st.session_state.setdefault("latest_puter_result", None)


def _extract_json_array(text: str, warning_message: Optional[str] = None) -> list:
    clean = re.sub(r"```(?:json)?", "", text or "").strip()
    match = re.search(r"\[[\s\S]*\]", clean)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    if warning_message:
        st.warning(warning_message)
    return []


def consolidate(chembl, dgidb, ot, iuphar):
    drugs: dict[str, dict] = {}

    def upsert(name, data, source):
        if not name:
            return
        key = _norm(name)
        if key not in drugs:
            drugs[key] = {
                "name": name,
                "best_ic50_nm": None,
                "max_phase": None,
                "clinical_status": "",
                "selectivity": "",
                "mechanism": "",
                "indication": "",
                "chembl_id": None,
                "iuphar_id": None,
                "ot_id": None,
                "ai_bonus": 5,
                "n_assays": 0,
                "n_databases": 0,
                "_sources": set(),
            }
        drug = drugs[key]
        for k, v in data.items():
            if k.startswith("_"):
                continue
            if v not in (None, ""):
                if k == "best_ic50_nm" and drug.get(k) is not None:
                    drug[k] = min(drug[k], float(v))
                elif k == "max_phase" and drug.get(k) is not None:
                    drug[k] = max(int(drug[k]), int(v) if v else 0)
                elif drug.get(k) in (None, ""):
                    drug[k] = v
        drug["n_assays"] += int(data.get("_n_assays", 1))
        drug["_sources"].add(source)
        drug["n_databases"] = len(drug["_sources"])

    mol_best_ic50: dict[str, float] = {}
    mol_n_assays: dict[str, int] = {}
    for activity in chembl.get("activities", []):
        mid = activity.get("chembl_id")
        name = activity.get("molecule")
        if not mid and not name:
            continue
        key = mid or _norm(name)
        value = activity.get("value")
        units = (activity.get("units", "nM") or "nM").replace("Â", "").replace("µ", "u").strip()
        value_nm = None
        if value is not None:
            value = float(value)
            if units.lower() == "um":
                value_nm = value * 1000
            elif units.lower() == "mm":
                value_nm = value * 1e6
            elif units.lower() == "nm":
                value_nm = value
            elif units.lower() == "pm":
                value_nm = value / 1000
        if value_nm is not None:
            if key not in mol_best_ic50 or value_nm < mol_best_ic50[key]:
                mol_best_ic50[key] = value_nm
            mol_n_assays[key] = mol_n_assays.get(key, 0) + 1

    mol_names: dict[str, str] = {}
    for activity in chembl.get("activities", []):
        mid = activity.get("chembl_id") or _norm(activity.get("molecule", ""))
        name = activity.get("molecule")
        if name and mid and mid not in mol_names:
            mol_names[mid] = name

    for mid, name in mol_names.items():
        mol_info = chembl.get("molecules", {}).get(mid, {})
        max_phase = mol_info.get("max_phase")
        status = {
            4: "FDA Approved",
            3: "Phase 3",
            2: "Phase 2",
            1: "Phase 1",
            0: "Preclinical",
        }.get(int(max_phase) if max_phase else -1, "")
        upsert(
            name,
            {
                "chembl_id": mid,
                "best_ic50_nm": mol_best_ic50.get(mid),
                "max_phase": max_phase,
                "clinical_status": status,
                "_n_assays": mol_n_assays.get(mid, 1),
            },
            "ChEMBL",
        )

    for item in dgidb:
        upsert(
            item.get("drug"),
            {
                "clinical_status": "FDA Approved" if item.get("approved") else "",
                "_n_assays": max(1, int(item.get("n_sources") or 1)),
            },
            "DGIdb",
        )

    for item in ot.get("drugs", []):
        max_phase = item.get("max_phase")
        status = "FDA Approved" if item.get("approved") else (f"Phase {int(max_phase)}" if max_phase else "")
        upsert(
            item.get("name"),
            {
                "ot_id": item.get("ot_id"),
                "max_phase": max_phase,
                "clinical_status": status,
                "mechanism": item.get("mechanism") or "",
                "indication": item.get("indication") or "",
                "_n_assays": 1,
            },
            "Open Targets",
        )

    for item in iuphar:
        upsert(
            item.get("drug"),
            {
                "iuphar_id": item.get("iuphar_id"),
                "best_ic50_nm": item.get("ic50_nm"),
                "clinical_status": "FDA Approved" if item.get("approved") else "",
                "mechanism": item.get("action") or "",
                "_n_assays": 1,
            },
            "IUPHAR",
        )

    return drugs


def _build_ai_summary(drugs_raw) -> list:
    summary = []
    for drug in list(drugs_raw.values())[:35]:
        summary.append(
            {
                "name": drug["name"],
                "ic50_nm": round(drug["best_ic50_nm"], 3) if drug.get("best_ic50_nm") else None,
                "status": drug.get("clinical_status"),
                "max_phase": drug.get("max_phase"),
                "n_dbs": drug.get("n_databases"),
                "mechanism": drug.get("mechanism"),
                "chembl_id": drug.get("chembl_id"),
            }
        )
    return summary


def _build_prompt(gene: str, itype: str, summary: list) -> tuple[str, str]:
    system = f"""You are a world-class medicinal chemist and pharmacologist.
Given raw database query results for the target "{gene}", produce a definitive,
expert-curated JSON array of the top 12-18 {itype}s.

Use your knowledge to:
- Verify current FDA approval status and clinical trial phases
- Add important compounds missing from the database results
- Confirm or correct IC50/Ki values and report the best published value
- Determine selectivity (High, Moderate, or Low) versus related targets

Return ONLY a valid JSON array with these exact keys per object:
  name              - most common drug or compound name
  clinical_status   - one of: FDA Approved | Phase 4 | Phase 3 | Phase 2 | Phase 1 | Preclinical | Research Tool
  best_ic50_nm      - best IC50 or Ki in nM as a float, or null
  selectivity       - High | Moderate | Low
  mechanism         - one precise sentence about target, pharmacology type, and effect
  indication        - primary clinical or research indication
  chembl_id         - CHEMBLXXXXX if known, else null
  notes             - key differentiator such as generation, resistance profile, or combo use
  n_assays          - estimated number of binding assays or literature data points
  n_databases       - number of databases with evidence (1-5)
  ai_bonus          - expert score from 0-10 for mechanistic validation, clinical importance,
                      research utility, and novelty. Reserve 9-10 for landmark drugs.

No markdown. No prose. No code fences. Start the response with [ and end it with ]."""

    user = f"""Target: {gene} | Interaction type: {itype}
Database candidates ({len(summary)} compounds):
{json.dumps(summary, indent=2)}

Return the expert-curated JSON array of the top {itype}s of {gene}.
Remember: respond ONLY with the JSON array starting with [ and ending with ]."""
    return system, user


def _ai_oai_compat(gene, itype, summary, api_key, model, base_url):
    try:
        from openai import OpenAI
    except ImportError:
        st.error("The `openai` package is not installed.")
        return []

    system, user = _build_prompt(gene, itype, summary)
    client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_array(
            text,
            "AI returned a response but no JSON array was found. Showing database-only results.",
        )
    except Exception as exc:
        st.error(f"AI API error ({base_url}): {exc}")
    return []


def _ai_ollama(gene, itype, summary, model):
    system, user = _build_prompt(gene, itype, summary)

    try:
        from openai import OpenAI

        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        resp = client.chat.completions.create(
            model=model or "llama3",
            max_tokens=4096,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_array(text)
    except Exception:
        pass

    try:
        full_prompt = f"{system}\n\n{user}"
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model or "llama3", "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        text = r.json().get("response", "")
        return _extract_json_array(text, "Ollama returned a response but no JSON array was found.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach Ollama at localhost:11434. Start it with `ollama serve`.")
    except Exception as exc:
        st.error(f"Ollama error: {exc}")
    return []


def _ai_gemini(gene, itype, summary, api_key, model):
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("The `google-generativeai` package is not installed.")
        return []

    system, user = _build_prompt(gene, itype, summary)
    genai.configure(api_key=api_key)

    def _try(with_search: bool):
        kwargs = {"model_name": model, "system_instruction": system}
        if with_search:
            kwargs["tools"] = ["google_search_retrieval"]
        gmodel = genai.GenerativeModel(**kwargs)
        response = gmodel.generate_content(user)
        return _extract_json_array(response.text or "")

    try:
        return _try(with_search=True)
    except Exception:
        pass
    try:
        return _try(with_search=False)
    except Exception as exc:
        st.error(f"Gemini API error: {exc}")
    return []


def ai_synthesize(gene, itype, drugs_raw, api_key, provider_cfg, ollama_model=""):
    summary = _build_ai_summary(drugs_raw)
    provider_id = provider_cfg["id"]

    if provider_id == "oai_compat":
        return _ai_oai_compat(gene, itype, summary, api_key, provider_cfg["model"], provider_cfg["base_url"])
    if provider_id == "ollama":
        return _ai_ollama(gene, itype, summary, ollama_model or "llama3")
    if provider_id == "gemini":
        return _ai_gemini(gene, itype, summary, api_key, provider_cfg["model"])
    return []


def _build_puter_command(gene: str, itype: str, drugs_raw, model: str) -> dict:
    summary = _build_ai_summary(drugs_raw)
    system, user = _build_prompt(gene, itype, summary)
    request_id = f"{_norm(gene)}-{itype}-{int(time.time() * 1000)}"
    return {
        "type": "run_puter_ai",
        "request_id": request_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "model": model or "gpt-5-nano",
            "temperature": 0.2,
            "max_tokens": 4096,
        },
    }


def _sync_puter_result(bridge_value):
    if not isinstance(bridge_value, dict):
        return bridge_value
    pending = st.session_state.get("pending_puter_request") or {}
    command = pending.get("command") or {}
    if (
        bridge_value.get("type") == "puter_result"
        and bridge_value.get("request_id")
        and bridge_value.get("request_id") == command.get("request_id")
    ):
        st.session_state["latest_puter_result"] = bridge_value
    return bridge_value


def build_df(drugs_raw, ai_drugs):
    merged: dict[str, dict] = {}

    for ai_drug in ai_drugs:
        key = _norm(ai_drug.get("name", ""))
        raw = drugs_raw.get(key, {})
        combo = {**raw}
        for k, v in ai_drug.items():
            if v not in (None, ""):
                if k == "best_ic50_nm" and combo.get("best_ic50_nm") is not None:
                    combo[k] = min(combo["best_ic50_nm"], float(v))
                else:
                    combo[k] = v
        merged[key] = combo

    for key, drug in drugs_raw.items():
        if key not in merged and drug.get("name"):
            merged[key] = drug

    rows = []
    for drug in merged.values():
        name = drug.get("name")
        if not name:
            continue
        score = compute_score(drug)
        chembl_id = drug.get("chembl_id") or ""
        encoded = requests.utils.quote(name)
        ic50 = drug.get("best_ic50_nm")
        potency = pic50_from_nm(ic50)
        status = drug.get("clinical_status") or "Unknown"
        rows.append(
            {
                "Drug / Compound": name,
                "Score": score,
                "Clinical Status": status,
                "Best IC50 (nM)": round(ic50, 3) if ic50 else None,
                "pIC50": round(potency, 2) if potency else None,
                "Selectivity": drug.get("selectivity") or "-",
                "Mechanism": drug.get("mechanism") or "-",
                "Indication": drug.get("indication") or "-",
                "Notes": drug.get("notes") or "-",
                "# Assays": int(drug.get("n_assays") or 1),
                "# Databases": int(drug.get("n_databases") or 1),
                "ChEMBL ID": chembl_id,
                "ChEMBL": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/" if chembl_id else "",
                "PubChem": f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded}",
                "DrugBank": f"https://go.drugbank.com/unearth/q?query={encoded}&searcher=drugs",
                "DGIdb": f"https://dgidb.org/results?searchTerms={encoded}",
                "IUPHAR": (
                    f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={drug['iuphar_id']}"
                    if drug.get("iuphar_id")
                    else f"https://www.guidetopharmacology.org/GRAC/ObjectDisplayForward?searchString={encoded}"
                ),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.index += 1
    return df


def main():
    settings = _load_local_settings()
    _init_session_defaults(settings)

    free_providers = [k for k, v in AI_PROVIDERS.items() if v["free"]]
    paid_providers = [k for k, v in AI_PROVIDERS.items() if not v["free"]]
    bridge_value = {}

    with st.sidebar:
        st.header("Configuration")
        st.subheader("AI Provider")

        tier_options = ["Free", "Paid"]
        if st.session_state["tier_name"] not in tier_options:
            st.session_state["tier_name"] = "Free"
        tier_name = st.radio(
            "Tier",
            tier_options,
            horizontal=True,
            label_visibility="collapsed",
            key="tier_name",
        )
        provider_list = free_providers if tier_name == "Free" else paid_providers
        if st.session_state["provider_name"] not in provider_list:
            st.session_state["provider_name"] = provider_list[0]
        provider_name = st.selectbox("Model", provider_list, label_visibility="collapsed", key="provider_name")
        provider_cfg = AI_PROVIDERS[provider_name]
        _update_local_settings(tier_name=tier_name, provider_name=provider_name)

        st.caption(provider_cfg["note"])

        if provider_cfg["id"] == "ollama":
            ollama_model = st.text_input(
                "Ollama model name",
                key="ollama_model",
                help="Run `ollama list` to see installed models. Pull one with `ollama pull llama3`.",
            )
            _update_local_settings(ollama_model=ollama_model)
            st.caption("Make sure Ollama is running: `ollama serve`.")
        else:
            ollama_model = st.session_state.get("ollama_model", "llama3")

        if provider_cfg["id"] == "puter_js":
            puter_model = st.text_input("Puter model", key="puter_model")
            _update_local_settings(puter_model=puter_model)
            pending_puter = st.session_state.get("pending_puter_request")
            puter_command = (pending_puter or {}).get("command") or {"type": "status"}
            bridge_value = PUTER_BRIDGE(
                command=puter_command,
                default={"type": "bridge_init"},
                key="puter_bridge_component",
            )
            bridge_value = _sync_puter_result(bridge_value)
            if isinstance(bridge_value, dict) and bridge_value.get("signed_in"):
                user = bridge_value.get("user") or {}
                label = user.get("username") or user.get("email") or user.get("uuid") or "Connected"
                st.success(f"Puter connected: {label}")
            else:
                st.info("Use the embedded Puter control below to sign in.")
            st.caption("Puter opens a popup, so its sign-in button is rendered inside the browser component.")
        else:
            puter_model = st.session_state.get("puter_model", "gpt-5-nano")

        api_key = ""
        if provider_cfg["key_required"]:
            api_widget_key = _get_api_widget_key(provider_name)
            if api_widget_key not in st.session_state:
                st.session_state[api_widget_key] = settings.get("api_keys", {}).get(provider_name, "")
            api_key = st.text_input(
                "API Key",
                type="password",
                help=provider_cfg["key_help"],
                placeholder=provider_cfg["key_placeholder"],
                key=api_widget_key,
            )
            save_col, clear_col = st.columns(2)
            if save_col.button("Save key locally", use_container_width=True):
                _save_provider_api_key(provider_name, api_key.strip())
                st.success(f"Saved locally to `{LOCAL_SETTINGS_PATH}`.")
            if clear_col.button("Forget saved key", use_container_width=True):
                _remove_provider_api_key(provider_name)
                st.session_state[api_widget_key] = ""
                api_key = ""
                st.success("Saved key removed.")
            st.caption(provider_cfg["key_help"])
            st.caption(f"Stored in plain text at `{LOCAL_SETTINGS_PATH}`.")

        if provider_cfg.get("supports_search"):
            st.success("Live search grounding available")
        elif provider_cfg["id"] == "puter_js":
            st.info("Browser-side AI via Puter.js")
        else:
            st.info("Uses model training knowledge without live search")

        st.divider()
        st.subheader("Filters")
        min_score = st.slider("Min score", 0.0, 10.0, 0.0, 0.5)
        status_filter = st.multiselect("Clinical status (empty = all)", STATUS_ORDER[:-1], default=[])
        max_ic50 = st.number_input("Max IC50 (nM)", 0.0, 1e7, 100_000.0, 1000.0)

        st.divider()
        st.subheader("Scoring weights")
        st.caption("Fixed weights used by the composite score:")
        st.progress(0.30, "Potency 30%")
        st.progress(0.30, "Clinical 30%")
        st.progress(0.20, "Evidence 20%")
        st.progress(0.10, "Selectivity 10%")
        st.progress(0.10, "AI bonus 10%")

        st.divider()
        st.caption("Databases: ChEMBL, DGIdb, Open Targets, IUPHAR, PubChem")
        st.caption(f"AI: {provider_name}")

    st.title("PharmaQuery Pro")
    st.markdown("Comprehensive drug-gene interaction discovery across 5 databases with AI-assisted synthesis.")

    c1, c2, c3 = st.columns([3, 1.5, 1])
    with c1:
        gene_input = st.text_input(
            "Gene / target",
            key="gene_input",
            label_visibility="collapsed",
            placeholder="Gene or target (for example: EGFR, mTOR, BCR-ABL, COX-2, dopamine D2)",
        )
    with c2:
        itype = st.selectbox(
            "Type",
            ["inhibitor", "agonist", "modulator"],
            format_func=str.capitalize,
            label_visibility="collapsed",
            key="itype_input",
        )
    with c3:
        go_btn = st.button("Search", type="primary", use_container_width=True)

    st.caption("Try: EGFR, BCR-ABL, mTOR, VEGFR2, HDAC1, CDK4, PI3Ka, ACE2, PCSK9, dopamine D2")
    st.divider()

    if go_btn and gene_input.strip():
        st.session_state["active_search"] = {"gene": gene_input.strip(), "itype": itype}
        st.session_state["pending_puter_request"] = None
        st.session_state["latest_puter_result"] = None
        _update_local_settings(last_gene=gene_input.strip(), last_itype=itype)

    active_search = st.session_state.get("active_search")
    if not active_search:
        st.info("Enter a gene symbol or receptor name above and click `Search`.")
        with st.expander("How scoring works"):
            st.markdown(
                """
| Component | Weight | Details |
|-----------|--------|---------|
| **Potency** | 30% | pIC50 scale: under 1 nM scores highest, above 10 uM scores lowest |
| **Clinical status** | 30% | FDA Approved = 30 points, down to Research Tool = 4 points |
| **Evidence breadth** | 20% | Log-scaled assay count plus multi-database consensus |
| **Selectivity** | 10% | High selectivity scores highest |
| **AI expert bonus** | 10% | Model-assigned domain score |

Final score is normalized to **0-10**.
"""
            )
        return

    if provider_cfg["key_required"] and not api_key.strip():
        st.error("Please enter your API key in the sidebar to enable AI synthesis.")
        return

    gene = active_search["gene"]
    itype = active_search["itype"]
    st.caption(f"Showing results for `{gene}` ({itype}).")

    progress = st.progress(0.0, "Starting database queries...")
    status_row = st.columns(5)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures_map = {
            ex.submit(query_chembl, gene, itype): ("ChEMBL", 0),
            ex.submit(query_dgidb, gene, itype): ("DGIdb", 1),
            ex.submit(query_open_targets, gene): ("Open Targets", 2),
            ex.submit(query_iuphar, gene, itype): ("IUPHAR", 3),
            ex.submit(query_pubchem, gene): ("PubChem", 4),
        }
        db_results = {}
        done = 0
        for fut in as_completed(futures_map):
            label, col_i = futures_map[fut]
            try:
                db_results[label] = fut.result()
                status_row[col_i].success(f"OK {label}")
            except Exception:
                db_results[label] = {} if label in ("ChEMBL", "Open Targets") else []
                status_row[col_i].error(f"Failed {label}")
            done += 1
            progress.progress(done / 6, f"Completed {label}")

    progress.progress(5 / 6, "Consolidating and deduplicating...")
    drugs_raw = consolidate(
        db_results.get("ChEMBL", {}),
        db_results.get("DGIdb", []),
        db_results.get("Open Targets", {}),
        db_results.get("IUPHAR", []),
    )

    ai_drugs = []
    if provider_cfg["id"] == "puter_js":
        if not (isinstance(bridge_value, dict) and bridge_value.get("signed_in")):
            progress.empty()
            st.error("Puter.js is selected, but no Puter session is connected. Sign in in the sidebar and search again.")
            return

        search_key = json.dumps({"gene": gene, "itype": itype, "model": puter_model}, sort_keys=True)
        pending_puter = st.session_state.get("pending_puter_request")
        latest_puter = st.session_state.get("latest_puter_result")

        if (
            latest_puter
            and pending_puter
            and latest_puter.get("request_id") == (pending_puter.get("command") or {}).get("request_id")
            and pending_puter.get("search_key") == search_key
        ):
            if latest_puter.get("ok"):
                ai_drugs = _extract_json_array(
                    latest_puter.get("text", ""),
                    "Puter.js returned a response but no JSON array was found. Showing database-only results.",
                )
            else:
                st.error(f"Puter.js error: {latest_puter.get('error', 'Unknown error')}")
            st.session_state["pending_puter_request"] = None
            st.session_state["latest_puter_result"] = None
        else:
            if not pending_puter or pending_puter.get("search_key") != search_key:
                st.session_state["pending_puter_request"] = {
                    "search_key": search_key,
                    "command": _build_puter_command(gene, itype, drugs_raw, puter_model),
                }
                st.session_state["latest_puter_result"] = None
                st.rerun()

            progress.progress(1.0, "Waiting for Puter.js to finish...")
            st.info("Puter.js is synthesizing the shortlist in your browser. Keep this tab open for a moment.")
            st.stop()
    else:
        progress.progress(5.5 / 6, f"{provider_name}: synthesizing...")
        ai_drugs = ai_synthesize(gene, itype, drugs_raw, api_key, provider_cfg, ollama_model)

    progress.progress(1.0, "Computing scores...")
    df = build_df(drugs_raw, ai_drugs)
    progress.empty()

    if df.empty:
        st.error(f"No results found for `{gene}`. Try the official gene symbol, such as `EGFR`.")
        return

    dff = df.copy()
    if min_score > 0:
        dff = dff[dff["Score"] >= min_score]
    if status_filter:
        dff = dff[dff["Clinical Status"].isin(status_filter)]
    if max_ic50 < 100_000:
        dff = dff[dff["Best IC50 (nM)"].isna() | (dff["Best IC50 (nM)"] <= max_ic50)]

    if dff.empty:
        st.warning("No compounds match the current filters. Relax the sidebar filters to see results.")
        return

    n_approved = (dff["Clinical Status"] == "FDA Approved").sum()
    avg_score = dff["Score"].mean()
    best_ic50 = dff["Best IC50 (nM)"].dropna().min()
    n_dbs_avg = dff["# Databases"].mean()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Compounds found", len(dff))
    m2.metric("FDA Approved", int(n_approved))
    m3.metric("Avg score", f"{avg_score:.1f} / 10")
    m4.metric("Best IC50", f"{best_ic50:.3f} nM" if pd.notna(best_ic50) else "N/A")
    m5.metric("Avg # databases", f"{n_dbs_avg:.1f}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Potency Chart", "Breakdown", "Export"])

    with tab1:
        display_cols = [
            "Drug / Compound",
            "Score",
            "Clinical Status",
            "Best IC50 (nM)",
            "pIC50",
            "Selectivity",
            "Mechanism",
            "Indication",
            "Notes",
            "# Assays",
            "# Databases",
        ]

        def _score_bg(val):
            if pd.isna(val):
                return ""
            if val >= 7.5:
                return "background-color:#e8f5e9;color:#1b5e20"
            if val >= 5.0:
                return "background-color:#e3f2fd;color:#0d47a1"
            return "background-color:#fff3e0;color:#bf360c"

        styled = (
            dff[display_cols]
            .style.map(_score_bg, subset=["Score"])
            .format(
                {
                    "Score": "{:.2f}",
                    "Best IC50 (nM)": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
                    "pIC50": lambda x: f"{x:.2f}" if pd.notna(x) else "-",
                }
            )
        )
        st.dataframe(styled, use_container_width=True, height=520)

        st.subheader("Source links")
        selected = st.selectbox("Select compound", dff["Drug / Compound"].tolist(), key="selected_compound")
        if selected:
            row = dff[dff["Drug / Compound"] == selected].iloc[0]
            cols = st.columns(5)
            if row["ChEMBL"]:
                cols[0].markdown(f"[ChEMBL]({row['ChEMBL']})")
            cols[1].markdown(f"[PubChem]({row['PubChem']})")
            cols[2].markdown(f"[DrugBank]({row['DrugBank']})")
            cols[3].markdown(f"[DGIdb]({row['DGIdb']})")
            cols[4].markdown(f"[IUPHAR]({row['IUPHAR']})")
            if row["Notes"] != "-":
                st.caption(f"Note: {row['Notes']}")

    with tab2:
        plot_df = dff.dropna(subset=["Best IC50 (nM)"]).copy()
        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x="pIC50",
                y="Score",
                color="Clinical Status",
                size="# Assays",
                size_max=30,
                hover_name="Drug / Compound",
                hover_data={"Best IC50 (nM)": ":.3f", "Mechanism": True, "Selectivity": True, "pIC50": ":.2f"},
                title=f"Potency vs composite score - {gene} {itype}s",
                labels={"pIC50": "pIC50 (higher = more potent)", "Score": "Composite score (0-10)"},
                color_discrete_map=STATUS_COLORS,
            )
            fig.update_layout(height=480, font_size=12, legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IC50 data available.")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(
                dff,
                x="Score",
                nbins=15,
                title="Score distribution",
                color_discrete_sequence=["#1565c0"],
            )
            fig_hist.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            vc = dff["Clinical Status"].value_counts()
            fig_pie = px.pie(
                values=vc.values,
                names=vc.index,
                title="Clinical status breakdown",
                color=vc.index,
                color_discrete_map=STATUS_COLORS,
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        top_n = min(15, len(dff))
        top = dff.head(top_n).sort_values("Score")
        fig_bar = px.bar(
            top,
            x="Score",
            y="Drug / Compound",
            orientation="h",
            color="Score",
            color_continuous_scale=["#ef5350", "#42a5f5", "#66bb6a"],
            text="Score",
            title=f"Top {top_n} by composite score",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_bar.update_layout(
            height=500,
            showlegend=False,
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.subheader("Download results")
        exp_cols = [
            "Drug / Compound",
            "Score",
            "Clinical Status",
            "Best IC50 (nM)",
            "pIC50",
            "Selectivity",
            "Mechanism",
            "Indication",
            "Notes",
            "# Assays",
            "# Databases",
            "ChEMBL ID",
            "ChEMBL",
            "PubChem",
            "DrugBank",
            "DGIdb",
            "IUPHAR",
        ]
        csv = dff[exp_cols].to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"pharmaquery_{gene}_{itype}.csv",
            mime="text/csv",
        )
        st.dataframe(dff[exp_cols], use_container_width=True, height=400)


if __name__ == "__main__":
    main()
