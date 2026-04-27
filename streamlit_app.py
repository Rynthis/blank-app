"""
PharmaQuery Pro ───────────────
Comprehensive drug-gene interaction discovery.
Queries ChEMBL · DGIdb · Open Targets · IUPHAR · PubChem in parallel,
then synthesizes and scores results with ChatGPT or Gemini + web search.

Run: streamlit run app.py
Requirements: pip install streamlit requests pandas plotly openai google-generativeai
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import math, json, re, time, unicodedata
from typing import Optional

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaQuery Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    background:#f8f9fa; border-radius:8px; padding:.5rem 1rem; border:1px solid #e0e0e0;
}
.stDataFrame { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────
CHEMBL  = "https://www.ebi.ac.uk/chembl/api/data"
DGIDB   = "https://dgidb.org/api/graphql"
OT      = "https://api.platform.opentargets.org/api/v4/graphql"
IUPHAR  = "https://www.guidetopharmacology.org/services"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 20

ITYPE_KEYWORDS = {
    "inhibitor": ["inhibitor","antagonist","blocker","inverse agonist","negative modulator",
                  "suppressor","inhibition","channel blocker"],
    "agonist":   ["agonist","activator","positive modulator","partial agonist","full agonist",
                  "superagonist","potentiator","activation"],
    "modulator": [],
}

STATUS_ORDER  = ["FDA Approved","Phase 4","Phase 3","Phase 2","Phase 1",
                 "Preclinical","Research Tool","Unknown"]
STATUS_COLORS = {
    "FDA Approved": "#2e7d32", "Phase 4": "#388e3c", "Phase 3": "#1565c0",
    "Phase 2": "#283593",      "Phase 1": "#e65100", "Preclinical": "#880e4f",
    "Research Tool": "#616161","Unknown": "#9e9e9e",
}

# ── AI provider config ───────────────────────────────────────────────────
AI_PROVIDERS = {
    "ChatGPT (GPT-5.4)": {
        "id": "openai",
        "model": "gpt-5.4",
        "key_placeholder": "sk-...",
        "key_help": "Get one at platform.openai.com",
        "supports_search": True,   # via tool calling + brave/serper, or GPT-4o browsing
    },
    "ChatGPT (GPT-5.5)": {
        "id": "openai",
        "model": "gpt-5.5",
        "key_placeholder": "sk-...",
        "key_help": "Get one at platform.openai.com",
        "supports_search": False,
    },
    "Gemini 3 Flash Preview": {
        "id": "gemini",
        "model": "gemini-3-flash-preview",
        "key_placeholder": "AIza...",
        "key_help": "Get one at aistudio.google.com",
        "supports_search": True,   # Gemini has built-in Google Search grounding
    },
    "Gemini 3.1 Pro Preview": {
        "id": "gemini",
        "model": "gemini-3.1-pro-preview",
        "key_placeholder": "AIza...",
        "key_help": "Get one at aistudio.google.com",
        "supports_search": True,
    },
}

# ══════════════════════════════════════════════════════════════════════════
# DATABASE QUERY FUNCTIONS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def query_chembl(gene: str, itype: str) -> dict:
    out = {"target": None, "activities": [], "molecules": {}}
    try:
        r = requests.get(f"{CHEMBL}/target/search.json",
                         params={"q": gene, "limit": 10}, timeout=TIMEOUT)
        targets = r.json().get("targets", [])
        target = (
            next((t for t in targets if t["target_type"] == "SINGLE PROTEIN"
                  and "Homo sapiens" in t.get("organism", "")), None)
            or next((t for t in targets if t["target_type"] == "SINGLE PROTEIN"), None)
            or (targets[0] if targets else None)
        )
        if not target:
            return out
        cid = target["target_chembl_id"]
        out["target"] = {"name": target.get("pref_name"),
                         "chembl_id": cid, "organism": target.get("organism")}

        r2 = requests.get(f"{CHEMBL}/activity.json", params={
            "target_chembl_id": cid,
            "standard_type__in": "IC50,Ki,EC50,Kd,pIC50",
            "standard_relation__in": "=,<,<=",
            "limit": 100, "order_by": "standard_value",
        }, timeout=TIMEOUT)
        acts = [a for a in r2.json().get("activities", []) if a.get("standard_value")]
        out["activities"] = [{
            "molecule":  a.get("molecule_pref_name") or a.get("molecule_chembl_id"),
            "chembl_id": a.get("molecule_chembl_id"),
            "meas_type": a.get("standard_type"),
            "value":     float(a["standard_value"]),
            "units":     a.get("standard_units", "nM"),
        } for a in acts]

        mol_ids = list({a["chembl_id"] for a in out["activities"] if a.get("chembl_id")})[:25]
        def _fetch_mol(mid):
            try:
                mr = requests.get(f"{CHEMBL}/molecule/{mid}.json", timeout=TIMEOUT)
                mol = mr.json()
                return mid, {"name": mol.get("pref_name"), "max_phase": mol.get("max_phase", 0)}
            except:
                return mid, {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for mid, info in ex.map(_fetch_mol, mol_ids):
                out["molecules"][mid] = info
    except:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def query_dgidb(gene: str, itype: str) -> list:
    try:
        q = """query($names:[String!]!){
            genes(names:$names){nodes{name interactions{
                drug{name approved conceptId}
                interactionScore interactionTypes{type directionality}
                sources{fullName}
            }}}
        }"""
        r = requests.post(DGIDB, json={"query": q, "variables": {"names": [gene.upper()]}},
                          timeout=TIMEOUT)
        nodes = r.json().get("data", {}).get("genes", {}).get("nodes", [])
        ixns  = nodes[0].get("interactions", []) if nodes else []
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
            results.append({
                "drug":     i.get("drug", {}).get("name"),
                "approved": i.get("drug", {}).get("approved"),
                "dgi_score": i.get("interactionScore") or 0,
                "types":    ", ".join(type_strs),
                "n_sources": len(i.get("sources") or []),
                "sources":  [s.get("fullName") for s in (i.get("sources") or [])],
            })
        return results
    except:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def query_open_targets(gene: str) -> dict:
    out = {"target_id": None, "drugs": []}
    try:
        sq = """query($q:String!){search(queryString:$q,entityNames:["target"],
            page:{index:0,size:3}){hits{id name}}}"""
        r  = requests.post(OT, json={"query": sq, "variables": {"q": gene}}, timeout=TIMEOUT)
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
        r2   = requests.post(OT, json={"query": dq, "variables": {"id": tid}}, timeout=TIMEOUT)
        rows = (r2.json().get("data", {}).get("target", {})
                .get("knownDrugs", {}).get("rows", []))
        for row in rows:
            drug = row.get("drug", {})
            out["drugs"].append({
                "name":        drug.get("name"),
                "ot_id":       drug.get("id"),
                "max_phase":   drug.get("maximumClinicalTrialPhase"),
                "approved":    drug.get("isApproved"),
                "mechanism":   row.get("mechanismOfAction"),
                "action_type": row.get("actionType"),
                "indication":  (row.get("disease") or {}).get("name"),
            })
    except:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def query_iuphar(gene: str, itype: str) -> list:
    results = []
    try:
        r = requests.get(f"{IUPHAR}/targets",
                         params={"geneSymbol": gene.upper(), "species": "Human"}, timeout=TIMEOUT)
        targets = r.json() if r.ok and r.content else []
        if not targets:
            r2 = requests.get(f"{IUPHAR}/targets/search/{requests.utils.quote(gene)}", timeout=TIMEOUT)
            targets = r2.json() if r2.ok and r2.content else []
        if not targets:
            return []
        tid = targets[0].get("targetId")
        r3  = requests.get(f"{IUPHAR}/interactions/target/{tid}", timeout=TIMEOUT)
        ixns = r3.json() if r3.ok and r3.content else []
        filters = ITYPE_KEYWORDS.get(itype, [])
        for i in ixns:
            itype_str = ((i.get("type") or "") + " " + (i.get("action") or "")).lower()
            if filters and not any(f in itype_str for f in filters):
                continue
            aff_type = i.get("affinityType") or ""
            aff_val  = i.get("affinity")
            ic50_nm  = None
            if aff_val and aff_type.startswith("p"):
                ic50_nm = (10 ** -float(aff_val)) * 1e9
            lid = i.get("ligandId")
            results.append({
                "drug":      i.get("ligandName"),
                "iuphar_id": lid,
                "action":    i.get("action") or i.get("type"),
                "aff_type":  aff_type,
                "aff_value": aff_val,
                "ic50_nm":   ic50_nm,
                "approved":  i.get("ligandApproved"),
                "url": (f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lid}"
                        if lid else None),
            })
    except:
        pass
    return results[:40]


@st.cache_data(ttl=3600, show_spinner=False)
def query_pubchem(gene: str) -> list:
    results = []
    try:
        r = requests.get(f"{PUBCHEM}/assay/target/genesymbol/{gene.upper()}/aids/JSON", timeout=TIMEOUT)
        aids = r.json().get("IdentifierList", {}).get("AID", [])[:5]
        for aid in aids:
            r2 = requests.get(f"{PUBCHEM}/assay/aid/{aid}/summary/JSON", timeout=TIMEOUT)
            assay_info = r2.json().get("AssaySummaries", {}).get("AssaySummary", [{}])[0]
            name = assay_info.get("Name", "")
            r3 = requests.get(f"{PUBCHEM}/assay/aid/{aid}/cids/JSON",
                              params={"cids_type": "active", "list_return": 10}, timeout=TIMEOUT)
            cids = r3.json().get("AssayLink", {}).get("CID", [])[:10]
            for cid in cids:
                results.append({"cid": cid, "assay": name, "aid": aid})
    except:
        pass
    return results

# ══════════════════════════════════════════════════════════════════════════
# SCORING  (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def pic50_from_nm(ic50_nm: Optional[float]) -> Optional[float]:
    if not ic50_nm or ic50_nm <= 0:
        return None
    return -math.log10(ic50_nm * 1e-9)

def _potency_pts(ic50_nm):
    p = pic50_from_nm(ic50_nm)
    if p is None: return 4.0
    if p >= 10:   return 30
    elif p >= 9:  return 27
    elif p >= 8:  return 23
    elif p >= 7:  return 18
    elif p >= 6:  return 12
    elif p >= 5:  return 7
    else:         return 3

def _clinical_pts(status, max_phase=None):
    if max_phase is not None:
        mp = int(max_phase) if max_phase else 0
        return {4:30, 3:25, 2:20, 1:15, 0:6}.get(mp, 5)
    if not status: return 4.0
    s = status.lower()
    for keys, pts in [
        (["fda approved","approved","marketed"], 30),
        (["phase 4","phase iv"], 29),
        (["phase 3","phase iii"], 25),
        (["phase 2/3"], 22),
        (["phase 2","phase ii"], 20),
        (["phase 1/2"], 17),
        (["phase 1","phase i"], 15),
        (["ind filed"], 12),
        (["preclinical"], 8),
        (["research tool","research"], 4),
        (["withdrawn","discontinued","failed"], 2),
    ]:
        if any(k in s for k in keys): return float(pts)
    return 4.0

def _evidence_pts(n_assays, n_databases):
    return min(10.0, math.log1p(n_assays)*2.3) + min(10.0, n_databases*2.0)

def _selectivity_pts(selectivity):
    return {"high":10,"moderate":6,"low":2}.get((selectivity or "").lower(), 4)

def compute_score(drug):
    pot  = _potency_pts(drug.get("best_ic50_nm"))
    clin = _clinical_pts(drug.get("clinical_status",""), drug.get("max_phase"))
    evid = _evidence_pts(int(drug.get("n_assays") or 1), int(drug.get("n_databases") or 1))
    sel  = _selectivity_pts(drug.get("selectivity",""))
    ai   = float(drug.get("ai_bonus") or 5)
    raw  = pot*0.30 + clin*0.30 + evid*0.20 + sel*0.10 + ai*0.10
    return round(min(10.0, raw), 2)

# ══════════════════════════════════════════════════════════════════════════
# DATA CONSOLIDATION  (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def _norm(name):
    if not name: return ""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii","ignore").decode()
    return re.sub(r"[^a-z0-9]","", n.lower())

def consolidate(chembl, dgidb, ot, iuphar):
    drugs: dict[str, dict] = {}

    def upsert(name, data, source):
        if not name: return
        key = _norm(name)
        if key not in drugs:
            drugs[key] = {
                "name": name, "best_ic50_nm": None, "max_phase": None,
                "clinical_status": "", "selectivity": "", "mechanism": "",
                "indication": "", "chembl_id": None, "iuphar_id": None,
                "ot_id": None, "ai_bonus": 5, "n_assays": 0, "n_databases": 0,
                "_sources": set(),
            }
        d = drugs[key]
        for k, v in data.items():
            if k.startswith("_"): continue
            if v not in (None, ""):
                if k == "best_ic50_nm" and d.get(k) is not None:
                    d[k] = min(d[k], float(v))
                elif k == "max_phase" and d.get(k) is not None:
                    d[k] = max(int(d[k]), int(v) if v else 0)
                elif d.get(k) in (None, ""):
                    d[k] = v
        d["n_assays"] += int(data.get("_n_assays", 1))
        d["_sources"].add(source)
        d["n_databases"] = len(d["_sources"])

    # ChEMBL
    mol_best_ic50: dict[str, float] = {}
    mol_n_assays:  dict[str, int]   = {}
    for a in chembl.get("activities", []):
        mid = a.get("chembl_id"); name = a.get("molecule")
        if not mid and not name: continue
        key = mid or _norm(name); v = a.get("value"); u = a.get("units","nM")
        v_nm = None
        if v is not None:
            v = float(v)
            if u in ("uM","µM"): v_nm = v*1000
            elif u == "mM":      v_nm = v*1e6
            elif u in ("nM","nm"): v_nm = v
            elif u == "pM":      v_nm = v/1000
        if v_nm is not None:
            if key not in mol_best_ic50 or v_nm < mol_best_ic50[key]:
                mol_best_ic50[key] = v_nm
            mol_n_assays[key] = mol_n_assays.get(key, 0) + 1

    mol_names: dict[str, str] = {}
    for a in chembl.get("activities", []):
        mid = a.get("chembl_id") or _norm(a.get("molecule",""))
        name = a.get("molecule")
        if name and mid and mid not in mol_names: mol_names[mid] = name

    for mid, name in mol_names.items():
        mol_info   = chembl.get("molecules", {}).get(mid, {})
        max_phase  = mol_info.get("max_phase")
        status = {4:"FDA Approved",3:"Phase 3",2:"Phase 2",1:"Phase 1",0:"Preclinical"}.get(
                  int(max_phase) if max_phase else -1, "")
        upsert(name, {
            "chembl_id": mid, "best_ic50_nm": mol_best_ic50.get(mid),
            "max_phase": max_phase, "clinical_status": status,
            "_n_assays": mol_n_assays.get(mid, 1),
        }, "ChEMBL")

    for i in dgidb:
        upsert(i.get("drug"), {
            "clinical_status": "FDA Approved" if i.get("approved") else "",
            "_n_assays": max(1, int(i.get("n_sources") or 1)),
        }, "DGIdb")

    for d in ot.get("drugs", []):
        mp = d.get("max_phase")
        st = "FDA Approved" if d.get("approved") else (f"Phase {int(mp)}" if mp else "")
        upsert(d.get("name"), {
            "ot_id": d.get("ot_id"), "max_phase": mp, "clinical_status": st,
            "mechanism": d.get("mechanism") or "", "indication": d.get("indication") or "",
            "_n_assays": 1,
        }, "Open Targets")

    for i in iuphar:
        upsert(i.get("drug"), {
            "iuphar_id": i.get("iuphar_id"), "best_ic50_nm": i.get("ic50_nm"),
            "clinical_status": "FDA Approved" if i.get("approved") else "",
            "mechanism": i.get("action") or "", "_n_assays": 1,
        }, "IUPHAR")

    return drugs

# ══════════════════════════════════════════════════════════════════════════
# AI SYNTHESIS  (ChatGPT / Gemini)
# ══════════════════════════════════════════════════════════════════════════

def _build_prompt(gene, itype, summary):
    """Shared prompt content for both providers."""
    system = f"""You are a world-class medicinal chemist and pharmacologist.
Given raw database query results for the target "{gene}", produce a definitive,
expert-curated JSON array of the top 12-18 {itype}s.

Use available knowledge to:
• Verify current FDA approval status and clinical trial phases
• Add important compounds missing from the database results
• Confirm or correct IC50/Ki values (report best published value)
• Determine selectivity (High/Moderate/Low vs. related targets)

Return ONLY a valid JSON array with these exact keys per object:
  name              – most common drug/compound name
  clinical_status   – one of: FDA Approved | Phase 4 | Phase 3 | Phase 2 | Phase 1 | Preclinical | Research Tool
  best_ic50_nm      – best IC50/Ki in nM as a float, or null
  selectivity       – High | Moderate | Low
  mechanism         – one precise sentence (target, pharmacology type, effect)
  indication        – primary clinical or research indication
  chembl_id         – CHEMBLXXXXX if known, else null
  notes             – key differentiator: generation, resistance profile, combo use, etc.
  n_assays          – estimated number of binding assays/data points in literature
  n_databases       – number of databases with evidence (1-5)
  ai_bonus          – YOUR expert score 0-10 for: mechanistic validation, clinical importance,
                      research utility, novelty. Be discriminating; reserve 9-10 for landmark drugs.

No markdown. No prose. Start with [ end with ]."""

    user = f"""Target: {gene} | Interaction type: {itype}
Database candidates ({len(summary)} compounds):
{json.dumps(summary, indent=2)}

Return expert-curated JSON array of the top {itype}s of {gene}."""
    return system, user


def _ai_openai(gene, itype, summary, api_key, model):
    """Call OpenAI ChatGPT."""
    try:
        from openai import OpenAI
    except ImportError:
        st.error("openai package not installed. Run: pip install openai")
        return []

    system, user = _build_prompt(gene, itype, summary)
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=5000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            # GPT-4o supports web browsing via the Responses API; here we use chat completions
            # which relies on the model's training knowledge. For live search, use the
            # Responses API or pass a search tool definition if your account supports it.
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
    return []


def _ai_gemini(gene, itype, summary, api_key, model):
    """Call Google Gemini (with Google Search grounding when available)."""
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("google-generativeai package not installed. Run: pip install google-generativeai")
        return []

    system, user = _build_prompt(gene, itype, summary)
    genai.configure(api_key=api_key)

    try:
        gmodel = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            # Google Search grounding — available for 1.5 Pro / Flash
            tools=["google_search_retrieval"],
        )
        response = gmodel.generate_content(user)
        text = response.text or ""
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        # Retry without search grounding if it fails (e.g. unsupported model tier)
        try:
            gmodel2 = genai.GenerativeModel(model_name=model, system_instruction=system)
            response2 = gmodel2.generate_content(user)
            text2 = response2.text or ""
            m2 = re.search(r"\[[\s\S]*\]", text2)
            if m2:
                return json.loads(m2.group(0))
        except Exception as e2:
            st.error(f"Gemini API error: {e2}")
    return []


def ai_synthesize(gene, itype, drugs_raw, api_key, provider_cfg):
    summary = []
    for d in list(drugs_raw.values())[:35]:
        summary.append({
            "name":      d["name"],
            "ic50_nm":   round(d["best_ic50_nm"], 3) if d.get("best_ic50_nm") else None,
            "status":    d.get("clinical_status"),
            "max_phase": d.get("max_phase"),
            "n_dbs":     d.get("n_databases"),
            "mechanism": d.get("mechanism"),
            "chembl_id": d.get("chembl_id"),
        })

    pid   = provider_cfg["id"]
    model = provider_cfg["model"]

    if pid == "openai":
        return _ai_openai(gene, itype, summary, api_key, model)
    elif pid == "gemini":
        return _ai_gemini(gene, itype, summary, api_key, model)
    return []

# ══════════════════════════════════════════════════════════════════════════
# BUILD FINAL DATAFRAME  (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def build_df(drugs_raw, ai_drugs):
    ai_lut = {_norm(d.get("name","")): d for d in ai_drugs}
    merged: dict[str, dict] = {}

    for ai_d in ai_drugs:
        key = _norm(ai_d.get("name",""))
        raw = drugs_raw.get(key, {})
        combo = {**raw}
        for k, v in ai_d.items():
            if v not in (None,""):
                if k == "best_ic50_nm" and combo.get("best_ic50_nm") is not None:
                    combo[k] = min(combo["best_ic50_nm"], float(v))
                else:
                    combo[k] = v
        merged[key] = combo

    for key, d in drugs_raw.items():
        if key not in merged and d.get("name"):
            merged[key] = d

    rows = []
    for d in merged.values():
        name = d.get("name")
        if not name: continue
        score = compute_score(d)
        cid   = d.get("chembl_id") or ""
        enc   = requests.utils.quote(name)
        ic50  = d.get("best_ic50_nm")
        p     = pic50_from_nm(ic50)
        status = d.get("clinical_status") or "Unknown"
        rows.append({
            "Drug / Compound":  name,
            "Score":            score,
            "Clinical Status":  status,
            "Best IC50 (nM)":   round(ic50, 3) if ic50 else None,
            "pIC50":            round(p, 2)     if p    else None,
            "Selectivity":      d.get("selectivity") or "—",
            "Mechanism":        d.get("mechanism")   or "—",
            "Indication":       d.get("indication")  or "—",
            "Notes":            d.get("notes")        or "—",
            "# Assays":         int(d.get("n_assays")    or 1),
            "# Databases":      int(d.get("n_databases") or 1),
            "ChEMBL ID":        cid,
            "ChEMBL":   f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/" if cid else "",
            "PubChem":  f"https://pubchem.ncbi.nlm.nih.gov/#query={enc}",
            "DrugBank": f"https://go.drugbank.com/unearth/q?query={enc}&searcher=drugs",
            "DGIdb":    f"https://dgidb.org/results?searchTerms={enc}",
            "IUPHAR":   (f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={d['iuphar_id']}"
                         if d.get("iuphar_id") else
                         f"https://www.guidetopharmacology.org/GRAC/ObjectDisplayForward?searchString={enc}"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df.index += 1
    return df

# ══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # AI provider selection
        provider_name = st.selectbox(
            "AI model",
            list(AI_PROVIDERS.keys()),
            help="ChatGPT uses the OpenAI API; Gemini uses the Google AI API."
        )
        provider_cfg = AI_PROVIDERS[provider_name]

        api_key = st.text_input(
            f"API Key ({provider_cfg['id'].upper()})",
            type="password",
            help=provider_cfg["key_help"],
            placeholder=provider_cfg["key_placeholder"],
        )
        if provider_cfg["id"] == "openai":
            st.caption("💡 Get a key at [platform.openai.com](https://platform.openai.com)")
        else:
            st.caption("💡 Get a key at [aistudio.google.com](https://aistudio.google.com)")

        st.divider()
        st.subheader("Filters")
        min_score    = st.slider("Min score", 0.0, 10.0, 0.0, 0.5)
        status_filter = st.multiselect("Clinical status (empty = all)", STATUS_ORDER[:-1], default=[])
        max_ic50     = st.number_input("Max IC50 (nM)", 0.0, 1e7, 100_000.0, 1000.0)

        st.divider()
        st.subheader("Scoring weights")
        st.caption("Fixed optimal weights:")
        st.progress(0.30, "Potency 30%")
        st.progress(0.30, "Clinical 30%")
        st.progress(0.20, "Evidence 20%")
        st.progress(0.10, "Selectiv 10%")
        st.progress(0.10, "AI bonus 10%")

        st.divider()
        st.caption("Databases: ChEMBL · DGIdb · Open Targets · IUPHAR · PubChem")
        if provider_cfg["supports_search"]:
            st.caption(f"✅ {provider_name} supports live search grounding")
        else:
            st.caption(f"ℹ️ {provider_name} uses training-data knowledge (no live search)")

    # ── Header ───────────────────────────────────────────────────────────
    st.title("🔬 PharmaQuery Pro")
    st.markdown(
        "Comprehensive drug-gene interaction discovery · "
        "5 databases · AI scoring · web synthesis"
    )

    # ── Search bar ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1.5, 1])
    with c1:
        gene = st.text_input(
            "Gene / target", label_visibility="collapsed",
            placeholder="Gene or target (e.g. EGFR, mTOR, BCR-ABL, COX-2, dopamine D2)"
        )
    with c2:
        itype = st.selectbox(
            "Type", ["inhibitor","agonist","modulator"],
            format_func=str.capitalize, label_visibility="collapsed"
        )
    with c3:
        go_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    st.caption("Try: EGFR · BCR-ABL · mTOR · VEGFR2 · HDAC1 · CDK4 · PI3Kα · ACE2 · PCSK9 · dopamine D2")
    st.divider()

    if not go_btn or not gene.strip():
        st.info("Enter a gene symbol or receptor name above and click **Search**.")
        with st.expander("How scoring works"):
            st.markdown("""
| Component | Weight | Details |
|-----------|--------|---------|
| **Potency** | 30% | pIC50 scale: < 1 nM = max, > 10 µM = min |
| **Clinical status** | 30% | FDA Approved = 30 pts, down to Research Tool = 4 pts |
| **Evidence breadth** | 20% | Log-scaled assay count + multi-database consensus |
| **Selectivity** | 10% | High vs. off-targets = 10 pts |
| **AI expert bonus** | 10% | AI model's domain-knowledge score |

Final score is normalized to **0–10**.
""")
        return

    if not api_key:
        st.error(f"Please enter your {provider_cfg['id'].upper()} API key in the sidebar to enable AI synthesis.")
        return

    gene = gene.strip()

    # ── Parallel DB queries ───────────────────────────────────────────────
    prog       = st.progress(0.0, "Starting database queries...")
    status_row = st.columns(5)
    db_labels  = ["ChEMBL","DGIdb","Open Targets","IUPHAR","PubChem"]

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures_map = {
            ex.submit(query_chembl, gene, itype):      ("ChEMBL", 0),
            ex.submit(query_dgidb, gene, itype):       ("DGIdb", 1),
            ex.submit(query_open_targets, gene):       ("Open Targets", 2),
            ex.submit(query_iuphar, gene, itype):      ("IUPHAR", 3),
            ex.submit(query_pubchem, gene):            ("PubChem", 4),
        }
        db_results = {}
        done = 0
        for fut in as_completed(futures_map):
            label, col_i = futures_map[fut]
            try:
                db_results[label] = fut.result()
                status_row[col_i].success(f"✓ {label}")
            except:
                db_results[label] = {} if label in ("ChEMBL","Open Targets") else []
                status_row[col_i].error(f"✗ {label}")
            done += 1
            prog.progress(done / 6, f"Completed {label}")

    # ── Consolidate ───────────────────────────────────────────────────────
    prog.progress(5/6, "Consolidating and deduplicating...")
    drugs_raw = consolidate(
        db_results.get("ChEMBL", {}),
        db_results.get("DGIdb", []),
        db_results.get("Open Targets", {}),
        db_results.get("IUPHAR", []),
    )

    # ── AI synthesis ──────────────────────────────────────────────────────
    prog.progress(5.5/6, f"{provider_name}: synthesizing...")
    ai_drugs = ai_synthesize(gene, itype, drugs_raw, api_key, provider_cfg)

    # ── Build dataframe ───────────────────────────────────────────────────
    prog.progress(1.0, "Computing scores...")
    df = build_df(drugs_raw, ai_drugs)
    prog.empty()

    if df.empty:
        st.error(f"No results found for **{gene}**. Try the official gene symbol (e.g. EGFR not 'EGF receptor').")
        return

    # ── Apply sidebar filters ─────────────────────────────────────────────
    dff = df.copy()
    if min_score > 0:      dff = dff[dff["Score"] >= min_score]
    if status_filter:      dff = dff[dff["Clinical Status"].isin(status_filter)]
    if max_ic50 < 100_000: dff = dff[dff["Best IC50 (nM)"].isna() | (dff["Best IC50 (nM)"] <= max_ic50)]

    # ── Summary metrics ───────────────────────────────────────────────────
    n_approved  = (dff["Clinical Status"] == "FDA Approved").sum()
    avg_score   = dff["Score"].mean() if not dff.empty else 0
    best_ic50   = dff["Best IC50 (nM)"].dropna().min()
    n_dbs_avg   = dff["# Databases"].mean() if not dff.empty else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Compounds found", len(dff))
    m2.metric("FDA Approved",    int(n_approved))
    m3.metric("Avg score",       f"{avg_score:.1f} / 10")
    m4.metric("Best IC50",       f"{best_ic50:.3f} nM" if pd.notna(best_ic50) else "N/A")
    m5.metric("Avg # databases", f"{n_dbs_avg:.1f}")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Results","📊 Potency Chart","🥧 Breakdown","⬇️ Export"])

    # ─ Tab 1 ─────────────────────────────────────────────────────────────
    with tab1:
        display_cols = ["Drug / Compound","Score","Clinical Status","Best IC50 (nM)",
                        "pIC50","Selectivity","Mechanism","Indication","Notes","# Assays","# Databases"]
        def _score_bg(val):
            if pd.isna(val): return ""
            if val >= 7.5: return "background-color:#e8f5e9;color:#1b5e20"
            if val >= 5.0: return "background-color:#e3f2fd;color:#0d47a1"
            return "background-color:#fff3e0;color:#bf360c"

        styled = (
            dff[display_cols]
            .style.map(_score_bg, subset=["Score"])
            .format({
                "Score":          "{:.2f}",
                "Best IC50 (nM)": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                "pIC50":          lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
        )
        st.dataframe(styled, use_container_width=True, height=520)

        st.subheader("Source links")
        selected = st.selectbox("Select compound", dff["Drug / Compound"].tolist())
        if selected:
            row  = dff[dff["Drug / Compound"] == selected].iloc[0]
            cols = st.columns(5)
            if row["ChEMBL"]: cols[0].markdown(f"[ChEMBL ↗]({row['ChEMBL']})")
            cols[1].markdown(f"[PubChem ↗]({row['PubChem']})")
            cols[2].markdown(f"[DrugBank ↗]({row['DrugBank']})")
            cols[3].markdown(f"[DGIdb ↗]({row['DGIdb']})")
            cols[4].markdown(f"[IUPHAR ↗]({row['IUPHAR']})")
            if row.get("Notes") and row["Notes"] != "—":
                st.caption(f"Note: {row['Notes']}")

    # ─ Tab 2 ─────────────────────────────────────────────────────────────
    with tab2:
        plot_df = dff.dropna(subset=["Best IC50 (nM)"]).copy()
        if not plot_df.empty:
            fig = px.scatter(
                plot_df, x="pIC50", y="Score",
                color="Clinical Status", size="# Assays", size_max=30,
                hover_name="Drug / Compound",
                hover_data={"Best IC50 (nM)":":.3f","Mechanism":True,"Selectivity":True,"pIC50":":.2f"},
                title=f"Potency vs Composite Score — {gene} {itype}s",
                labels={"pIC50":"pIC50 (higher = more potent →)","Score":"Composite Score (0–10)"},
                color_discrete_map=STATUS_COLORS,
            )
            fig.update_layout(height=480, font_size=12, legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IC50 data available.")

    # ─ Tab 3 ─────────────────────────────────────────────────────────────
    with tab3:
        ca, cb = st.columns(2)
        with ca:
            fig_hist = px.histogram(dff, x="Score", nbins=15, title="Score distribution",
                                    color_discrete_sequence=["#1565c0"])
            fig_hist.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        with cb:
            vc = dff["Clinical Status"].value_counts()
            fig_pie = px.pie(values=vc.values, names=vc.index, title="Clinical status breakdown",
                             color=vc.index, color_discrete_map=STATUS_COLORS)
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        top_n = min(15, len(dff))
        top   = dff.head(top_n).sort_values("Score")
        fig_bar = px.bar(
            top, x="Score", y="Drug / Compound", orientation="h",
            color="Score", color_continuous_scale=["#ef5350","#42a5f5","#66bb6a"],
            text="Score", title=f"Top {top_n} by composite score",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_bar.update_layout(height=500, showlegend=False,
                              yaxis=dict(categoryorder="total ascending"),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─ Tab 4 ─────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Download results")
        exp_cols = ["Drug / Compound","Score","Clinical Status","Best IC50 (nM)","pIC50",
                    "Selectivity","Mechanism","Indication","Notes","# Assays","# Databases",
                    "ChEMBL ID","ChEMBL","PubChem","DrugBank","DGIdb","IUPHAR"]
        csv = dff[exp_cols].to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv,
                           file_name=f"pharmaquery_{gene}_{itype}.csv", mime="text/csv")
        st.dataframe(dff[exp_cols], use_container_width=True, height=400)


if __name__ == "__main__":
    main()
