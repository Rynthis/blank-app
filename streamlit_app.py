""" PharmaQuery Pro
Comprehensive drug-gene interaction discovery.
Queries ChEMBL · DGIdb · Open Targets · IUPHAR · PubChem in parallel,
then synthesizes and scores results with ChatGPT or Gemini + web search.
Run: streamlit run app.py
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import math, json, re, time, unicodedata
from typing import Optional

# NEW: choose your provider
from openai import OpenAI
import google.generativeai as genai

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaQuery Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
div[data-testid="metric-container"] { background:#f8f9fa; border-radius:8px; padding:.5rem 1rem; border:1px solid #e0e0e0; }
.stDataFrame { border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
DGIDB = "https://dgidb.org/api/graphql"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
IUPHAR = "https://www.guidetopharmacology.org/services"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 20

ITYPE_KEYWORDS = {
    "inhibitor": ["inhibitor", "antagonist", "blocker", "inverse agonist", "negative modulator", "suppressor", "inhibition", "channel blocker"],
    "agonist": ["agonist", "activator", "positive modulator", "partial agonist", "full agonist", "superagonist", "potentiator", "activation"],
    "modulator": [],
}
STATUS_ORDER = ["FDA Approved","Phase 4","Phase 3","Phase 2","Phase 1", "Preclinical","Research Tool","Unknown"]
STATUS_COLORS = {
    "FDA Approved": "#2e7d32", "Phase 4": "#388e3c", "Phase 3": "#1565c0",
    "Phase 2": "#283593", "Phase 1": "#e65100", "Preclinical": "#880e4f",
    "Research Tool": "#616161", "Unknown": "#9e9e9e",
}

# ══════════════════════════════════════════════════════════════════════════
# DATABASE QUERY FUNCTIONS (unchanged)
# ══════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def query_chembl(gene: str, itype: str) -> dict:
    out = {"target": None, "activities": [], "molecules": {}}
    try:
        r = requests.get(f"{CHEMBL}/target/search.json", params={"q": gene, "limit": 10}, timeout=TIMEOUT)
        targets = r.json().get("targets", [])
        target = (next((t for t in targets if t["target_type"] == "SINGLE PROTEIN" and "Homo sapiens" in t.get("organism", "")), None) or next((t for t in targets if t["target_type"] == "SINGLE PROTEIN"), None) or (targets[0] if targets else None))
        if not target: return out
        cid = target["target_chembl_id"]
        out["target"] = {"name": target.get("pref_name"), "chembl_id": cid, "organism": target.get("organism")}
        r2 = requests.get(f"{CHEMBL}/activity.json", params={"target_chembl_id": cid, "standard_type__in": "IC50,Ki,EC50,Kd,pIC50", "standard_relation__in": "=,<,<=", "limit": 100, "order_by": "standard_value",}, timeout=TIMEOUT)
        acts = [a for a in r2.json().get("activities", []) if a.get("standard_value")]
        out["activities"] = [{"molecule": a.get("molecule_pref_name") or a.get("molecule_chembl_id"), "chembl_id": a.get("molecule_chembl_id"), "meas_type": a.get("standard_type"), "value": float(a["standard_value"]), "units": a.get("standard_units", "nM"),} for a in acts]
        mol_ids = list({a["chembl_id"] for a in out["activities"] if a.get("chembl_id")})[:25]
        def _fetch_mol(mid):
            try:
                mr = requests.get(f"{CHEMBL}/molecule/{mid}.json", timeout=TIMEOUT)
                mol = mr.json()
                return mid, {"name": mol.get("pref_name"), "max_phase": mol.get("max_phase", 0)}
            except: return mid, {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for mid, info in ex.map(_fetch_mol, mol_ids): out["molecules"][mid] = info
    except Exception: pass
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def query_dgidb(gene: str, itype: str) -> list:
    try:
        q = """query($names:[String!]!){ genes(names:$names){nodes{name interactions{ drug{name approved conceptId} interactionScore interactionTypes{type directionality} sources{fullName} }}} }"""
        r = requests.post(DGIDB, json={"query": q, "variables": {"names": [gene.upper()]}}, timeout=TIMEOUT)
        nodes = r.json().get("data", {}).get("genes", {}).get("nodes", [])
        ixns = nodes[0].get("interactions", []) if nodes else []
        filters = ITYPE_KEYWORDS.get(itype, [])
        results = []
        for i in ixns:
            type_strs = [((t.get("directionality") or "") + " " + (t.get("type") or "")).strip().lower() for t in (i.get("interactionTypes") or [])]
            combined = " ".join(type_strs)
            if filters and not any(f in combined for f in filters): continue
            results.append({"drug": i.get("drug", {}).get("name"), "approved": i.get("drug", {}).get("approved"), "dgi_score": i.get("interactionScore") or 0, "types": ", ".join(type_strs), "n_sources": len(i.get("sources") or []), "sources": [s.get("fullName") for s in (i.get("sources") or [])],})
        return results
    except: return []

@st.cache_data(ttl=3600, show_spinner=False)
def query_open_targets(gene: str) -> dict:
    out = {"target_id": None, "drugs": []}
    try:
        sq = """query($q:String!){search(queryString:$q,entityNames:["target"], page:{index:0,size:3}){hits{id name}}}"""
        r = requests.post(OT, json={"query": sq, "variables": {"q": gene}}, timeout=TIMEOUT)
        hits = r.json().get("data", {}).get("search", {}).get("hits", [])
        if not hits: return out
        tid = hits[0]["id"]; out["target_id"] = tid
        dq = """query($id:String!){target(ensemblId:$id){ approvedName knownDrugs{rows{ drug{id name maximumClinicalTrialPhase isApproved} mechanismOfAction actionType phase status disease{name} }} }}"""
        r2 = requests.post(OT, json={"query": dq, "variables": {"id": tid}}, timeout=TIMEOUT)
        rows = (r2.json().get("data", {}).get("target", {}).get("knownDrugs", {}).get("rows", []))
        for row in rows:
            drug = row.get("drug", {})
            out["drugs"].append({"name": drug.get("name"), "ot_id": drug.get("id"), "max_phase": drug.get("maximumClinicalTrialPhase"), "approved": drug.get("isApproved"), "mechanism": row.get("mechanismOfAction"), "action_type": row.get("actionType"), "indication": (row.get("disease") or {}).get("name"),})
    except: pass
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def query_iuphar(gene: str, itype: str) -> list:
    results = []
    try:
        r = requests.get(f"{IUPHAR}/targets", params={"geneSymbol": gene.upper(), "species": "Human"}, timeout=TIMEOUT)
        targets = r.json() if r.ok and r.content else []
        if not targets:
            r2 = requests.get(f"{IUPHAR}/targets/search/{requests.utils.quote(gene)}", timeout=TIMEOUT)
            targets = r2.json() if r2.ok and r2.content else []
        if not targets: return []
        tid = targets[0].get("targetId")
        r3 = requests.get(f"{IUPHAR}/interactions/target/{tid}", timeout=TIMEOUT)
        ixns = r3.json() if r3.ok and r3.content else []
        filters = ITYPE_KEYWORDS.get(itype, [])
        for i in ixns:
            itype_str = ((i.get("type") or "") + " " + (i.get("action") or "")).lower()
            if filters and not any(f in itype_str for f in filters): continue
            aff_type = i.get("affinityType") or ""; aff_val = i.get("affinity"); ic50_nm = None
            if aff_val and aff_type.startswith("p"): ic50_nm = (10 ** -float(aff_val)) * 1e9
            lid = i.get("ligandId")
            results.append({"drug": i.get("ligandName"), "iuphar_id": lid, "action": i.get("action") or i.get("type"), "aff_type": aff_type, "aff_value": aff_val, "ic50_nm": ic50_nm, "approved": i.get("ligandApproved"), "url": f"https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId={lid}" if lid else None,})
    except: pass
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
            r3 = requests.get(f"{PUBCHEM}/assay/aid/{aid}/cids/JSON", params={"cids_type": "active", "list_return": 10}, timeout=TIMEOUT)
            cids = r3.json().get("AssayLink", {}).get("CID", [])[:10]
            for cid in cids: results.append({"cid": cid, "assay": name, "aid": aid})
    except: pass
    return results

# ══════════════════════════════════════════════════════════════════════════
# SCORING (unchanged)
# ══════════════════════════════════════════════════════════════════════════
def pic50_from_nm(ic50_nm: Optional[float]) -> Optional[float]:
    if not ic50_nm or ic50_nm <= 0: return None
    return -math.log10(ic50_nm * 1e-9)

def _potency_pts(ic50_nm: Optional[float]) -> float:
    p = pic50_from_nm(ic50_nm)
    if p is None: return 4.0
    if p >= 10: return 30
    elif p >= 9: return 27
    elif p >= 8: return 23
    elif p >= 7: return 18
    elif p >= 6: return 12
    elif p >= 5: return 7
    else: return 3

def _clinical_pts(status: str, max_phase: Optional[int] = None) -> float:
    if max_phase is not None:
        mp = int(max_phase) if max_phase else 0
        return {4: 30, 3: 25, 2: 20, 1: 15, 0: 6}.get(mp, 5)
    if not status: return 4.0
    s = status.lower()
    for keys, pts in [(["fda approved","approved","marketed"], 30), (["phase 4","phase iv"], 29), (["phase 3","phase iii"], 25), (["phase 2/3"], 22), (["phase 2","phase ii"], 20), (["phase 1/2"], 17), (["phase 1","phase i"], 15), (["ind filed"], 12), (["preclinical"], 8), (["research tool","research"], 4), (["withdrawn","discontinued","failed"], 2),]:
        if any(k in s for k in keys): return float(pts)
    return 4.0

def _evidence_pts(n_assays: int, n_databases: int) -> float:
    assay_pts = min(10.0, math.log1p(n_assays) * 2.3)
    db_pts = min(10.0, n_databases * 2.0)
    return assay_pts + db_pts

def _selectivity_pts(selectivity: str) -> float:
    return {"high": 10, "moderate": 6, "low": 2}.get((selectivity or "").lower(), 4)

def compute_score(drug: dict) -> float:
    pot = _potency_pts(drug.get("best_ic50_nm"))
    clin = _clinical_pts(drug.get("clinical_status",""), drug.get("max_phase"))
    evid = _evidence_pts(int(drug.get("n_assays") or 1), int(drug.get("n_databases") or 1))
    sel = _selectivity_pts(drug.get("selectivity",""))
    ai = float(drug.get("ai_bonus") or 5) * 1.0
    raw = pot*0.30 + clin*0.30 + evid*0.20 + sel*0.10 + ai*0.10
    return round(min(10.0, raw), 2)

# ══════════════════════════════════════════════════════════════════════════
# CONSOLIDATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════
def _norm(name: str) -> str:
    if not name: return ""
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", n.lower())

def consolidate(chembl: dict, dgidb: list, ot: dict, iuphar: list) -> dict:
    drugs: dict[str, dict] = {}
    def upsert(name, data: dict, source: str):
        if not name: return
        key = _norm(name)
        if key not in drugs:
            drugs[key] = {"name": name, "best_ic50_nm": None, "max_phase": None, "clinical_status": "", "selectivity": "", "mechanism": "", "indication": "", "chembl_id": None, "iuphar_id": None, "ot_id": None, "ai_bonus": 5, "n_assays": 0, "n_databases": 0, "_sources": set(),}
        d = drugs[key]
        for k, v in data.items():
            if k.startswith("_"): continue
            if v not in (None, ""):
                if k == "best_ic50_nm" and d.get(k) is not None: d[k] = min(d[k], float(v))
                elif k == "max_phase" and d.get(k) is not None: d[k] = max(int(d[k]), int(v) if v else 0)
                elif d.get(k) in (None, ""): d[k] = v
        d["n_assays"] += int(data.get("_n_assays", 1))
        d["_sources"].add(source)
        d["n_databases"] = len(d["_sources"])
    # ChEMBL, DGIdb, OT, IUPHAR merging – same as your original (omitted for brevity, keep your full block here)
    #... paste your original consolidate body here...
    # For space, I assume you keep the exact code you posted
    return drugs # placeholder – replace with your full consolidate implementation

# ══════════════════════════════════════════
# AI SYNTHESIS – NOW ChatGPT OR GEMINI
# ══════════════════════════════════════════════════════════════════════════
def ai_synthesize(gene: str, itype: str, drugs_raw: dict, api_key: str, provider: str) -> list:
    summary = []
    for d in list(drugs_raw.values())[:35]:
        summary.append({"name": d["name"], "ic50_nm": round(d["best_ic50_nm"], 3) if d.get("best_ic50_nm") else None, "status": d.get("clinical_status"), "max_phase": d.get("max_phase"), "n_dbs": d.get("n_databases"), "mechanism": d.get("mechanism"), "chembl_id": d.get("chembl_id"),})

    system = f"""You are a world-class medicinal chemist. Given raw database results for target "{gene}", produce a definitive JSON array of the top 12-18 {itype}s. Use web search to verify FDA status, clinical phases, add missing compounds, and confirm IC50/Ki. Return ONLY valid JSON array with keys: name, clinical_status, best_ic50_nm, selectivity, mechanism, indication, chembl_id, notes, n_assays, n_databases, ai_bonus. ai_bonus is 0-10 for expert importance."""

    user = f"""Target: {gene} | Type: {itype}
Database candidates: {json.dumps(summary, indent=2)}
Search the web and return expert-curated JSON array."""

    try:
        if provider == "ChatGPT (OpenAI)":
            client = OpenAI(api_key=api_key)
            # gpt-4o with web search
            resp = client.responses.create(
                model="gpt-4o",
                instructions=system,
                input=user,
                tools=[{"type": "web_search_preview"}],
                max_output_tokens=5000,
                temperature=0.2,
            )
            text = resp.output_text
        else: # Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-pro")
            # Enable Google Search grounding
            resp = model.generate_content(
                [system, user],
                tools=[{"type": "google_search_retrieval"}],
                generation_config={"temperature": 0.2, "max_output_tokens": 5000}
            )
            text = resp.text

        m = re.search(r"\[[\s\S]*\]", text)
        if m: return json.loads(m.group(0))
    except Exception as e:
        st.error(f"AI synthesis error: {e}")
    return []

# ══════════════════════════════════════════════════════════════════════════
# BUILD DF and UI – same as yours, just update sidebar
# ══════════════════════════════════════════
def build_df(drugs_raw: dict, ai_drugs: list) -> pd.DataFrame:
    # keep your original build_df implementation
    pass # replace with your full function

def main():
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox("AI Provider", ["ChatGPT (OpenAI)", "Gemini (Google)"])
        api_key = st.text_input(f"{provider} API Key", type="password", placeholder="sk-... or AIza...")
        st.divider()
        #... keep rest of your sidebar...
        st.caption(f"AI model: {provider} + live web search")

    st.title("🔬 PharmaQuery Pro")
    st.markdown("Comprehensive drug-gene interaction discovery · 5 databases · AI scoring · live web synthesis")

    c1, c2, c3 = st.columns([3, 1.5, 1])
    with c1: gene = st.text_input("Gene / target", label_visibility="collapsed", placeholder="EGFR, mTOR, BCR-ABL")
    with c2: itype = st.selectbox("Type", ["inhibitor", "agonist", "modulator"], label_visibility="collapsed")
    with c3: go_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if not go_btn or not gene.strip():
        st.info("Enter a gene and click Search.")
        return
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
        return

    #... keep your parallel queries, consolidate, etc....
    # when calling AI:
    # ai_drugs = ai_synthesize(gene, itype, drugs_raw, api_key, provider)
    # prog.progress(5.5/6, f"{provider}: synthesizing + web search...")
    #... rest unchanged...

if __name__ == "__main__": main()
