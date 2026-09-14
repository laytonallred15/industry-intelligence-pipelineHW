
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from ddgs import DDGS
import requests
import re, os, tempfile

st.set_page_config(page_title="Industry Intelligence Pipeline", page_icon="📊", layout="wide")
st.title("📊 Industry Intelligence Pipeline")
st.caption("Enter a company, upload industry reports, and generate a structured, traceable industry brief.")

st.info(
    "NAICS is looked up automatically. You do NOT need to upload the NAICS manual. "
    "The tool searches for likely company codes, then verifies candidates against the official 2022 U.S. Census NAICS Manual."
)

CENSUS_MANUAL_URL = "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf"

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    return [x.strip() for x in re.split(r'(?<=[.!?;])\s+(?=[A-Z0-9•])', text) if len(x.strip()) > 20]

def web_search(query, max_results=6):
    rows = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                rows.append({
                    "title": clean_text(r.get("title", "")),
                    "url": r.get("href", ""),
                    "snippet": clean_text(r.get("body", ""))
                })
    except Exception:
        pass
    return rows

# ============================================================
# AUTOMATIC NAICS LOOKUP
# ============================================================

def find_candidate_naics_codes(company):
    """
    Discover likely 6-digit NAICS codes from public search results.
    These are only candidate codes; final verification comes from Census.
    """
    queries = [
        f'"{company}" NAICS code',
        f'"{company}" "NAICS"',
        f'"{company}" industry classification NAICS',
        f'"{company}" primary business products'
    ]

    evidence = []
    candidate_counts = {}

    for q in queries:
        for r in web_search(q, max_results=8):
            text = f'{r["title"]} {r["snippet"]}'
            codes = re.findall(r'(?<!\d)(\d{6})(?!\d)', text)
            for code in codes:
                candidate_counts[code] = candidate_counts.get(code, 0) + 1
            evidence.append({**r, "query": q, "codes": codes})

    ranked = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
    return [code for code, _ in ranked[:12]], evidence

@st.cache_resource(show_spinner=False)
def load_census_manual():
    """
    Downloads the official Census 2022 NAICS Manual once per Streamlit session
    and returns extracted page text.
    """
    response = requests.get(CENSUS_MANUAL_URL, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        path = tmp.name

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        pages.append({"page": i, "text": txt})

    try:
        os.remove(path)
    except Exception:
        pass

    return pages

def census_verify_code(code, census_pages):
    """
    Locate a candidate code in the official Census NAICS Manual.
    Returns the best nearby official text window.
    """
    exact = re.compile(rf'(?<!\d){re.escape(code)}(?!\d)')
    matches = []

    for p in census_pages:
        m = exact.search(p["text"])
        if not m:
            continue

        start = max(0, m.start() - 160)
        end = min(len(p["text"]), m.end() + 1100)
        window = clean_text(p["text"][start:end])

        # Reward the main industry description, not only alphabetic-index entries.
        score = 0
        low = window.lower()
        if "this industry comprises establishments primarily engaged" in low:
            score += 12
        if "see industry description for" in low:
            score += 4
        if "cross-references" in low:
            score += 3
        if p["page"] < 650:  # Main manual before alphabetic index
            score += 3

        matches.append({
            "code": code,
            "page": p["page"],
            "text": window,
            "score": score,
            "source_url": CENSUS_MANUAL_URL
        })

    if not matches:
        return None

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[0]

def discover_company_profile(company):
    """
    Uses public web search to capture a short description of the company's
    actual business. This helps rank competing NAICS candidates.
    """
    results = []
    for q in [
        f'"{company}" products company overview',
        f'"{company}" annual report business products',
        f'"{company}" primary business manufactures sells'
    ]:
        results.extend(web_search(q, max_results=5))

    snippets = []
    seen = set()
    for r in results:
        s = r["snippet"]
        if s and s.lower() not in seen:
            seen.add(s.lower())
            snippets.append(s)

    return " ".join(snippets[:6]), results[:10]

def term_overlap_score(profile, census_text):
    """
    Lightweight ranking that compares meaningful words in the detected company
    profile with official Census description text.
    """
    stop = {
        "company","companies","business","products","product","market","industry",
        "sales","sells","selling","design","designs","designed","including",
        "offers","provides","and","the","for","with","from","into","that","this",
        "are","its","their","has","have","was","were","www","com"
    }

    profile_terms = set(
        w for w in re.findall(r"[a-z]{3,}", profile.lower())
        if w not in stop
    )
    census_terms = set(re.findall(r"[a-z]{3,}", census_text.lower()))

    overlap = profile_terms & census_terms
    score = len(overlap)

    # Stronger weight for specific golf/sporting terms if naturally present.
    for phrase in [
        "golf", "golf balls", "golf clubs", "sporting", "athletic",
        "apparel", "footwear", "retail", "wholesale", "manufacturing"
    ]:
        if phrase in profile.lower() and phrase in census_text.lower():
            score += 6

    return score, sorted(overlap)

def automatic_naics_lookup(company):
    profile, profile_sources = discover_company_profile(company)
    candidate_codes, discovery_sources = find_candidate_naics_codes(company)

    if not candidate_codes:
        return {
            "profile": profile,
            "profile_sources": profile_sources,
            "ranked": [],
            "discovery_sources": discovery_sources,
            "error": "No six-digit NAICS candidates were discovered from public search results."
        }

    census_pages = load_census_manual()
    verified = []

    for code in candidate_codes:
        official = census_verify_code(code, census_pages)
        if not official:
            continue

        overlap_score, overlap_terms = term_overlap_score(profile, official["text"])

        # Candidate frequency in search results
        appearances = sum(1 for r in discovery_sources if code in r.get("codes", []))

        official["rank_score"] = overlap_score + appearances * 4
        official["overlap_terms"] = overlap_terms
        official["appearances"] = appearances
        verified.append(official)

    verified.sort(key=lambda x: x["rank_score"], reverse=True)

    return {
        "profile": profile,
        "profile_sources": profile_sources,
        "ranked": verified,
        "discovery_sources": discovery_sources,
        "error": None
    }

# ============================================================
# UPLOADED REPORT READING
# ============================================================

def extract_pdf(file):
    reader = PdfReader(file)
    rows = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        if txt:
            rows.append({"document": file.name, "location": f"p. {i}", "text": txt})
    return rows

def extract_docx(file):
    doc = Document(file)
    txt = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{"document": file.name, "location": "document", "text": txt}] if txt else []

def extract_txt(file):
    data = file.read()
    try:
        txt = data.decode("utf-8")
    except Exception:
        txt = data.decode("latin-1", errors="ignore")
    txt = clean_text(txt)
    return [{"document": file.name, "location": "document", "text": txt}] if txt else []

def extract_csv(file):
    df = pd.read_csv(file)
    txt = clean_text(df.astype(str).to_csv(index=False))
    return [{"document": file.name, "location": "table", "text": txt}] if txt else []

def read_upload(file):
    ext = os.path.splitext(file.name.lower())[1]
    if ext == ".pdf": return extract_pdf(file)
    if ext == ".docx": return extract_docx(file)
    if ext in [".txt", ".md"]: return extract_txt(file)
    if ext == ".csv": return extract_csv(file)
    return []

def chunk_pages(pages, chunk_size=1500, overlap=200):
    chunks = []
    for p in pages:
        txt = p["text"]
        if len(txt) <= chunk_size:
            chunks.append(dict(p))
            continue
        start = 0
        while start < len(txt):
            end = min(len(txt), start + chunk_size)
            chunks.append({
                "document": p["document"],
                "location": p["location"],
                "text": txt[start:end]
            })
            if end == len(txt):
                break
            start = end - overlap
    return chunks

def build_index(chunks):
    if not chunks:
        return None, None
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=40000)
    m = v.fit_transform([c["text"] for c in chunks])
    return v, m

def retrieve(query, chunks, v, m, top_k=12):
    if not chunks or v is None:
        return []
    q = v.transform([query])
    sims = cosine_similarity(q, m)[0]
    idx = sims.argsort()[::-1][:top_k]
    out = []
    for i in idx:
        if sims[i] <= 0:
            continue
        row = dict(chunks[i])
        row["score"] = float(sims[i])
        out.append(row)
    return out

# ============================================================
# ASSIGNMENT SIGNALS
# ============================================================

MONEY = r"(?:\$|US\$)?\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|bn|mn|B|M)?"
PERCENT = r"\b\d+(?:\.\d+)?\s?%"
YEAR = r"\b(?:19|20)\d{2}\b"

SIGNALS = {
    "industry_size": {
        "label":"Industry Size",
        "query":"industry market size total revenue market value sales",
        "keywords":["market size","industry revenue","market value","sales","revenue","industry size"],
        "patterns":[MONEY,YEAR]
    },
    "growth": {
        "label":"Five-Year Growth Trajectory",
        "query":"five year growth historical growth CAGR annualized growth forecast",
        "keywords":["growth","cagr","annualized","five-year","five year","forecast"],
        "patterns":[PERCENT,YEAR]
    },
    "competitors": {
        "label":"Top Competitors and Market Share Estimates",
        "query":"major companies competitors market share leading companies concentration",
        "keywords":["market share","competitor","major companies","leading companies","largest company","concentration"],
        "patterns":[PERCENT]
    },
    "regulation": {
        "label":"Regulatory or Compliance Pressure",
        "query":"regulation regulatory compliance law government safety environmental import tariff",
        "keywords":["regulation","regulatory","compliance","law","government","environmental","tariff","safety"],
        "patterns":[]
    },
    "supply_chain": {
        "label":"Supply Chain Concentration or Fragility",
        "query":"supply chain suppliers raw materials imports sourcing concentration shortage logistics",
        "keywords":["supply chain","supplier","raw material","import","sourcing","shortage","logistics","concentration"],
        "patterns":[PERCENT]
    },
    "customers": {
        "label":"Customer Concentration or Fragmentation",
        "query":"customers buyers market segments customer concentration downstream demand",
        "keywords":["customer","buyer","market segment","downstream","consumer","concentration","demand"],
        "patterns":[PERCENT]
    },
    "trend": {
        "label":"Biggest Trend of the Next Five Years",
        "query":"future outlook next five years trend forecast technology ecommerce participation premiumization sustainability",
        "keywords":["trend","outlook","forecast","future","five years","five-year","driver","expected"],
        "patterns":[PERCENT,YEAR]
    },
    "threat": {
        "label":"Biggest Threat",
        "query":"industry threat risk challenge decline substitutes competition economic downturn cost pressure",
        "keywords":["threat","risk","challenge","decline","substitute","competition","pressure","downturn","cost"],
        "patterns":[]
    }
}

def sentence_score(sentence, keywords, patterns):
    low = sentence.lower()
    score = sum(2 for k in keywords if k.lower() in low)
    for pat in patterns:
        if re.search(pat, sentence, flags=re.I):
            score += 3
    return score

def best_sentences(hits, keywords, patterns, limit=4):
    rows, seen = [], set()
    for h in hits:
        for sent in split_sentences(h["text"]):
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            score = sentence_score(sent, keywords, patterns)
            if score > 0:
                rows.append({
                    "sentence": sent,
                    "score": score,
                    "document": h["document"],
                    "location": h["location"]
                })
    rows.sort(key=lambda x:x["score"], reverse=True)
    return rows[:limit]

def build_report_signals(chunks, v, m):
    signals = {}
    missing = []

    for key,cfg in SIGNALS.items():
        hits = retrieve(cfg["query"], chunks, v, m, 12)
        evidence = best_sentences(hits, cfg["keywords"], cfg["patterns"], 4)
        summary = " ".join(e["sentence"] for e in evidence) if evidence else "Not found in uploaded files."
        sources = []
        for e in evidence:
            s = f'{e["document"]}, {e["location"]}'
            if s not in sources:
                sources.append(s)

        signals[key] = {
            "label": cfg["label"],
            "summary": summary,
            "evidence": evidence,
            "sources": sources
        }

        if not evidence:
            missing.append({
                "signal": cfg["label"],
                "needed": "Additional industry-report or primary-source evidence."
            })

    return signals, missing

# ============================================================
# EXPORT
# ============================================================

def render_markdown(company, naics_code, naics_text, neighbor_code, neighbor_text, signals, missing):
    md = [
        f"# Industry Intelligence Brief: {company}",
        "",
        "## NAICS Classification",
        f"**Selected NAICS code:** {naics_code or 'Not selected'}",
        "",
        f"**Official Census evidence:** {naics_text or 'Not available'}",
        "",
        f"**Neighboring / alternative code:** {neighbor_code or 'Not selected'}",
        "",
        f"**Neighboring-code evidence:** {neighbor_text or 'Not available'}",
        "",
        f"**Official source:** {CENSUS_MANUAL_URL}",
        ""
    ]

    for key in ["industry_size","growth","competitors","regulation","supply_chain","customers","trend","threat"]:
        item = signals[key]
        md += [f"## {item['label']}", item["summary"], ""]
        if item["sources"]:
            md.append("**Sources:**")
            md += [f"- {s}" for s in item["sources"]]
        else:
            md.append("**Source:** Not found in uploaded files.")
        md.append("")

    md.append("## What the Pipeline Could Not Find")
    if missing:
        md += [f"- **{x['signal']}** — Needed: {x['needed']}" for x in missing]
    else:
        md.append("- No required category was completely missing.")

    return "\n".join(md)

def markdown_to_pdf(md_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40,leftMargin=40,topMargin=45,bottomMargin=45)
    styles = getSampleStyleSheet()
    story = []

    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1,8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph("• "+line[2:].replace("**",""), styles["BodyText"]))
        else:
            safe = line.replace("**","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1,4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# USER INTERFACE
# ============================================================

st.subheader("1. Company")
company = st.text_input("Company name", placeholder="Example: Callaway Golf Company")

st.subheader("2. Industry Reports")
uploads = st.file_uploader(
    "Upload IBISWorld, MarketResearch.com, and any other supporting reports",
    type=["pdf","docx","txt","md","csv"],
    accept_multiple_files=True
)
st.caption("You do not need to upload the NAICS manual. The tool handles NAICS separately.")

run = st.button(
    "🚀 Run Industry Intelligence Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not company or not uploads
)

if run:
    # Automatic NAICS
    with st.spinner("Looking up and verifying the NAICS code..."):
        try:
            naics_result = automatic_naics_lookup(company)
        except Exception as e:
            naics_result = {
                "profile":"",
                "profile_sources":[],
                "ranked":[],
                "discovery_sources":[],
                "error":str(e)
            }

    # Industry reports
    pages = []
    for f in uploads:
        try:
            pages.extend(read_upload(f))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    if not pages:
        st.error("No readable text was extracted from the uploaded reports.")
        st.stop()

    chunks = chunk_pages(pages)
    v,m = build_index(chunks)
    signals, missing = build_report_signals(chunks, v, m)

    st.session_state["naics_result"] = naics_result
    st.session_state["signals"] = signals
    st.session_state["missing"] = missing
    st.session_state["company"] = company
    st.success("Pipeline complete.")

if "signals" in st.session_state:
    company = st.session_state["company"]
    naics_result = st.session_state["naics_result"]
    signals = st.session_state["signals"]
    missing = st.session_state["missing"]

    st.divider()
    st.header(f"3. Structured Industry Brief — {company}")

    # ---------------- NAICS ----------------
    st.subheader("Automatic NAICS Classification")

    ranked = naics_result.get("ranked", [])

    if naics_result.get("profile"):
        with st.expander("Detected company business profile"):
            st.write(naics_result["profile"])

    if ranked:
        option_labels = [
            f'{x["code"]} — Census p. {x["page"]} — score {x["rank_score"]}'
            for x in ranked[:6]
        ]
        chosen_label = st.selectbox("Best NAICS candidate", option_labels)
        chosen = ranked[option_labels.index(chosen_label)]

        st.markdown(f'### Selected: {chosen["code"]}')
        st.write(chosen["text"])
        st.markdown(
            f'**Official source:** [2022 U.S. Census NAICS Manual]({chosen["source_url"]}), '
            f'PDF p. {chosen["page"]}'
        )

        alternatives = [x for x in ranked if x["code"] != chosen["code"]]
        if alternatives:
            alt_labels = [""] + [
                f'{x["code"]} — Census p. {x["page"]}'
                for x in alternatives[:5]
            ]
            alt_choice = st.selectbox("Neighboring / alternative code to compare", alt_labels)
            if alt_choice:
                alt_code = alt_choice.split(" — ")[0]
                neighbor = next(x for x in alternatives if x["code"] == alt_code)
                st.write(neighbor["text"])
                st.markdown(
                    f'**Official source:** [2022 U.S. Census NAICS Manual]({neighbor["source_url"]}), '
                    f'PDF p. {neighbor["page"]}'
                )
            else:
                neighbor = None
        else:
            neighbor = None
    else:
        st.warning(
            "The automatic search could not produce a Census-verified six-digit code. "
            "Review the company name or try again."
        )
        chosen = None
        neighbor = None
        if naics_result.get("error"):
            st.caption(naics_result["error"])

    # ---------------- REQUIRED SIGNALS ----------------
    for key in ["industry_size","growth","competitors","regulation","supply_chain","customers","trend","threat"]:
        item = signals[key]
        with st.expander(item["label"], expanded=True):
            st.markdown("**Extracted answer**")
            st.write(item["summary"])

            if item["evidence"]:
                st.markdown("**Traceable evidence**")
                for e in item["evidence"]:
                    st.markdown(
                        f'> {e["sentence"]}\n\n'
                        f'**Source:** {e["document"]}, {e["location"]}'
                    )
            else:
                st.warning("Not found in uploaded files.")

    st.subheader("What the Pipeline Could Not Find")
    if missing:
        for x in missing:
            st.write(f'• **{x["signal"]}** — {x["needed"]}')
    else:
        st.write("All required categories had supporting evidence.")

    # ---------------- EXPORT ----------------
    st.divider()
    st.header("4. Export")

    chosen_code = chosen["code"] if chosen else ""
    chosen_text = chosen["text"] if chosen else ""
    neighbor_code = neighbor["code"] if neighbor else ""
    neighbor_text = neighbor["text"] if neighbor else ""

    md = render_markdown(
        company,
        chosen_code,
        chosen_text,
        neighbor_code,
        neighbor_text,
        signals,
        missing
    )

    stem = re.sub(r'[^A-Za-z0-9_-]+','_',company)

    st.download_button(
        "⬇️ Download Markdown Brief",
        md,
        file_name=f"{stem}_industry_brief.md",
        mime="text/markdown",
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download PDF Brief",
        markdown_to_pdf(md),
        file_name=f"{stem}_industry_brief.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    rows = []
    for _, item in signals.items():
        for e in item["evidence"]:
            rows.append({
                "Signal": item["label"],
                "Evidence": e["sentence"],
                "Document": e["document"],
                "Location": e["location"]
            })

    if rows:
        df = pd.DataFrame(rows)
        st.download_button(
            "⬇️ Download Source Log (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{stem}_source_log.csv",
            mime="text/csv",
            use_container_width=True
        )
