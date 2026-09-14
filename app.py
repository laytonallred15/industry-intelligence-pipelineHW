
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
import re, os

st.set_page_config(page_title="Industry Intelligence Pipeline", page_icon="📊", layout="wide")
st.title("📊 Industry Intelligence Pipeline")
st.caption("Upload company/industry evidence and generate a structured, traceable industry brief.")

st.info(
    "This version uses stricter NAICS logic. It prioritizes Census/NAICS files, matches the company's "
    "primary products to official industry descriptions, and requires the user to confirm the final code."
)

# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def split_sentences(text):
    text = clean_text(text)
    if not text:
        return []
    return [x.strip() for x in re.split(r'(?<=[.!?;])\s+(?=[A-Z0-9•])', text) if len(x.strip()) > 20]

def extract_pdf(file):
    reader = PdfReader(file)
    rows = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        if txt:
            rows.append({
                "document": file.name,
                "location": f"p. {i}",
                "page": i,
                "text": txt,
                "is_naics_source": any(k in file.name.lower() for k in ["naics", "census"])
            })
    return rows

def extract_docx(file):
    doc = Document(file)
    txt = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{
        "document": file.name, "location": "document", "page": None, "text": txt,
        "is_naics_source": any(k in file.name.lower() for k in ["naics", "census"])
    }] if txt else []

def extract_txt(file):
    data = file.read()
    try:
        txt = data.decode("utf-8")
    except Exception:
        txt = data.decode("latin-1", errors="ignore")
    txt = clean_text(txt)
    return [{
        "document": file.name, "location": "document", "page": None, "text": txt,
        "is_naics_source": any(k in file.name.lower() for k in ["naics", "census"])
    }] if txt else []

def extract_csv(file):
    df = pd.read_csv(file)
    txt = clean_text(df.astype(str).to_csv(index=False))
    return [{
        "document": file.name, "location": "table", "page": None, "text": txt,
        "is_naics_source": any(k in file.name.lower() for k in ["naics", "census"])
    }] if txt else []

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
            piece = dict(p)
            piece["text"] = txt[start:end]
            chunks.append(piece)
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

def retrieve(query, chunks, v, m, top_k=12, only_naics=False):
    if not chunks or v is None:
        return []
    pool_idx = [i for i,c in enumerate(chunks) if (c.get("is_naics_source") if only_naics else True)]
    if not pool_idx:
        return []
    q = v.transform([query])
    sims = cosine_similarity(q, m)[0]
    ranked = sorted(pool_idx, key=lambda i: sims[i], reverse=True)[:top_k]
    out = []
    for i in ranked:
        if sims[i] <= 0: 
            continue
        row = dict(chunks[i])
        row["score"] = float(sims[i])
        out.append(row)
    return out

# ============================================================
# IMPROVED NAICS LOGIC
# ============================================================

NAICS_RE = re.compile(r"\b(\d{6})\b")

def normalize_terms(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    stop = {
        "company","business","designs","sells","sale","sales","products","product",
        "manufactures","manufacturing","manufacturer","related","and","the","of",
        "a","an","to","for","with","in","on"
    }
    return [w for w in words if len(w) > 2 and w not in stop]

def extract_naics_entries(pages):
    """
    Pull official-looking NAICS entries from Census/NAICS files.
    Associates a six-digit code with nearby description text.
    """
    source_pages = [p for p in pages if p.get("is_naics_source")]
    if not source_pages:
        source_pages = pages

    entries = []
    seen = set()

    for p in source_pages:
        text = p["text"]
        # Capture each code plus a window following it.
        for match in NAICS_RE.finditer(text):
            code = match.group(1)
            start = max(0, match.start() - 120)
            end = min(len(text), match.end() + 850)
            window = clean_text(text[start:end])

            # Avoid repeated index noise by requiring at least some alphabetic description.
            if len(re.findall(r"[A-Za-z]{4,}", window)) < 5:
                continue

            key = (code, p["document"], p["location"], window[:180])
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                "code": code,
                "text": window,
                "document": p["document"],
                "location": p["location"]
            })
    return entries

def score_naics_entry(entry, company, description):
    text = entry["text"].lower()
    desc_terms = normalize_terms(description)
    company_terms = normalize_terms(company)

    score = 0.0

    # Strong preference for official description language.
    if "this industry comprises establishments primarily engaged" in text:
        score += 8
    if "see industry description for" in text:
        score += 2

    # Match the user's stated primary activities/products.
    for term in desc_terms:
        if term in text:
            score += 4

    # Company-name match can help if a supporting uploaded file explicitly names a code.
    for term in company_terms:
        if term in text:
            score += 1

    # Golf-specific keywords are not hard-coded to a NAICS code;
    # they simply reward direct evidence when the company description uses them.
    phrase_weights = {
        "golf": 8,
        "golf ball": 12,
        "golf balls": 12,
        "golf club": 12,
        "golf clubs": 12,
        "sporting and athletic goods": 10,
        "sporting goods": 7,
        "athletic goods": 7,
        "apparel": 3,
        "footwear": 3,
        "retail": 2,
        "wholesale": 2
    }
    desc_lower = description.lower()
    for phrase, weight in phrase_weights.items():
        if phrase in desc_lower and phrase in text:
            score += weight

    # Prefer entries from a file clearly identified as Census / NAICS.
    if any(k in entry["document"].lower() for k in ["naics", "census"]):
        score += 8

    return score

def get_naics_candidates(company, description, pages, limit=8):
    entries = extract_naics_entries(pages)
    for e in entries:
        e["score"] = score_naics_entry(e, company, description)

    # Collapse duplicate codes and keep the best evidence window for each.
    best = {}
    for e in entries:
        if e["code"] not in best or e["score"] > best[e["code"]]["score"]:
            best[e["code"]] = e

    ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)

    # Require meaningful evidence instead of returning random codes.
    ranked = [r for r in ranked if r["score"] >= 8]
    return ranked[:limit]

# ============================================================
# REQUIRED SIGNALS
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
        "query":"future outlook five years trend forecast technology ecommerce participation premiumization sustainability",
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
    s = sentence.lower()
    score = sum(2 for k in keywords if k.lower() in s)
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
                    "sentence":sent,
                    "score":score,
                    "document":h["document"],
                    "location":h["location"]
                })
    rows.sort(key=lambda x:x["score"], reverse=True)
    return rows[:limit]

def unique_sources(evidence):
    out, seen = [], set()
    for e in evidence:
        s = f'{e["document"]}, {e["location"]}'
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def build_brief(company, description, pages, chunks, v, m):
    brief = {
        "company":company,
        "naics_candidates":get_naics_candidates(company, description, pages),
        "signals":{},
        "missing":[]
    }

    for key,cfg in SIGNALS.items():
        hits = retrieve(cfg["query"], chunks, v, m, 12)
        evidence = best_sentences(hits, cfg["keywords"], cfg["patterns"])
        summary = " ".join(x["sentence"] for x in evidence) if evidence else "Not found in uploaded files."
        brief["signals"][key] = {
            "label":cfg["label"],
            "summary":summary,
            "evidence":evidence,
            "sources":unique_sources(evidence)
        }
        if not evidence:
            brief["missing"].append({
                "signal":cfg["label"],
                "needed":"A source that directly addresses this signal."
            })

    if not brief["naics_candidates"]:
        brief["missing"].append({
            "signal":"NAICS Code",
            "needed":"Upload the relevant U.S. Census NAICS manual pages and provide a short description of the company's primary products/activities."
        })

    return brief

# ============================================================
# EXPORT
# ============================================================

def render_markdown(brief, selected_code, selected_title, neighbor_code, reason, neighbor_reason):
    md = [
        f"# Industry Intelligence Brief: {brief['company']}",
        "",
        "## NAICS Classification",
        f"**Selected NAICS:** {selected_code or 'Not selected'} {selected_title or ''}",
        "",
        f"**Why this code:** {reason or 'Not completed'}",
        "",
        f"**Neighboring code considered:** {neighbor_code or 'Not selected'}",
        "",
        f"**Why not the neighboring code:** {neighbor_reason or 'Not completed'}",
        "",
        "**Verification:** Final selection should be checked against the official U.S. Census NAICS manual.",
        ""
    ]

    for key in ["industry_size","growth","competitors","regulation","supply_chain","customers","trend","threat"]:
        item = brief["signals"][key]
        md += [f"## {item['label']}", item["summary"], ""]
        if item["sources"]:
            md.append("**Sources:**")
            md += [f"- {s}" for s in item["sources"]]
        else:
            md.append("**Source:** Not found in uploaded files.")
        md.append("")

    md.append("## What the Pipeline Could Not Find")
    if brief["missing"]:
        md += [f"- **{m['signal']}** — Needed: {m['needed']}" for m in brief["missing"]]
    else:
        md.append("- No required category was completely missing.")
    return "\n".join(md)

def markdown_to_pdf(md_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()
    story = []
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1,8))
        elif line.startswith("# "):
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
# UI
# ============================================================

st.subheader("1. Company")
company = st.text_input("Company name", placeholder="Example: Callaway Golf Company")
description = st.text_area(
    "Primary products / activities",
    placeholder="Example: Manufactures and sells golf clubs, golf balls, golf bags, and other golf equipment.",
    help="This is used to match the company to the correct official NAICS description."
)

st.subheader("2. Upload Sources")
uploads = st.file_uploader(
    "Upload IBISWorld, MarketResearch.com, Census/NAICS, and other supporting files",
    type=["pdf","docx","txt","md","csv"],
    accept_multiple_files=True
)

st.warning(
    "For the NAICS step, upload the 2022 Census NAICS manual (or the relevant pages). "
    "The tool will prefer files with 'NAICS' or 'Census' in the filename."
)

run = st.button(
    "🚀 Analyze Uploaded Files",
    type="primary",
    use_container_width=True,
    disabled=not company or not description or not uploads
)

if run:
    pages = []
    for f in uploads:
        try:
            pages.extend(read_upload(f))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    if not pages:
        st.error("No readable text was extracted.")
        st.stop()

    chunks = chunk_pages(pages)
    v,m = build_index(chunks)

    with st.spinner("Analyzing evidence and ranking NAICS candidates..."):
        brief = build_brief(company, description, pages, chunks, v, m)

    st.session_state["brief"] = brief
    st.success("Analysis complete.")

if "brief" in st.session_state:
    brief = st.session_state["brief"]

    st.divider()
    st.header(f"3. Structured Industry Brief — {brief['company']}")

    st.subheader("NAICS Classification")
    candidates = brief["naics_candidates"]

    selected_code = ""
    selected_title = ""
    neighbor_code = ""

    if candidates:
        st.write("The candidates below are ranked from the uploaded Census/NAICS evidence.")

        labels = []
        for c in candidates:
            snippet = c["text"][:180]
            labels.append(f'{c["code"]} — {snippet}... [{c["document"]}, {c["location"]}]')

        selected_label = st.selectbox("Select the best NAICS code", labels)
        selected = candidates[labels.index(selected_label)]
        selected_code = selected["code"]

        st.markdown("**Evidence used by the tool:**")
        st.write(selected["text"])
        st.markdown(f'**Source:** {selected["document"]}, {selected["location"]}')

        neighbor_choices = [""] + [c["code"] for c in candidates if c["code"] != selected_code]
        neighbor_code = st.selectbox("Select a neighboring / alternative code", neighbor_choices)
    else:
        st.warning("No defensible six-digit NAICS candidate was found.")
        selected_code = st.text_input("Enter the verified NAICS code manually")
        neighbor_code = st.text_input("Enter a neighboring code manually")

    selected_title = st.text_input("Official NAICS title", placeholder="Example: Sporting and Athletic Goods Manufacturing")
    reason = st.text_area("Why this is the correct NAICS code")
    neighbor_reason = st.text_area("Why the neighboring code is less appropriate")

    for key in ["industry_size","growth","competitors","regulation","supply_chain","customers","trend","threat"]:
        item = brief["signals"][key]
        with st.expander(item["label"], expanded=True):
            st.write(item["summary"])
            if item["evidence"]:
                st.markdown("**Traceable evidence**")
                for e in item["evidence"]:
                    st.markdown(f'> {e["sentence"]}\n\n**Source:** {e["document"]}, {e["location"]}')
            else:
                st.warning("Not found in uploaded files.")

    st.subheader("What the Pipeline Could Not Find")
    if brief["missing"]:
        for m in brief["missing"]:
            st.write(f'• **{m["signal"]}** — {m["needed"]}')
    else:
        st.write("All required categories had supporting evidence.")

    st.divider()
    st.header("4. Export")
    md = render_markdown(brief, selected_code, selected_title, neighbor_code, reason, neighbor_reason)
    stem = re.sub(r'[^A-Za-z0-9_-]+','_',brief["company"])

    st.download_button("⬇️ Download Markdown Brief", md, file_name=f"{stem}_industry_brief.md", mime="text/markdown", use_container_width=True)
    st.download_button("⬇️ Download PDF Brief", markdown_to_pdf(md), file_name=f"{stem}_industry_brief.pdf", mime="application/pdf", use_container_width=True)
