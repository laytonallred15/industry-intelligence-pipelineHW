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
st.caption("Upload company and industry evidence, then generate a structured, traceable industry brief.")
st.info("This tool analyzes only the files you upload. Every extracted signal is tied to a document and page/section. Missing evidence is flagged instead of invented.")

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def split_sentences(text):
    text = clean_text(text)
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text) if len(p.strip()) > 25]

def extract_pdf(file):
    reader = PdfReader(file)
    rows = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        if txt:
            rows.append({"document": file.name, "location": f"p. {i}", "page": i, "text": txt})
    return rows

def extract_docx(file):
    doc = Document(file)
    txt = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{"document": file.name, "location": "document", "page": None, "text": txt}] if txt else []

def extract_txt(file):
    data = file.read()
    try:
        txt = data.decode("utf-8")
    except Exception:
        txt = data.decode("latin-1", errors="ignore")
    txt = clean_text(txt)
    return [{"document": file.name, "location": "document", "page": None, "text": txt}] if txt else []

def extract_csv(file):
    df = pd.read_csv(file)
    txt = clean_text(df.astype(str).to_csv(index=False))
    return [{"document": file.name, "location": "table", "page": None, "text": txt}] if txt else []

def read_upload(file):
    ext = os.path.splitext(file.name.lower())[1]
    if ext == ".pdf": return extract_pdf(file)
    if ext == ".docx": return extract_docx(file)
    if ext in [".txt", ".md"]: return extract_txt(file)
    if ext == ".csv": return extract_csv(file)
    return []

def chunk_pages(pages, chunk_size=1400, overlap=200):
    chunks = []
    for p in pages:
        text = p["text"]
        if len(text) <= chunk_size:
            chunks.append(dict(p))
        else:
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                chunks.append({"document": p["document"], "location": p["location"], "page": p["page"], "text": text[start:end]})
                if end == len(text): break
                start = max(start + 1, end - overlap)
    return chunks

def build_index(chunks):
    corpus = [c["text"] for c in chunks]
    if not corpus: return None, None
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=30000)
    m = v.fit_transform(corpus)
    return v, m

def retrieve(query, chunks, vectorizer, matrix, top_k=10):
    if not chunks or vectorizer is None: return []
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, matrix)[0]
    idx = sims.argsort()[::-1][:top_k]
    out = []
    for i in idx:
        if sims[i] <= 0: continue
        row = dict(chunks[i]); row["score"] = float(sims[i]); out.append(row)
    return out

MONEY = r"(?:\$|US\$)?\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|bn|mn|B|M)?"
PERCENT = r"\b\d+(?:\.\d+)?\s?%"
YEAR = r"\b(?:19|20)\d{2}\b"
NAICS_CODE = r"\b\d{6}\b"

def sentence_score(sentence, keywords, patterns):
    s = sentence.lower()
    score = sum(2 for k in keywords if k.lower() in s)
    for pat in patterns:
        if re.search(pat, sentence, flags=re.I): score += 3
    return score

def best_sentences(hits, keywords, patterns, limit=4):
    candidates, seen = [], set()
    for h in hits:
        for sent in split_sentences(h["text"]):
            key = sent.lower()
            if key in seen: continue
            seen.add(key)
            score = sentence_score(sent, keywords, patterns)
            if score > 0:
                candidates.append({"sentence": sent, "score": score, "document": h["document"], "location": h["location"], "page": h["page"]})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]

def citation(e):
    return f'{e["document"]}, {e["location"]}'

def unique_citations(evidence):
    out, seen = [], set()
    for e in evidence:
        c = citation(e)
        if c not in seen:
            seen.add(c); out.append(c)
    return out

SIGNALS = {
    "industry_size": {"label":"Industry Size","query":"industry market size total revenue market value sales","keywords":["market size","industry revenue","market value","sales","revenue","industry size"],"patterns":[MONEY,YEAR]},
    "growth": {"label":"Five-Year Growth Trajectory","query":"five year growth historical growth CAGR annualized growth forecast","keywords":["growth","cagr","annualized","five-year","five year","forecast"],"patterns":[PERCENT,YEAR]},
    "competitors": {"label":"Top Competitors and Market Share Estimates","query":"major companies competitors market share leading companies concentration","keywords":["market share","competitor","major companies","leading companies","largest company","concentration"],"patterns":[PERCENT]},
    "regulation": {"label":"Regulatory or Compliance Pressure","query":"regulation regulatory compliance law government safety environmental import tariff","keywords":["regulation","regulatory","compliance","law","government","environmental","tariff","safety"],"patterns":[]},
    "supply_chain": {"label":"Supply Chain Concentration or Fragility","query":"supply chain suppliers raw materials imports sourcing concentration shortage logistics","keywords":["supply chain","supplier","raw material","import","sourcing","shortage","logistics","concentration"],"patterns":[PERCENT]},
    "customers": {"label":"Customer Concentration or Fragmentation","query":"customers buyers market segments customer concentration downstream demand","keywords":["customer","buyer","market segment","downstream","consumer","concentration","demand"],"patterns":[PERCENT]},
    "trend": {"label":"Biggest Trend of the Next Five Years","query":"future outlook five years trend forecast technology ecommerce participation premiumization sustainability","keywords":["trend","outlook","forecast","future","five years","five-year","driver","expected"],"patterns":[PERCENT,YEAR]},
    "threat": {"label":"Biggest Threat","query":"industry threat risk challenge decline substitutes competition economic downturn cost pressure","keywords":["threat","risk","challenge","decline","substitute","competition","pressure","downturn","cost"],"patterns":[]}
}

def extract_naics_candidates(company, desc, chunks, v, m):
    hits = retrieve(f"{company} {desc} NAICS classification primary business industry description establishments primarily engaged", chunks, v, m, 15)
    candidates, seen = [], set()
    for h in hits:
        for sent in split_sentences(h["text"]):
            for code in re.findall(NAICS_CODE, sent):
                key = (code, sent[:180])
                if key in seen: continue
                seen.add(key)
                candidates.append({"code":code,"description":sent,"document":h["document"],"location":h["location"],"score":h["score"]})
    candidates.sort(key=lambda x:x["score"], reverse=True)
    return candidates[:8]

def build_brief(company, desc, chunks, v, m):
    result = {"company":company,"naics_candidates":extract_naics_candidates(company, desc, chunks, v, m),"signals":{},"missing":[]}
    for key,cfg in SIGNALS.items():
        hits = retrieve(cfg["query"], chunks, v, m, 12)
        evidence = best_sentences(hits, cfg["keywords"], cfg["patterns"], 4)
        summary = " ".join(e["sentence"] for e in evidence) if evidence else "Not found in uploaded files."
        result["signals"][key] = {"label":cfg["label"],"summary":summary,"evidence":evidence,"sources":unique_citations(evidence)}
        if not evidence:
            result["missing"].append({"signal":cfg["label"],"needed":"A source that directly addresses this signal."})
    if not result["naics_candidates"]:
        result["missing"].append({"signal":"NAICS Code","needed":"Relevant U.S. Census NAICS manual pages or another authoritative NAICS source."})
    return result

def render_markdown(brief, selected_naics, neighboring_naics, naics_reason, neighbor_reason):
    md = [f"# Industry Intelligence Brief: {brief['company']}", "", "## NAICS Classification",
          f"**Selected NAICS code:** {selected_naics or 'Not selected'}", "",
          f"**Why this code:** {naics_reason or 'Not completed'}", "",
          f"**Neighboring code considered:** {neighboring_naics or 'Not selected'}", "",
          f"**Why not the neighboring code:** {neighbor_reason or 'Not completed'}", "",
          "**NAICS verification:** Final selection should be checked against the official U.S. Census NAICS manual.", ""]
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
        md.append("- No required category was completely missing from the uploaded evidence.")
    md += ["", "## Method / Traceability Note",
           "The pipeline searches the uploaded documents using TF-IDF retrieval, then selects the most relevant sentences for each required signal. Each extracted claim keeps the original document and page/section location. The tool does not create unsupported numbers; if evidence is missing, it is reported as missing."]
    return "\n".join(md)

def markdown_to_pdf(md_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40,leftMargin=40,topMargin=45,bottomMargin=45)
    styles = getSampleStyleSheet(); story = []
    for raw in md_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1,8)); continue
        if line.startswith("# "): story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "): story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("- "): story.append(Paragraph("• "+line[2:].replace("**",""), styles["BodyText"]))
        else:
            safe = line.replace("**","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, styles["BodyText"]))
        story.append(Spacer(1,4))
    doc.build(story); buffer.seek(0); return buffer.getvalue()

st.subheader("1. Company")
company = st.text_input("Company name", placeholder="Example: Callaway Golf Company")
desc = st.text_area("Short company description (recommended)", placeholder="Example: Designs and sells golf clubs, golf balls, bags, accessories, apparel, and related golf products.")

st.subheader("2. Upload Sources")
st.write("Upload your IBISWorld report, MarketResearch.com report, relevant Census/NAICS pages, and any other supporting files.")
uploads = st.file_uploader("Upload PDFs, DOCX, TXT/MD, or CSV files", type=["pdf","docx","txt","md","csv"], accept_multiple_files=True)
st.caption("Include the Census NAICS pages that contain the likely code and neighboring code so the pipeline can compare them.")

run = st.button("🚀 Analyze Uploaded Files", type="primary", use_container_width=True, disabled=not company or not uploads)

if run:
    pages, errors = [], []
    with st.spinner("Reading uploaded files..."):
        for f in uploads:
            try: pages.extend(read_upload(f))
            except Exception as e: errors.append(f"{f.name}: {e}")
    for err in errors: st.warning(err)
    if not pages:
        st.error("No readable text was extracted from the uploaded files."); st.stop()
    chunks = chunk_pages(pages)
    v,m = build_index(chunks)
    with st.spinner("Extracting required industry signals..."):
        brief = build_brief(company, desc, chunks, v, m)
    st.session_state["brief"] = brief
    st.success("Analysis complete.")

if "brief" in st.session_state:
    brief = st.session_state["brief"]
    st.divider()
    st.header(f"3. Structured Industry Brief — {brief['company']}")
    st.subheader("NAICS Classification")
    candidates = brief["naics_candidates"]
    if candidates:
        labels = [f'{c["code"]} — {c["description"][:140]}... [{c["document"]}, {c["location"]}]' for c in candidates]
        selected_label = st.selectbox("Select the best NAICS candidate", labels)
        idx = labels.index(selected_label); selected = candidates[idx]; selected_naics = selected["code"]
        st.markdown(f'**Evidence:** {selected["description"]}  \n**Source:** {selected["document"]}, {selected["location"]}')
        neighbor_options = [""] + [c["code"] for i,c in enumerate(candidates) if i != idx]
        neighboring_naics = st.selectbox("Select a neighboring / alternative NAICS code", neighbor_options)
    else:
        st.warning("No six-digit NAICS code was found in the uploaded files.")
        selected_naics = st.text_input("Enter the NAICS code manually")
        neighboring_naics = st.text_input("Enter a neighboring NAICS code")
    naics_reason = st.text_area("One sentence: Why is this the correct NAICS code?", placeholder="Explain why the company's primary activity fits this code.")
    neighbor_reason = st.text_area("One sentence: Why not the neighboring code?", placeholder="Explain the key activity that makes the neighboring code less appropriate.")

    for key in ["industry_size","growth","competitors","regulation","supply_chain","customers","trend","threat"]:
        item = brief["signals"][key]
        with st.expander(item["label"], expanded=True):
            st.markdown("**Extracted answer**"); st.write(item["summary"])
            if item["evidence"]:
                st.markdown("**Traceable evidence**")
                for e in item["evidence"]:
                    st.markdown(f'> {e["sentence"]}\n\n**Source:** {e["document"]}, {e["location"]}')
            else:
                st.warning("Not found in the uploaded files.")

    st.subheader("What the Pipeline Could Not Find")
    if brief["missing"]:
        for m in brief["missing"]: st.write(f'• **{m["signal"]}** — {m["needed"]}')
    else:
        st.write("All required categories had at least some supporting evidence.")

    st.divider(); st.header("4. Export")
    md = render_markdown(brief, selected_naics, neighboring_naics, naics_reason, neighbor_reason)
    stem = re.sub(r'[^A-Za-z0-9_-]+','_',brief['company'])
    st.download_button("⬇️ Download Markdown Brief", md, file_name=f"{stem}_industry_brief.md", mime="text/markdown", use_container_width=True)
    st.download_button("⬇️ Download PDF Brief", markdown_to_pdf(md), file_name=f"{stem}_industry_brief.pdf", mime="application/pdf", use_container_width=True)

    rows = []
    for _,item in brief["signals"].items():
        for e in item["evidence"]:
            rows.append({"Signal":item["label"],"Extracted Evidence":e["sentence"],"Document":e["document"],"Location":e["location"]})
    if rows:
        df = pd.DataFrame(rows)
        st.download_button("⬇️ Download Source Log (CSV)", df.to_csv(index=False).encode("utf-8"), file_name=f"{stem}_source_log.csv", mime="text/csv", use_container_width=True)

    st.subheader("Final submission check")
    st.markdown('''
- Open the website in a private/incognito window to confirm the link works without login.
- Verify the selected NAICS code using the official Census NAICS manual.
- Check that each number is directly supported by the cited page/section.
- Keep the missing-information section if the uploaded reports do not support a required signal.
- Put the public website link at the top of page one of your assignment PDF.
''')
