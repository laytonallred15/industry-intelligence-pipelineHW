
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from ddgs import DDGS
import re
import time
from io import BytesIO

st.set_page_config(
    page_title="Industry Intelligence Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Industry Intelligence Pipeline")
st.caption("Enter a company, upload industry reports, and generate a structured, traceable industry brief.")

st.info(
    "This tool is designed for traceability. It does not invent unsupported numbers. "
    "Search snippets are treated as leads, and uploaded report evidence is shown with document/page references."
)

# -----------------------------
# Helpers
# -----------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def extract_pdf_pages(uploaded_file):
    reader = PdfReader(uploaded_file)
    rows = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text())
        rows.append({
            "document": uploaded_file.name,
            "page": i,
            "text": txt
        })
    return rows

def web_search(query, max_results=5):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({
                    "title": clean_text(r.get("title", "")),
                    "url": r.get("href", ""),
                    "snippet": clean_text(r.get("body", "")),
                    "query": query
                })
    except Exception as e:
        out.append({
            "title": "Search error",
            "url": "",
            "snippet": str(e),
            "query": query
        })
    return out

def dedupe_results(rows):
    seen = set()
    out = []
    for r in rows:
        key = r.get("url") or (r.get("title"), r.get("snippet"))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def score_page(text, keywords):
    t = text.lower()
    return sum(t.count(k.lower()) for k in keywords)

def best_pdf_evidence(pages, keywords, limit=4):
    scored = []
    for p in pages:
        s = score_page(p["text"], keywords)
        if s > 0:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for _, p in scored[:limit]:
        text = p["text"]
        # Find a useful excerpt near the first keyword
        lower = text.lower()
        pos = -1
        for k in keywords:
            pos = lower.find(k.lower())
            if pos != -1:
                break
        if pos == -1:
            pos = 0
        start = max(0, pos - 220)
        end = min(len(text), pos + 700)
        excerpt = text[start:end].strip()
        results.append({
            "document": p["document"],
            "page": p["page"],
            "excerpt": excerpt
        })
    return results

def top_web_evidence(results, limit=4):
    return results[:limit]

def source_line_for_web(r):
    if r.get("url"):
        return f'{r.get("title","Source")} — {r.get("url")}'
    return r.get("title", "Source")

def make_markdown(company, brief):
    md = [f"# Industry Intelligence Brief: {company}\n"]
    md.append("## Proposed NAICS Classification\n")
    md.append(brief["naics"]["summary"] or "Not yet verified.")
    md.append("\n\n**Source leads:**")
    for s in brief["naics"]["web"]:
        md.append(f"- {source_line_for_web(s)}")
    md.append("\n> Verify the final code against the official U.S. Census NAICS manual.\n")

    for key, title in [
        ("industry_size", "Industry Size and Five-Year Growth Trajectory"),
        ("competitors", "Top Competitors and Market Share Estimates"),
        ("regulation", "Regulatory or Compliance Pressure"),
        ("supply_chain", "Supply Chain Concentration or Fragility"),
        ("customers", "Customer Concentration or Fragmentation"),
        ("trend", "Biggest Trend of the Next Five Years"),
        ("threat", "Biggest Threat"),
    ]:
        item = brief[key]
        md.append(f"\n## {title}\n")
        md.append(item["summary"] or "No verified conclusion generated automatically.")
        if item["pdf"]:
            md.append("\n\n**Uploaded report evidence:**")
            for p in item["pdf"]:
                md.append(f'- {p["document"]}, p. {p["page"]}: {p["excerpt"]}')
        if item["web"]:
            md.append("\n\n**Public web source leads:**")
            for w in item["web"]:
                md.append(f"- {w.get('snippet','')}\n  - {source_line_for_web(w)}")

    md.append("\n## What the Pipeline Could Not Find\n")
    missing = []
    for key, title in [
        ("industry_size", "industry size / growth"),
        ("competitors", "competitors / market share"),
        ("regulation", "regulatory pressure"),
        ("supply_chain", "supply-chain evidence"),
        ("customers", "customer concentration"),
        ("trend", "five-year trend"),
        ("threat", "industry threat")
    ]:
        item = brief[key]
        if not item["pdf"] and not item["web"]:
            missing.append(title)
    if missing:
        for x in missing:
            md.append(f"- {x}: additional industry-report or primary-source evidence is needed.")
    else:
        md.append("- No category was completely empty, but all important claims should still be verified against the underlying sources.")

    md.append("""
## Method and Limitation Note

The pipeline uses targeted public-web searches plus keyword-based extraction from uploaded industry reports.
It preserves source titles/URLs for web evidence and document/page numbers for uploaded PDFs.
The tool intentionally does not convert an unverified search snippet into a definitive industry statistic.
Final numbers, market shares, and NAICS classifications should be checked against the underlying source before submission.
""")
    return "\n".join(md)

# -----------------------------
# Input area
# -----------------------------
left, right = st.columns([2, 1])

with left:
    company = st.text_input(
        "Company name",
        placeholder="Example: Nike, Tesla, Diamond K Gypsum"
    )

with right:
    max_results = st.selectbox(
        "Web sources per search",
        [3, 5, 8],
        index=1
    )

uploads = st.file_uploader(
    "Upload IBISWorld / MarketResearch / other industry report PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

run = st.button(
    "🚀 Run Industry Intelligence Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not company
)

# -----------------------------
# Pipeline
# -----------------------------
if run:
    pdf_pages = []
    for f in uploads:
        try:
            pdf_pages.extend(extract_pdf_pages(f))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")

    categories = {
        "naics": {
            "queries": [
                f'"{company}" NAICS code',
                f'"{company}" industry NAICS',
                f'"{company}" site:census.gov NAICS'
            ],
            "keywords": ["NAICS", "primary business", "industry"]
        },
        "industry_size": {
            "queries": [
                f'"{company}" industry market size five year growth',
                f'"{company}" industry CAGR market revenue',
                f'"{company}" industry outlook growth'
            ],
            "keywords": ["market size", "industry revenue", "revenue", "growth", "CAGR", "five-year", "five year"]
        },
        "competitors": {
            "queries": [
                f'"{company}" competitors market share',
                f'"{company}" major competitors industry market share'
            ],
            "keywords": ["market share", "competitor", "major player", "competition", "largest companies"]
        },
        "regulation": {
            "queries": [
                f'"{company}" industry regulation compliance',
                f'"{company}" regulatory risks government rules'
            ],
            "keywords": ["regulation", "regulatory", "compliance", "law", "EPA", "OSHA", "FDA", "rule"]
        },
        "supply_chain": {
            "queries": [
                f'"{company}" supply chain risks suppliers raw materials',
                f'"{company}" supplier concentration supply chain'
            ],
            "keywords": ["supplier", "supply chain", "raw material", "input", "shortage", "import", "logistics"]
        },
        "customers": {
            "queries": [
                f'"{company}" customer concentration customer segments',
                f'"{company}" major customers industry'
            ],
            "keywords": ["customer", "buyer", "client", "end market", "consumer", "concentration"]
        },
        "trend": {
            "queries": [
                f'"{company}" industry trends next five years',
                f'"{company}" industry outlook future trends'
            ],
            "keywords": ["trend", "outlook", "forecast", "future", "five years", "five-year", "growth driver"]
        },
        "threat": {
            "queries": [
                f'"{company}" industry threats risks',
                f'"{company}" competitive risks industry outlook'
            ],
            "keywords": ["threat", "risk", "decline", "challenge", "pressure", "competition", "substitute"]
        }
    }

    brief = {}
    progress = st.progress(0)
    status = st.empty()

    keys = list(categories.keys())
    for i, key in enumerate(keys, start=1):
        cfg = categories[key]
        status.write(f"Researching: {key.replace('_', ' ').title()}")

        web_rows = []
        for q in cfg["queries"]:
            web_rows.extend(web_search(q, max_results=max_results))
            time.sleep(0.15)
        web_rows = dedupe_results(web_rows)[:max_results]

        pdf_rows = best_pdf_evidence(pdf_pages, cfg["keywords"], limit=4) if pdf_pages else []

        # Conservative auto-summary using evidence snippets rather than fabricated synthesis
        if pdf_rows:
            summary = (
                "Relevant uploaded-report evidence was found. Review the excerpts and use the cited document/page "
                "to write the final conclusion."
            )
        elif web_rows:
            summary = (
                "Relevant public-web source leads were found. Open and verify the underlying sources before "
                "using a statistic or conclusion in the final brief."
            )
        else:
            summary = "No reliable evidence was found automatically."

        brief[key] = {
            "summary": summary,
            "web": web_rows,
            "pdf": pdf_rows
        }

        progress.progress(i / len(keys))

    status.empty()
    progress.empty()
    st.session_state["brief"] = brief
    st.session_state["company"] = company
    st.session_state["pdf_pages"] = pdf_pages
    st.success("Pipeline complete.")

# -----------------------------
# Results
# -----------------------------
if "brief" in st.session_state:
    brief = st.session_state["brief"]
    company = st.session_state["company"]

    st.divider()
    st.header(f"Structured Industry Brief — {company}")

    labels = {
        "naics": "NAICS Classification",
        "industry_size": "Industry Size & Five-Year Growth",
        "competitors": "Top Competitors & Market Share",
        "regulation": "Regulatory / Compliance Pressure",
        "supply_chain": "Supply Chain Concentration / Fragility",
        "customers": "Customer Concentration / Fragmentation",
        "trend": "Biggest Five-Year Trend",
        "threat": "Biggest Threat"
    }

    for key in ["naics", "industry_size", "competitors", "regulation", "supply_chain", "customers", "trend", "threat"]:
        item = brief[key]
        with st.expander(labels[key], expanded=(key in ["naics", "industry_size"])):
            st.write(item["summary"])

            if item["pdf"]:
                st.markdown("**Uploaded report evidence**")
                for p in item["pdf"]:
                    st.markdown(f'**{p["document"]}, p. {p["page"]}**')
                    st.write(p["excerpt"])

            if item["web"]:
                st.markdown("**Public web source leads**")
                df = pd.DataFrame(item["web"])[["title", "snippet", "url"]]
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"url": st.column_config.LinkColumn("Source URL")}
                )

    st.divider()
    st.subheader("Export")
    md = make_markdown(company, brief)

    st.download_button(
        "⬇️ Download structured brief (Markdown)",
        data=md,
        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+','_',company)}_industry_brief.md",
        mime="text/markdown",
        use_container_width=True
    )

    rows = []
    for category, item in brief.items():
        for p in item["pdf"]:
            rows.append({
                "category": labels.get(category, category),
                "source_type": "Uploaded PDF",
                "source": p["document"],
                "page": p["page"],
                "evidence": p["excerpt"],
                "url": ""
            })
        for w in item["web"]:
            rows.append({
                "category": labels.get(category, category),
                "source_type": "Web",
                "source": w.get("title",""),
                "page": "",
                "evidence": w.get("snippet",""),
                "url": w.get("url","")
            })

    csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download source log (CSV)",
        data=csv,
        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+','_',company)}_source_log.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "For your assignment, verify the final NAICS code against the Census manual and verify every number against its original source."
    )
