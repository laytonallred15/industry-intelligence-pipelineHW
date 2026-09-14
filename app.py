
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from ddgs import DDGS
import requests
import tempfile
import re, os

st.set_page_config(
    page_title="Industry Intelligence Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Industry Intelligence Pipeline")
st.caption("Company → NAICS → uploaded industry reports → clean structured brief")

st.info(
    "This version is intentionally strict. It extracts only the specific information needed for each assignment section, "
    "keeps the output concise, and flags missing evidence instead of dumping large blocks of text."
)

CENSUS_MANUAL_URL = "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf"

# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def safe_float(value):
    try:
        return float(value.replace(",", ""))
    except:
        return None

def sentence_split(text):
    text = clean_text(text)
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text) if len(x.strip()) > 20]

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

@st.cache_resource(show_spinner=False)
def load_census_manual():
    response = requests.get(CENSUS_MANUAL_URL, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        path = tmp.name

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append({
            "page": i,
            "text": clean_text(page.extract_text() or "")
        })

    try:
        os.remove(path)
    except:
        pass

    return pages

def discover_company_profile(company):
    results = []
    for q in [
        f'"{company}" products company overview',
        f'"{company}" annual report business products',
        f'"{company}" manufactures sells'
    ]:
        results.extend(web_search(q, max_results=5))

    snippets = []
    seen = set()
    for r in results:
        s = r["snippet"]
        if s and s.lower() not in seen:
            seen.add(s.lower())
            snippets.append(s)

    return " ".join(snippets[:6])

def find_candidate_codes(company):
    candidate_counts = {}

    for q in [
        f'"{company}" NAICS code',
        f'"{company}" "NAICS"',
        f'"{company}" industry classification NAICS'
    ]:
        for r in web_search(q, max_results=8):
            txt = f'{r["title"]} {r["snippet"]}'
            for code in re.findall(r'(?<!\d)(\d{6})(?!\d)', txt):
                candidate_counts[code] = candidate_counts.get(code, 0) + 1

    return [c for c,_ in sorted(candidate_counts.items(), key=lambda x:x[1], reverse=True)[:12]]

def census_match(code, census_pages):
    exact = re.compile(rf'(?<!\d){re.escape(code)}(?!\d)')
    matches = []

    for p in census_pages:
        m = exact.search(p["text"])
        if not m:
            continue

        start = max(0, m.start() - 150)
        end = min(len(p["text"]), m.end() + 900)
        window = clean_text(p["text"][start:end])
        low = window.lower()

        score = 0
        if "this industry comprises establishments primarily engaged" in low:
            score += 12
        if "see industry description for" in low:
            score += 3
        if "cross-references" in low:
            score += 2
        if p["page"] < 650:
            score += 3

        matches.append({
            "code": code,
            "page": p["page"],
            "text": window,
            "score": score
        })

    if not matches:
        return None

    matches.sort(key=lambda x:x["score"], reverse=True)
    return matches[0]

def rank_naics(company):
    profile = discover_company_profile(company)
    codes = find_candidate_codes(company)

    if not codes:
        return profile, []

    census_pages = load_census_manual()
    ranked = []

    profile_words = set(re.findall(r"[a-z]{3,}", profile.lower()))

    for code in codes:
        m = census_match(code, census_pages)
        if not m:
            continue

        census_words = set(re.findall(r"[a-z]{3,}", m["text"].lower()))
        overlap = profile_words & census_words
        score = m["score"] + len(overlap)

        for phrase in [
            "golf", "golf balls", "golf clubs", "sporting", "athletic",
            "apparel", "footwear", "retail", "wholesale"
        ]:
            if phrase in profile.lower() and phrase in m["text"].lower():
                score += 7

        m["rank_score"] = score
        ranked.append(m)

    ranked.sort(key=lambda x:x["rank_score"], reverse=True)
    return profile, ranked[:6]

# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_pdf(file):
    reader = PdfReader(file)
    rows = []

    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        if txt:
            rows.append({
                "document": file.name,
                "page": i,
                "location": f"p. {i}",
                "text": txt
            })

    return rows

def extract_docx(file):
    doc = Document(file)
    txt = clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return [{
        "document": file.name,
        "page": None,
        "location": "document",
        "text": txt
    }] if txt else []

def extract_txt(file):
    data = file.read()
    try:
        txt = data.decode("utf-8")
    except:
        txt = data.decode("latin-1", errors="ignore")

    txt = clean_text(txt)
    return [{
        "document": file.name,
        "page": None,
        "location": "document",
        "text": txt
    }] if txt else []

def extract_csv(file):
    df = pd.read_csv(file)
    txt = clean_text(df.astype(str).to_csv(index=False))
    return [{
        "document": file.name,
        "page": None,
        "location": "table",
        "text": txt
    }] if txt else []

def read_upload(file):
    ext = os.path.splitext(file.name.lower())[1]
    if ext == ".pdf":
        return extract_pdf(file)
    if ext == ".docx":
        return extract_docx(file)
    if ext in [".txt", ".md"]:
        return extract_txt(file)
    if ext == ".csv":
        return extract_csv(file)
    return []

# ============================================================
# SOURCE PRIORITY
# ============================================================

def source_priority(document):
    d = document.lower()
    if "ibis" in d or "33992" in d:
        return 5
    if "marketresearch" in d or "market research" in d or "golf" in d:
        return 4
    if "annual" in d or "10-k" in d or "10k" in d:
        return 3
    return 1

# ============================================================
# CLEAN STRUCTURED EXTRACTION
# ============================================================

def find_pages(pages, phrases):
    rows = []
    for p in pages:
        low = p["text"].lower()
        score = source_priority(p["document"]) * 2
        score += sum(3 for phrase in phrases if phrase.lower() in low)

        if score > source_priority(p["document"]) * 2:
            rows.append({**p, "score": score})

    rows.sort(key=lambda x:x["score"], reverse=True)
    return rows

def find_best_match(patterns, pages, phrases=None):
    candidates = find_pages(pages, phrases or [])
    if not candidates:
        candidates = pages

    hits = []
    for p in candidates:
        for pat in patterns:
            for m in re.finditer(pat, p["text"], flags=re.I):
                hits.append({
                    "match": clean_text(m.group(0)),
                    "groups": m.groups(),
                    "document": p["document"],
                    "page": p["page"],
                    "location": p["location"],
                    "score": p.get("score", 0)
                })

    hits.sort(key=lambda x:x["score"], reverse=True)
    return hits[0] if hits else None

def get_context(page_text, anchor, before=180, after=420):
    idx = page_text.lower().find(anchor.lower())
    if idx < 0:
        return ""
    start = max(0, idx-before)
    end = min(len(page_text), idx+after)
    return clean_text(page_text[start:end])

def extract_industry_size(pages):
    # Strongest pattern: "Revenue $10.0bn 2020-25 ... 2025-30 ..."
    revenue_patterns = [
        r"Revenue\s+\$?([\d,.]+)\s*(bn|billion|mn|million|m)\b",
        r"industry revenue[^$]{0,50}\$?([\d,.]+)\s*(bn|billion|mn|million|m)\b",
        r"market size[^$]{0,50}\$?([\d,.]+)\s*(bn|billion|mn|million|m)\b"
    ]

    revenue_hit = find_best_match(
        revenue_patterns,
        pages,
        phrases=["At a Glance", "Revenue", "market size", "industry revenue"]
    )

    historical = find_best_match(
        [r"(2020[-–]25|2020[-–]2025)\s*[■•:\-]?\s*([\d.]+)%"],
        pages,
        phrases=["At a Glance", "Revenue", "five-year growth"]
    )

    forecast = find_best_match(
        [r"(2025[-–]30|2025[-–]2030)\s*[■•:\-]?\s*([\d.]+)%"],
        pages,
        phrases=["At a Glance", "Revenue", "forecast"]
    )

    result = {
        "current_size": "Not found",
        "current_year": "Not found",
        "historical_cagr": "Not found",
        "historical_period": "Not found",
        "forecast_cagr": "Not found",
        "forecast_period": "Not found",
        "source": "Not found"
    }

    if revenue_hit:
        value, unit = revenue_hit["groups"][0], revenue_hit["groups"][1]
        result["current_size"] = f"${value}{unit}"
        result["current_year"] = "2025" if "2025" in revenue_hit["match"] or "At a Glance" else "Most recent reported year"
        result["source"] = f'{revenue_hit["document"]}, {revenue_hit["location"]}'

    if historical:
        result["historical_period"] = historical["groups"][0]
        result["historical_cagr"] = historical["groups"][1] + "%"
        if result["source"] == "Not found":
            result["source"] = f'{historical["document"]}, {historical["location"]}'

    if forecast:
        result["forecast_period"] = forecast["groups"][0]
        result["forecast_cagr"] = forecast["groups"][1] + "%"
        if result["source"] == "Not found":
            result["source"] = f'{forecast["document"]}, {forecast["location"]}'

    return result


def extract_growth(pages):
    """
    Extract the best forward-looking five-year-ish industry growth trajectory.

    Priority:
    1. Golf-specific forecast CAGR ending around 2030-2032
    2. Other explicit forecast CAGR values
    3. IBISWorld-style 2025-2030 forecast CAGR
    4. Historical CAGR only as supporting context

    This avoids accidentally using price inflation or product-specific growth as the industry growth rate.
    """

    def growth_source_priority(document):
        d = document.lower()
        if "gol" in d or "golf" in d or "barnes" in d or "ghto" in d:
            return 10
        if "marketresearch" in d or "market research" in d:
            return 9
        if "ibis" in d or "33992" in d:
            return 7
        return 2

    forecast_candidates = []
    historical_candidates = []

    for p in pages:
        text = p["text"]
        base = growth_source_priority(p["document"])

        # Best case: explicit "CAGR 2027-2032 4.9%"
        for m in re.finditer(r"CAGR\s+(20\d{2})[-–](20\d{2})\s+(-?\d+(?:\.\d+)?)%", text, flags=re.I):
            start_year = int(m.group(1))
            end_year = int(m.group(2))
            pct = float(m.group(3))
            years = end_year - start_year

            score = base
            if 4 <= years <= 6:
                score += 12
            if end_year >= 2030:
                score += 8
            if start_year >= 2026:
                score += 6

            context_start = max(0, m.start() - 250)
            context_end = min(len(text), m.end() + 250)
            context = clean_text(text[context_start:context_end])
            context_low = context.lower()

            if "market" in context_low:
                score += 3
            if "forecast" in context_low or "projection" in context_low:
                score += 4

            if any(bad in context_low for bad in [
                "price", "prices", "payroll", "wages", "playground equipment",
                "health insurance", "transportation cost", "operating cost"
            ]):
                score -= 12

            forecast_candidates.append({
                "period": f"{start_year}-{end_year}",
                "cagr": pct,
                "document": p["document"],
                "page": p.get("page"),
                "location": p["location"],
                "score": score
            })

        # IBISWorld style: Revenue ... 2025-30 ... 0.6%
        for m in re.finditer(
            r"Revenue[^.]{0,180}(2025[-–]30|2025[-–]2030)\s*[■•:\-]?\s*(-?\d+(?:\.\d+)?)%",
            text,
            flags=re.I
        ):
            forecast_candidates.append({
                "period": m.group(1).replace("–", "-"),
                "cagr": float(m.group(2)),
                "document": p["document"],
                "page": p.get("page"),
                "location": p["location"],
                "score": base + 10
            })

        # Historical revenue CAGR
        for m in re.finditer(
            r"Revenue[^.]{0,180}(2020[-–]25|2020[-–]2025)\s*[■•:\-]?\s*(-?\d+(?:\.\d+)?)%",
            text,
            flags=re.I
        ):
            historical_candidates.append({
                "period": m.group(1).replace("–", "-"),
                "cagr": float(m.group(2)),
                "document": p["document"],
                "page": p.get("page"),
                "location": p["location"],
                "score": base + 8
            })

    forecast_candidates.sort(key=lambda x: x["score"], reverse=True)
    historical_candidates.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "forecast_period": "Not found",
        "forecast_cagr": "Not found",
        "historical_period": "Not found",
        "historical_cagr": "Not found",
        "direction": "Not found",
        "interpretation": "Not found",
        "source": "Not found"
    }

    if forecast_candidates:
        best = forecast_candidates[0]
        result["forecast_period"] = best["period"]
        result["forecast_cagr"] = f'{best["cagr"]:.1f}%'
        result["source"] = f'{best["document"]}, {best["location"]}'

        pct = best["cagr"]
        if pct > 5:
            result["direction"] = "Strong growth"
            result["interpretation"] = "The market is projected to grow at a strong pace over the forecast period."
        elif pct > 2:
            result["direction"] = "Moderate growth"
            result["interpretation"] = "The market is projected to grow at a moderate and sustained pace over the forecast period."
        elif pct > 0:
            result["direction"] = "Slow growth"
            result["interpretation"] = "The market is projected to remain positive but grow slowly over the forecast period."
        elif pct == 0:
            result["direction"] = "Flat"
            result["interpretation"] = "The market is projected to remain relatively flat over the forecast period."
        else:
            result["direction"] = "Declining"
            result["interpretation"] = "The market is projected to contract over the forecast period."

    if historical_candidates:
        best_hist = historical_candidates[0]
        result["historical_period"] = best_hist["period"]
        result["historical_cagr"] = f'{best_hist["cagr"]:.1f}%'

    return result


def extract_competitors(pages):
    # Look specifically for major player tables / market-share text.
    candidates = find_pages(
        pages,
        ["Major Players", "Market Share (%)", "market share", "Company Revenue"]
    )

    competitors = []
    seen = set()

    # Pattern designed for lines like:
    # Acushnet Holdings Corp. 659.1 6.6 Callaway Golf 568.7 5.7
    company_share_patterns = [
        r"([A-Z][A-Za-z0-9&.,' \-]{2,60}?)\s+\$?[\d,]+(?:\.\d+)?\s+(\d+(?:\.\d+)?)\s*(?:%|(?=\s+[A-Z]|$))",
        r"([A-Z][A-Za-z0-9&.,' \-]{2,60}?)\s+(\d+(?:\.\d+)?)%"
    ]

    for p in candidates:
        text = p["text"]

        # Prefer the text after "Major Players" if present.
        anchor = text.lower().find("major players")
        if anchor >= 0:
            text = text[anchor:anchor+1200]

        for pat in company_share_patterns:
            for m in re.finditer(pat, text):
                name = clean_text(m.group(1))
                share = m.group(2)

                # Filters for obvious junk.
                bad = [
                    "market share", "revenue", "employees", "businesses",
                    "profit", "wages", "other companies", "products services",
                    "company revenue"
                ]
                if any(b in name.lower() for b in bad):
                    continue
                if len(name.split()) > 8:
                    continue

                key = name.lower()
                if key in seen:
                    continue

                seen.add(key)
                competitors.append({
                    "company": name,
                    "market_share": share + "%",
                    "source": f'{p["document"]}, {p["location"]}'
                })

        if len(competitors) >= 5:
            break

    # Special fallback for IBISWorld compressed tables:
    # Find "Acushnet Holdings Corp. 659.1 6.6 Callaway Golf 568.7 5.7"
    for p in candidates:
        text = p["text"]
        if "Acushnet Holdings Corp." in text and "Callaway Golf" in text:
            m1 = re.search(r"Acushnet Holdings Corp\.\s+[\d,.]+\s+([\d.]+)", text)
            m2 = re.search(r"Callaway Golf\s+[\d,.]+\s+([\d.]+)", text)

            if m1 and "acushnet holdings corp." not in seen:
                competitors.insert(0, {
                    "company": "Acushnet Holdings Corp.",
                    "market_share": m1.group(1) + "%",
                    "source": f'{p["document"]}, {p["location"]}'
                })
                seen.add("acushnet holdings corp.")

            if m2 and "callaway golf" not in seen:
                competitors.append({
                    "company": "Callaway Golf",
                    "market_share": m2.group(1) + "%",
                    "source": f'{p["document"]}, {p["location"]}'
                })
                seen.add("callaway golf")

    competitors = competitors[:5]

    return {
        "competitors": competitors,
        "count_found": len(competitors),
        "missing_note": (
            "Not enough named competitors with reliable market-share percentages were found in the uploaded files. "
            "Upload a golf-equipment-specific competitive-landscape report to complete the 3–5 company requirement."
            if len(competitors) < 3 else ""
        )
    }

def extract_regulation(pages):
    candidates = find_pages(
        pages,
        ["regulation", "regulatory", "compliance", "Consumer Product Safety", "environmental"]
    )

    evidence = []
    seen = set()

    for p in candidates:
        for sent in sentence_split(p["text"]):
            low = sent.lower()
            if any(k in low for k in ["regulation", "regulatory", "compliance", "consumer product safety", "environmental law"]):
                if sent.lower() not in seen:
                    seen.add(sent.lower())
                    evidence.append({
                        "text": sent,
                        "source": f'{p["document"]}, {p["location"]}'
                    })
            if len(evidence) >= 3:
                break
        if len(evidence) >= 3:
            break

    assessment = "Not found"
    if evidence:
        text = " ".join(x["text"].lower() for x in evidence)
        assessment = "Moderate to High" if any(x in text for x in ["stringent", "fines", "recalls", "limitations"]) else "Moderate"

    return {
        "assessment": assessment,
        "evidence": evidence
    }

def extract_supply_chain(pages):
    candidates = find_pages(
        pages,
        ["supply chain", "imports", "import penetration", "tariff", "supplier power", "raw materials"]
    )

    evidence = []
    seen = set()

    for p in candidates:
        for sent in sentence_split(p["text"]):
            low = sent.lower()
            if any(k in low for k in ["supply chain", "import", "tariff", "supplier", "raw material"]):
                if sent.lower() not in seen:
                    seen.add(sent.lower())
                    evidence.append({
                        "text": sent,
                        "source": f'{p["document"]}, {p["location"]}'
                    })
            if len(evidence) >= 3:
                break
        if len(evidence) >= 3:
            break

    assessment = "Not found"
    if evidence:
        joined = " ".join(x["text"].lower() for x in evidence)
        if any(k in joined for k in ["45.0%", "disrupt", "tariff", "imports"]):
            assessment = "Moderately fragile / import-dependent"
        else:
            assessment = "Some supply-chain exposure identified"

    return {
        "assessment": assessment,
        "evidence": evidence
    }

def extract_customers(pages):
    candidates = find_pages(
        pages,
        ["Customer Class Concentration", "buyer power", "customers", "buyers", "downstream"]
    )

    evidence = []
    seen = set()

    for p in candidates:
        for sent in sentence_split(p["text"]):
            low = sent.lower()
            if any(k in low for k in ["customer", "buyer", "downstream", "consumer preferences", "customer class concentration"]):
                if sent.lower() not in seen:
                    seen.add(sent.lower())
                    evidence.append({
                        "text": sent,
                        "source": f'{p["document"]}, {p["location"]}'
                    })
            if len(evidence) >= 3:
                break
        if len(evidence) >= 3:
            break

    assessment = "Not found"
    joined = " ".join(x["text"].lower() for x in evidence)

    if "customer class concentration low" in joined:
        assessment = "Fragmented / low concentration"
    elif evidence:
        assessment = "Broad customer base indicated"

    return {
        "assessment": assessment,
        "evidence": evidence
    }

def extract_trend(pages):
    # Prefer future-facing golf-specific evidence.
    candidates = find_pages(
        pages,
        ["forecast", "2030", "2031", "2032", "younger consumers", "participation", "next five years"]
    )

    trend_candidates = []

    for p in candidates:
        for sent in sentence_split(p["text"]):
            low = sent.lower()

            score = 0
            if any(y in sent for y in ["2030", "2031", "2032"]):
                score += 3
            if any(k in low for k in ["younger consumers", "participation", "e-commerce", "technology", "premium", "sustainability"]):
                score += 5
            if "forecast" in low or "expected" in low:
                score += 2

            # Reject obviously unrelated evidence.
            if any(k in low for k in ["health insurance", "playground equipment", "medical"]):
                score -= 8

            if score > 0:
                trend_candidates.append({
                    "text": sent,
                    "source": f'{p["document"]}, {p["location"]}',
                    "score": score
                })

    trend_candidates.sort(key=lambda x:x["score"], reverse=True)

    if trend_candidates:
        best = trend_candidates[0]
        return {
            "trend": best["text"],
            "why_it_matters": "This trend could influence demand, product strategy, and competitive positioning over the next five years.",
            "source": best["source"]
        }

    return {
        "trend": "Not found",
        "why_it_matters": "A more golf-specific forward-looking report is needed.",
        "source": "Not found"
    }

def extract_threat(pages):
    candidates = find_pages(
        pages,
        ["threat", "competition", "foreign producers", "price-based competition", "imports", "substitutes"]
    )

    scored = []

    for p in candidates:
        for sent in sentence_split(p["text"]):
            low = sent.lower()
            score = 0

            if "foreign producers" in low:
                score += 6
            if "price-based competition" in low:
                score += 6
            if "imports" in low:
                score += 3
            if "challenge" in low or "threat" in low:
                score += 3

            if score > 0:
                scored.append({
                    "text": sent,
                    "source": f'{p["document"]}, {p["location"]}',
                    "score": score
                })

    scored.sort(key=lambda x:x["score"], reverse=True)

    if scored:
        best = scored[0]
        return {
            "threat": best["text"],
            "why_it_matters": "This pressure can reduce pricing power, compress margins, and make it harder for U.S.-based producers to compete.",
            "source": best["source"]
        }

    return {
        "threat": "Not found",
        "why_it_matters": "Additional industry-risk evidence is needed.",
        "source": "Not found"
    }

# ============================================================
# REPORT BUILD
# ============================================================

def build_clean_brief(pages):
    return {
        "industry_size": extract_industry_size(pages),
        "growth": extract_growth(pages),
        "competitors": extract_competitors(pages),
        "regulation": extract_regulation(pages),
        "supply_chain": extract_supply_chain(pages),
        "customers": extract_customers(pages),
        "trend": extract_trend(pages),
        "threat": extract_threat(pages)
    }

# ============================================================
# MARKDOWN / PDF EXPORT
# ============================================================

def render_markdown(company, chosen, neighbor, brief):
    md = [f"# Industry Intelligence Brief: {company}", ""]

    md.append("## NAICS Classification")
    if chosen:
        md.append(f"**Selected NAICS:** {chosen['code']}")
        md.append("")
        md.append(f"**Official Census evidence:** {chosen['text']}")
        md.append("")
        md.append(f"**Source:** 2022 U.S. Census NAICS Manual, PDF p. {chosen['page']}")
    else:
        md.append("No Census-verified NAICS code was selected.")

    if neighbor:
        md.append("")
        md.append(f"**Neighboring / alternative code:** {neighbor['code']}")
        md.append("")
        md.append(f"**Alternative-code evidence:** {neighbor['text']}")

    md.append("")
    md.append("## Industry Size")
    s = brief["industry_size"]
    md += [
        f"**Current industry size:** {s['current_size']}",
        f"**Current year:** {s['current_year']}",
        f"**Historical CAGR:** {s['historical_cagr']} ({s['historical_period']})",
        f"**Forecast CAGR:** {s['forecast_cagr']} ({s['forecast_period']})",
        f"**Source:** {s['source']}",
        ""
    ]

    md.append("## Five-Year Growth Trajectory")
    g = brief["growth"]
    md += [
        f"**Forecast period:** {g['forecast_period']}",
        f"**Forecast CAGR:** {g['forecast_cagr']}",
        f"**Historical CAGR:** {g['historical_cagr']} ({g['historical_period']})",
        f"**Direction:** {g['direction']}",
        f"**Interpretation:** {g['interpretation']}",
        f"**Source:** {g['source']}",
        ""
    ]

    md.append("## Top Competitors and Market Share Estimates")
    c = brief["competitors"]
    if c["competitors"]:
        for x in c["competitors"]:
            md.append(f"- **{x['company']}** — {x['market_share']} — Source: {x['source']}")
    else:
        md.append("- No reliable named competitor market shares found.")
    if c["missing_note"]:
        md.append("")
        md.append(f"**Missing information:** {c['missing_note']}")
    md.append("")

    md.append("## Regulatory / Compliance Pressure")
    r = brief["regulation"]
    md.append(f"**Assessment:** {r['assessment']}")
    for e in r["evidence"]:
        md.append(f"- {e['text']} — Source: {e['source']}")
    if not r["evidence"]:
        md.append("- Not found in uploaded files.")
    md.append("")

    md.append("## Supply-Chain Concentration or Fragility")
    sc = brief["supply_chain"]
    md.append(f"**Assessment:** {sc['assessment']}")
    for e in sc["evidence"]:
        md.append(f"- {e['text']} — Source: {e['source']}")
    if not sc["evidence"]:
        md.append("- Not found in uploaded files.")
    md.append("")

    md.append("## Customer Concentration or Fragmentation")
    cu = brief["customers"]
    md.append(f"**Assessment:** {cu['assessment']}")
    for e in cu["evidence"]:
        md.append(f"- {e['text']} — Source: {e['source']}")
    if not cu["evidence"]:
        md.append("- Not found in uploaded files.")
    md.append("")

    md.append("## Biggest Trend of the Next Five Years")
    t = brief["trend"]
    md += [
        f"**Trend:** {t['trend']}",
        f"**Why it matters:** {t['why_it_matters']}",
        f"**Source:** {t['source']}",
        ""
    ]

    md.append("## Biggest Threat")
    th = brief["threat"]
    md += [
        f"**Threat:** {th['threat']}",
        f"**Why it matters:** {th['why_it_matters']}",
        f"**Source:** {th['source']}",
        ""
    ]

    md.append("## What the Pipeline Could Not Find")
    missing = []

    if s["current_size"] == "Not found":
        missing.append("Current industry size")
    if g["forecast_cagr"] == "Not found":
        missing.append("Five-year industry revenue forecast")
    if c["count_found"] < 3:
        missing.append("At least 3 named competitors with reliable market-share estimates")
    if not r["evidence"]:
        missing.append("Regulatory/compliance evidence")
    if not sc["evidence"]:
        missing.append("Supply-chain evidence")
    if not cu["evidence"]:
        missing.append("Customer concentration evidence")
    if t["trend"] == "Not found":
        missing.append("Forward-looking five-year trend")
    if th["threat"] == "Not found":
        missing.append("Industry threat")

    if missing:
        for x in missing:
            md.append(f"- {x}: additional source evidence is needed.")
    else:
        md.append("- No required category was completely missing.")

    return "\n".join(md)

def markdown_to_pdf(md_text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    story = []

    for raw in md_text.splitlines():
        line = raw.strip()

        if not line:
            story.append(Spacer(1, 7))
            continue

        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("- "):
            safe = line[2:].replace("**","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph("• " + safe, styles["BodyText"]))
        else:
            safe = line.replace("**","").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            story.append(Paragraph(safe, styles["BodyText"]))

        story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# UI
# ============================================================

st.subheader("1. Company")
company = st.text_input(
    "Company name",
    placeholder="Example: Callaway Golf Company"
)

st.subheader("2. Upload Industry Reports")
uploads = st.file_uploader(
    "Upload IBISWorld, MarketResearch.com, and any other useful industry files",
    type=["pdf","docx","txt","md","csv"],
    accept_multiple_files=True
)

st.caption(
    "Best results: upload at least one broad industry report (for size/growth/regulation) "
    "and one golf-specific competitive or forecast report (for competitor shares and future trends)."
)

run = st.button(
    "🚀 Run Industry Intelligence Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not company or not uploads
)

if run:
    pages = []

    with st.spinner("Reading uploaded industry reports..."):
        for f in uploads:
            try:
                pages.extend(read_upload(f))
            except Exception as e:
                st.warning(f"Could not read {f.name}: {e}")

    if not pages:
        st.error("No readable content was extracted from the uploaded files.")
        st.stop()

    with st.spinner("Looking up NAICS and extracting clean assignment signals..."):
        try:
            profile, ranked_naics = rank_naics(company)
        except Exception:
            profile, ranked_naics = "", []

        brief = build_clean_brief(pages)

    st.session_state["company"] = company
    st.session_state["profile"] = profile
    st.session_state["ranked_naics"] = ranked_naics
    st.session_state["brief"] = brief
    st.success("Pipeline complete.")

if "brief" in st.session_state:
    company = st.session_state["company"]
    ranked_naics = st.session_state["ranked_naics"]
    brief = st.session_state["brief"]

    st.divider()
    st.header(f"3. Structured Industry Brief — {company}")

    # NAICS
    st.subheader("NAICS Classification")

    chosen = None
    neighbor = None

    if ranked_naics:
        labels = [
            f'{x["code"]} — Census p. {x["page"]}'
            for x in ranked_naics
        ]

        chosen_label = st.selectbox("Best NAICS candidate", labels)
        chosen = ranked_naics[labels.index(chosen_label)]

        st.markdown(f'**Selected NAICS:** {chosen["code"]}')
        st.write(chosen["text"])
        st.markdown(
            f'**Source:** [2022 U.S. Census NAICS Manual]({CENSUS_MANUAL_URL}), PDF p. {chosen["page"]}'
        )

        alternatives = [x for x in ranked_naics if x["code"] != chosen["code"]]

        if alternatives:
            alt_labels = [""] + [
                f'{x["code"]} — Census p. {x["page"]}'
                for x in alternatives
            ]

            alt = st.selectbox("Neighboring / alternative code", alt_labels)

            if alt:
                code = alt.split(" — ")[0]
                neighbor = next(x for x in alternatives if x["code"] == code)
                st.write(neighbor["text"])
    else:
        st.warning("Automatic NAICS lookup did not return a Census-verified code.")

    # Industry Size
    st.subheader("Industry Size")
    s = brief["industry_size"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Industry Size", s["current_size"])
    c2.metric("Historical CAGR", s["historical_cagr"])
    c3.metric("Forecast CAGR", s["forecast_cagr"])

    st.write(f'**Historical period:** {s["historical_period"]}')
    st.write(f'**Forecast period:** {s["forecast_period"]}')
    st.write(f'**Source:** {s["source"]}')

    # Growth
    st.subheader("Five-Year Growth Trajectory")
    g = brief["growth"]

    c1, c2 = st.columns(2)
    c1.metric("Forecast CAGR", g["forecast_cagr"])
    c2.metric("Direction", g["direction"])

    st.write(f'**Forecast period:** {g["forecast_period"]}')
    st.write(f'**Historical CAGR:** {g["historical_cagr"]} ({g["historical_period"]})')
    st.write(f'**Interpretation:** {g["interpretation"]}')
    st.write(f'**Source:** {g["source"]}')

    # Competitors
    st.subheader("Top Competitors and Market Share Estimates")
    comp = brief["competitors"]

    if comp["competitors"]:
        df = pd.DataFrame(comp["competitors"])
        st.dataframe(
            df.rename(columns={
                "company":"Company",
                "market_share":"Market Share",
                "source":"Source"
            }),
            use_container_width=True,
            hide_index=True
        )

    if comp["missing_note"]:
        st.warning(comp["missing_note"])

    # Regulation
    st.subheader("Regulatory / Compliance Pressure")
    reg = brief["regulation"]
    st.markdown(f'**Assessment:** {reg["assessment"]}')
    for e in reg["evidence"]:
        st.write(f'• {e["text"]}')
        st.caption(e["source"])
    if not reg["evidence"]:
        st.warning("Not found in uploaded files.")

    # Supply chain
    st.subheader("Supply-Chain Concentration or Fragility")
    sc = brief["supply_chain"]
    st.markdown(f'**Assessment:** {sc["assessment"]}')
    for e in sc["evidence"]:
        st.write(f'• {e["text"]}')
        st.caption(e["source"])
    if not sc["evidence"]:
        st.warning("Not found in uploaded files.")

    # Customers
    st.subheader("Customer Concentration or Fragmentation")
    cu = brief["customers"]
    st.markdown(f'**Assessment:** {cu["assessment"]}')
    for e in cu["evidence"]:
        st.write(f'• {e["text"]}')
        st.caption(e["source"])
    if not cu["evidence"]:
        st.warning("Not found in uploaded files.")

    # Trend
    st.subheader("Biggest Trend of the Next Five Years")
    tr = brief["trend"]
    st.markdown(f'**Trend:** {tr["trend"]}')
    st.markdown(f'**Why it matters:** {tr["why_it_matters"]}')
    st.write(f'**Source:** {tr["source"]}')

    # Threat
    st.subheader("Biggest Threat")
    th = brief["threat"]
    st.markdown(f'**Threat:** {th["threat"]}')
    st.markdown(f'**Why it matters:** {th["why_it_matters"]}')
    st.write(f'**Source:** {th["source"]}')

    # Missing info
    st.subheader("What the Pipeline Could Not Find")

    missing = []
    if s["current_size"] == "Not found":
        missing.append("Current industry size")
    if g["forecast_cagr"] == "Not found":
        missing.append("Five-year industry revenue forecast")
    if comp["count_found"] < 3:
        missing.append("3–5 named competitors with reliable market-share estimates")
    if not reg["evidence"]:
        missing.append("Regulatory/compliance evidence")
    if not sc["evidence"]:
        missing.append("Supply-chain evidence")
    if not cu["evidence"]:
        missing.append("Customer concentration evidence")
    if tr["trend"] == "Not found":
        missing.append("Forward-looking five-year trend")
    if th["threat"] == "Not found":
        missing.append("Industry threat")

    if missing:
        for x in missing:
            st.write(f"• {x}")
    else:
        st.write("All required categories had usable evidence.")

    # Export
    st.divider()
    st.header("4. Export")

    md = render_markdown(company, chosen, neighbor, brief)
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
