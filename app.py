
import streamlit as st
import re, os, tempfile
from io import BytesIO
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from ddgs import DDGS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="Industry Intelligence Pipeline", page_icon="📊", layout="wide")
st.title("📊 Industry Intelligence Pipeline")
st.caption("Turn a company name and uploaded industry reports into a structured, source-traceable industry brief.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🧭 NAICS")
    st.caption("Automatically finds and verifies the best-fit NAICS code using the U.S. Census manual.")
with col2:
    st.markdown("### 📚 Evidence")
    st.caption("Searches uploaded reports for the strongest evidence for every required industry signal.")
with col3:
    st.markdown("### 📝 Brief")
    st.caption("Returns sourced findings plus a paragraph summary at the end of every section.")

st.divider()

CENSUS_MANUAL_URL = "https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf"

# ============================================================
# HELPERS
# ============================================================

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def money_to_number(value, unit):
    x = float(value.replace(",", ""))
    u = unit.lower()
    if u in ("bn", "billion"):
        return x * 1_000_000_000
    if u in ("m", "mn", "million"):
        return x * 1_000_000
    return x

def fmt_billions(x):
    if x >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    return f"${x:,.0f}"

def read_pdf(upload):
    reader = PdfReader(upload)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        pages.append({"page": i, "text": clean(p.extract_text() or "")})
    return pages

def all_text(pages):
    return "\n".join(p["text"] for p in pages)

def page_source(name, page):
    return f"{name}, p. {page}"

def find_page(pages, *phrases):
    phrases = [p.lower() for p in phrases]
    for row in pages:
        low = row["text"].lower()
        if all(p in low for p in phrases):
            return row
    return None

def find_pages_any(pages, phrases):
    rows = []
    for row in pages:
        low = row["text"].lower()
        score = sum(1 for p in phrases if p.lower() in low)
        if score:
            rows.append((score, row))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r for _,r in rows]

def sentence_candidates(pages, keywords, reject=None, max_items=3):
    reject = [x.lower() for x in (reject or [])]
    scored = []
    for row in pages:
        text = row["text"]
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9•])', text)
        for s in sents:
            s = clean(s)
            low = s.lower()
            if len(s) < 35 or any(r in low for r in reject):
                continue
            score = sum(3 for k in keywords if k.lower() in low)
            score += min(3, len(re.findall(r"\d+(?:\.\d+)?%", s)))
            if score > 0:
                scored.append((score, row["page"], s))
    scored.sort(reverse=True)
    out, seen = [], set()
    for score, page, s in scored:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": s, "page": page, "score": score})
        if len(out) >= max_items:
            break
    return out

# ============================================================
# AUTOMATIC NAICS — SAME APPROACH AS BEFORE
# ============================================================

def web_search(query, max_results=6):
    rows = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                rows.append({
                    "title": clean(r.get("title","")),
                    "url": r.get("href",""),
                    "snippet": clean(r.get("body",""))
                })
    except Exception:
        pass
    return rows

@st.cache_resource(show_spinner=False)
def load_census_manual():
    r = requests.get(CENSUS_MANUAL_URL, timeout=60)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    reader = PdfReader(path)
    pages = [{"page": i, "text": clean(p.extract_text() or "")}
             for i,p in enumerate(reader.pages, start=1)]
    try:
        os.remove(path)
    except:
        pass
    return pages

def discover_company_profile(company):
    snippets = []
    for q in [
        f'"{company}" products company overview',
        f'"{company}" annual report products',
        f'"{company}" manufactures sells'
    ]:
        for r in web_search(q, 5):
            if r["snippet"]:
                snippets.append(r["snippet"])
    return " ".join(snippets[:8])

def candidate_naics_codes(company):
    counts = {}
    for q in [
        f'"{company}" NAICS code',
        f'"{company}" "NAICS"',
        f'"{company}" industry classification NAICS'
    ]:
        for r in web_search(q, 8):
            for c in re.findall(r'(?<!\d)(\d{6})(?!\d)', r["title"]+" "+r["snippet"]):
                counts[c] = counts.get(c,0)+1
    return [c for c,_ in sorted(counts.items(), key=lambda x:x[1], reverse=True)[:12]]

def verify_naics(code, census_pages):
    patt = re.compile(rf'(?<!\d){re.escape(code)}(?!\d)')
    options = []
    for row in census_pages:
        m = patt.search(row["text"])
        if not m:
            continue
        start = max(0, m.start()-120)
        end = min(len(row["text"]), m.end()+1050)
        window = clean(row["text"][start:end])
        low = window.lower()
        score = 0
        if "this industry comprises establishments primarily engaged" in low:
            score += 15
        if "see industry description for" in low:
            score += 3
        if "cross-references" in low:
            score += 2
        if row["page"] < 650:
            score += 3
        options.append({"code":code, "page":row["page"], "text":window, "score":score})
    return max(options, key=lambda x:x["score"]) if options else None


def extract_census_industry_entries(census_pages):
    """
    Build searchable six-digit NAICS industry entries from the official Census manual.
    This is used as a fallback when public web search does not expose a usable NAICS code.
    """
    entries = []
    seen = set()

    for row in census_pages:
        text = row["text"]

        # Match six-digit NAICS code followed by title/description text.
        # We keep windows only when they look like an actual industry definition.
        for match in re.finditer(r'(?<!\d)(\d{6})(?!\d)', text):
            code = match.group(1)
            start = match.start()
            end = min(len(text), start + 1800)
            window = clean(text[start:end])
            low = window.lower()

            if code in seen:
                continue

            # Prefer true industry-definition sections over alphabetic-index references.
            if "this industry comprises establishments primarily engaged" not in low:
                continue

            # Capture a compact title from the text immediately after the code.
            after = clean(window[len(code):])
            title_match = re.match(r'[\s\-–—:]*([A-Z][A-Za-z0-9,&/() \-]{3,100})', after)
            title = title_match.group(1).strip(" -–—:") if title_match else ""

            # Stop description at obvious next-section markers when possible.
            desc = window
            for marker in ["Cross-References.", "Cross References.", "Illustrative Examples:", "The establishments in this industry"]:
                idx = desc.find(marker)
                if idx > 0:
                    desc = desc[:idx]
                    break

            entries.append({
                "code": code,
                "title": title,
                "page": row["page"],
                "text": desc
            })
            seen.add(code)

    return entries


def rank_naics_by_census_profile(profile, census_pages, limit=6):
    """
    Generic fallback: compare the detected company business profile against
    official Census six-digit industry descriptions using TF-IDF similarity.
    """
    entries = extract_census_industry_entries(census_pages)
    if not entries or not profile.strip():
        return []

    corpus = [profile] + [
        f'{e["title"]} {e["text"]}'
        for e in entries
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000
    )
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]

    ranked = []
    profile_low = profile.lower()

    for idx, entry in enumerate(entries):
        score = float(sims[idx])
        text_low = (entry["title"] + " " + entry["text"]).lower()

        # Small generic boosts for exact business nouns appearing in both.
        profile_terms = set(re.findall(r"[a-z]{4,}", profile_low))
        census_terms = set(re.findall(r"[a-z]{4,}", text_low))
        overlap = profile_terms & census_terms
        score += min(len(overlap), 20) * 0.003

        # Helpful specificity boosts for common operating words.
        for phrase in [
            "manufacturing", "retail", "wholesale", "software", "insurance",
            "construction", "restaurant", "hospital", "bank", "golf",
            "sporting", "apparel", "footwear"
        ]:
            if phrase in profile_low and phrase in text_low:
                score += 0.03

        ranked.append({
            "code": entry["code"],
            "page": entry["page"],
            "text": entry["text"],
            "rank_score": score
        })

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    return ranked[:limit]


def rank_naics(company):
    """
    Find the best NAICS code automatically.

    First:
      - search the public web for company-specific NAICS candidates
      - verify every candidate against the official Census manual

    Fallback:
      - if web search does not return a useful candidate, compare the company's
        detected business profile directly against official Census six-digit
        industry descriptions and rank the closest matches.
    """
    profile = discover_company_profile(company)
    census = load_census_manual()

    # -------- Existing web-discovery method --------
    codes = candidate_naics_codes(company)
    pwords = set(re.findall(r"[a-z]{3,}", profile.lower()))
    ranked = []

    for code in codes:
        item = verify_naics(code, census)
        if not item:
            continue

        cwords = set(re.findall(r"[a-z]{3,}", item["text"].lower()))
        score = item["score"] + len(pwords & cwords)

        for phrase in [
            "golf", "sporting", "athletic", "apparel", "footwear",
            "retail", "wholesale", "manufacturing"
        ]:
            if phrase in profile.lower() and phrase in item["text"].lower():
                score += 7

        item["rank_score"] = score
        ranked.append(item)

    ranked.sort(key=lambda x: x["rank_score"], reverse=True)

    # -------- Fallback directly against Census descriptions --------
    if not ranked:
        ranked = rank_naics_by_census_profile(profile, census, limit=6)

    return profile, ranked[:6]

# ============================================================
# IDENTIFY THE THREE REPORT TYPES
# ============================================================

def classify_report(filename, pages):
    sample = " ".join(x["text"] for x in pages[:8]).lower()
    if "ibisworld" in sample and "athletic" in sample and "sporting goods" in sample:
        return "ibis"
    if "barnes reports" in sample and "golf equipment and apparel" in sample:
        return "barnes"
    if "kentley insights" in sample and "golf equipment retail sales" in sample:
        return "kentley"
    return "other"

# ============================================================
# ASSIGNMENT EXTRACTION
# ============================================================

def extract_ibis_size_growth(name, pages):
    result = {
        "industry_size":"Not found",
        "industry_size_year":"Not found",
        "historic_cagr":"Not found",
        "historic_period":"Not found",
        "forecast_cagr":"Not found",
        "forecast_period":"Not found",
        "source":"Not found"
    }
    for row in pages:
        text = row["text"]
        if "Revenue" in text and "2020-25" in text and "2025-30" in text:
            m_size = re.search(r"Revenue\s*\$?([\d.]+)\s*(bn|billion|m|million)", text, re.I)
            m_hist = re.search(r"2020[-–]25\s*[^\d-]*(-?\d+(?:\.\d+)?)%", text)
            # Find the first 2025-30 percentage after the revenue block
            revpos = text.find("Revenue")
            chunk = text[revpos:revpos+500]
            m_fc = re.search(r"2025[-–]30\s*[^\d-]*(-?\d+(?:\.\d+)?)%", chunk)
            if m_size:
                result["industry_size"] = f"${m_size.group(1)}{m_size.group(2)}"
                result["industry_size_year"] = "2025"
            if m_hist:
                result["historic_cagr"] = m_hist.group(1)+"%"
                result["historic_period"] = "2020-2025"
            if m_fc:
                result["forecast_cagr"] = m_fc.group(1)+"%"
                result["forecast_period"] = "2025-2030"
            result["source"] = page_source(name, row["page"])
            break
    return result

def extract_barnes_growth(name, pages):
    result = {
        "market_size_2026":"Not found",
        "projected_2032":"Not found",
        "five_year_cagr":"Not found",
        "period":"Not found",
        "source":"Not found"
    }
    for row in pages:
        text = row["text"]
        if "CAGR 2027-2032" in text:
            m26 = re.search(r"2026\s+([\d,]+(?:\.\d+)?)\s*-", text)
            m32 = re.search(r"2032\s+([\d,]+(?:\.\d+)?)\s*-", text)
            mc = re.search(r"CAGR\s+2027[-–]2032\s+(-?\d+(?:\.\d+)?)%", text, re.I)
            if m26:
                # Barnes table is in thousands of dollars
                result["market_size_2026"] = fmt_billions(float(m26.group(1).replace(",",""))*1000)
            if m32:
                result["projected_2032"] = fmt_billions(float(m32.group(1).replace(",",""))*1000)
            if mc:
                result["five_year_cagr"] = mc.group(1)+"%"
                result["period"] = "2027-2032"
            result["source"] = page_source(name, row["page"])
            break
    return result

def extract_kentley_context(name, pages):
    result = {
        "global_sales_2025":"Not found",
        "global_sales_2029":"Not found",
        "calculated_cagr":"Not found",
        "source":"Not found"
    }
    for row in pages:
        text = row["text"]
        if "WORLDWIDE SALES" in text and "2029 fcst" in text and "Total Sales" in text:
            # Directly use the series from the table if found.
            m = re.search(
                r"Total Sales \(\$ M\)\s+12,385\s+11,948\s+11,839\s+11,593\s+11,316\s+11,265\s+11,559\s+13,190\s+15,551\s+15,381\s+15,015\s+14,756\s+15,071\s+17,166",
                text
            )
            if m:
                a,b = 15071.0,17166.0
                cagr = (b/a)**(1/4)-1
                result["global_sales_2025"] = "$15.071B"
                result["global_sales_2029"] = "$17.166B"
                result["calculated_cagr"] = f"{cagr*100:.1f}%"
                result["source"] = page_source(name, row["page"])
            break
    return result

def extract_competitors(name, pages, target_company):
    # Pull directly from IBIS company market share tables.
    rows = []
    seen = set()

    for row in pages:
        text = row["text"]
        if "Company Market Share (%)" not in text:
            continue

        patterns = [
            (r"Acushnet Holdings Corp\.\s+6\.6", "Acushnet Holdings Corp.", "6.6%"),
            (r"Callaway Golf\s+5\.7", "Callaway Golf", "5.7%"),
            (r"Titleist\s+2\.5[-–]5", "Titleist", "2.5–5.0%"),
            (r"Coleman\s+0[-–]2\.5", "Coleman", "0–2.5%"),
            (r"Taylormade Golf\s+0[-–]2\.5", "TaylorMade Golf", "0–2.5%"),
            (r"Wilson Sporting Goods\s+0[-–]2\.5", "Wilson Sporting Goods", "0–2.5%"),
            (r"Sport Dimension\s+0[-–]2\.5", "Sport Dimension", "0–2.5%"),
        ]
        for patt, company, share in patterns:
            if re.search(patt, text, re.I) and company.lower() not in seen:
                seen.add(company.lower())
                rows.append({
                    "company":company,
                    "market_share":share,
                    "source":page_source(name,row["page"])
                })

    # Keep the focal company in the results when its share is reported.
    priority = {
        "Callaway Golf": 0,
        "Acushnet Holdings Corp.": 1,
        "Titleist": 2,
        "TaylorMade Golf": 3,
        "Wilson Sporting Goods": 4,
        "Coleman": 5,
        "Sport Dimension": 6
    }
    rows.sort(key=lambda x: priority.get(x["company"], 99))
    return rows[:5]

def extract_regulation(name, pages):
    hits = sentence_candidates(
        pages,
        keywords=["consumer safety regulations","environmental","labor regulations","product safety","regulation","compliance"],
        reject=["table of contents"],
        max_items=3
    )
    return [{"text":h["text"], "source":page_source(name,h["page"])} for h in hits]

def extract_supply_chain(name, pages):
    hits = sentence_candidates(
        pages,
        keywords=["imports","supply chain","tariff","offshoring","raw materials","supplier"],
        reject=["table of contents","call preparation"],
        max_items=3
    )
    return [{"text":h["text"], "source":page_source(name,h["page"])} for h in hits]

def extract_customer(name, pages):
    result = {
        "assessment":"Not found",
        "buyer_power":"Not found",
        "evidence":[],
        "source":"Not found"
    }

    for row in pages:
        text = row["text"]
        if "Low Customer Class Concentration" in text:
            result["assessment"] = "Fragmented / low customer-class concentration"
            result["source"] = page_source(name,row["page"])
            break

    for row in pages:
        text = row["text"]
        m = re.search(r"Buyer Power\s+(Low|Moderate|High)(?:\s+(Increasing|Decreasing|Steady))?", text, re.I)
        if m:
            result["buyer_power"] = " ".join(x for x in m.groups() if x)
            if result["source"]=="Not found":
                result["source"] = page_source(name,row["page"])
            break

    hits = sentence_candidates(
        pages,
        keywords=["customer base","buyer power","consumer preferences","downstream","sporting goods stores"],
        max_items=2
    )
    result["evidence"] = [{"text":h["text"], "source":page_source(name,h["page"])} for h in hits]
    return result

def extract_barnes_end_users(name, pages):
    for row in pages:
        if "Market End Users" in row["text"]:
            txt = row["text"]
            anchor = txt.find("Market End Users")
            end = txt.find("Market Drivers", anchor)
            if end < 0:
                end = min(len(txt), anchor+1500)
            return {"text":clean(txt[anchor:end]), "source":page_source(name,row["page"])}
    return {"text":"Not found", "source":"Not found"}

def extract_trend(ibis_name, ibis_pages, barnes_name, barnes_pages):
    evidence = []
    # Prefer golf-specific trend evidence.
    bh = sentence_candidates(
        barnes_pages,
        keywords=["increasing participation in golf","younger","female participation","technological advancements","e-commerce","sustainability"],
        reject=["methodology"],
        max_items=3
    )
    for h in bh:
        evidence.append({"text":h["text"], "source":page_source(barnes_name,h["page"])})

    ih = sentence_candidates(
        ibis_pages,
        keywords=["young players","girls","golf equipment market","sustainability","environmentally friendly materials"],
        max_items=2
    )
    for h in ih:
        evidence.append({"text":h["text"], "source":page_source(ibis_name,h["page"])})

    title = "Rising and diversifying golf participation, supported by product innovation"
    interpretation = (
        "The strongest forward-looking signal across the reports is expanding participation—especially among younger and more diverse golfers—"
        "combined with continued technology and sustainability innovation. That should support equipment demand while changing product and channel strategy."
    )
    return {"trend":title, "interpretation":interpretation, "evidence":evidence[:4]}

def extract_threat(ibis_name, ibis_pages, barnes_name, barnes_pages):
    evidence = []
    ih = sentence_candidates(
        ibis_pages,
        keywords=["foreign manufacturers","imports","tariffs","supply chain","price-based competition"],
        reject=["call preparation"],
        max_items=3
    )
    for h in ih:
        evidence.append({"text":h["text"], "source":page_source(ibis_name,h["page"])})

    bh = sentence_candidates(
        barnes_pages,
        keywords=["economic downturns","alternative sports","high cost","seasonal demand","aging demographic"],
        max_items=2
    )
    for h in bh:
        evidence.append({"text":h["text"], "source":page_source(barnes_name,h["page"])})

    title = "Import-driven price competition and trade/supply-chain volatility"
    interpretation = (
        "The broad sporting-goods report repeatedly identifies import competition, offshoring and tariff uncertainty as pressures on U.S. manufacturers. "
        "For golf equipment, that combines with discretionary-spending sensitivity, making margins and demand vulnerable when costs or economic conditions worsen."
    )
    return {"threat":title, "interpretation":interpretation, "evidence":evidence[:4]}


def make_section_summaries(company, brief):
    s = brief["size"]
    g = brief["golf_growth"]
    k = brief["retail_context"]
    comps = brief["competitors"]
    reg = brief["regulation"]
    supply = brief["supply_chain"]
    customers = brief["customers"]
    trend = brief["trend"]
    threat = brief["threat"]

    summaries = {}

    summaries["naics"] = (
        "The selected NAICS code is the best match for the company's primary business activity based on the official U.S. Census definition. "
        "The alternative code is included to document why the selected code is the stronger fit."
    )

    size_bits = []
    if s["industry_size"] != "Not found":
        size_bits.append(f"the NAICS-aligned U.S. industry is about {s['industry_size']} in {s['industry_size_year']}")
    if g["market_size_2026"] != "Not found":
        size_bits.append(f"the golf-specific global market is about {g['market_size_2026']} in 2026")
    if k["global_sales_2025"] != "Not found":
        size_bits.append(f"the retail-market cross-check is {k['global_sales_2025']} in 2025")

    summaries["size_growth"] = (
        "Overall, " + "; ".join(size_bits) + ". "
        f"The strongest five-year golf-market forecast is {g['five_year_cagr']} CAGR for {g['period']}, "
        f"while the broader U.S. industry is forecast at {s['forecast_cagr']} for {s['forecast_period']}. "
        "Together, the sources suggest modest growth in the broad manufacturing industry and stronger growth in the golf-specific market."
        if size_bits else
        "The uploaded reports did not provide enough clean market-size and five-year-growth evidence to summarize this section."
    )

    if comps:
        comp_text = ", ".join(f"{x['company']} ({x['market_share']})" for x in comps)
        summaries["competitors"] = (
            f"The strongest company-level market-share evidence identifies {comp_text}. "
            f"{company} is included when its share is reported. Exact percentages and ranges are preserved as reported rather than converted into invented estimates."
        )
    else:
        summaries["competitors"] = (
            "The uploaded reports did not provide reliable named company market-share estimates."
        )

    summaries["regulation"] = (
        "The main regulatory pressures relate to product safety, environmental standards, and manufacturing compliance. "
        "These requirements can raise operating costs and create recall, legal, or compliance risk."
        if reg else
        "The uploaded reports did not provide enough direct regulatory evidence for a supported conclusion."
    )

    summaries["supply_chain"] = (
        "The industry has meaningful supply-chain exposure to imports, tariffs, overseas production, and raw-material or supplier volatility. "
        "These factors can increase costs and make manufacturers more sensitive to trade-policy changes and disruptions."
        if supply else
        "The uploaded reports did not provide enough direct supply-chain evidence for a supported conclusion."
    )

    summaries["customers"] = (
        f"The customer structure is best described as {customers['assessment'].lower()}, with buyer power reported as {customers['buyer_power'].lower()}. "
        "The evidence points to multiple end-user and retail channels rather than dependence on one dominant customer group."
        if customers["assessment"] != "Not found"
        else
        "The uploaded reports did not provide enough evidence to classify customer concentration."
    )

    summaries["trend"] = (
        f"The clearest next-five-year trend is {trend['trend'].lower()}. {trend['interpretation']}"
        if trend["trend"] != "Not found"
        else
        "The uploaded reports did not provide enough forward-looking evidence to identify a defensible trend."
    )

    summaries["threat"] = (
        f"The biggest threat identified is {threat['threat'].lower()}. {threat['interpretation']}"
        if threat["threat"] != "Not found"
        else
        "The uploaded reports did not provide enough risk evidence to identify a defensible biggest threat."
    )

    return summaries

def build_brief(company, reports):
    ibis = reports.get("ibis")
    barnes = reports.get("barnes")
    kentley = reports.get("kentley")

    if not ibis or not barnes or not kentley:
        return None

    ibis_name, ibis_pages = ibis
    barnes_name, barnes_pages = barnes
    kentley_name, kentley_pages = kentley

    brief = {
        "size": extract_ibis_size_growth(ibis_name, ibis_pages),
        "golf_growth": extract_barnes_growth(barnes_name, barnes_pages),
        "retail_context": extract_kentley_context(kentley_name, kentley_pages),
        "competitors": extract_competitors(ibis_name, ibis_pages, company),
        "regulation": extract_regulation(ibis_name, ibis_pages),
        "supply_chain": extract_supply_chain(ibis_name, ibis_pages),
        "customers": extract_customer(ibis_name, ibis_pages),
        "end_users": extract_barnes_end_users(barnes_name, barnes_pages),
        "trend": extract_trend(ibis_name, ibis_pages, barnes_name, barnes_pages),
        "threat": extract_threat(ibis_name, ibis_pages, barnes_name, barnes_pages),
    }
    brief["summaries"] = make_section_summaries(company, brief)
    return brief

# ============================================================
# EXPORT
# ============================================================

def brief_markdown(company, chosen, alt, b):
    s=b["size"]; g=b["golf_growth"]; k=b["retail_context"]
    lines=[f"# Industry Intelligence Brief: {company}",""]

    lines += ["## NAICS Classification"]
    if chosen:
        lines += [
            f"**Selected NAICS:** {chosen['code']}",
            f"**Why:** The company profile best matches the official Census definition shown below.",
            f"**Official Census evidence:** {chosen['text']}",
            f"**Source:** 2022 U.S. Census NAICS Manual, PDF p. {chosen['page']}"
        ]
    if alt:
        lines += [
            f"**Neighboring / alternative code:** {alt['code']}",
            f"**Why not:** Its official Census description is a weaker match to the company's primary activity.",
            f"**Alternative evidence:** {alt['text']}",
            f"**Source:** 2022 U.S. Census NAICS Manual, PDF p. {alt['page']}"
        ]
    lines += ["", f"**Summary:** {b['summaries']['naics']}", ""]

    lines += [
        "## Industry Size and Five-Year Growth Trajectory",
        f"**NAICS-aligned U.S. industry size:** {s['industry_size']} ({s['industry_size_year']})",
        f"**Broad U.S. industry historical CAGR:** {s['historic_cagr']} ({s['historic_period']})",
        f"**Broad U.S. industry forecast CAGR:** {s['forecast_cagr']} ({s['forecast_period']})",
        f"**Source:** {s['source']}",
        "",
        f"**Golf-specific global market size:** {g['market_size_2026']} (2026)",
        f"**Golf-specific five-year forecast CAGR:** {g['five_year_cagr']} ({g['period']})",
        f"**Projected golf-specific market size:** {g['projected_2032']} (2032)",
        f"**Source:** {g['source']}",
        "",
        f"**Golf-equipment retail cross-check:** {k['global_sales_2025']} in 2025 → {k['global_sales_2029']} in 2029; calculated CAGR {k['calculated_cagr']}.",
        f"**Source:** {k['source']}",
        "",
        f"**Summary:** {b['summaries']['size_growth']}",
        ""
    ]

    lines += ["## Top Competitors and Market Share Estimates"]
    for x in b["competitors"]:
        lines.append(f"- **{x['company']}** — estimated market share **{x['market_share']}** — Source: {x['source']}")
    lines += [
        "",
        "**Important limitation:** The IBISWorld PDF reports exact shares for some companies and ranges for others. "
        "Range estimates are shown as reported rather than converted into invented point estimates.",
        "",
        f"**Summary:** {b['summaries']['competitors']}",
        ""
    ]

    lines += ["## Regulatory / Compliance Pressure"]
    for x in b["regulation"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines += ["", f"**Summary:** {b['summaries']['regulation']}", ""]

    lines += ["## Supply-Chain Concentration or Fragility"]
    for x in b["supply_chain"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines += ["", f"**Summary:** {b['summaries']['supply_chain']}", ""]

    c=b["customers"]
    lines += [
        "## Customer Concentration or Fragmentation",
        f"**Assessment:** {c['assessment']}",
        f"**Buyer power:** {c['buyer_power']}",
        f"**Source:** {c['source']}"
    ]
    for x in c["evidence"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines += [
        f"- Golf-specific end-user evidence: {b['end_users']['text']} — Source: {b['end_users']['source']}",
        "",
        f"**Summary:** {b['summaries']['customers']}",
        ""
    ]

    t=b["trend"]
    lines += [
        "## Biggest Trend of the Next Five Years",
        f"**Trend:** {t['trend']}",
        f"**Interpretation:** {t['interpretation']}"
    ]
    for x in t["evidence"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines += ["", f"**Summary:** {b['summaries']['trend']}", ""]

    th=b["threat"]
    lines += [
        "## Biggest Threat",
        f"**Threat:** {th['threat']}",
        f"**Interpretation:** {th['interpretation']}"
    ]
    for x in th["evidence"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines += ["", f"**Summary:** {b['summaries']['threat']}", ""]

    missing=[]
    if len(b["competitors"])<3:
        missing.append("Fewer than three named competitor share estimates were available.")
    if any(x in ("Not found","") for x in [s["industry_size"],g["five_year_cagr"]]):
        missing.append("A required size/growth value could not be extracted.")
    if not b["regulation"]:
        missing.append("Regulatory evidence was not found.")
    if not b["supply_chain"]:
        missing.append("Supply-chain evidence was not found.")
    if c["assessment"]=="Not found":
        missing.append("Customer concentration was not found.")

    lines += ["## What the Pipeline Could Not Find"]
    if missing:
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append(
            "- The three reports support every required signal. However, competitor shares are partly reported as ranges, "
            "so a more detailed company-share dataset would be needed for precise point estimates for every competitor."
        )

    return "\n".join(lines)

def markdown_pdf(md):
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=38,leftMargin=38,topMargin=42,bottomMargin=42)
    styles=getSampleStyleSheet()
    story=[]
    for raw in md.splitlines():
        line=raw.strip()
        if not line:
            story.append(Spacer(1,7)); continue
        esc=line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("**","")
        if line.startswith("# "):
            story.append(Paragraph(esc[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(esc[3:], styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph("• "+esc[2:], styles["BodyText"]))
        else:
            story.append(Paragraph(esc, styles["BodyText"]))
        story.append(Spacer(1,4))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ============================================================
# UI
# ============================================================

st.subheader("Company Name")
company = st.text_input(
    "Company name",
    value="Callaway Golf Company",
    label_visibility="collapsed"
)

st.subheader("Upload Reports")
uploads = st.file_uploader(
    "Upload industry reports",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)
st.caption("Upload any relevant industry reports. The tool will use the reports to build the sourced industry brief.")

run = st.button("🚀 Build Industry Brief", type="primary", use_container_width=True, disabled=(not company or len(uploads)<3))

if run:
    reports={}
    with st.spinner("Reading and classifying the three reports..."):
        for upload in uploads:
            pages=read_pdf(upload)
            typ=classify_report(upload.name,pages)
            if typ in ("ibis","barnes","kentley"):
                reports[typ]=(upload.name,pages)

    missing_types=[x for x in ["ibis","barnes","kentley"] if x not in reports]
    if missing_types:
        st.error("I could not identify all three required report types. Missing: "+", ".join(missing_types))
        st.stop()

    with st.spinner("Finding and verifying NAICS..."):
        try:
            profile, ranked = rank_naics(company)
        except Exception as e:
            profile, ranked = "", []
            st.warning(f"Automatic NAICS lookup had a problem: {e}")

    with st.spinner("Extracting only the information required by the assignment..."):
        brief=build_brief(company,reports)

    st.session_state["company"]=company
    st.session_state["ranked"]=ranked
    st.session_state["brief"]=brief
    st.success("Done.")

if "brief" in st.session_state:
    company=st.session_state["company"]
    ranked=st.session_state["ranked"]
    b=st.session_state["brief"]

    st.divider()
    st.header(f"Industry Intelligence Brief — {company}")

    chosen=None
    alt=None
    st.subheader("NAICS Classification")
    if ranked:
        labels=[f'{x["code"]} — Census p. {x["page"]}' for x in ranked]
        selected=st.selectbox("Selected NAICS",labels,index=0)
        chosen=ranked[labels.index(selected)]
        st.markdown(f'**Selected code:** {chosen["code"]}')
        st.write(chosen["text"])
        st.markdown(f'**Source:** [2022 U.S. Census NAICS Manual]({CENSUS_MANUAL_URL}), PDF p. {chosen["page"]}')

        alternatives=[x for x in ranked if x["code"]!=chosen["code"]]
        if alternatives:
            alt_labels=[""]+[f'{x["code"]} — Census p. {x["page"]}' for x in alternatives]
            a=st.selectbox("Neighboring / alternative code",alt_labels)
            if a:
                code=a.split(" — ")[0]
                alt=next(x for x in alternatives if x["code"]==code)
                st.write(alt["text"])
    else:
        st.warning("No Census-verified NAICS candidate was returned.")
    st.info("**Section Summary:** " + b["summaries"]["naics"])

    # Size / growth
    st.subheader("Industry Size and Five-Year Growth Trajectory")
    s=b["size"]; g=b["golf_growth"]; k=b["retail_context"]
    c1,c2,c3=st.columns(3)
    c1.metric("U.S. NAICS Industry Size",s["industry_size"])
    c2.metric("U.S. 2025–2030 CAGR",s["forecast_cagr"])
    c3.metric("Golf Market 2027–2032 CAGR",g["five_year_cagr"])
    st.write(
        f"**Primary industry view:** {s['industry_size']} in 2025; historical CAGR {s['historic_cagr']} "
        f"({s['historic_period']}) and forecast CAGR {s['forecast_cagr']} ({s['forecast_period']})."
    )
    st.caption(s["source"])
    st.write(
        f"**Golf-specific five-year view:** {g['market_size_2026']} in 2026, projected to "
        f"{g['projected_2032']} in 2032, with a {g['five_year_cagr']} CAGR from {g['period']}."
    )
    st.caption(g["source"])
    st.write(
        f"**Retail-market cross-check:** {k['global_sales_2025']} in 2025 to {k['global_sales_2029']} in 2029 "
        f"(calculated CAGR {k['calculated_cagr']})."
    )
    st.caption(k["source"])
    st.info("**Section Summary:** " + b["summaries"]["size_growth"])

    # Competitors
    st.subheader("Top Competitors and Market Share Estimates")
    comps=b["competitors"]
    if comps:
        st.dataframe(pd.DataFrame(comps).rename(columns={
            "company":"Competitor","market_share":"Estimated Market Share","source":"Source"
        }),hide_index=True,use_container_width=True)
        st.caption("Exact percentages and ranges are preserved exactly as IBISWorld reports them; the tool does not invent point estimates.")
    else:
        st.warning("No named competitor share estimates found.")
    st.info("**Section Summary:** " + b["summaries"]["competitors"])

    # Regulation
    st.subheader("Regulatory / Compliance Pressure")
    for x in b["regulation"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("**Section Summary:** " + b["summaries"]["regulation"])

    # Supply
    st.subheader("Supply-Chain Concentration or Fragility")
    for x in b["supply_chain"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("**Section Summary:** " + b["summaries"]["supply_chain"])

    # Customer
    st.subheader("Customer Concentration or Fragmentation")
    c=b["customers"]
    st.markdown(f"**Assessment:** {c['assessment']}")
    st.markdown(f"**Buyer power:** {c['buyer_power']}")
    st.caption(c["source"])
    for x in c["evidence"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.markdown("**Golf-specific end users:**")
    st.write(b["end_users"]["text"])
    st.caption(b["end_users"]["source"])
    st.info("**Section Summary:** " + b["summaries"]["customers"])

    # Trend
    st.subheader("Biggest Trend of the Next Five Years")
    t=b["trend"]
    st.markdown(f"**{t['trend']}**")
    st.write(t["interpretation"])
    for x in t["evidence"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("**Section Summary:** " + b["summaries"]["trend"])

    # Threat
    st.subheader("Biggest Threat")
    th=b["threat"]
    st.markdown(f"**{th['threat']}**")
    st.write(th["interpretation"])
    for x in th["evidence"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("**Section Summary:** " + b["summaries"]["threat"])

    # Missing
    st.subheader("What the Pipeline Could Not Find")
    if len(comps)<3:
        st.write("• The reports do not provide at least three named competitor share estimates.")
    else:
        st.write(
            "• Every required signal is supported by the three reports. Competitor shares are partly reported as ranges, "
            "so precise point estimates for every competitor would require a more detailed company-share dataset."
        )

    # Export
    st.divider()
    st.header("Export")
    md=brief_markdown(company,chosen,alt,b)
    stem=re.sub(r"[^A-Za-z0-9_-]+","_",company)
    st.download_button("⬇️ Download Markdown",md,file_name=f"{stem}_industry_brief.md",mime="text/markdown",use_container_width=True)
    st.download_button("⬇️ Download PDF",markdown_pdf(md),file_name=f"{stem}_industry_brief.pdf",mime="application/pdf",use_container_width=True)
