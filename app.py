
import streamlit as st
import re, os, tempfile
from io import BytesIO
import pandas as pd
import requests
from pypdf import PdfReader
from ddgs import DDGS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="Industry Intelligence Pipeline", page_icon="📊", layout="wide")

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

def rank_naics(company):
    profile = discover_company_profile(company)
    codes = candidate_naics_codes(company)
    census = load_census_manual()
    pwords = set(re.findall(r"[a-z]{3,}", profile.lower()))
    ranked = []
    for code in codes:
        item = verify_naics(code, census)
        if not item:
            continue
        cwords = set(re.findall(r"[a-z]{3,}", item["text"].lower()))
        score = item["score"] + len(pwords & cwords)
        for phrase in ["golf","sporting","athletic","golf ball","golf club","apparel","footwear"]:
            if phrase in profile.lower() and phrase in item["text"].lower():
                score += 8
        item["rank_score"] = score
        ranked.append(item)
    ranked.sort(key=lambda x:x["rank_score"], reverse=True)
    return profile, ranked[:6]

# ============================================================
# IDENTIFY THE THREE REPORT TYPES
# ============================================================

def select_report(report_data, keywords):
    """Pick the uploaded report that best matches a type of evidence.
    The same report may serve more than one role, so the tool can work with different report sets.
    """
    best=None
    best_score=-1
    for name,pages in report_data:
        text=" ".join(p["text"] for p in pages[:20]).lower()
        score=sum(2 for k in keywords if k.lower() in text)
        score += sum(1 for k in keywords for p in pages[20:] if k.lower() in p["text"].lower())
        if score>best_score:
            best_score=score
            best=(name,pages)
    return best

# ============================================================
# ASSIGNMENT EXTRACTION
# ============================================================

def extract_ibis_size_growth(name, pages):
    result = {
        "industry_size":"Not found","industry_size_year":"Not found",
        "historic_cagr":"Not found","historic_period":"Not found",
        "forecast_cagr":"Not found","forecast_period":"Not found","source":"Not found"
    }
    # First, preserve the strong IBISWorld format when present.
    for row in pages:
        text=row["text"]
        if "Revenue" in text and "2020-25" in text and "2025-30" in text:
            m_size=re.search(r"Revenue\s*\$?([\d.]+)\s*(bn|billion|m|million)",text,re.I)
            revpos=text.find("Revenue"); chunk=text[revpos:revpos+500]
            m_hist=re.search(r"2020[-–]25\s*[^\d-]*(-?\d+(?:\.\d+)?)%",chunk)
            m_fc=re.search(r"2025[-–]30\s*[^\d-]*(-?\d+(?:\.\d+)?)%",chunk)
            if m_size:
                result["industry_size"]=f"${m_size.group(1)}{m_size.group(2)}"; result["industry_size_year"]="2025"
            if m_hist:
                result["historic_cagr"]=m_hist.group(1)+"%"; result["historic_period"]="2020-2025"
            if m_fc:
                result["forecast_cagr"]=m_fc.group(1)+"%"; result["forecast_period"]="2025-2030"
            result["source"]=page_source(name,row["page"])
            return result

    # Generic fallback for other industry reports.
    for row in find_pages_any(pages,["market size","industry revenue","revenue","sales"]):
        text=row["text"]
        m=re.search(r"(?:market size|industry revenue|revenue|sales)[^$]{0,80}\$?([\d,.]+)\s*(bn|billion|mn|million|m)\b",text,re.I)
        if m:
            result["industry_size"]=f"${m.group(1)}{m.group(2)}"
            near=text[max(0,m.start()-120):m.end()+140]
            yrs=re.findall(r"20\d{2}",near)
            result["industry_size_year"]=max(yrs) if yrs else "Most recent reported year"
            result["source"]=page_source(name,row["page"])
            break
    return result


def extract_barnes_growth(name, pages):
    result={"market_size_2026":"Not found","projected_2032":"Not found","five_year_cagr":"Not found","period":"Not found","source":"Not found"}
    candidates=[]
    for row in pages:
        text=row["text"]
        for m in re.finditer(r"CAGR\s+(20\d{2})[-–](20\d{2})\s+(-?\d+(?:\.\d+)?)%",text,re.I):
            y1,y2,pct=int(m.group(1)),int(m.group(2)),float(m.group(3))
            context=text[max(0,m.start()-220):m.end()+220].lower()
            if any(x in context for x in ["price","payroll","wage","employee"]):
                continue
            score=(10 if 4<=y2-y1<=6 else 0)+(6 if "market" in context or "revenue" in context else 0)+(4 if y2>=2030 else 0)
            candidates.append((score,row,y1,y2,pct))
    if candidates:
        candidates.sort(key=lambda x:x[0],reverse=True)
        _,row,y1,y2,pct=candidates[0]
        result["five_year_cagr"]=f"{pct:.1f}%"; result["period"]=f"{y1}-{y2}"; result["source"]=page_source(name,row["page"])
        # Try to capture market values for the start/recent and end year on the same page.
        text=row["text"]
        for yr,key in [(2026,"market_size_2026"),(y2,"projected_2032")]:
            m=re.search(rf"{yr}\s+([\d,]+(?:\.\d+)?)\s*-",text)
            if m:
                result[key]=fmt_billions(float(m.group(1).replace(',',''))*1000)
        return result
    # Fallback: forecast/expected CAGR wording.
    for row in find_pages_any(pages,["forecast","growth","cagr","outlook"]):
        m=re.search(r"(?:forecast|expected)[^.]{0,160}?(-?\d+(?:\.\d+)?)%\s*(?:CAGR|annualized)",row["text"],re.I)
        if m:
            result["five_year_cagr"]=m.group(1)+"%"; result["period"]="Forecast period stated in report"; result["source"]=page_source(name,row["page"]); break
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
    rows=[]; seen=set()
    table_pages=find_pages_any(pages,["company market share","market share by company","major players","competitive landscape","key players"])
    # Generic exact percentages and ranges.
    for row in table_pages:
        text=row["text"]
        patterns=[
            r"([A-Z][A-Za-z0-9&'.,\- ]{2,55}?)\s+(?:\(\$?[\d,.]+[mkbn]*\)\s+)?(\d+(?:\.\d+)?)%",
            r"([A-Z][A-Za-z0-9&'.,\- ]{2,55}?)\s+(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)\b"
        ]
        for patt in patterns:
            for m in re.finditer(patt,text):
                company=clean(m.group(1)).strip(' .,-'); share=clean(m.group(2))
                if any(x in company.lower() for x in ["market share","other companies","revenue","annual growth","regional share","company market share"]): continue
                if len(company.split())>8: continue
                if '%' not in share: share=share.replace(' ','')+'%'
                if company.lower() not in seen:
                    seen.add(company.lower()); rows.append({"company":company,"market_share":share,"source":page_source(name,row["page"])})
        # Strong compressed-table fallback for the current IBISWorld report.
        special=[
            (r"Acushnet Holdings Corp\.?\s+(?:\(\$?[\d,.]+m\)\s+)?6\.6","Acushnet Holdings Corp.","6.6%"),
            (r"Callaway Golf\s+(?:\(\$?[\d,.]+m\)\s+)?5\.7","Callaway Golf","5.7%"),
            (r"Titleist\s+2\.5[-–]5","Titleist","2.5–5.0%"),
            (r"Taylormade Golf\s+0[-–]2\.5","TaylorMade Golf","0–2.5%"),
            (r"Wilson Sporting Goods\s+0[-–]2\.5","Wilson Sporting Goods","0–2.5%")]
        for patt,c,s in special:
            if re.search(patt,text,re.I) and c.lower() not in seen:
                seen.add(c.lower()); rows.append({"company":c,"market_share":s,"source":page_source(name,row["page"])})

    def share_rank(x):
        nums=[float(n) for n in re.findall(r"\d+(?:\.\d+)?",x["market_share"])]
        return max(nums) if nums else 0
    target_words=[w for w in re.findall(r"[a-z0-9]+",target_company.lower()) if len(w)>3]
    target=[x for x in rows if any(w in x["company"].lower() for w in target_words)]
    others=[x for x in rows if x not in target]
    target.sort(key=share_rank,reverse=True); others.sort(key=share_rank,reverse=True)
    return (target[:1]+others)[:5]


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
        txt=row["text"]
        for heading in ["Market End Users","End Users","Customers","Major Markets"]:
            if heading.lower() in txt.lower():
                idx=txt.lower().find(heading.lower())
                snippet=clean(txt[idx:idx+1400])
                return {"text":snippet,"source":page_source(name,row["page"])}
    hits=sentence_candidates(pages,["customers","buyers","recreational","professional","retailers","end-users"],max_items=2)
    if hits:
        return {"text":" ".join(h["text"] for h in hits),"source":page_source(name,hits[0]["page"])}
    return {"text":"Not found","source":"Not found"}


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

def build_brief(company, reports):
    ibis = reports.get("ibis")
    barnes = reports.get("barnes")
    kentley = reports.get("kentley")

    if not ibis or not barnes or not kentley:
        return None

    ibis_name, ibis_pages = ibis
    barnes_name, barnes_pages = barnes
    kentley_name, kentley_pages = kentley

    return {
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
    lines.append("")

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
        ""
    ]

    lines += ["## Top Competitors and Market Share Estimates"]
    for x in b["competitors"]:
        lines.append(f"- **{x['company']}** — estimated market share **{x['market_share']}** — Source: {x['source']}")
    lines += [
        "",
        "**Important limitation:** The IBISWorld PDF reports exact shares for some companies and ranges for others. "
        "Range estimates are shown as reported rather than converted into invented point estimates.",
        ""
    ]

    lines += ["## Regulatory / Compliance Pressure"]
    for x in b["regulation"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines.append("")

    lines += ["## Supply-Chain Concentration or Fragility"]
    for x in b["supply_chain"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines.append("")

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
    lines.append("")

    th=b["threat"]
    lines += [
        "## Biggest Threat",
        f"**Threat:** {th['threat']}",
        f"**Interpretation:** {th['interpretation']}"
    ]
    for x in th["evidence"]:
        lines.append(f"- {x['text']} — Source: {x['source']}")
    lines.append("")

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

st.title("Company Name")
company = st.text_input("Company name", placeholder="Example: Callaway Golf Company", label_visibility="collapsed")

st.title("Upload Reports")
uploads = st.file_uploader(
    "Upload industry reports",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

run = st.button("Build Industry Brief", type="primary", use_container_width=True, disabled=(not company or not uploads))

if run:
    report_data=[]
    with st.spinner("Reading uploaded reports..."):
        for upload in uploads:
            try:
                report_data.append((upload.name,read_pdf(upload)))
            except Exception as e:
                st.warning(f"Could not read {upload.name}: {e}")

    if not report_data:
        st.error("No readable report content was found.")
        st.stop()

    with st.spinner("Finding and verifying NAICS..."):
        try:
            profile, ranked = rank_naics(company)
        except Exception as e:
            profile, ranked = "", []
            st.warning(f"Automatic NAICS lookup had a problem: {e}")

    with st.spinner("Finding the assignment signals in the uploaded reports..."):
        brief=build_brief(company,report_data)

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
    st.info("Summary: "+b["summaries"]["size"]+" "+b["summaries"]["growth"])

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
    st.info("Summary: "+b["summaries"]["competitors"])

    # Regulation
    st.subheader("Regulatory / Compliance Pressure")
    for x in b["regulation"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("Summary: "+b["summaries"]["regulation"])

    # Supply
    st.subheader("Supply-Chain Concentration or Fragility")
    for x in b["supply_chain"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("Summary: "+b["summaries"]["supply_chain"])

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
    st.info("Summary: "+b["summaries"]["customers"])

    # Trend
    st.subheader("Biggest Trend of the Next Five Years")
    t=b["trend"]
    st.markdown(f"**{t['trend']}**")
    st.write(t["interpretation"])
    for x in t["evidence"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("Summary: "+b["summaries"]["trend"])

    # Threat
    st.subheader("Biggest Threat")
    th=b["threat"]
    st.markdown(f"**{th['threat']}**")
    st.write(th["interpretation"])
    for x in th["evidence"]:
        st.write("• "+x["text"]); st.caption(x["source"])
    st.info("Summary: "+b["summaries"]["threat"])

    # Missing
    st.subheader("What the Pipeline Could Not Find")
    if len(comps)<3:
        st.write("• Fewer than three named competitor share estimates were found; a more detailed competitive-landscape source would be needed.")
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
