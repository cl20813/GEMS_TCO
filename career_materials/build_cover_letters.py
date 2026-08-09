from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
GOLDMAN_SACHS = ROOT / "goldman_sachs"
GOLDMAN_SACHS.mkdir(parents=True, exist_ok=True)
GOOGLE_YOUTUBE = ROOT / "google_youtube"
GOOGLE_YOUTUBE.mkdir(parents=True, exist_ok=True)
GOLDMAN_CORE_PLANNING = ROOT / "goldman_sachs_core_planning"
GOLDMAN_CORE_PLANNING.mkdir(parents=True, exist_ok=True)

FONT = "Calibri"
BLACK = RGBColor(0x11, 0x11, 0x11)
MUTED = RGBColor(0x4A, 0x55, 0x68)


def set_font(run, size=11, bold=False, italic=False, color=BLACK):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), FONT)
    rpr.rFonts.set(qn("w:hAnsi"), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    return run


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    normal._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for name, size, color, before, after in (
        ("Heading 1", 16, RGBColor(0x2E, 0x74, 0xB5), 16, 8),
        ("Heading 2", 13, RGBColor(0x2E, 0x74, 0xB5), 12, 6),
        ("Heading 3", 12, RGBColor(0x1F, 0x4D, 0x78), 8, 4),
    ):
        style = doc.styles[name]
        style.font.name = FONT
        style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
        style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
        style.font.size = Pt(size)
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.10


def add_paragraph(doc, text="", after=6, line=1.10, bold=False, size=11, color=BLACK):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = line
    set_font(paragraph.add_run(text), size=size, bold=bold, color=color)
    return paragraph


def add_bottom_rule(paragraph, color="7A8793", size="8", space="1"):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.append(pbdr)


def build_goldman_sachs_cover_letter():
    doc = Document()
    configure_document(doc)

    properties = doc.core_properties
    properties.title = "Joonwon Lee - Goldman Sachs GCEM Quantitative Strategist Cover Letter"
    properties.subject = "Application for Quantitative Strategist, Global Currency and Emerging Markets"
    properties.author = "Joonwon Lee"
    properties.keywords = (
        "Goldman Sachs, GCEM, quantitative strategist, stochastic modeling, machine learning, "
        "market making, model calibration, Python"
    )
    properties.comments = ""

    # proposal_centerpiece-inspired application header; intentionally restrained for ATS readability.
    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name.paragraph_format.space_before = Pt(0)
    name.paragraph_format.space_after = Pt(2)
    name.paragraph_format.line_spacing = 1.0
    set_font(name.add_run("JOONWON LEE"), size=18, bold=True)

    contact = doc.add_paragraph()
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    contact.paragraph_format.space_before = Pt(0)
    contact.paragraph_format.space_after = Pt(10)
    contact.paragraph_format.line_spacing = 1.0
    set_font(
        contact.add_run(
            "Piscataway, NJ  |  612-438-9144  |  joonwon.lee22@gmail.com  |  github.com/cl20813"
        ),
        size=9.5,
        color=MUTED,
    )
    add_bottom_rule(contact)

    add_paragraph(doc, "July 27, 2026", after=8)
    add_paragraph(
        doc,
        "Goldman Sachs, Global Banking & Markets\nNew York, NY",
        after=8,
        line=1.05,
    )
    add_paragraph(
        doc,
        "Re: Quantitative Strategist, Global Currency and Emerging Markets (GCEM)",
        after=10,
        bold=True,
    )
    add_paragraph(doc, "Dear GCEM Hiring Team,", after=8)

    add_paragraph(
        doc,
        "I am applying for the Quantitative Strategist position within Goldman Sachs' Global Currency "
        "and Emerging Markets team. I am a Statistics Ph.D. candidate at Rutgers University and expect "
        "to complete my degree in December 2026. My training in stochastic modeling, machine learning, "
        "numerical optimization, and simulation, together with experience in quantitative finance, "
        "insurance pricing, and algorithmic market making, aligns with a desk-strat role combining "
        "quantitative research, production-oriented coding, and close collaboration with traders.",
        after=8,
    )

    add_paragraph(
        doc,
        "At JPMorgan Chase, I developed and validated a Sequential Probability Ratio Test framework for "
        "binary risk indicators. I translated a monitoring problem into statistically controlled decision "
        "rules and used simulation to evaluate Type I and Type II error, detection delay, and early-decision "
        "tradeoffs. Brownian-motion approximations characterized stopping-time uncertainty and helped me "
        "communicate the model's behavior and limitations.",
        after=8,
    )

    add_paragraph(
        doc,
        "My doctoral research develops scalable approximate-likelihood inference and statistical diagnostics "
        "for large, dependent datasets. I built an advection-aware Vecchia approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with fixed-budget conditioning to fit "
        "up to 145,008 satellite observations per day. The Python/PyTorch implementation uses GLS profiling, "
        "L-BFGS optimization, CPU/GPU execution, and restartable HPC workflows. Although the application is "
        "atmospheric data, the work has trained me to calibrate and test models under dependence, missingness, "
        "noise, and computational constraints.",
        after=8,
    )

    add_paragraph(
        doc,
        "At Travelers, I developed an end-to-end LightGBM pricing pipeline across more than 2.46 million "
        "property-risk records, benchmarked it against a generalized linear model, and presented performance "
        "and feature-attribution findings as pricing and risk-segmentation recommendations. I also placed in "
        "the top 2.39% of the IMC Prosperity Algorithmic Trading Competition by designing a market-making "
        "strategy with inventory-aware quoting, dynamic liquidation rules, and benchmark-driven evaluation.",
        after=8,
    )

    add_paragraph(
        doc,
        "I would welcome the opportunity to bring this combination of statistical depth, practical Python "
        "engineering, and model-validation discipline to GCEM. I am eager to deepen my product knowledge "
        "across Rates, FX, and Credit while contributing to machine-learning research and robust trading "
        "tools. Thank you for your consideration.",
        after=10,
    )

    signature = doc.add_paragraph()
    signature.paragraph_format.space_before = Pt(0)
    signature.paragraph_format.space_after = Pt(0)
    signature.paragraph_format.line_spacing = 1.05
    set_font(signature.add_run("Sincerely,\n"), size=11)
    set_font(signature.add_run("Joonwon Lee"), size=11, bold=True)

    output = GOLDMAN_SACHS / "Joonwon_Lee_Goldman_Sachs_GCEM_Cover_Letter.docx"
    doc.save(output)
    return output


def build_google_youtube_cover_letter():
    doc = Document()
    configure_document(doc)

    properties = doc.core_properties
    properties.title = "Joonwon Lee - Google YouTube Marketing Business Data Scientist Cover Letter"
    properties.subject = "Application for Business Data Scientist, YouTube Marketing"
    properties.author = "Joonwon Lee"
    properties.keywords = (
        "Google, YouTube, business data science, hypothesis testing, regression analysis, "
        "causal inference, propensity-score matching, machine learning"
    )
    properties.comments = ""

    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name.paragraph_format.space_before = Pt(0)
    name.paragraph_format.space_after = Pt(2)
    name.paragraph_format.line_spacing = 1.0
    set_font(name.add_run("JOONWON LEE"), size=18, bold=True)

    contact = doc.add_paragraph()
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    contact.paragraph_format.space_before = Pt(0)
    contact.paragraph_format.space_after = Pt(10)
    contact.paragraph_format.line_spacing = 1.0
    set_font(
        contact.add_run(
            "Piscataway, NJ  |  612-438-9144  |  joonwon.lee22@gmail.com  |  github.com/cl20813"
        ),
        size=9.5,
        color=MUTED,
    )
    add_bottom_rule(contact)

    add_paragraph(doc, "July 28, 2026", after=8)
    add_paragraph(
        doc,
        "Google, YouTube Marketing\nSan Bruno, CA",
        after=8,
        line=1.05,
    )
    add_paragraph(
        doc,
        "Re: Business Data Scientist, YouTube Marketing",
        after=10,
        bold=True,
    )
    add_paragraph(doc, "Dear YouTube Marketing Hiring Team,", after=8)

    add_paragraph(
        doc,
        "I am applying for the Business Data Scientist position with YouTube Marketing. I am a Statistics "
        "Ph.D. candidate at Rutgers University, expecting to graduate in December 2026, and hold an M.A. "
        "in Economics. My background combines hypothesis testing, regression analysis, causal inference, "
        "and machine learning - methods central to measuring marketing impact and translating evidence into "
        "business recommendations.",
        after=8,
    )

    add_paragraph(
        doc,
        "At JPMorgan Chase, I developed and validated a Sequential Probability Ratio Test for binary risk "
        "indicators, converting a monitoring question into testable hypotheses and measurable decision rules. "
        "I designed simulation studies to evaluate Type I and Type II error, detection delay, and early-decision "
        "tradeoffs, then communicated the model's behavior and limitations. At Travelers, I developed an "
        "end-to-end LightGBM gradient-boosting pipeline across more than 2.46 million property-risk records "
        "and benchmarked it against generalized linear regression. I evaluated model performance and feature "
        "attribution and presented the findings as pricing and risk-segmentation recommendations.",
        after=8,
    )

    add_paragraph(
        doc,
        "My economics research provides direct causal-inference experience. I estimated the effect of the "
        "Earned Income Tax Credit on household labor supply using fixed-effects regression and propensity-score "
        "matching with observed household covariates. This work strengthened my ability to define treatment "
        "and outcome measures, construct an appropriate comparison group, address confounding, and interpret "
        "estimated effects within the assumptions of an observational study.",
        after=8,
    )

    add_paragraph(
        doc,
        "These experiences prepare me to design rigorous measurement studies, use regression and machine "
        "learning to identify drivers of impact, and communicate clear recommendations to Product and Marketing "
        "partners. I would welcome the opportunity to apply this combination of statistical rigor and practical "
        "Python modeling to YouTube's marketing and user-growth questions. Thank you for your consideration.",
        after=10,
    )

    signature = doc.add_paragraph()
    signature.paragraph_format.space_before = Pt(0)
    signature.paragraph_format.space_after = Pt(0)
    signature.paragraph_format.line_spacing = 1.05
    set_font(signature.add_run("Sincerely,\n"), size=11)
    set_font(signature.add_run("Joonwon Lee"), size=11, bold=True)

    output = GOOGLE_YOUTUBE / "Joonwon_Lee_Google_YouTube_Marketing_Cover_Letter.docx"
    doc.save(output)
    return output


def build_goldman_core_planning_cover_letter():
    doc = Document()
    configure_document(doc)

    properties = doc.core_properties
    properties.title = "Joonwon Lee - Goldman Sachs Core Planning and Analysis Strats Cover Letter"
    properties.subject = "Application for Associate, Quantitative Strategist, Core Planning and Analysis Strats"
    properties.author = "Joonwon Lee"
    properties.keywords = (
        "Goldman Sachs, Core Planning and Analysis Strats, quantitative strategist, stochastic modeling, "
        "simulation, model validation, causal inference, Python, C++ integration"
    )
    properties.comments = ""

    # standard_business_brief with a restrained proposal_centerpiece-style applicant header.
    name = doc.add_paragraph()
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name.paragraph_format.space_before = Pt(0)
    name.paragraph_format.space_after = Pt(2)
    name.paragraph_format.line_spacing = 1.0
    set_font(name.add_run("JOONWON LEE"), size=18, bold=True)

    contact = doc.add_paragraph()
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    contact.paragraph_format.space_before = Pt(0)
    contact.paragraph_format.space_after = Pt(10)
    contact.paragraph_format.line_spacing = 1.0
    set_font(
        contact.add_run(
            "Piscataway, NJ  |  612-438-9144  |  joonwon.lee22@gmail.com  |  github.com/cl20813"
        ),
        size=9.5,
        color=MUTED,
    )
    add_bottom_rule(contact)

    add_paragraph(doc, "August 7, 2026", after=8)
    add_paragraph(
        doc,
        "Goldman Sachs, Corporate Planning & Management\nNew York, NY",
        after=8,
        line=1.05,
    )
    add_paragraph(
        doc,
        "Re: Associate, Quantitative Strategist, Core Planning and Analysis Strats",
        after=10,
        bold=True,
    )
    add_paragraph(doc, "Dear Core Planning and Analysis Strats Hiring Team,", after=8)

    add_paragraph(
        doc,
        "I am applying for the Associate Quantitative Strategist position within the Core Planning and "
        "Analysis Strats team. I am a Statistics Ph.D. candidate at Rutgers University, expecting to graduate "
        "in January 2027. My background combines stochastic and time-dependent modeling, simulation, statistical "
        "diagnostics, model validation, and scalable Python engineering.",
        after=8,
    )

    add_paragraph(
        doc,
        "At JPMorgan Chase, I developed and validated a Sequential Probability Ratio Test for binary risk "
        "indicators, translating a monitoring requirement into explicit hypotheses and statistically controlled "
        "decision rules. I used Brownian-motion approximations, dynamic-programming probability propagation, and "
        "simulation to evaluate stopping behavior, Type I/II error, and model limitations. At Travelers, I built "
        "an end-to-end LightGBM pricing pipeline across more than 2.46 million records, benchmarked it against a "
        "generalized linear model, and communicated the results to business stakeholders.",
        after=8,
    )

    add_paragraph(
        doc,
        "My doctoral research develops scalable likelihood-based inference and diagnostics for large, dependent "
        "datasets. I built an advection-aware Vecchia approximation that fits up to 145,008 observations per day, "
        "designed simulation-based diagnostics that identify model failures by frequency scale, and implemented "
        "restartable Python/PyTorch workflows on CPU, GPU, and HPC systems. I also integrated compiled C++ ordering "
        "routines into Python through pybind11. Separately, I studied the effect of the Earned Income Tax Credit on "
        "household labor supply using fixed-effects regression and propensity-score matching.",
        after=8,
    )

    add_paragraph(
        doc,
        "I am drawn to this role's combination of quantitative planning models, rigorous validation, and analytical "
        "automation. My experience building modular research pipelines provides a strong foundation for production "
        "modeling, and I would be excited to extend that foundation to agentic systems for automated analysis and "
        "reporting. Thank you for your consideration.",
        after=10,
    )

    signature = doc.add_paragraph()
    signature.paragraph_format.space_before = Pt(0)
    signature.paragraph_format.space_after = Pt(0)
    signature.paragraph_format.line_spacing = 1.05
    set_font(signature.add_run("Sincerely,\n"), size=11)
    set_font(signature.add_run("Joonwon Lee"), size=11, bold=True)

    output = GOLDMAN_CORE_PLANNING / "Joonwon_Lee_Goldman_Sachs_Core_Planning_Cover_Letter.docx"
    doc.save(output)
    return output


if __name__ == "__main__":
    print(build_goldman_sachs_cover_letter())
    print(build_google_youtube_cover_letter())
    print(build_goldman_core_planning_cover_letter())
