from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "final"
FINAL.mkdir(parents=True, exist_ok=True)
WELLS_FARGO = ROOT / "wells_fargo"
WELLS_FARGO.mkdir(parents=True, exist_ok=True)
MORGAN_STANLEY = ROOT / "morgan_stanley"
MORGAN_STANLEY.mkdir(parents=True, exist_ok=True)
MICROSOFT = ROOT / "microsoft"
MICROSOFT.mkdir(parents=True, exist_ok=True)
BALYASNY = ROOT / "balyasny"
BALYASNY.mkdir(parents=True, exist_ok=True)
GOLDMAN_SACHS = ROOT / "goldman_sachs"
GOLDMAN_SACHS.mkdir(parents=True, exist_ok=True)
GOOGLE_YOUTUBE = ROOT / "google_youtube"
GOOGLE_YOUTUBE.mkdir(parents=True, exist_ok=True)
GOOGLE_ADS_METRICS = ROOT / "google_ads_metrics"
GOOGLE_ADS_METRICS.mkdir(parents=True, exist_ok=True)
GOOGLE_STUDENT_RESEARCHER = ROOT / "google_student_researcher"
GOOGLE_STUDENT_RESEARCHER.mkdir(parents=True, exist_ok=True)
GOOGLE_DATA_SCIENTIST = ROOT / "google_data_scientist"
GOOGLE_DATA_SCIENTIST.mkdir(parents=True, exist_ok=True)
VALLEY_BANK = ROOT / "valley_bank"
VALLEY_BANK.mkdir(parents=True, exist_ok=True)
BLACKROCK = ROOT / "blackrock"
BLACKROCK.mkdir(parents=True, exist_ok=True)
BLACKROCK_SAE = ROOT / "blackrock_systematic_active_equity"
BLACKROCK_SAE.mkdir(parents=True, exist_ok=True)
RADIX_TRADING = ROOT / "radix_trading"
RADIX_TRADING.mkdir(parents=True, exist_ok=True)
THREE_RED_PARTNERS = ROOT / "3red_partners"
THREE_RED_PARTNERS.mkdir(parents=True, exist_ok=True)
UPSTART = ROOT / "upstart"
UPSTART.mkdir(parents=True, exist_ok=True)
AKUNA_CAPITAL = ROOT / "akuna_capital"
AKUNA_CAPITAL.mkdir(parents=True, exist_ok=True)
JPMORGAN_SPG_QTR = ROOT / "jpmorgan_spg_qtr"
JPMORGAN_SPG_QTR.mkdir(parents=True, exist_ok=True)
GOLDMAN_CORE_PLANNING = ROOT / "goldman_sachs_core_planning"
GOLDMAN_CORE_PLANNING.mkdir(parents=True, exist_ok=True)
WALLEYE_SINGLE_STOCK_VOL = ROOT / "walleye_single_stock_volatility"
WALLEYE_SINGLE_STOCK_VOL.mkdir(parents=True, exist_ok=True)
SENTILINK = ROOT / "sentilink"
SENTILINK.mkdir(parents=True, exist_ok=True)
TEZA_TECHNOLOGIES = ROOT / "teza_technologies"
TEZA_TECHNOLOGIES.mkdir(parents=True, exist_ok=True)
FREDDIE_MAC = ROOT / "freddie_mac"
FREDDIE_MAC.mkdir(parents=True, exist_ok=True)
OPENAI_SAFETY = ROOT / "openai_safety"
OPENAI_SAFETY.mkdir(parents=True, exist_ok=True)
CAPITAL_ONE_MODEL_RISK = ROOT / "capital_one_model_risk_audit"
CAPITAL_ONE_MODEL_RISK.mkdir(parents=True, exist_ok=True)
CAPITAL_ONE_QUANT_MODELER = ROOT / "capital_one_principal_quant_modeler"
CAPITAL_ONE_QUANT_MODELER.mkdir(parents=True, exist_ok=True)
WELLS_FARGO_QA_2027 = ROOT / "wells_fargo_qa_program_2027"
WELLS_FARGO_QA_2027.mkdir(parents=True, exist_ok=True)
TOWER_RESEARCH = ROOT / "tower_research_capital"
TOWER_RESEARCH.mkdir(parents=True, exist_ok=True)
IMC_TRADING = ROOT / "imc_trading"
IMC_TRADING.mkdir(parents=True, exist_ok=True)
NEW_YORK_LIFE = ROOT / "new_york_life"
NEW_YORK_LIFE.mkdir(parents=True, exist_ok=True)
VOLEON = ROOT / "voleon"
VOLEON.mkdir(parents=True, exist_ok=True)
STATE_STREET = ROOT / "state_street"
STATE_STREET.mkdir(parents=True, exist_ok=True)
LIBERTY_MUTUAL = ROOT / "liberty_mutual"
LIBERTY_MUTUAL.mkdir(parents=True, exist_ok=True)
VIRTU_FINANCIAL = ROOT / "virtu_financial"
VIRTU_FINANCIAL.mkdir(parents=True, exist_ok=True)
BARCLAYS_ELECTRONIC_TRADING = ROOT / "barclays_electronic_trading"
BARCLAYS_ELECTRONIC_TRADING.mkdir(parents=True, exist_ok=True)
CUBIST = ROOT / "cubist_point72"
CUBIST.mkdir(parents=True, exist_ok=True)
CITADEL = ROOT / "citadel"
CITADEL.mkdir(parents=True, exist_ok=True)
TWO_SIGMA = ROOT / "two_sigma"
TWO_SIGMA.mkdir(parents=True, exist_ok=True)

FONT = "Arial"
BLACK = RGBColor(0x11, 0x11, 0x11)
NAVY = RGBColor(0x24, 0x3B, 0x53)
MUTED = RGBColor(0x4A, 0x55, 0x68)
RULE = "7A8793"


def set_run_font(run, size, bold=False, italic=False, color=BLACK):
    run.font.name = FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color
    run.font.underline = False
    return run


def set_cell_free_document_defaults(doc, margin_x, margin_top, margin_bottom):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(margin_x)
    section.right_margin = Inches(margin_x)
    section.top_margin = Inches(margin_top)
    section.bottom_margin = Inches(margin_bottom)
    section.header_distance = Inches(0.25)
    section.footer_distance = Inches(0.28)
    section.gutter = Inches(0)
    return 8.5 - 2 * margin_x


def set_style(style, size, color=BLACK, bold=False, before=0, after=0, line=1.0):
    style.font.name = FONT
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    style.font.size = Pt(size)
    style.font.color.rgb = color
    style.font.bold = bold
    pf = style.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line


def configure_styles(doc, body_size, body_line):
    styles = doc.styles
    set_style(styles["Normal"], body_size, BLACK, False, 0, 0, body_line)
    set_style(styles["Title"], 18, BLACK, True, 0, 1, 1.0)
    set_style(styles["Subtitle"], 9.5, MUTED, False, 0, 2, 1.0)
    set_style(styles["Heading 1"], 11, NAVY, True, 5, 2, 1.0)
    set_style(styles["Heading 2"], 9.8, BLACK, True, 2, 1, 1.0)
    set_style(styles["Heading 3"], 9.4, MUTED, True, 1, 0, 1.0)
    if "Compact Bullet" not in styles:
        bullet_style = styles.add_style("Compact Bullet", WD_STYLE_TYPE.PARAGRAPH)
    else:
        bullet_style = styles["Compact Bullet"]
    set_style(bullet_style, body_size, BLACK, False, 0, 0.5, body_line)
    bullet_style.paragraph_format.left_indent = Inches(0.22)
    bullet_style.paragraph_format.first_line_indent = Inches(-0.12)
    bullet_style.paragraph_format.keep_together = True


def add_bullet_numbering(doc, left_twips=320, hanging_twips=180):
    numbering = doc.part.numbering_part.element
    abstract_ids = [
        int(x.get(qn("w:abstractNumId")))
        for x in numbering.findall(qn("w:abstractNum"))
        if x.get(qn("w:abstractNumId")) is not None
    ]
    num_ids = [
        int(x.get(qn("w:numId")))
        for x in numbering.findall(qn("w:num"))
        if x.get(qn("w:numId")) is not None
    ]
    abstract_id = max(abstract_ids or [0]) + 1
    num_id = max(num_ids or [0]) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet")
    lvl.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "\u2022")
    lvl.append(lvl_text)
    suff = OxmlElement("w:suff")
    suff.set(qn("w:val"), "space")
    lvl.append(suff)
    ppr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), str(left_twips))
    tabs.append(tab)
    ppr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), str(left_twips))
    ind.set(qn("w:hanging"), str(hanging_twips))
    ppr.append(ind)
    lvl.append(ppr)
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rpr.append(rfonts)
    lvl.append(rpr)
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_num_id = OxmlElement("w:abstractNumId")
    abstract_num_id.set(qn("w:val"), str(abstract_id))
    num.append(abstract_num_id)
    numbering.append(num)
    return num_id


def apply_num(paragraph, num_id):
    ppr = paragraph._p.get_or_add_pPr()
    numpr = ppr.find(qn("w:numPr"))
    if numpr is None:
        numpr = OxmlElement("w:numPr")
        ppr.append(numpr)
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    numid = OxmlElement("w:numId")
    numid.set(qn("w:val"), str(num_id))
    numpr.append(ilvl)
    numpr.append(numid)


def add_hyperlink(paragraph, text, url, size, color=BLACK, bold=False):
    part = paragraph.part
    rid = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rpr.append(rfonts)
    size_el = OxmlElement("w:sz")
    size_el.set(qn("w:val"), str(int(size * 2)))
    rpr.append(size_el)
    color_el = OxmlElement("w:color")
    color_el.set(qn("w:val"), f"{color[0]:02X}{color[1]:02X}{color[2]:02X}")
    rpr.append(color_el)
    if bold:
        rpr.append(OxmlElement("w:b"))
    run.append(rpr)
    text_el = OxmlElement("w:t")
    text_el.text = text
    run.append(text_el)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def set_bottom_border(paragraph, color=RULE, size="5", space="1"):
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.append(pbdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)


def add_name_header(doc, subtitle=None, compact=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0.5)
    set_run_font(p.add_run("JOONWON LEE"), 17.5 if compact else 18.5, bold=True)

    if subtitle:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1.5)
        set_run_font(p.add_run(subtitle), 9.0 if compact else 9.6, bold=True, color=NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2.0 if compact else 3.0)
    size = 9.0 if compact else 8.7
    set_run_font(p.add_run("Piscataway, NJ 08854  |  612-438-9144  |  "), size, color=MUTED)
    add_hyperlink(p, "joonwon.lee22@gmail.com", "mailto:joonwon.lee22@gmail.com", size, MUTED)
    set_run_font(p.add_run("  |  "), size, color=MUTED)
    add_hyperlink(p, "github.com/cl20813", "https://github.com/cl20813", size, MUTED)


def add_section_heading(doc, text, compact=False):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.space_before = Pt(3.0 if compact else 4.2)
    p.paragraph_format.space_after = Pt(1.3 if compact else 1.7)
    p.paragraph_format.keep_with_next = True
    set_run_font(p.add_run(text.upper()), 10.4 if compact else 10.8, bold=True, color=NAVY)
    set_bottom_border(p)
    return p


def add_tabbed_line(
    doc,
    left,
    right,
    width,
    size,
    bold_left=True,
    color=BLACK,
    before=0,
    after=0.4,
    keep_with_next=True,
):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.keep_with_next = keep_with_next
    pf.tab_stops.add_tab_stop(Inches(width), WD_TAB_ALIGNMENT.RIGHT)
    set_run_font(p.add_run(left), size, bold=bold_left, color=color)
    set_run_font(p.add_run("\t" + right), size, bold=False, color=color)
    return p


def add_body_paragraph(doc, text, size, after=0.8, line=1.0, color=BLACK):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    set_run_font(p.add_run(text), size, color=color)
    return p


def add_labeled_paragraph(doc, label, text, size, after=0.6, line=1.0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    set_run_font(p.add_run(label), size, bold=True)
    set_run_font(p.add_run(text), size)
    return p


def add_bullet(doc, text, num_id, size, after=0.6, line=1.0, bold_lead=None):
    p = doc.add_paragraph(style="Compact Bullet")
    apply_num(p, num_id)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    p.paragraph_format.keep_together = True
    if bold_lead and text.startswith(bold_lead):
        set_run_font(p.add_run(bold_lead), size, bold=True)
        set_run_font(p.add_run(text[len(bold_lead) :]), size)
    else:
        set_run_font(p.add_run(text), size)
    return p


def add_page_number_footer(section, label, size=8):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    set_run_font(p.add_run(label + "  |  Page "), size, color=MUTED)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), FONT)
    rfonts.set(qn("w:hAnsi"), FONT)
    rpr.append(rfonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(size * 2))
    rpr.append(sz)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "4A5568")
    rpr.append(color)
    r.append(rpr)
    fld.append(r)
    p._p.append(fld)


def set_core_properties(doc, title, subject, keywords):
    cp = doc.core_properties
    cp.title = title
    cp.subject = subject
    cp.author = "Joonwon Lee"
    cp.keywords = keywords
    cp.comments = ""


def build_finance_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Finance and Quantitative Research Resume",
        "One-page finance and quantitative modeling resume",
        "quantitative finance, stochastic processes, model diagnostics, Vecchia, sequential testing",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher with experience in stochastic processes, "
        "scalable inference, sequential testing, and model risk. Builds Python/PyTorch models and "
        "diagnostic systems for large, dependent datasets on GPU/HPC and AWS.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming: ",
        "Python (NumPy, Pandas, PyTorch, LightGBM, scikit-learn), R, SQL/MySQL, Linux, Git, HPC, AWS EC2",
        9.4,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, MLE, Gaussian processes, approximate likelihood, spectral methods, Monte "
        "Carlo simulation, sequential testing, model validation, GLM/GBM",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Designed a sequential anomaly-detection framework for binary risk indicators by representing "
        "cumulative evidence as a random walk and using Brownian-motion approximations to quantify "
        "stopping-time uncertainty and early-decision tradeoffs.",
        num_id,
        9.35,
        after=0.75,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline for mid-market property risk - data processing, "
        "GLM benchmarking, validation, and diagnostics - across 2.46M+ records on AWS EC2.",
        num_id,
        9.35,
        after=0.75,
        line=1.0,
    )
    add_bullet(
        doc,
        "Translated model diagnostics and feature-attribution patterns into pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.35,
        after=0.75,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Projects", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Inference: Developed an advection-aware Vecchia likelihood for nonseparable "
        "spatiotemporal Gaussian processes, replacing dense covariance operations with fixed-budget "
        "conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.3,
        after=0.65,
        line=1.0,
        bold_lead="Scalable Inference:",
    )
    add_bullet(
        doc,
        "Model Diagnostics: Designed scale- and frequency-resolved diagnostics to identify "
        "misspecification in covariance smoothness, noise, spatial and temporal dependence, and "
        "space-time interactions.",
        num_id,
        9.3,
        after=0.65,
        line=1.0,
        bold_lead="Model Diagnostics:",
    )
    add_bullet(
        doc,
        "Gaussian Process Modeling: Built and compared Matérn and generalized Cauchy covariance "
        "models that capture multiscale dependence, temporal evolution, and advection.",
        num_id,
        9.3,
        after=0.65,
        line=1.0,
        bold_lead="Gaussian Process Modeling:",
    )
    add_bullet(
        doc,
        "Spectral and Physical Validation: Derived missing-data-aware spectral diagnostics and used "
        "directional cross-variograms to evaluate advection and wind-direction consistency.",
        num_id,
        9.3,
        after=0.65,
        line=1.0,
        bold_lead="Spectral and Physical Validation:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented estimation, simulation, and diagnostic pipelines in "
        "Python/PyTorch with CPU/GPU execution and restartable HPC workflows.",
        num_id,
        9.3,
        after=0.75,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built a market-making strategy with inventory-aware quoting, dynamic liquidation rules, and "
        "benchmark-driven performance evaluation.",
        num_id,
        9.3,
        after=0.65,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        9.0,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = FINAL / "Joonwon_Lee_Finance_Quant_Resume.docx"
    doc.save(out)
    return out


def build_wells_fargo_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Wells Fargo Securities Quantitative Analytics Resume",
        "Targeted resume for Securities Quantitative Analytics Associate",
        "quantitative analytics, stochastic modeling, model development, model validation, simulation, optimization",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative analytics researcher with experience developing and "
        "validating stochastic and machine-learning models for financial risk and large dependent datasets. "
        "Builds Python/SQL/PyTorch workflows for simulation, optimization, model diagnostics, and performance "
        "evaluation on AWS and HPC.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Analytics: ",
        "Stochastic modeling, optimization, Monte Carlo simulation, computational statistics, predictive modeling, "
        "machine learning, model development and validation, performance assessment and evaluation testing",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a sequential anomaly-detection framework for binary risk indicators by "
        "representing cumulative evidence as a random walk and using Brownian-motion approximations to "
        "quantify stopping-time uncertainty.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Performed simulation-based comparisons of fixed-sample and sequential rules across detection delay, "
        "false-positive/false-negative control, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM predictive-modeling pipeline for mid-market property risk, including "
        "data processing, GLM benchmarking, validation, and performance diagnostics across 2.46M+ records on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Translated model diagnostics and feature-attribution results into pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Model Development and Validation", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Model Development: Developed an advection-aware Vecchia likelihood for nonseparable spatiotemporal "
        "Gaussian processes, replacing dense covariance operations with fixed-budget conditioning to fit up "
        "to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Development:",
    )
    add_bullet(
        doc,
        "Calibration and Validation: Calibrated Matérn and generalized Cauchy covariance models and built "
        "scale- and frequency-resolved diagnostics to detect misspecification in smoothness, noise, and dependence.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Calibration and Validation:",
    )
    add_bullet(
        doc,
        "Simulation and Evaluation Testing: Validated a mask-aware expected cross-periodogram over 1,000 "
        "Monte Carlo simulations (MAD 0.00648) and analyzed missing-data effects on covariance parameter estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Simulation and Evaluation Testing:",
    )
    add_bullet(
        doc,
        "Computational Implementation: Implemented reusable Python/PyTorch pipelines with GLS mean profiling, "
        "L-BFGS optimization, CPU/GPU execution, and restartable HPC jobs.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
        bold_lead="Computational Implementation:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation rules, "
        "and benchmark-driven performance analysis.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = WELLS_FARGO / "Joonwon_Lee_Wells_Fargo_Securities_Quantitative_Analytics_Resume.docx"
    doc.save(out)
    return out


def build_morgan_stanley_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Morgan Stanley Applied Mathematician Resume",
        "Targeted resume for Applied Mathematician - Electronic Trading and Quantitative Finance",
        "applied mathematics, stochastic processes, noisy data, sparse estimation, simulation, numerical optimization",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and applied quantitative researcher specializing in stochastic processes, "
        "statistical modeling of sparse and noisy dependent data, numerical optimization, and simulation. "
        "Develops Python/PyTorch prototypes that connect mathematical models with empirical diagnostics, "
        "scalable computation, and risk decisions.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), R, SQL/MySQL, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Mathematical and Statistical Methods: ",
        "Probability, stochastic processes, Gaussian processes, MLE/GLS, numerical optimization (L-BFGS), "
        "Monte Carlo simulation, spectral methods, predictive modeling, missing-data and model diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed a sequential anomaly-detection framework for binary risk indicators by representing "
        "cumulative evidence as a random walk and using Brownian-motion approximations to quantify "
        "stopping-time uncertainty.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Used simulation to compare fixed-sample and sequential policies across detection delay, "
        "false-positive/false-negative control, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM predictive-modeling pipeline for mid-market property risk, including "
        "data processing, GLM benchmarking, validation, and diagnostics across 2.46M+ records on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Explained model diagnostics and feature-attribution results to business stakeholders and translated "
        "them into pricing and risk-segmentation recommendations.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Applied Research and Trading Projects", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation rules, "
        "and benchmark-driven performance analysis.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Sparse and Noisy Data Modeling: Developed an advection-aware Vecchia approximation for "
        "nonseparable Gaussian processes with structured missingness, replacing dense O(n^3) covariance "
        "operations with fixed-budget conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Sparse and Noisy Data Modeling:",
    )
    add_bullet(
        doc,
        "Estimation and Optimization: Calibrated Matérn and generalized Cauchy covariance models using "
        "GLS mean profiling and L-BFGS; designed block-prefix experiments to separate approximation error "
        "from covariance-family misspecification.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Estimation and Optimization:",
    )
    add_bullet(
        doc,
        "Simulation and Assumption Testing: Validated a mask-aware expected cross-periodogram over 1,000 "
        "Monte Carlo simulations (MAD 0.00648) and built scale- and frequency-resolved diagnostics for "
        "missing-data and covariance misspecification.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Simulation and Assumption Testing:",
    )
    add_bullet(
        doc,
        "Scalable Prototyping: Implemented reusable Python/PyTorch likelihood and diagnostic pipelines with "
        "CPU/GPU execution, cached fit artifacts, and restartable HPC jobs.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
        bold_lead="Scalable Prototyping:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = MORGAN_STANLEY / "Joonwon_Lee_Morgan_Stanley_Applied_Mathematician_Resume.docx"
    doc.save(out)
    return out


def build_balyasny_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Balyasny Commodities Quantitative Analyst Resume",
        "Targeted resume for Balyasny Asset Management Commodities Quantitative Analyst",
        "commodities quantitative analysis, portfolio risk, time series, stochastic modeling, "
        "optimization, quantitative research, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in time-series and "
        "spatiotemporal modeling, stochastic processes, scalable likelihood inference, and statistical "
        "risk analysis. Develops Python/PyTorch models and simulation-based diagnostics for large, noisy "
        "datasets and translates quantitative results into decision-relevant risk insights.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Time-series and spatiotemporal modeling, stochastic processes, MLE/GLS, numerical optimization, "
        "Monte Carlo simulation, hypothesis testing, spectral analysis, predictive modeling, model validation, risk analytics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, translating a business monitoring problem into statistically controlled decision rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation-based evaluation of Type I/II error, detection delay, and early-decision "
        "tradeoffs, using Brownian-motion approximations to characterize stopping-time uncertainty.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline for mid-market property risk, including data "
        "preparation, GLM benchmarking, model validation, and performance diagnostics across 2.46M+ records on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading Projects", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation "
        "rules, and benchmark-driven performance analysis.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Time-Series and Spatiotemporal Modeling",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Statistical Modeling: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with fixed-budget conditioning "
        "to fit up to 145,008 environmental observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Scalable Statistical Modeling:",
    )
    add_bullet(
        doc,
        "Model Risk and Diagnostics: Compared Matérn and generalized Cauchy dependence models and designed "
        "scale- and frequency-resolved tests to identify misspecification in smoothness, noise, and temporal dependence.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Model Risk and Diagnostics:",
    )
    add_bullet(
        doc,
        "Large-Scale Data Engineering: Built workflows for missing-data handling, nearest-neighbor construction, "
        "dependency-aware ordering, quality filtering, and irregular-grid resampling.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Large-Scale Data Engineering:",
    )
    add_bullet(
        doc,
        "Estimation and Computation: Implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS "
        "optimization, Monte Carlo simulation, CPU/GPU execution, and restartable HPC jobs.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
        bold_lead="Estimation and Computation:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = BALYASNY / "Joonwon_Lee_Balyasny_Commodities_Quantitative_Analyst_Resume.docx"
    doc.save(out)
    return out


def build_goldman_sachs_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Goldman Sachs GCEM Quantitative Strategist Resume",
        "Targeted resume for Quantitative Strategist, Global Currency and Emerging Markets",
        "quantitative strategy, global markets, market making, pricing, hedging, machine learning, "
        "stochastic modeling, numerical optimization, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in stochastic processes, "
        "machine learning, scalable likelihood inference, numerical optimization, and simulation. Builds "
        "Python/PyTorch models and statistical diagnostic systems for large, dependent datasets, translating "
        "mathematical research into validated computational methods. Experience spans quantitative finance, "
        "insurance pricing, and algorithmic market making.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, time-series and spatiotemporal modeling, MLE/GLS, numerical optimization, "
        "Monte Carlo simulation, hypothesis testing, spectral analysis, predictive modeling, GLM/GBM, model validation",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, translating a business monitoring problem into statistically controlled decision rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation-based evaluation of Type I/II error, detection delay, and early-decision "
        "tradeoffs, using Brownian-motion approximations to characterize stopping-time uncertainty.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline for mid-market property risk, including data "
        "preparation, GLM benchmarking, model validation, and performance diagnostics across 2.46M+ records on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading Projects", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Designed and evaluated a market-making algorithm with inventory-aware bid/ask quoting, dynamic "
        "liquidation rules, and benchmark-driven performance analysis.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic-Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Model Development and Calibration: Developed an advection-aware Vecchia likelihood approximation "
        "for nonseparable Gaussian processes, replacing dense covariance operations with fixed-budget "
        "conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Model Development and Calibration:",
    )
    add_bullet(
        doc,
        "Optimization and Model Selection: Calibrated Matérn and generalized Cauchy dependence models using "
        "GLS profiling and L-BFGS optimization and compared competing specifications through empirical diagnostics.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Optimization and Model Selection:",
    )
    add_bullet(
        doc,
        "Simulation and Model Risk: Built Monte Carlo and scale- and frequency-resolved diagnostics to identify "
        "misspecification in smoothness, noise, missingness, and temporal dependence.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Simulation and Model Risk:",
    )
    add_bullet(
        doc,
        "Quantitative Engineering: Implemented reusable Python/PyTorch likelihood and diagnostic pipelines "
        "with dependency-aware data ordering, CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
        bold_lead="Quantitative Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = GOLDMAN_SACHS / "Joonwon_Lee_Goldman_Sachs_GCEM_Quantitative_Strategist_Resume.docx"
    doc.save(out)
    return out


def build_microsoft_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Microsoft Data Scientist Resume",
        "Targeted resume for Microsoft Data Scientist",
        "data science, machine learning, predictive modeling, model validation, large-scale analytics, stakeholder communication",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist specializing in scalable approximate-likelihood "
        "inference, hypothesis testing, and predictive modeling for large, dependent datasets. Builds "
        "end-to-end Python/PyTorch pipelines for missing-data handling, nearest-neighbor construction, "
        "dependency-aware ordering, irregular-grid processing, optimization, and model validation. "
        "Translates complex business and risk questions into well-defined statistical problems, measurable "
        "decision rules, and actionable recommendations.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected 12/2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Data Science and Machine Learning: ",
        "Data preparation, predictive modeling, GLM/GBM, hypothesis testing, optimization, numerical methods, "
        "model selection and validation, performance evaluation, Monte Carlo simulation, statistical model diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, translating a business monitoring problem into statistically controlled decision rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation-based evaluation of Type I/II error, detection delay, and early-decision "
        "tradeoffs, using Brownian-motion approximations to characterize stopping-time uncertainty and "
        "communicate model limitations.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Research and Engineering", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Large-Scale Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with fixed-budget conditioning to fit "
        "up to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Large-Scale Modeling:",
    )
    add_bullet(
        doc,
        "Data Quality and Missingness: Built observation-mask and quality-filtering workflows for noisy, "
        "partially observed satellite data and analyzed how missingness and grid resampling affect "
        "statistical estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Data Quality and Missingness:",
    )
    add_bullet(
        doc,
        "Model Evaluation and Diagnostics: Compared Matérn and generalized Cauchy models and designed "
        "scale- and frequency-resolved tests to identify misspecification in smoothness, noise, and dependence.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Evaluation and Diagnostics:",
    )
    add_bullet(
        doc,
        "Scalable Research Engineering: Implemented reusable Python/PyTorch pipelines with GLS profiling, "
        "L-BFGS optimization, CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
        bold_lead="Scalable Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = MICROSOFT / "Joonwon_Lee_Microsoft_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_google_youtube_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Google YouTube Marketing Business Data Scientist Resume",
        "Targeted resume for Business Data Scientist, YouTube Marketing",
        "business data science, regression analysis, causal inference, experimentation, hypothesis testing, "
        "predictive modeling, marketing measurement, machine learning, Python, R, SQL",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist specializing in regression, causal inference, "
        "hypothesis testing, predictive modeling, and statistical measurement for large, complex datasets. "
        "Builds reproducible Python workflows across AWS and HPC environments, connecting data preparation, "
        "model validation, and interpretable analysis to business decisions. Experience translating complex "
        "questions into measurable outcomes and actionable recommendations for stakeholders.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.A. in Economics, Hanyang University, Seoul, South Korea",
        "Feb 2019  |  GPA: 3.9/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Statistical Analysis and Machine Learning: ",
        "Regression (GLM/GLS), causal inference (fixed effects, propensity-score matching), predictive modeling, "
        "LightGBM, hypothesis testing, experimental design, sequential testing, Monte Carlo simulation, "
        "model selection and validation, feature attribution, missing-data analysis",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, translating a business monitoring problem into testable hypotheses and measurable decision rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation studies to evaluate operating characteristics including Type I/II error, "
        "detection delay, and early-decision tradeoffs; communicated model behavior and limitations.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, generalized linear regression benchmarking, model validation, and performance evaluation on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations, translating technical results into clear actions for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Research and Analytical Projects", compact=True)
    add_tabbed_line(
        doc,
        "M.A. Economics Research - EITC and Household Labor Supply",
        "Hanyang University",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Causal Inference: Estimated the effect of the Earned Income Tax Credit (EITC) on household labor "
        "supply using fixed-effects regression and propensity-score matching with observed household covariates.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
        bold_lead="Causal Inference:",
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Large-Scale Statistical Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Regression and Estimation: Profiled regression mean structures using generalized least squares and "
        "calibrated flexible Gaussian-process models for large, dependent datasets.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Regression and Estimation:",
    )
    add_bullet(
        doc,
        "Data Preparation and Quality: Built workflows for observation masking, quality filtering, "
        "nearest-neighbor construction, dependency-aware ordering, and irregular-grid resampling for noisy, "
        "partially observed satellite data.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Data Preparation and Quality:",
    )
    add_bullet(
        doc,
        "Measurement and Model Evaluation: Designed scale- and frequency-resolved tests to identify "
        "misspecification and quantify how missingness, noise, and dependence affect statistical estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Measurement and Model Evaluation:",
    )
    add_bullet(
        doc,
        "Scalable Computing: Implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS "
        "optimization, CPU/GPU execution, cached artifacts, and restartable HPC jobs for up to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
        bold_lead="Scalable Computing:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Linear Algebra, Statistical Computing, "
        "Stochastic Processes, Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = GOOGLE_YOUTUBE / "Joonwon_Lee_Google_YouTube_Marketing_Business_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_google_ads_metrics_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Google Ads Metrics Research Data Scientist Resume",
        "Targeted resume for Research Data Scientist, Ads Metrics, Core Metrics at Google",
        "research data science, metrics, experiment design, measurement methodology, statistical modeling, "
        "data quality, hypothesis testing, causal inference, Python, SQL",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and research data scientist specializing in statistical measurement, "
        "experiment design, hypothesis testing, and scalable modeling for large, complex datasets. Translates "
        "product and business questions into tractable metrics, mathematical models, and validation plans; "
        "builds reproducible Python workflows for data preparation, quality assessment, and model evaluation.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Data Workflows: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), Linux, Git, AWS EC2, HPC/SLURM; "
        "data restructuring, quality validation, and reproducible pipelines; SQL/MySQL (working knowledge)",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Statistical Measurement and Modeling: ",
        "Experimental design, causal inference (fixed effects, propensity-score matching), hypothesis and "
        "sequential testing, regression (GLM/GLS), predictive modeling, Monte Carlo simulation, model selection "
        "and validation, performance evaluation, missing-data analysis",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Translated a business monitoring question into explicit hypotheses and measurable decision rules, "
        "developing and validating a Sequential Probability Ratio Test for binary risk indicators.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Modeled the cumulative log-likelihood-ratio process as a random walk and applied Brownian-motion "
        "approximation and dynamic-programming probability propagation to evaluate boundary-crossing "
        "probabilities, stopping-time distributions, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM regression benchmarking, model validation, and performance evaluation on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations, translating technical results into clear actions for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Research and Analytical Projects", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Statistical Measurement, Data Quality, and Scalable Modeling",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Measurement Methodology: Designed scale- and frequency-resolved tests to identify model "
        "misspecification and quantify how missingness, noise, and dependence affect statistical estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Measurement Methodology:",
    )
    add_bullet(
        doc,
        "Data Preparation and Quality: Built workflows for observation masking, quality filtering, "
        "nearest-neighbor construction, dependency-aware ordering, and irregular-grid resampling for noisy, "
        "partially observed satellite data.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Data Preparation and Quality:",
    )
    add_bullet(
        doc,
        "Scalable Model Development: Profiled regression mean structures with GLS and calibrated flexible "
        "Gaussian-process models using reusable Python/PyTorch pipelines, L-BFGS optimization, CPU/GPU execution, "
        "and restartable HPC jobs for up to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
        bold_lead="Scalable Model Development:",
    )
    add_tabbed_line(
        doc,
        "Economic Policy Research - EITC and Household Labor Supply",
        "Hanyang University",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the effect of the Earned Income Tax Credit on household labor supply using fixed-effects "
        "regression and propensity-score matching with observed household covariates.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Linear Algebra, Statistical Computing, "
        "Stochastic Processes, Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = GOOGLE_ADS_METRICS / "Joonwon_Lee_Google_Ads_Metrics_Research_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_google_data_scientist_resume():
    """Create a computational-statistics-focused resume for Google's Data Scientist role."""
    source = GOOGLE_ADS_METRICS / "Joonwon_Lee_Google_Ads_Metrics_Research_Data_Scientist_Resume.docx"
    if not source.exists():
        build_google_ads_metrics_resume()

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Google Data Scientist Resume",
        "Targeted resume for Data Scientist at Google YouTube",
        "computational statistics, scalable inference, predictive modeling, statistical machine learning, "
        "data integration, data integrity, experimentation, Python, PyTorch, SQL, stakeholder communication",
    )

    replacements = {
        "Statistics Ph.D. candidate and research data scientist": (
            "Computational statistician and Statistics Ph.D. candidate specializing in scalable likelihood inference, "
            "probabilistic and predictive modeling, and statistical diagnostics for large, dependent datasets. Builds "
            "reproducible Python/PyTorch workflows for data integration, model estimation, simulation, and validation; "
            "translates business questions into measurable statistical problems, evaluation metrics, and defensible results."
        ),
        "Programming and Data Workflows:": (
            "Programming and Statistical Computing: Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), "
            "C++/pybind11 integration, Linux, Git, AWS EC2, HPC/SLURM; SQL/MySQL (working knowledge); CPU/GPU computing, "
            "data restructuring, and reproducible pipelines"
        ),
        "Statistical Measurement and Modeling:": (
            "Computational Statistics and Modeling: Scalable likelihood approximation, Gaussian processes, MLE/GLS, "
            "numerical optimization, Monte Carlo simulation, statistical diagnostics, hypothesis testing, regression "
            "(GLM/GBM), machine learning, model selection and validation, uncertainty quantification, and missing-data analysis"
        ),
        "JPMorgan Chase - Summer Quantitative Analytics Associate": (
            "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review\tSummer 2026"
        ),
        "RESEARCH AND ANALYTICAL PROJECTS": "COMPUTATIONAL STATISTICS RESEARCH",
        "Ph.D. Dissertation Research -": (
            "Ph.D. Dissertation Research - Scalable Statistical Computing and Model Diagnostics\tSep 2024 - Present"
        ),
        "Measurement Methodology:": (
            "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
            "space-time Gaussian processes, using ordered local conditioning for likelihood-based parameter estimation "
            "and uncertainty quantification with up to 145,008 observations per day."
        ),
        "Data Preparation and Quality:": (
            "Data Integration and Integrity: Built workflows for quality filtering, missingness tracking, time-dependent "
            "coordinate offsets, nearest-center matching, spatial-threshold regular-grid construction, and "
            "dependency-aware ordering for noisy satellite data."
        ),
        "Scalable Model Development:": (
            "Statistical Diagnostics and Computing: Developed scale- and frequency-resolved diagnostics that localize "
            "covariance misspecification and distinguish persistent dependence from high-frequency noise and acquisition "
            "artifacts; implemented Python/PyTorch pipelines with L-BFGS optimization, CPU/GPU and HPC execution, and "
            "C++/pybind11 integration."
        ),
    }

    for paragraph in doc.paragraphs:
        for prefix, new_text in replacements.items():
            if not paragraph.text.startswith(prefix):
                continue
            first_run = paragraph.runs[0] if paragraph.runs else None
            for run in list(paragraph.runs):
                paragraph._p.remove(run._r)

            if "\t" in new_text:
                left, right = new_text.split("\t", 1)
                left_run = paragraph.add_run(left)
                if first_run is not None:
                    set_run_font(
                        left_run,
                        first_run.font.size.pt if first_run.font.size else 9.2,
                        bold=True,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                paragraph.add_run("\t")
                right_run = paragraph.add_run(right)
                set_run_font(
                    right_run,
                    first_run.font.size.pt if first_run and first_run.font.size else 9.2,
                    bold=False,
                    italic=False,
                    color=first_run.font.color.rgb if first_run and first_run.font.color.rgb else BLACK,
                )
            elif first_run is not None and first_run.bold and ": " in new_text:
                lead, detail = new_text.split(": ", 1)
                lead_run = paragraph.add_run(f"{lead}: ")
                set_run_font(
                    lead_run,
                    first_run.font.size.pt if first_run.font.size else 9.2,
                    bold=True,
                    italic=bool(first_run.italic),
                    color=first_run.font.color.rgb or BLACK,
                )
                detail_run = paragraph.add_run(detail)
                set_run_font(
                    detail_run,
                    first_run.font.size.pt if first_run.font.size else 9.2,
                    bold=False,
                    italic=bool(first_run.italic),
                    color=first_run.font.color.rgb or BLACK,
                )
            else:
                replacement = paragraph.add_run(new_text)
                if first_run is not None:
                    set_run_font(
                        replacement,
                        first_run.font.size.pt if first_run.font.size else 9.2,
                        bold=bool(first_run.bold),
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
            break

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if "Expected Dec 2026" in run.text:
                run.text = run.text.replace("Expected Dec 2026", "Expected May 2027")

    out = GOOGLE_DATA_SCIENTIST / "Joonwon_Lee_Google_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_valley_bank_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Valley Bank Quantitative Model Analyst Resume",
        "Targeted resume for Quantitative Model Analyst at Valley Bank",
        "quantitative model analyst, model validation, statistical analysis, financial modeling, "
        "data integrity, model reliability, regression, hypothesis testing, simulation, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative modeler specializing in statistical model development "
        "and validation, hypothesis testing, regression, simulation, and model diagnostics. Experience "
        "evaluating financial-risk monitoring and insurance-pricing models, assessing data integrity and "
        "completeness, and working with large dependent datasets. Builds reproducible Python workflows and "
        "communicates model assumptions, performance, and limitations clearly.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL/MySQL, R, Linux, Git, AWS EC2, HPC/SLURM",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Model Development and Validation: ",
        "Regression (GLM/GLS), predictive modeling, hypothesis and sequential testing, stochastic modeling, "
        "Monte Carlo simulation, numerical optimization, model selection and validation, performance "
        "assessment, data-quality and missing-data diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, translating a monitoring problem into explicit hypotheses and statistically controlled decision rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation studies to evaluate Type I/II error, detection delay, stopping-time uncertainty, "
        "and early-decision tradeoffs; documented model behavior, assumptions, and limitations.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and presented pricing and "
        "risk-segmentation recommendations to business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Model Development and Validation Research", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Statistical Model Development and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Model Development and Estimation: Developed a scalable Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with fixed-budget conditioning "
        "to fit up to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Development and Estimation:",
    )
    add_bullet(
        doc,
        "Data Integrity and Completeness: Built observation-mask, quality-filtering, nearest-neighbor, "
        "dependency-ordering, and irregular-grid workflows; quantified how missingness and resampling affect estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Data Integrity and Completeness:",
    )
    add_bullet(
        doc,
        "Model Validation and Diagnostics: Compared Matérn and generalized Cauchy specifications and designed "
        "scale- and frequency-resolved tests to identify misspecification in smoothness, noise, and dependence.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Validation and Diagnostics:",
    )
    add_bullet(
        doc,
        "Quantitative Engineering: Implemented reproducible Python/PyTorch pipelines with GLS profiling, "
        "L-BFGS optimization, CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
        bold_lead="Quantitative Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = VALLEY_BANK / "Joonwon_Lee_Valley_Bank_Quantitative_Model_Analyst_Resume.docx"
    doc.save(out)
    return out


def build_blackrock_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - BlackRock Investment Institute Portfolio Research Associate Resume",
        "Targeted resume for BlackRock Investment Institute, Portfolio Research, Associate",
        "portfolio research, stochastic simulation, capital markets, asset allocation, quantitative research, "
        "Python, statistical modeling, scalable research workflows",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher with experience in stochastic simulation, "
        "scalable likelihood-based modeling, statistical diagnostics, and financial risk analytics. Builds "
        "modular Python/PyTorch research workflows for large, noisy dependent datasets and translates model "
        "results into clear, decision-relevant insights. Brings finance-sector experience and a strong "
        "interest in capital markets, asset allocation, and systematic investment research.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), Linux, Git, AWS EC2, HPC/SLURM; "
        "reproducible pipelines, cached artifacts, CPU/GPU execution, restartable workflows",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Research: ",
        "Stochastic processes, Monte Carlo simulation, time-series and spatiotemporal modeling, Gaussian "
        "processes, MLE/GLS, numerical optimization, regression, spectral analysis, predictive modeling, "
        "model validation, scenario and sensitivity analysis",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test (SPRT) framework for binary risk "
        "indicators, modeling the cumulative log-likelihood-ratio process as a random walk with statistically "
        "controlled decision boundaries.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied a Brownian-motion approximation and dynamic-programming probability propagation to quantify "
        "boundary-crossing probabilities, stopping-time distributions, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "GLM benchmarking, model validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Markets", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Research Workflows",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Stochastic Modeling and Simulation: Developed an advection-aware Vecchia likelihood approximation "
        "for nonseparable Gaussian processes, replacing dense covariance operations with fixed-budget "
        "conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Stochastic Modeling and Simulation:",
    )
    add_bullet(
        doc,
        "Research Diagnostics: Compared Matérn and generalized Cauchy dependence models and designed "
        "scale- and frequency-resolved tests to identify misspecification in smoothness, noise, and dependence.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Research Diagnostics:",
    )
    add_bullet(
        doc,
        "Data and Workflow Engineering: Built modular Python/PyTorch pipelines for quality filtering, "
        "missing-data handling, nearest-neighbor construction, dependency-aware ordering, irregular-grid "
        "resampling, cached outputs, and restartable HPC execution.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Data and Workflow Engineering:",
    )
    add_bullet(
        doc,
        "Estimation and Scalability: Implemented GLS profiling, L-BFGS optimization, Monte Carlo simulation, "
        "and CPU/GPU execution for repeatable large-scale model estimation and evaluation.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
        bold_lead="Estimation and Scalability:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation "
        "rules, and benchmark-driven performance analysis.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = BLACKROCK / "Joonwon_Lee_BlackRock_BII_Portfolio_Research_Associate_Resume.docx"
    doc.save(out)
    return out


def build_blackrock_sae_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.7, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - BlackRock Systematic Active Equity Quantitative Researcher Resume",
        "Targeted resume for Associate, Quantitative Researcher - Systematic Active Equity at BlackRock",
        "systematic investing, quantitative research, hypothesis testing, econometrics, machine learning, simulation, "
        "large datasets, research pipelines, model evaluation, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in hypothesis-driven empirical research, "
        "scalable likelihood-based modeling, econometrics, machine learning, simulation, and statistical diagnostics. "
        "Builds reproducible Python/PyTorch pipelines for large, complex datasets and evaluates when apparent signals "
        "reflect persistent structure versus noise or model misspecification. Brings financial-services experience and "
        "a strong interest in systematic equity research.",
        9.8,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, C++/pybind11 integration, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Research: ",
        "Hypothesis testing, econometrics, regression, GLM/GBM, machine learning, stochastic processes, MLE/GLS, "
        "Monte Carlo simulation, numerical optimization, spectral analysis, feature attribution, model selection, "
        "validation, sensitivity analysis, and performance evaluation",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, converting a "
        "monitoring question into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated predictive performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Markets", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy using market-based fair-value estimation, inventory-aware "
        "quoting, rolling-window exposure controls, and benchmark-driven performance analysis.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Statistical Research and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Statistical Modeling: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with ordered local conditioning to "
        "fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Statistical Modeling:",
    )
    add_bullet(
        doc,
        "Hypothesis Evaluation and Diagnostics: Designed simulation studies and scale- and frequency-resolved tools "
        "to distinguish persistent dependence from low-frequency misspecification, high-frequency noise, and data artifacts.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Hypothesis Evaluation and Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Pipeline: Built reusable Python/PyTorch workflows for data-quality filtering, missingness tracking, "
        "time-dependent alignment, estimation, simulation, cached outputs, and restartable CPU/GPU and HPC execution; "
        "integrated compiled C++ routines through pybind11.",
        num_id,
        9.15,
        after=0.55,
        line=1.0,
        bold_lead="Research Pipeline:",
    )
    add_tabbed_line(
        doc,
        "Econometrics Project - Earned Income Tax Credit and Household Labor Supply",
        "Graduate Research",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the effect of the Earned Income Tax Credit on household labor supply using fixed-effects "
        "regression and propensity-score matching with observed household covariates.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Linear Algebra",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = BLACKROCK_SAE / "Joonwon_Lee_BlackRock_SAE_Resume.docx"
    doc.save(out)
    return out


def build_upstart_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Upstart Applied Scientist Resume",
        "Targeted resume for Applied Scientist, Unsecured Underwriting Machine Learning at Upstart",
        "applied scientist, underwriting machine learning, predictive modeling, supervised learning, "
        "model validation, feature engineering, financial risk, Python, LightGBM",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and applied quantitative scientist with experience developing and "
        "validating predictive and statistical models for financial risk, insurance pricing, and large, "
        "complex datasets. Builds reproducible Python/PyTorch and LightGBM workflows, designs rigorous "
        "performance and reliability evaluations, and translates research findings into implemented analyses "
        "and clear recommendations for business stakeholders.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Machine Learning: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), Linux, Git, AWS EC2, HPC/SLURM; "
        "supervised learning, GLM/GBM, feature engineering, feature attribution, numerical optimization, "
        "reproducible model-development workflows",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Statistical Modeling and Validation: ",
        "Probability, regression (GLM/GLS), hypothesis and sequential testing, Monte Carlo simulation, "
        "model selection and validation, predictive-performance evaluation, sensitivity analysis, "
        "missing-data and model-misspecification diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test framework for binary risk indicators, "
        "modeling the cumulative log-likelihood-ratio process as a random walk with statistically controlled "
        "decision boundaries.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation and dynamic-programming probability propagation to evaluate "
        "boundary-crossing probabilities, stopping-time distributions, Type I/II error, and early-decision "
        "tradeoffs, documenting model behavior and limitations.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ property-risk records, including "
        "data preparation, feature processing, GLM benchmarking, model validation, and predictive-performance "
        "evaluation on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed model performance and feature-attribution patterns and presented pricing and "
        "risk-segmentation recommendations to business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Applied Statistical Research", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development and Reliability Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Model Development: Developed a scalable Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with fixed-budget conditioning to fit up "
        "to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Development:",
    )
    add_bullet(
        doc,
        "Validation and Reliability: Compared Matérn and generalized Cauchy specifications and designed "
        "scale- and frequency-resolved tests to identify misspecification in smoothness, noise, and dependence.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Validation and Reliability:",
    )
    add_bullet(
        doc,
        "Data Quality and Robustness: Built workflows for observation masking, quality filtering, "
        "missing-data handling, nearest-neighbor construction, dependency-aware ordering, and irregular-grid "
        "resampling; quantified the effect of data imperfections on estimates.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Data Quality and Robustness:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS "
        "optimization, Monte Carlo simulation, CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.2,
        after=0.7,
        line=1.0,
        bold_lead="Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Linear Algebra, Statistical Computing, Stochastic Processes, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = UPSTART / "Joonwon_Lee_Upstart_Applied_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_akuna_capital_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Akuna Capital Junior Quantitative Researcher 2027 Resume",
        "Targeted resume for Junior Quantitative Researcher, 2027 at Akuna Capital",
        "quantitative research, market making, stochastic processes, probability, simulation, optimization, "
        "statistical modeling, trading strategies, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in stochastic processes, "
        "likelihood-based inference, Monte Carlo simulation, numerical optimization, and scalable statistical "
        "modeling. Develops Python/PyTorch research prototypes and diagnostics for large, noisy dependent data, "
        "with finance-sector experience and demonstrated interest in market making and systematic trading.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), Linux, Git, AWS EC2, HPC/SLURM; "
        "CPU/GPU computing, reproducible research pipelines, algorithm development",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Probability, stochastic processes, MLE/GLS, Monte Carlo simulation, numerical optimization, "
        "Gaussian processes, time-series and spatiotemporal modeling, spectral analysis, hypothesis and "
        "sequential testing, predictive modeling, model validation",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test framework for binary risk indicators, "
        "modeling the cumulative log-likelihood-ratio process as a random walk with statistically controlled "
        "decision boundaries.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation and dynamic-programming probability propagation to compute "
        "boundary-crossing probabilities, stopping-time distributions, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model validation, and predictive-performance evaluation on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed model performance and feature-attribution patterns and presented pricing and "
        "risk-segmentation recommendations to business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation "
        "rules, and benchmark-driven performance analysis.",
        num_id,
        9.2,
        after=0.65,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Statistical Modeling: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with fixed-budget conditioning "
        "to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Statistical Modeling:",
    )
    add_bullet(
        doc,
        "Simulation and Diagnostics: Compared Matérn and generalized Cauchy models and designed scale- and "
        "frequency-resolved tests to identify misspecification in smoothness, noise, and dependence.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Simulation and Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS "
        "optimization, Monte Carlo simulation, CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
        bold_lead="Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = AKUNA_CAPITAL / "Joonwon_Lee_Akuna_Capital_Junior_Quantitative_Researcher_2027_Resume.docx"
    doc.save(out)
    return out


def build_jpmorgan_spg_qtr_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - JPMorgan SPG Quantitative Trading and Research Associate Resume",
        "Targeted resume for the Securitized Products Group Quantitative Trading and Research team at JPMorgan Chase",
        "stochastic modeling, model diagnostics, model validation, data processing, root-cause analysis, "
        "Python, C++ integration, machine learning, market making, risk analytics",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher with experience in stochastic modeling, "
        "likelihood-based inference, statistical diagnostics, model validation, and large-scale data processing. "
        "Builds reproducible Python/PyTorch workflows for noisy, irregular dependent data, integrates compiled "
        "C++ components where needed, and investigates the root causes of unexpected model and data behavior.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, "
        "Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable data/modeling pipelines",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, likelihood-based inference, Gaussian processes, Monte Carlo simulation, "
        "numerical optimization, regression and machine learning, hypothesis testing, spectral analysis, "
        "model diagnostics and validation",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test framework for binary risk indicators, "
        "translating a monitoring problem into explicit hypotheses, likelihood-ratio decision boundaries, "
        "and statistically controlled stopping rules.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation and dynamic-programming probability propagation to quantify "
        "boundary-crossing probabilities, stopping-time behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Inference: Developed an advection-aware Vecchia approximation that factorizes a large "
        "Gaussian-process likelihood into ordered local conditional models, replacing dense covariance "
        "operations to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Scalable Inference:",
    )
    add_bullet(
        doc,
        "Data Processing and Root-Cause Analysis: Identified systematic measurement geometry, including a "
        "cross-scan latitude slope with distance from instrument nadir and hour-specific westward grid drift; "
        "applied time-specific longitude offsets, quality filtering, missingness tracking, and KD-tree center matching.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Data Processing and Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Inference-Driven Diagnostics: Designed scale- and frequency-resolved tools to pinpoint whether model "
        "misspecification arises from low-frequency structure, high-frequency behavior, or space-time dependence.",
        num_id,
        9.15,
        after=0.5,
        line=1.0,
        bold_lead="Inference-Driven Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Engineering: Integrated compiled C++ max-min ordering routines into Python through pybind11 "
        "and built reusable Python/PyTorch workflows with CPU/GPU execution, cached artifacts, and restartable HPC jobs.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy combining market-based fair-value estimation with "
        "inventory-aware quoting and rolling-window soft/hard liquidation controls.",
        num_id,
        9.15,
        after=0.65,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = JPMORGAN_SPG_QTR / "Joonwon_Lee_JPMorgan_SPG_QTR_Associate_Resume.docx"
    doc.save(out)
    return out


def build_goldman_core_planning_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Goldman Sachs Core Planning and Analysis Quantitative Strategist Resume",
        "Targeted resume for Associate Quantitative Strategist, Core Planning and Analysis Strats at Goldman Sachs",
        "quantitative strategy, stochastic modeling, time-series analysis, simulation, uncertainty quantification, "
        "model validation, causal inference, Python, C++ integration, cloud analytics",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in stochastic modeling, scalable "
        "likelihood-based inference, simulation, statistical diagnostics, and end-to-end model validation. Builds "
        "reproducible Python/PyTorch workflows for large, time-dependent datasets, integrates compiled C++ "
        "components for performance-critical routines, and translates analytical problems into testable model specifications.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, "
        "Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable analytical pipelines",
        9.3,
        after=0.35,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, time-series and spatiotemporal modeling, Gaussian processes, MLE/GLS, Monte Carlo "
        "simulation, numerical optimization, regression and machine learning, causal inference, hypothesis testing, "
        "spectral analysis, model selection, diagnostics, and validation",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test framework for binary risk indicators, "
        "translating a monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and "
        "statistically controlled decision rules.",
        num_id,
        9.25,
        after=0.55,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping-time behavior, Type I/II error, and model limitations.",
        num_id,
        9.25,
        after=0.65,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.55,
        line=1.0,
    )
    add_bullet(
        doc,
        "Presented model diagnostics and feature-attribution findings as pricing and risk-segmentation "
        "recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.65,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Projects", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Time-Dependent Modeling: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, using ordered conditional models to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Time-Dependent Modeling:",
    )
    add_bullet(
        doc,
        "Simulation and Model Diagnostics: Designed controlled simulation studies and scale- and frequency-resolved "
        "diagnostics to identify misspecification in low-frequency structure, high-frequency behavior, noise, and dependence.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Simulation and Model Diagnostics:",
    )
    add_bullet(
        doc,
        "Data and Model Lifecycle: Built workflows for quality filtering, missingness tracking, time-dependent grid "
        "alignment, regular-grid construction, estimation, optimization, sensitivity analysis, cached outputs, and restartable HPC execution.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Data and Model Lifecycle:",
    )
    add_bullet(
        doc,
        "Research Engineering: Integrated compiled C++ max-min ordering routines into Python through pybind11 and "
        "implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS optimization, and CPU/GPU execution.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "Econometrics and Causal Inference Project - Earned Income Tax Credit and Labor Supply",
        "Graduate Research",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the effect of the Earned Income Tax Credit on household labor supply using fixed-effects "
        "regression and propensity-score matching, with explicit treatment, outcome, comparison-group, and confounding assumptions.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = GOLDMAN_CORE_PLANNING / "Joonwon_Lee_Goldman_Sachs_Core_Planning_Quantitative_Strategist_Resume.docx"
    doc.save(out)
    return out


def build_walleye_single_stock_vol_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.60, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.6, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Walleye Capital Single Stock Volatility Quantitative Researcher Resume",
        "Targeted resume for Quantitative Researcher, Single Stock Volatility at Walleye Capital",
        "quantitative research, volatility, market making, financial time series, stochastic modeling, "
        "machine learning, statistical diagnostics, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in stochastic processes, time-dependent "
        "modeling, scalable likelihood-based inference, simulation, machine learning, and statistical diagnostics. "
        "Builds reproducible Python/PyTorch research pipelines and has hands-on experience with fair-value-driven "
        "market making, inventory-aware quoting, and systematic risk controls.",
        9.6,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.5,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, "
        "Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing, reusable and restartable pipelines",
        9.3,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, time-series and spatiotemporal modeling, Gaussian processes, MLE/GLS, Monte Carlo "
        "simulation, numerical optimization, regression and machine learning, spectral analysis, model diagnostics and validation",
        9.3,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Trading Research: ",
        "Fair-value estimation, market making, inventory-aware quoting, rolling-window risk monitoring, and "
        "soft/hard liquidation controls",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test framework for binary risk indicators, "
        "translating a monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and statistically controlled stopping rules.",
        num_id,
        9.2,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.2,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed model performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy combining market-based fair-value estimation with "
        "inventory-aware quoting and rolling-window soft/hard liquidation controls.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Stochastic Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with ordered local conditioning to fit up to "
        "145,008 observations per day.",
        num_id,
        9.1,
        after=0.45,
        line=1.0,
        bold_lead="Stochastic Modeling:",
    )
    add_bullet(
        doc,
        "Simulation and Diagnostics: Designed scale- and frequency-resolved tools to identify whether model "
        "misspecification arises from low-frequency structure, high-frequency behavior, noise, or dependence.",
        num_id,
        9.1,
        after=0.45,
        line=1.0,
        bold_lead="Simulation and Diagnostics:",
    )
    add_bullet(
        doc,
        "Data and Research Engineering: Built Python/PyTorch pipelines for quality filtering, missingness tracking, "
        "time-dependent alignment, estimation, and HPC execution; integrated compiled C++ ordering routines through pybind11.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
        bold_lead="Data and Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        8.95,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = WALLEYE_SINGLE_STOCK_VOL / "Joonwon_Lee_Walleye_Single_Stock_Volatility_Quantitative_Researcher_Resume.docx"
    doc.save(out)
    return out


def build_sentilink_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.70, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.8, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - SentiLink Quantitative Researcher Resume",
        "Targeted resume for Quantitative Researcher, PhD New Grad at SentiLink",
        "applied machine learning, financial risk, model development, data quality, root-cause analysis, "
        "statistical testing, Python, scalable research engineering",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and applied quantitative researcher with experience developing and validating "
        "machine-learning and statistical models for financial risk, insurance pricing, and large, noisy dependent "
        "data. Builds reproducible Python/PyTorch workflows spanning data quality, feature processing, model training, "
        "simulation, diagnostics, and monitoring, and translates ambiguous problems into testable model specifications "
        "and clear recommendations.",
        9.9,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, C++/pybind11 integration, "
        "Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable analytical pipelines",
        9.5,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Machine Learning and Statistics: ",
        "Supervised learning, GLM/GBM, regression, feature processing and attribution, model selection and validation, "
        "hypothesis testing, Monte Carlo simulation, stochastic processes, data-quality diagnostics, and numerical optimization",
        9.5,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.4,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and model limitations.",
        num_id,
        9.4,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.4,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed model performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.4,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Applied Research and Engineering", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development, Data Quality, and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, replacing dense covariance operations with ordered local conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.3,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Data Quality and Root-Cause Analysis: Built workflows for quality filtering, missingness tracking, "
        "time-dependent coordinate offsets, nearest-neighbor center matching, and regular-grid construction to "
        "separate systematic measurement artifacts from statistical model failures.",
        num_id,
        9.3,
        after=0.45,
        line=1.0,
        bold_lead="Data Quality and Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Model Evaluation and Monitoring: Designed scale- and frequency-resolved diagnostics to localize whether "
        "misspecification arises from low-frequency structure, high-frequency behavior, noise, or dependence.",
        num_id,
        9.3,
        after=0.45,
        line=1.0,
        bold_lead="Model Evaluation and Monitoring:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch pipelines with CPU/GPU execution, cached "
        "artifacts, and restartable HPC jobs; integrated compiled C++ ordering routines through pybind11.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Linear Algebra, Econometrics",
        9.15,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.15,
        after=0,
    )

    out = SENTILINK / "Joonwon_Lee_SentiLink_Quantitative_Researcher_Resume.docx"
    doc.save(out)
    return out


def build_teza_technologies_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.7, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Teza Technologies Quantitative Researcher PhD Resume",
        "Targeted resume for Quantitative Researcher, PhD at Teza Technologies",
        "quantitative research, systematic trading, market making, stochastic modeling, time-series analysis, "
        "hypothesis testing, machine learning, research infrastructure, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in time-dependent stochastic modeling, "
        "scalable likelihood-based inference, hypothesis testing, simulation, machine learning, and statistical "
        "diagnostics. Builds reproducible Python/PyTorch research infrastructure for large, granular datasets and has "
        "hands-on market-making strategy experience with fair-value estimation, inventory-aware quoting, and systematic risk controls.",
        9.8,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, "
        "Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Research: ",
        "Stochastic processes, time-series and spatiotemporal modeling, Gaussian processes, MLE/GLS, Monte Carlo "
        "simulation, numerical optimization, regression and machine learning, spectral methods, hypothesis testing, "
        "feature evaluation, model diagnostics and validation",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Trading Research: ",
        "Fair-value estimation, market making, inventory-aware quoting, rolling-window risk monitoring, and "
        "soft/hard liquidation controls",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and statistically controlled stopping rules.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Applied Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy combining market-based fair-value estimation with "
        "inventory-aware quoting.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Implemented rolling-window exposure controls that penalized persistent inventory and triggered soft/hard "
        "liquidation to limit position risk.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with ordered local conditioning to fit up to "
        "145,008 observations per day.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Frequency-Resolved Model Diagnostics: Designed scale- and frequency-resolved tools to identify whether observed "
        "model failure arises from low-frequency structure, high-frequency behavior, noise, or dependence.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Frequency-Resolved Model Diagnostics:",
    )
    add_bullet(
        doc,
        "Granular Data Pipeline: Built quality-filtering, missingness-tracking, time-dependent alignment, and "
        "nearest-neighbor grid-matching workflows to correct systematic measurement and geometry shifts.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Granular Data Pipeline:",
    )
    add_bullet(
        doc,
        "Research Infrastructure: Built reusable Python/PyTorch workflows for estimation, simulation, and restartable "
        "CPU/GPU and HPC execution; integrated compiled C++ ordering routines "
        "through pybind11.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
        bold_lead="Research Infrastructure:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = TEZA_TECHNOLOGIES / "Joonwon_Lee_Teza_Quantitative_Researcher_PhD_Resume.docx"
    doc.save(out)
    return out


def build_freddie_mac_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.72, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.9, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Freddie Mac Quantitative Analytics Senior Resume",
        "Targeted resume for Quantitative Analytics Senior, Single Family Modeling at Freddie Mac",
        "predictive modeling, property risk, financial risk, econometrics, machine learning, Monte Carlo simulation, "
        "optimization, model development, model validation, data processing, Python",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative modeler with experience developing and validating predictive and statistical "
        "models for financial-risk monitoring, commercial property insurance pricing, and large dependent datasets. "
        "Combines GLM/GBM, econometrics, Monte Carlo simulation, optimization, and statistical diagnostics with "
        "reproducible Python/PyTorch workflows, translating business and risk questions into independently developed, testable models.",
        10.0,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.8,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.8,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Data: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, C++/pybind11 integration, Linux, "
        "Git, AWS EC2, HPC/SLURM; data cleaning, feature processing, missingness handling, and CPU/GPU computing",
        9.55,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Modeling and Analytics: ",
        "Predictive modeling, GLM/GBM, econometrics, machine learning, MLE/GLS, Monte Carlo simulation, numerical "
        "optimization, hypothesis testing, stochastic processes, model selection, validation, and performance diagnostics",
        9.55,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.55,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "business monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.45,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and Monte Carlo "
        "simulation to evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and model limitations.",
        num_id,
        9.45,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.55,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ commercial property-risk records, including "
        "data preparation, feature processing, GLM benchmarking, model selection, validation, and diagnostics on AWS EC2.",
        num_id,
        9.45,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated predictive performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.45,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Modeling and Research", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Independent Model Development: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with ordered local conditioning to "
        "fit up to 145,008 observations per day.",
        num_id,
        9.35,
        after=0.45,
        line=1.0,
        bold_lead="Independent Model Development:",
    )
    add_bullet(
        doc,
        "Data Processing and Cleaning: Built workflows for quality filtering, missingness tracking, time-dependent "
        "coordinate offsets, nearest-neighbor center matching, and regular-grid construction to correct systematic measurement shifts.",
        num_id,
        9.35,
        after=0.45,
        line=1.0,
        bold_lead="Data Processing and Cleaning:",
    )
    add_bullet(
        doc,
        "Model Evaluation: Designed simulation studies and frequency-resolved diagnostics to identify whether "
        "misspecification arises from low-frequency structure, high-frequency behavior, noise, or dependence.",
        num_id,
        9.35,
        after=0.45,
        line=1.0,
        bold_lead="Model Evaluation:",
    )
    add_bullet(
        doc,
        "Scalable Research Engineering: Implemented reusable Python/PyTorch pipelines with GLS profiling, L-BFGS "
        "optimization, CPU/GPU execution, cached artifacts, and restartable HPC jobs; integrated C++ routines through pybind11.",
        num_id,
        9.35,
        after=0.6,
        line=1.0,
        bold_lead="Scalable Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "Econometrics Project - Earned Income Tax Credit and Household Labor Supply",
        "Graduate Research",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the effect of the Earned Income Tax Credit on household labor supply using fixed-effects "
        "regression and propensity-score matching with observed household covariates.",
        num_id,
        9.35,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Econometrics, Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Linear Algebra",
        9.2,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.2,
        after=0,
    )

    out = FREDDIE_MAC / "Joonwon_Lee_Freddie_Mac_Resume.docx"
    doc.save(out)
    return out


def build_radix_trading_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.7, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Radix Trading Quantitative Researcher Resume",
        "Targeted resume for Quantitative Researcher, Full-Time PhD+ at Radix Trading",
        "quantitative research, probabilistic thinking, signal and noise, hypothesis testing, simulation, "
        "market making, research framework, Python, C++ integration",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in probabilistic modeling, scalable "
        "likelihood-based inference, hypothesis testing, simulation, and signal-versus-noise diagnostics. Builds "
        "reproducible Python/PyTorch research frameworks for large, complex datasets, rapidly tests empirical ideas, "
        "and adapts models when evidence reveals noise, misspecification, or changing dependence. Brings hands-on "
        "market-making strategy experience and a strong interest in research-driven trading.",
        9.8,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Research: ",
        "Probability and stochastic processes, hypothesis testing, MLE/GLS, Monte Carlo simulation, numerical "
        "optimization, Gaussian processes, time-series and spatiotemporal modeling, regression and machine learning, "
        "spectral analysis, model selection, validation, and sensitivity analysis",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, converting a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and translated findings into pricing and "
        "risk-segmentation recommendations for business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy using market-based fair-value estimation and inventory-aware quoting.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Implemented rolling-window exposure controls that penalized persistent inventory and triggered soft/hard "
        "liquidation to limit position risk.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Probabilistic Modeling and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Probabilistic Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, using ordered conditional models to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Probabilistic Modeling:",
    )
    add_bullet(
        doc,
        "Signal-versus-Noise Diagnostics: Designed simulation studies and scale- and frequency-resolved tools to "
        "determine whether apparent structure reflects persistent dependence, noise, missingness, or model misspecification.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Signal-versus-Noise Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Framework: Built reusable Python/PyTorch workflows for data-quality filtering, missingness tracking, "
        "time-dependent alignment, estimation, simulation, cached outputs, and restartable CPU/GPU and HPC execution.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Research Framework:",
    )
    add_bullet(
        doc,
        "Performance Engineering: Integrated compiled C++ max-min ordering routines into the Python research pipeline "
        "through pybind11 for performance-critical dependency ordering.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
        bold_lead="Performance Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = RADIX_TRADING / "Joonwon_Lee_Radix_Resume.docx"
    doc.save(out)
    return out


def build_3red_partners_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.7, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - 3Red Partners Graduate Quantitative Researcher Resume",
        "Targeted resume for Graduate Quantitative Researcher, 2027 Incoming Graduate at 3Red Partners",
        "quantitative research, trading strategies, statistical modeling, machine learning, hypothesis testing, "
        "Python, C++ integration, research pipelines, performance evaluation",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in probabilistic modeling, hypothesis "
        "testing, simulation, machine learning, and signal-versus-noise diagnostics. Builds reproducible Python/PyTorch "
        "research pipelines for large datasets, integrates compiled C++ routines for performance-critical components, "
        "and evaluates models through controlled experiments and performance diagnostics. Brings two financial-services "
        "internships and hands-on market-making strategy experience.",
        9.8,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.65,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Probability and stochastic processes, hypothesis testing, MLE/GLS, Monte Carlo simulation, numerical "
        "optimization, time-series and spatiotemporal modeling, regression, GLM/GBM, machine learning, spectral "
        "analysis, model selection, validation, feature attribution, and performance evaluation",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, converting a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and translated findings into pricing and "
        "risk-segmentation recommendations for business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy using market-based fair-value estimation and inventory-aware quoting.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Implemented rolling-window exposure controls that penalized persistent inventory and triggered soft/hard "
        "liquidation to limit position risk.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Probabilistic Modeling and Research Systems",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, using ordered conditional models to fit up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Empirical Testing and Diagnostics: Designed simulation studies and scale- and frequency-resolved tools to "
        "determine whether apparent structure reflects persistent dependence, noise, missingness, or misspecification.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Empirical Testing and Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Infrastructure: Built reusable Python/PyTorch workflows for data-quality filtering, time-dependent "
        "alignment, estimation, simulation, cached outputs, and restartable CPU/GPU and HPC execution.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Research Infrastructure:",
    )
    add_bullet(
        doc,
        "C++ Integration: Integrated compiled C++ max-min ordering routines into the Python pipeline through pybind11 "
        "for performance-critical dependency ordering.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
        bold_lead="C++ Integration:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = THREE_RED_PARTNERS / "Joonwon_Lee_3Red_Resume.docx"
    doc.save(out)
    return out


def build_openai_safety_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.55, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - OpenAI Data Scientist, Safety Resume",
        "Targeted resume for Data Scientist, Safety at OpenAI",
        "statistical testing, uncertainty quantification, causal inference, observational analysis, false positives, "
        "false negatives, noisy data, root-cause diagnostics, Python, machine learning",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist specializing in rigorous statistical testing, causal and "
        "observational analysis, uncertainty quantification, and diagnostics for noisy, incomplete data. Develops "
        "Python/PyTorch analytical systems that translate ambiguous risk questions into measurable hypotheses, "
        "quantify false-positive/false-negative tradeoffs, and distinguish systematic data artifacts from model "
        "failures. Experience communicating model evidence and limitations in financial-risk and insurance settings. "
        "Available to begin full-time employment in February 2027.",
        9.65,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.55,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.55,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Data: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, C++/pybind11 integration, Linux, "
        "Git, AWS EC2, HPC/SLURM; data cleaning, missing-data workflows, nearest-neighbor matching, and large-scale pipelines",
        9.25,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Statistical and Causal Methods: ",
        "Sequential hypothesis testing, Type I/II error analysis, uncertainty quantification, Monte Carlo simulation, "
        "fixed-effects regression, propensity-score matching, GLM/GBM, machine learning, model validation, diagnostics, "
        "and performance evaluation",
        9.25,
        after=0.35,
    )

    add_section_heading(doc, "Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating an "
        "ambiguous monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled decisions.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "quantify boundary-crossing and stopping-time uncertainty, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Translated performance diagnostics and feature-attribution findings into pricing and risk-segmentation "
        "recommendations for business stakeholders, including clear communication of model limitations.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Statistical Research and Applied Analysis", compact=True)
    add_tabbed_line(
        doc,
        "Longitudinal Observational Study - EITC and Household Labor Supply",
        "2021",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the relationship between Earned Income Tax Credit exposure and household labor supply using "
        "fixed-effects regression and propensity-score matching with household income and demographic covariates; "
        "defined treatment and comparison groups and interpreted results under observational-study assumptions.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Data Quality, Root-Cause Diagnostics, and Scalable Modeling",
        "Sep 2024 - Present",
        width,
        9.3,
        after=0.35,
    )
    add_bullet(
        doc,
        "Messy-Data Pipeline: Built observation masks, quality filters, spatial-distance thresholds, hour-specific "
        "longitude offsets, and nearest-neighbor center matching to map incomplete satellite observations onto a regular grid.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Messy-Data Pipeline:",
    )
    add_bullet(
        doc,
        "Root-Cause Diagnostics: Distinguished random measurement noise from systematic acquisition artifacts, including "
        "scan-geometry trends with distance from instrument nadir and hour-specific coordinate shifts, before attributing "
        "remaining discrepancies to statistical-model misspecification.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Root-Cause Diagnostics:",
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, using ordered conditional models to fit up to 145,008 observations per day in reusable "
        "Python/PyTorch CPU/GPU and HPC workflows.",
        num_id,
        9.05,
        after=0.6,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Econometrics",
        8.95,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = OPENAI_SAFETY / "Joonwon_Lee_OpenAI_Safety_Resume.docx"
    doc.save(out)
    return out


def build_capital_one_model_risk_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.65, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Capital One Model Risk Audit Resume",
        "Targeted resume for Senior Associate, Data Scientist - Model Risk Audit at Capital One",
        "model development, model validation, model risk, machine learning, statistical testing, Python, AWS, "
        "large-scale data, performance diagnostics, stakeholder communication",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist with experience in statistical model development, independent "
        "validation, hypothesis testing, and machine learning across financial risk, insurance pricing, and large-scale "
        "dependent data. Builds reproducible Python workflows on AWS and HPC, evaluates model assumptions and performance, "
        "and translates technical findings into clear business and model-risk recommendations.",
        9.75,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, R, Linux, Git, AWS EC2, HPC/SLURM; "
        "CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.35,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Data Science and Model Risk: ",
        "Statistical model development and validation, hypothesis testing, MLE/GLS, Monte Carlo simulation, regression, "
        "GLM/GBM, machine learning, time-series and spatiotemporal modeling, model selection, diagnostics, feature "
        "attribution, and performance evaluation",
        9.35,
        after=0.35,
    )

    add_section_heading(doc, "Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and independently validated a Sequential Probability Ratio Test for binary risk indicators, "
        "translating a monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled decisions.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs; "
        "documented model assumptions, performance, and limitations.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and presented findings as pricing and "
        "risk-segmentation recommendations for business stakeholders.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Research and Engineering", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Large-Scale Model Development: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with ordered conditional models to fit "
        "up to 145,008 observations per day.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Large-Scale Model Development:",
    )
    add_bullet(
        doc,
        "Data Quality and Processing: Built observation masks, quality filters, spatial-distance thresholds, "
        "hour-specific coordinate offsets, and nearest-neighbor matching to convert incomplete, irregular satellite "
        "observations into analysis-ready grids.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Data Quality and Processing:",
    )
    add_bullet(
        doc,
        "Model Validation and Root-Cause Analysis: Designed scale- and frequency-resolved diagnostics to distinguish "
        "systematic acquisition artifacts and measurement noise from covariance-model misspecification and to identify "
        "where model assumptions fail.",
        num_id,
        9.15,
        after=0.45,
        line=1.0,
        bold_lead="Model Validation and Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch workflows with GLS profiling, L-BFGS optimization, "
        "CPU/GPU execution, cached artifacts, and restartable HPC jobs for estimation, simulation, and diagnostics.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "Longitudinal Observational Study - EITC and Household Labor Supply",
        "2021",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the relationship between Earned Income Tax Credit exposure and household labor supply using "
        "fixed-effects regression and propensity-score matching with household income and demographic covariates; "
        "evaluated comparison-group construction, confounding assumptions, and interpretation of model results.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Econometrics",
        9.0,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = CAPITAL_ONE_MODEL_RISK / "Joonwon_Lee_Capital_One_Resume.docx"
    doc.save(out)
    return out


def build_capital_one_principal_quant_modeler_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.65, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Capital One Principal Quantitative Modeler Resume",
        "Targeted resume for Principal Quantitative Modeler, Credit Risk Management Modeling at Capital One",
        "credit risk modeling, econometrics, machine learning, panel data, time series, large datasets, model "
        "development, validation, monitoring, Python, R, SQL, AWS, stakeholder communication",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative modeler with experience developing and validating statistical "
        "and machine-learning models for financial risk, insurance pricing, and large dependent datasets. Combines "
        "econometrics, stochastic processes, likelihood-based inference, simulation, and model diagnostics with "
        "reproducible Python/PyTorch workflows on AWS and HPC. Translates business and monitoring questions into "
        "transparent model specifications, performance evidence, and actionable recommendations.",
        9.75,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), R, SQL, C++/pybind11 integration, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.3,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Statistical and Econometric Modeling: ",
        "Linear and logistic regression, GLM/GBM, time-series and spatiotemporal modeling, panel/longitudinal and "
        "cross-sectional analysis, fixed effects, propensity-score matching, machine learning, MLE/GLS, Monte Carlo "
        "simulation, hypothesis testing, model selection, validation, sensitivity analysis, and performance diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "quantify Type I/II error, boundary-crossing probabilities, stopping behavior, uncertainty, and early-decision "
        "tradeoffs; documented assumptions, performance, and limitations.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ property-risk records, including data "
        "preparation, feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated predictive performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Research and Quantitative Modeling", compact=True)
    add_tabbed_line(
        doc,
        "Longitudinal Observational Study - EITC and Household Labor Supply",
        "2021",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Analyzed the relationship between Earned Income Tax Credit exposure and household labor supply using panel "
        "data, fixed-effects regression, and propensity-score matching with household income and demographic covariates; "
        "evaluated comparison-group construction, confounding assumptions, and interpretation of results.",
        num_id,
        9.1,
        after=0.55,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Large-Scale Model Development: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, replacing dense covariance operations with ordered conditional models to fit "
        "up to 145,008 observations per day.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Large-Scale Model Development:",
    )
    add_bullet(
        doc,
        "Data Engineering: Built quality filters, missingness tracking, time-specific coordinate offsets, "
        "nearest-neighbor center matching, and spatial-threshold regular-grid construction for incomplete, irregular data.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Data Engineering:",
    )
    add_bullet(
        doc,
        "Model Diagnostics and Monitoring: Designed scale- and frequency-resolved tools to distinguish systematic "
        "data artifacts and measurement noise from model misspecification and identify where assumptions fail.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Model Diagnostics and Monitoring:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch workflows with numerical optimization, CPU/GPU "
        "execution, cached artifacts, and restartable HPC jobs for estimation, simulation, and diagnostics; integrated "
        "compiled C++ ordering routines through pybind11.",
        num_id,
        9.05,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.95,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.95,
        after=0,
    )

    out = CAPITAL_ONE_QUANT_MODELER / "Joonwon_Lee_Capital_One_Principal_Quantitative_Modeler_Resume.docx"
    doc.save(out)
    return out


def build_wells_fargo_qa_2027_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.65, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Wells Fargo 2027 Quantitative Analytics Program Resume",
        "Targeted resume for the 2027 Quantitative Analytics Program - Capital Markets (PhD) at Wells Fargo",
        "capital markets, quantitative analytics, model development, model validation, Monte Carlo simulation, "
        "pricing and risk models, Python, C++ integration, scalable computing, stochastic modeling",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in stochastic modeling, likelihood-based "
        "inference, Monte Carlo simulation, hypothesis testing, and independent model validation. Develops scalable "
        "Python/PyTorch analytical systems for complex dependent data and communicates model assumptions, performance, "
        "and risk tradeoffs clearly. Brings financial model-risk and insurance-pricing experience with a strong interest "
        "in capital-markets pricing, portfolio risk, and quantitative research.",
        9.7,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, R, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.3,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Probability and stochastic processes, Monte Carlo simulation, hypothesis testing, MLE/GLS, numerical "
        "optimization, linear algebra, regression, GLM/GBM, machine learning, time-series and spatiotemporal modeling, "
        "model selection, validation, and performance diagnostics",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and independently validated a Sequential Probability Ratio Test for binary risk indicators, "
        "translating a monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled decisions.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs; "
        "documented model assumptions, performance, and limitations.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Translated model diagnostics and feature-attribution findings into pricing and risk-segmentation recommendations "
        "for business stakeholders.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Markets", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Model Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, replacing dense covariance operations with ordered conditional models to fit up to 145,008 "
        "observations per day.",
        num_id,
        9.1,
        after=0.4,
        line=1.0,
        bold_lead="Scalable Inference:",
    )
    add_bullet(
        doc,
        "Model Diagnostics: Designed scale- and frequency-resolved tools to identify whether discrepancies arise from "
        "low- or high-frequency behavior, missingness, measurement artifacts, or covariance-model misspecification.",
        num_id,
        9.1,
        after=0.4,
        line=1.0,
        bold_lead="Model Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Engineering: Built reusable Python/PyTorch workflows for data-quality filtering, optimization, "
        "simulation, cached outputs, and restartable CPU/GPU and HPC execution; integrated performance-critical C++ "
        "ordering routines through pybind11.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy using market-based fair-value estimation, inventory-aware "
        "quoting, rolling-window exposure controls, and dynamic liquidation rules to manage position risk.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, Advanced "
        "Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        9.0,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = WELLS_FARGO_QA_2027 / "Joonwon_Lee_Wells_Fargo_QA_2027_Resume.docx"
    doc.save(out)
    return out


def build_tower_research_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.65, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Tower Research Quantitative Trader Researcher 2027 Resume",
        "Targeted resume for 2027 Quantitative Trader / Researcher at Tower Research Capital",
        "quantitative research, stochastic modeling, simulation, statistical diagnostics, Python, C++ integration, "
        "large-scale data, spectral analysis, hypothesis testing, market making",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate specializing in scalable likelihood inference, stochastic modeling, and statistical "
        "diagnostics for dependent data. Builds reproducible Python/PyTorch research tools for large datasets, integrates "
        "compiled C++ through pybind11 for performance-critical computation, and evaluates models through hypothesis "
        "testing and simulation.",
        9.7,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, SQL, Linux, Git, "
        "AWS EC2, HPC/SLURM; CPU/GPU computing and restartable research pipelines",
        9.3,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Probability and stochastic processes, Monte Carlo simulation, hypothesis testing, MLE/GLS, numerical "
        "optimization, time-series and spatiotemporal modeling, spectral and autocorrelation analysis, regression, "
        "machine learning, model validation, and performance evaluation",
        9.3,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and translated findings into pricing and "
        "risk-segmentation recommendations for business stakeholders.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Statistical Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation for nonseparable space-time "
        "covariance structures, using ordered local conditional models for likelihood-based parameter estimation and "
        "uncertainty quantification with up to 145,008 observations per day.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Scalable Inference:",
    )
    add_bullet(
        doc,
        "Stochastic Diagnostics: Developed statistical diagnostic tools for dependent stochastic processes that localize "
        "where fitted models fail across scale and frequency, distinguishing persistent dependence from high-frequency "
        "noise, missingness, acquisition artifacts, and covariance misspecification.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Stochastic Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Infrastructure: Built reusable Python/PyTorch workflows for data-quality filtering, time-dependent "
        "alignment, optimization, simulation, and restartable CPU/GPU and HPC execution; integrated "
        "performance-critical C++ ordering routines through pybind11.",
        num_id,
        9.05,
        after=0.6,
        line=1.0,
        bold_lead="Research Infrastructure:",
    )
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and backtested an inventory-aware market-making strategy around empirically calibrated reference values; "
        "tracked position-limit saturation over a rolling 10-step window and triggered more aggressive inventory-reducing "
        "quotes to manage persistent exposure.",
        num_id,
        9.1,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, Advanced "
        "Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        9.0,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = TOWER_RESEARCH / "Joonwon_Lee_Tower_QTQR_2027_Resume.docx"
    doc.save(out)
    return out


def build_imc_quantitative_researcher_2027_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.68, margin_top=0.50, margin_bottom=0.50)
    configure_styles(doc, body_size=9.65, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - IMC Quantitative Researcher 2027 Resume",
        "Targeted resume for Quantitative Researcher at IMC Trading",
        "quantitative research, stochastic modeling, hypothesis testing, simulation, market making, Python, "
        "C++ integration, scalable likelihood inference, statistical diagnostics",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative researcher specializing in probabilistic modeling, hypothesis "
        "testing, simulation, scalable likelihood-based inference, and statistical diagnostics. Builds reproducible "
        "Python/PyTorch research systems for large dependent datasets and integrates compiled C++ for "
        "performance-critical computation. Demonstrated interest in market making through a top-2.39% finish in IMC "
        "Prosperity. Available for full-time employment in August 2027.",
        9.65,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.6,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, R, SQL, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable research pipelines",
        9.25,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Probability and stochastic processes, Monte Carlo simulation, hypothesis testing, MLE/GLS, numerical "
        "optimization, time-series and spatiotemporal modeling, spectral and autocorrelation analysis, regression, "
        "machine learning, model validation, and performance evaluation",
        9.25,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled sequential decisions.",
        num_id,
        9.15,
        after=0.4,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, stopping behavior, Type I/II error, and early-decision tradeoffs.",
        num_id,
        9.15,
        after=0.55,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
        "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.15,
        after=0.55,
        line=1.0,
    )

    add_section_heading(doc, "Quantitative Research and Trading", compact=True)
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy using market-based fair-value estimation and inventory-aware "
        "quoting; implemented rolling-window exposure controls and dynamic liquidation rules to manage position risk.",
        num_id,
        9.1,
        after=0.55,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Stochastic Modeling and Statistical Diagnostics",
        "Sep 2024 - Present",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, using ordered local conditional models to fit up to 145,008 observations per day.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Scalable Inference:",
    )
    add_bullet(
        doc,
        "Empirical Testing: Designed scale- and frequency-resolved diagnostics to distinguish persistent dependence "
        "from high-frequency noise, missingness, acquisition artifacts, and model misspecification.",
        num_id,
        9.05,
        after=0.4,
        line=1.0,
        bold_lead="Empirical Testing:",
    )
    add_bullet(
        doc,
        "Research Tools: Built reusable Python/PyTorch workflows for data-quality filtering, time-dependent alignment, "
        "optimization, simulation, and restartable CPU/GPU and HPC execution; integrated performance-critical C++ "
        "ordering routines through pybind11.",
        num_id,
        9.05,
        after=0.55,
        line=1.0,
        bold_lead="Research Tools:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, Advanced "
        "Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        9.0,
        after=0.25,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = IMC_TRADING / "Joonwon_Lee_IMC_Quantitative_Researcher_2027_Resume.docx"
    doc.save(out)
    return out


def build_virtu_quantitative_strategist_resume():
    """Create the Virtu-targeted version from the established one-page quant layout."""
    source = IMC_TRADING / "Joonwon_Lee_IMC_Quantitative_Researcher_2027_Resume.docx"
    if not source.exists():
        raise FileNotFoundError(f"Run build_imc_quantitative_researcher_2027_resume() first: {source}")

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Virtu Quantitative Strategist Resume",
        "Targeted resume for Quantitative Strategist (PhD) at Virtu Financial",
        "quantitative research, predictive modeling, trading strategies, portfolio risk, stochastic modeling, "
        "hypothesis testing, simulation, Python, C++ integration, market making, statistical diagnostics",
    )

    replacements = {
        "Statistics Ph.D. candidate and quantitative researcher specializing": (
            "Statistics Ph.D. candidate and quantitative researcher specializing in scalable likelihood inference and "
            "statistical diagnostics. Builds reproducible Python/PyTorch research tools for large dependent datasets, "
            "integrates compiled C++ for performance-critical computation, and evaluates models under uncertainty."
        ),
        "Built and evaluated a market-making strategy": (
            "Built and backtested an inventory-aware market-making strategy around empirically calibrated reference "
            "values; tracked position-limit saturation over a rolling 10-step window and triggered more aggressive "
            "inventory-reducing quotes to manage persistent exposure."
        ),
    }

    for paragraph in doc.paragraphs:
        for prefix, new_text in replacements.items():
            if paragraph.text.startswith(prefix):
                first_run = paragraph.runs[0] if paragraph.runs else None
                for run in list(paragraph.runs):
                    paragraph._p.remove(run._r)
                if first_run is not None and first_run.bold and ": " in new_text:
                    lead, detail = new_text.split(": ", 1)
                    lead_run = paragraph.add_run(f"{lead}: ")
                    set_run_font(
                        lead_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=True,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                    detail_run = paragraph.add_run(detail)
                    set_run_font(
                        detail_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=False,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                else:
                    new_run = paragraph.add_run(new_text)
                    if first_run is not None:
                        set_run_font(
                            new_run,
                            first_run.font.size.pt if first_run.font.size else 9.15,
                            bold=bool(first_run.bold),
                            italic=bool(first_run.italic),
                            color=first_run.font.color.rgb or BLACK,
                        )
                break

    out = VIRTU_FINANCIAL / "Joonwon_Lee_Virtu_Quantitative_Strategist_Resume.docx"
    doc.save(out)
    return out


def build_barclays_electronic_trading_resume():
    """Create a targeted one-page resume for Barclays' 2027 Electronic Trading program."""
    source = VIRTU_FINANCIAL / "Joonwon_Lee_Virtu_Quantitative_Strategist_Resume.docx"
    if not source.exists():
        raise FileNotFoundError(f"Run build_virtu_quantitative_strategist_resume() first: {source}")

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Barclays Electronic Trading Graduate Associate Resume",
        "Targeted resume for the Barclays Electronic Trading Associate Graduate Program 2027",
        "electronic trading, quantitative research, statistical modeling, machine learning, Python, C++ integration, "
        "model implementation, simulation, market making, scalable research tools",
    )

    replacements = {
        "Statistics Ph.D. candidate and quantitative researcher specializing": (
            "Statistics Ph.D. candidate specializing in scalable likelihood inference and statistical diagnostics "
            "for stochastically dependent data. Develops mathematical models and reproducible Python/PyTorch research "
            "tools, with C++/pybind11 integration for performance-critical computation."
        ),
        "Developed an end-to-end LightGBM pipeline across 2.46M+ records": (
            "Developed an end-to-end LightGBM pipeline across 2.46M+ records, including data preparation, feature "
            "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2; "
            "presented findings as pricing and risk-segmentation recommendations to business stakeholders."
        ),
        "Developed and validated a Sequential Probability Ratio Test": (
            "Developed a sequential risk-monitoring framework for binary indicators, using Brownian-motion "
            "approximation, dynamic-programming probability propagation, and simulation to quantify stopping-time "
            "uncertainty and early-decision tradeoffs."
        ),
        "Combined Brownian-motion approximation": (
            "Specified and validated a Sequential Probability Ratio Test with explicit hypotheses and "
            "likelihood-ratio boundaries; evaluated boundary-crossing probabilities, stopping behavior, and Type I/II "
            "error."
        ),
        "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation": (
            "Scalable Inference: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
            "space-time covariance structures, using ordered local conditional models to support likelihood-based "
            "parameter estimation and uncertainty quantification for up to 145,008 observations per day."
        ),
        "Empirical Testing: Designed scale- and frequency-resolved diagnostics": (
            "Stochastic Diagnostics: Developed statistical diagnostic tools for dependent stochastic processes that "
            "localize where a fitted model fails across scale and frequency, distinguishing persistent dependence from "
            "high-frequency noise, missingness, acquisition artifacts, and covariance misspecification."
        ),
    }

    for paragraph in doc.paragraphs:
        for prefix, new_text in replacements.items():
            if paragraph.text.startswith(prefix):
                first_run = paragraph.runs[0] if paragraph.runs else None
                for run in list(paragraph.runs):
                    paragraph._p.remove(run._r)
                if first_run is not None and first_run.bold and ": " in new_text:
                    lead, detail = new_text.split(": ", 1)
                    lead_run = paragraph.add_run(f"{lead}: ")
                    set_run_font(
                        lead_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=True,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                    detail_run = paragraph.add_run(detail)
                    set_run_font(
                        detail_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=False,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                else:
                    new_run = paragraph.add_run(new_text)
                    if first_run is not None:
                        set_run_font(
                            new_run,
                            first_run.font.size.pt if first_run.font.size else 9.15,
                            bold=bool(first_run.bold),
                            italic=bool(first_run.italic),
                            color=first_run.font.color.rgb or BLACK,
                        )
                break

    paragraphs = list(doc.paragraphs)
    imc_index = next(
        i for i, paragraph in enumerate(paragraphs)
        if paragraph.text.startswith("IMC Prosperity Algorithmic Trading Competition")
    )
    dissertation_index = next(
        i for i, paragraph in enumerate(paragraphs)
        if paragraph.text.startswith("Ph.D. Dissertation Research")
    )
    additional = next(paragraph for paragraph in paragraphs if paragraph.text == "ADDITIONAL")
    for paragraph in paragraphs[imc_index:dissertation_index]:
        additional._p.addprevious(paragraph._p)

    out = BARCLAYS_ELECTRONIC_TRADING / "Joonwon_Lee_Barclays_Electronic_Trading_Resume.docx"
    doc.save(out)
    return out


def build_cubist_entry_level_quantitative_researcher_resume():
    """Create a targeted one-page resume for Cubist's Entry-Level Quantitative Researcher role."""
    source = BARCLAYS_ELECTRONIC_TRADING / "Joonwon_Lee_Barclays_Electronic_Trading_Resume.docx"
    if not source.exists():
        raise FileNotFoundError(f"Run build_barclays_electronic_trading_resume() first: {source}")

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Cubist Entry-Level Quantitative Researcher Resume",
        "Targeted resume for Entry-Level Quantitative Researcher at Cubist Systematic Strategies",
        "quantitative research, predictive modeling, hypothesis testing, backtesting, statistical diagnostics, "
        "Python, SQL, C++ integration, systematic investing, reproducible research tools",
    )

    replacements = {
        "Statistics Ph.D. candidate specializing": (
            "Statistics Ph.D. candidate specializing in scalable likelihood inference, predictive modeling, and "
            "statistical diagnostics for stochastically dependent data. Develops rigorous hypothesis-driven research "
            "workflows and reproducible Python/PyTorch tools for large datasets, with C++/pybind11 integration for "
            "performance-critical computation."
        ),
        "Quantitative Methods: Probability and stochastic processes": (
            "Quantitative Methods: Hypothesis testing, predictive modeling, regression and machine learning, "
            "backtesting, probability and stochastic processes, Monte Carlo simulation, MLE/GLS, numerical "
            "optimization, time-series and spatiotemporal modeling, spectral and autocorrelation analysis, model "
            "validation, and performance evaluation"
        ),
        "QUANTITATIVE RESEARCH AND TRADING": "QUANTITATIVE RESEARCH PROJECTS",
    }

    for paragraph in doc.paragraphs:
        for prefix, new_text in replacements.items():
            if paragraph.text.startswith(prefix):
                first_run = paragraph.runs[0] if paragraph.runs else None
                for run in list(paragraph.runs):
                    paragraph._p.remove(run._r)
                if first_run is not None and first_run.bold and ": " in new_text:
                    lead, detail = new_text.split(": ", 1)
                    lead_run = paragraph.add_run(f"{lead}: ")
                    set_run_font(
                        lead_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=True,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                    detail_run = paragraph.add_run(detail)
                    set_run_font(
                        detail_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=False,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                else:
                    new_run = paragraph.add_run(new_text)
                    if first_run is not None:
                        set_run_font(
                            new_run,
                            first_run.font.size.pt if first_run.font.size else 9.15,
                            bold=bool(first_run.bold),
                            italic=bool(first_run.italic),
                            color=first_run.font.color.rgb or BLACK,
                        )
                break

    out = CUBIST / "Joonwon_Lee_Cubist_Quantitative_Researcher_Resume.docx"
    doc.save(out)
    return out


def build_citadel_quantitative_researcher_resume():
    """Create a targeted one-page resume for Citadel's PhD Graduate QR role."""
    source = CUBIST / "Joonwon_Lee_Cubist_Quantitative_Researcher_Resume.docx"
    if not source.exists():
        raise FileNotFoundError(
            f"Run build_cubist_entry_level_quantitative_researcher_resume() first: {source}"
        )

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Citadel Quantitative Researcher Resume",
        "Targeted resume for Quantitative Researcher - PhD Graduate at Citadel",
        "quantitative research, probability, statistics, predictive modeling, hypothesis testing, backtesting, "
        "stochastic processes, scalable likelihood inference, Python, PyTorch, C++, independent research",
    )

    old_summary = "Statistics Ph.D. candidate specializing"
    new_summary = (
        "Statistics Ph.D. candidate specializing in scalable likelihood inference, predictive modeling, and "
        "statistical diagnostics for stochastically dependent data. Develops hypothesis-driven research and "
        "translates mathematical models into reproducible Python/PyTorch workflows for large datasets, with "
        "C++/pybind11 integration for performance-critical computation."
    )
    for paragraph in doc.paragraphs:
        if not paragraph.text.startswith(old_summary):
            continue
        first_run = paragraph.runs[0] if paragraph.runs else None
        for run in list(paragraph.runs):
            paragraph._p.remove(run._r)
        replacement = paragraph.add_run(new_summary)
        if first_run is not None:
            set_run_font(
                replacement,
                first_run.font.size.pt if first_run.font.size else 9.15,
                bold=bool(first_run.bold),
                italic=bool(first_run.italic),
                color=first_run.font.color.rgb or BLACK,
            )
        break

    out = CITADEL / "Joonwon_Lee_Citadel_Quantitative_Researcher_Resume.docx"
    doc.save(out)
    return out


def build_two_sigma_quantitative_researcher_resume():
    """Create a targeted one-page resume for Two Sigma's Quantitative Researcher role."""
    source = CITADEL / "Joonwon_Lee_Citadel_Quantitative_Researcher_Resume.docx"
    if not source.exists():
        raise FileNotFoundError(f"Run build_citadel_quantitative_researcher_resume() first: {source}")

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Two Sigma Quantitative Researcher Resume",
        "Targeted resume for Quantitative Researcher at Two Sigma",
        "quantitative research, scientific method, hypothesis testing, probabilistic modeling, predictive modeling, "
        "statistical diagnostics, stochastic processes, scalable likelihood inference, Python, PyTorch, C++",
    )

    old_summary = "Statistics Ph.D. candidate specializing"
    new_summary = (
        "Statistics Ph.D. candidate specializing in scalable likelihood inference, predictive modeling, and "
        "statistical diagnostics for stochastically dependent data. Uses rigorous hypothesis testing and probabilistic "
        "modeling to study large real-world datasets and builds reproducible Python/PyTorch research workflows, with "
        "C++/pybind11 integration for performance-critical computation."
    )
    for paragraph in doc.paragraphs:
        if not paragraph.text.startswith(old_summary):
            continue
        first_run = paragraph.runs[0] if paragraph.runs else None
        for run in list(paragraph.runs):
            paragraph._p.remove(run._r)
        replacement = paragraph.add_run(new_summary)
        if first_run is not None:
            set_run_font(
                replacement,
                first_run.font.size.pt if first_run.font.size else 9.15,
                bold=bool(first_run.bold),
                italic=bool(first_run.italic),
                color=first_run.font.color.rgb or BLACK,
            )
        break

    out = TWO_SIGMA / "Joonwon_Lee_Two_Sigma_Quantitative_Researcher_Resume.docx"
    doc.save(out)
    return out


def build_tower_2027_from_two_sigma_resume():
    """Create Tower's 2027 resume directly from the verified Two Sigma version."""
    source = TWO_SIGMA / "Joonwon_Lee_Two_Sigma_Quantitative_Researcher_Resume.docx"
    if not source.exists():
        build_two_sigma_quantitative_researcher_resume()

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Tower Research Quantitative Trader Researcher 2027 Resume",
        "Targeted resume for Quantitative Trader Researcher 2027 at Tower Research Capital",
        "quantitative research, predictive modeling, hypothesis testing, probabilistic modeling, statistical "
        "diagnostics, stochastic processes, scalable likelihood inference, simulation, Python, PyTorch, C++",
    )

    out = TOWER_RESEARCH / "Joonwon_Lee_Tower_QTQR_2027_Resume.docx"
    doc.save(out)
    return out


def build_google_student_researcher_phd_resume():
    """Create a research-first resume for Google's 2027 PhD Student Researcher program."""
    source = TWO_SIGMA / "Joonwon_Lee_Two_Sigma_Quantitative_Researcher_Resume.docx"
    if not source.exists():
        build_two_sigma_quantitative_researcher_resume()

    doc = Document(source)
    set_core_properties(
        doc,
        "Joonwon Lee - Google Student Researcher PhD 2027 Resume",
        "Targeted resume for Student Researcher PhD Winter Summer 2027 at Google",
        "student researcher, statistics, scalable inference, probabilistic modeling, Gaussian processes, "
        "statistical diagnostics, machine learning, scientific computing, Python, PyTorch, C++",
    )

    replacements = {
        "Statistics Ph.D. candidate specializing": (
            "Statistics Ph.D. candidate specializing in scalable statistical inference, probabilistic modeling, "
            "and statistical diagnostics for stochastically dependent data. Develops new methods for large, noisy, "
            "partially observed datasets and builds reproducible Python/PyTorch research systems, with C++/pybind11 "
            "integration for performance-critical computation."
        ),
        "Programming and Research Systems:": (
            "Programming and Research Systems: Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), "
            "C++/pybind11 integration, Linux, Git, AWS EC2, HPC/SLURM; CPU/GPU computing and restartable research pipelines"
        ),
        "Quantitative Methods:": (
            "Quantitative Methods: Gaussian processes, probability and stochastic processes, scalable likelihood "
            "approximation, hypothesis testing, Monte Carlo simulation, MLE/GLS, numerical optimization, spatiotemporal "
            "modeling, spectral and eigenvalue diagnostics, regression and machine learning, model validation, and "
            "uncertainty quantification"
        ),
        "QUANTITATIVE RESEARCH PROJECTS": "RESEARCH EXPERIENCE",
        "Ph.D. Dissertation Research -": (
            "Ph.D. Dissertation Research - Scalable Statistical Inference and Diagnostics\tSep 2024 - Present"
        ),
        "Research Tools:": (
            "Research Systems: Built end-to-end Python/PyTorch workflows for quality filtering, missingness tracking, "
            "time-dependent coordinate alignment, regular-grid matching, optimization, simulation, and restartable "
            "CPU/GPU and HPC execution; integrated performance-critical C++ ordering routines through pybind11."
        ),
    }

    for paragraph in doc.paragraphs:
        for prefix, new_text in replacements.items():
            if not paragraph.text.startswith(prefix):
                continue
            first_run = paragraph.runs[0] if paragraph.runs else None
            for run in list(paragraph.runs):
                paragraph._p.remove(run._r)

            if "\t" in new_text:
                left, right = new_text.split("\t", 1)
                left_run = paragraph.add_run(left)
                if first_run is not None:
                    set_run_font(
                        left_run,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=True,
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
                paragraph.add_run("\t")
                right_run = paragraph.add_run(right)
                set_run_font(
                    right_run,
                    first_run.font.size.pt if first_run and first_run.font.size else 9.15,
                    bold=False,
                    italic=False,
                    color=first_run.font.color.rgb if first_run and first_run.font.color.rgb else BLACK,
                )
            elif first_run is not None and first_run.bold and ": " in new_text:
                lead, detail = new_text.split(": ", 1)
                lead_run = paragraph.add_run(f"{lead}: ")
                set_run_font(
                    lead_run,
                    first_run.font.size.pt if first_run.font.size else 9.15,
                    bold=True,
                    italic=bool(first_run.italic),
                    color=first_run.font.color.rgb or BLACK,
                )
                detail_run = paragraph.add_run(detail)
                set_run_font(
                    detail_run,
                    first_run.font.size.pt if first_run.font.size else 9.15,
                    bold=False,
                    italic=bool(first_run.italic),
                    color=first_run.font.color.rgb or BLACK,
                )
            else:
                replacement = paragraph.add_run(new_text)
                if first_run is not None:
                    set_run_font(
                        replacement,
                        first_run.font.size.pt if first_run.font.size else 9.15,
                        bold=bool(first_run.bold),
                        italic=bool(first_run.italic),
                        color=first_run.font.color.rgb or BLACK,
                    )
            break

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if "Expected May 2027" in run.text:
                run.text = run.text.replace("Expected May 2027", "Expected 05/2027")

    paragraphs = list(doc.paragraphs)
    imc_index = next(
        i for i, paragraph in enumerate(paragraphs)
        if paragraph.text.startswith("IMC Prosperity Algorithmic Trading Competition")
    )
    additional_index = next(i for i, paragraph in enumerate(paragraphs) if paragraph.text == "ADDITIONAL")
    for paragraph in paragraphs[imc_index:additional_index]:
        paragraph._element.getparent().remove(paragraph._element)

    paragraphs = list(doc.paragraphs)
    research_heading_index = next(i for i, paragraph in enumerate(paragraphs) if paragraph.text == "RESEARCH EXPERIENCE")
    industry_heading = next(paragraph for paragraph in paragraphs if paragraph.text == "QUANTITATIVE INDUSTRY EXPERIENCE")
    additional_index = next(i for i, paragraph in enumerate(paragraphs) if paragraph.text == "ADDITIONAL")
    for paragraph in paragraphs[research_heading_index:additional_index]:
        industry_heading._p.addprevious(paragraph._p)

    out = GOOGLE_STUDENT_RESEARCHER / "Joonwon_Lee_Google_Student_Researcher_PhD_Resume.docx"
    doc.save(out)
    return out


def build_new_york_life_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.70, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.8, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - New York Life Senior Associate Data Scientist Resume",
        "Targeted resume for Senior Associate - Data Scientist at New York Life",
        "data science, machine learning, model lifecycle, model validation, data quality, model governance, Python, "
        "PyTorch, AWS, stakeholder communication, statistical diagnostics",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist with experience developing and validating statistical and "
        "machine-learning solutions for financial risk, insurance pricing, and large, noisy datasets. Builds "
        "reproducible Python/PyTorch workflows spanning data preparation, feature processing, model training, "
        "validation, simulation, and monitoring diagnostics, and translates technical results into clear decisions "
        "for business stakeholders.",
        9.9,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Feb 2027  |  GPA: 3.8/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, C++/pybind11 integration, Linux, Git, "
        "AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.45,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Data Science and Model Lifecycle: ",
        "Data preparation, feature processing, supervised learning, GLM/GBM, regression, model selection and "
        "validation, performance evaluation, feature attribution, hypothesis testing, Monte Carlo simulation, "
        "numerical optimization, model diagnostics, documentation, and stakeholder communication",
        9.45,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.35,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate boundary-crossing probabilities, Type I/II error, stopping behavior, uncertainty, and model limitations.",
        num_id,
        9.35,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.35,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated model performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.35,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Research and Engineering", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development and Statistical Diagnostics",
        "Sep 2024 - Present",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, replacing dense covariance operations with ordered local conditioning to fit up to 145,008 observations per day.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Data Quality and Root-Cause Analysis: Built workflows for quality filtering, missingness tracking, "
        "time-specific coordinate offsets, nearest-neighbor center matching, and irregular-to-regular grid mapping "
        "to separate systematic acquisition artifacts from random noise and model misspecification.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
        bold_lead="Data Quality and Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Model Evaluation and Monitoring: Designed scale- and frequency-resolved diagnostics to localize whether "
        "model discrepancies arise from low-frequency structure, high-frequency behavior, noise, missingness, or dependence.",
        num_id,
        9.25,
        after=0.45,
        line=1.0,
        bold_lead="Model Evaluation and Monitoring:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch pipelines with CPU/GPU execution, cached artifacts, "
        "and restartable HPC jobs; integrated compiled C++ ordering routines through pybind11.",
        num_id,
        9.25,
        after=0.6,
        line=1.0,
        bold_lead="Research Engineering:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Linear Algebra, Microeconomics, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = NEW_YORK_LIFE / "Joonwon_Lee_New_York_Life_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_voleon_data_scientist_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.70, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.8, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Voleon Data Scientist Resume",
        "Targeted resume for Data Scientist at The Voleon Group",
        "data quality, root-cause analysis, statistical diagnostics, anomaly monitoring, data curation, Python, "
        "Pandas, R, SQL, Linux, Git, trading systems, research systems, automated analysis pipelines",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist specializing in data quality, statistical diagnostics, and "
        "root-cause analysis for large, noisy, and dependent datasets. Builds reproducible Python/PyTorch workflows "
        "for data curation, anomaly investigation, simulation, and ongoing analytical monitoring. Experienced in "
        "translating ambiguous financial and scientific questions into mathematically defined analyses and clearly "
        "communicating findings to technical and business stakeholders.",
        9.9,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Feb 2027  |  GPA: 3.8/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), R, SQL, C++/pybind11 integration, Bash, "
        "Linux/Unix, Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analysis pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Data Analysis and Monitoring: ",
        "Data curation, aggregation, exploratory analysis, visualization, data validation, anomaly detection, "
        "root-cause analysis, statistical testing, regression, GLM/GBM, Monte Carlo simulation, model diagnostics, "
        "feature attribution, performance evaluation, and stakeholder reporting",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, converting a "
        "monitoring problem into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "quantify Type I/II error, boundary-crossing probabilities, stopping behavior, uncertainty, "
        "and early-decision tradeoffs.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM analysis pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Research and Analytical Systems", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Data Quality, Scalable Analysis, and Statistical Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Data Curation and Quality: Built workflows for quality filtering, missingness tracking, time-specific "
        "coordinate offsets, nearest-neighbor center matching, and spatial-threshold regular-grid construction for "
        "large satellite datasets.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Data Curation and Quality:",
    )
    add_bullet(
        doc,
        "Root-Cause Analysis: Investigated scan-geometry and time-dependent alignment effects and designed scale- and "
        "frequency-resolved diagnostics to distinguish systematic acquisition artifacts from random noise, missingness, "
        "and statistical-model misspecification.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Scalable Statistical Analysis: Developed an advection-aware Vecchia likelihood approximation for "
        "nonseparable Gaussian processes, using ordered local conditioning to analyze up to 145,008 observations per day.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Statistical Analysis:",
    )
    add_bullet(
        doc,
        "Analysis Automation: Implemented reusable Python/PyTorch pipelines for preprocessing, estimation, simulation, "
        "cached outputs, and restartable CPU/GPU and HPC jobs; integrated compiled C++ ordering routines through pybind11.",
        num_id,
        9.2,
        after=0.6,
        line=1.0,
        bold_lead="Analysis Automation:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Linear Algebra, Microeconomics, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = VOLEON / "Joonwon_Lee_Voleon_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_state_street_credit_risk_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.70, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.8, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - State Street Quantitative Analyst AVP Resume",
        "Targeted resume for Quantitative Analyst, Assistant Vice President - Credit Risk Modeler at State Street",
        "credit risk modeling, model development, model validation, financial risk, econometrics, stochastic "
        "processes, simulation, Python, R, SQL, C++ integration, model governance, statistical diagnostics",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and quantitative modeler with industry experience in financial-risk model "
        "validation, insurance pricing, and scalable statistical model development. Combines stochastic processes, "
        "likelihood-based inference, econometrics, simulation, and statistical diagnostics with reproducible "
        "Python/PyTorch research systems. Translates model assumptions, performance, uncertainty, and limitations "
        "into clear findings for technical and business stakeholders.",
        9.9,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected Feb 2027  |  GPA: 3.8/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Research Systems: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), R, SQL, C++/pybind11 integration, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, cached artifacts, and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Quantitative Methods: ",
        "Stochastic processes, probability, MLE/GLS, regression, GLM/GBM, econometrics, fixed effects, propensity-score "
        "matching, Monte Carlo simulation, hypothesis testing, numerical optimization, model selection and validation, "
        "sensitivity analysis, feature attribution, and statistical diagnostics",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Quantitative Industry Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate Type I/II error, boundary-crossing probabilities, stopping behavior, uncertainty, and model limitations.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.5,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ property-risk records, including data "
        "preparation, feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.5,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.3,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Research and Quantitative Modeling", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Statistical Model Development and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Model Development: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "Gaussian processes, replacing dense covariance operations with ordered local conditioning to fit up to "
        "145,008 observations per day.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Scalable Model Development:",
    )
    add_bullet(
        doc,
        "Assumption Review and Diagnostics: Compared alternative covariance specifications and designed scale- and "
        "frequency-resolved tools to determine whether discrepancies arise from low-frequency structure, "
        "high-frequency behavior, noise, missingness, or model misspecification.",
        num_id,
        9.2,
        after=0.45,
        line=1.0,
        bold_lead="Assumption Review and Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch workflows for data-quality filtering, estimation, "
        "simulation, numerical optimization, cached outputs, and restartable CPU/GPU and HPC execution; integrated "
        "compiled C++ ordering routines through pybind11.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "Econometric Study - Earned Income Tax Credit and Household Labor Supply",
        "2021",
        width,
        9.35,
        after=0.35,
    )
    add_bullet(
        doc,
        "Estimated the association between EITC exposure and household labor supply using longitudinal observational "
        "data, fixed-effects regression, and propensity-score matching with household income and demographic covariates.",
        num_id,
        9.15,
        after=0.6,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Probability Theory, Stochastic Processes, Machine Learning, Linear Algebra, Statistical Computing, Advanced "
        "Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        9.0,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.0,
        after=0,
    )

    out = STATE_STREET / "Joonwon_Lee_State_Street_Quantitative_Analyst_Resume.docx"
    doc.save(out)
    return out


def build_liberty_mutual_data_scientist_resume():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.70, margin_top=0.52, margin_bottom=0.52)
    configure_styles(doc, body_size=9.8, body_line=1.01)
    num_id = add_bullet_numbering(doc, left_twips=320, hanging_twips=180)
    set_core_properties(
        doc,
        "Joonwon Lee - Liberty Mutual Data Scientist Resume",
        "Targeted resume for Data Scientist - Property and Specialty Product Design and Modeling",
        "property insurance pricing, predictive modeling, GLM, LightGBM, hypothesis testing, statistical diagnostics, "
        "model validation, Python, AWS, stakeholder communication",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate and data scientist with hands-on property-insurance pricing experience and a "
        "rigorous background in predictive modeling, hypothesis testing, and statistical diagnostics. Built an "
        "end-to-end GLM/LightGBM workflow across 2.46M+ property-risk records and reusable Python/PyTorch pipelines "
        "for large, noisy datasets. Translates model evidence, uncertainty, and diagnostics into clear recommendations "
        "for business stakeholders.",
        9.9,
        after=1.0,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.7,
        after=0.35,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Platforms: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), SQL, R, Linux, Git, AWS EC2, HPC/SLURM; "
        "reproducible and restartable analytical pipelines",
        9.4,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Predictive Modeling and Validation: ",
        "Data preparation, feature processing, regression, GLM/GBM, supervised learning, model selection, calibration "
        "and validation, performance evaluation, feature attribution, hypothesis testing, Monte Carlo simulation, "
        "statistical diagnostics, documentation, and stakeholder communication",
        9.4,
        after=0.35,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, translating a "
        "monitoring requirement into explicit hypotheses, likelihood-ratio boundaries, and controlled stopping decisions.",
        num_id,
        9.3,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Combined Brownian-motion approximation, dynamic-programming probability propagation, and simulation to "
        "evaluate Type I/II error, boundary-crossing probabilities, stopping behavior, uncertainty, and model limitations.",
        num_id,
        9.3,
        after=0.55,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.45,
        after=0.35,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM property-pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2.",
        num_id,
        9.3,
        after=0.45,
        line=1.0,
    )
    add_bullet(
        doc,
        "Evaluated predictive performance and feature-attribution patterns and presented pricing and risk-segmentation "
        "recommendations to business stakeholders.",
        num_id,
        9.3,
        after=0.55,
        line=1.0,
    )

    add_section_heading(doc, "Research and Analytical Systems", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Model Development and Statistical Diagnostics",
        "Sep 2024 - Present",
        width,
        9.4,
        after=0.35,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable Gaussian "
        "processes, replacing dense covariance operations with ordered local conditioning to fit up to 145,008 "
        "observations per day.",
        num_id,
        9.2,
        after=0.4,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Data Quality and Root-Cause Analysis: Built workflows for quality filtering, missingness tracking, "
        "time-specific coordinate offsets, nearest-neighbor matching, and irregular-to-regular grid mapping to "
        "separate systematic acquisition artifacts from random noise and model misspecification.",
        num_id,
        9.2,
        after=0.4,
        line=1.0,
        bold_lead="Data Quality and Root-Cause Analysis:",
    )
    add_bullet(
        doc,
        "Model Diagnostics: Designed scale- and frequency-resolved tests to identify whether discrepancies arise from "
        "low-frequency structure, high-frequency behavior, noise, missingness, or dependence; implemented reusable "
        "Python/PyTorch workflows for estimation, simulation, validation, and HPC execution.",
        num_id,
        9.2,
        after=0.55,
        line=1.0,
        bold_lead="Model Diagnostics:",
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Probability Theory, Stochastic Processes, Statistical Computing, Advanced Theory of "
        "Statistics I-II, Data Structures and Algorithms, Linear Algebra, Microeconomics, Econometrics",
        9.05,
        after=0.3,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        9.05,
        after=0,
    )

    out = LIBERTY_MUTUAL / "Joonwon_Lee_Liberty_Mutual_Data_Scientist_Resume.docx"
    doc.save(out)
    return out


def build_research_cv():
    doc = Document()
    width = set_cell_free_document_defaults(doc, margin_x=0.62, margin_top=0.48, margin_bottom=0.52)
    configure_styles(doc, body_size=9.05, body_line=1.02)
    num_id = add_bullet_numbering(doc, left_twips=330, hanging_twips=185)
    set_core_properties(
        doc,
        "Joonwon Lee - Quantitative Research CV",
        "Detailed quantitative research CV for research-intensive industry roles",
        "quantitative research, statistical machine learning, Gaussian processes, model risk, spectral methods",
    )
    add_page_number_footer(doc.sections[0], "Joonwon Lee - Quantitative Research CV", size=7)

    add_name_header(
        doc,
        subtitle="QUANTITATIVE RESEARCH  |  STATISTICAL MACHINE LEARNING  |  MODEL RISK",
        compact=False,
    )

    add_section_heading(doc, "Research Profile")
    add_body_paragraph(
        doc,
        "Statistics Ph.D. candidate developing scalable Gaussian process inference and model diagnostics "
        "for high-dimensional dependent data. Research combines approximate likelihoods, spectral methods, "
        "stochastic-process theory, and GPU/HPC implementation; industry experience spans financial risk "
        "monitoring and commercial insurance pricing.",
        9.05,
        after=1.2,
        line=1.03,
    )

    add_section_heading(doc, "Education")
    add_tabbed_line(
        doc,
        "Rutgers University, Piscataway, NJ - Ph.D. in Statistics",
        "Expected Dec 2026  |  GPA: 3.8/4.0",
        width,
        9.05,
        after=0.4,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "University of Minnesota, Minneapolis, MN - M.S. in Statistics",
        "Sep 2019 - Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.05,
        after=0.4,
        keep_with_next=False,
    )

    add_section_heading(doc, "Research Experience")
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Gaussian Process Inference and Diagnostics",
        "Sep 2024 - Present",
        width,
        9.05,
        after=0.3,
    )
    add_body_paragraph(
        doc,
        "Application domain: geostationary satellite total-column ozone, with 18,126 spatial cells and "
        "eight daytime snapshots per day. The methods target large, noisy, partially observed stochastic "
        "fields where dense covariance methods are infeasible and transfer to alternative-data and "
        "model-risk settings.",
        8.85,
        after=0.8,
        line=1.02,
        color=MUTED,
    )
    add_bullet(
        doc,
        "Scalable inference: Developed a nonseparable, advection-aware Vecchia approximation with 4x4 "
        "target blocks and a 6/4/3 temporal conditioning pattern. The fixed conditioning budget replaces "
        "dense O(n^3) covariance operations with approximately linear scaling in n, enabling daily fits "
        "over 145,008 rows (about 141,000 observed after quality filtering).",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Scalable inference:",
    )
    add_bullet(
        doc,
        "Covariance model selection: Built max-min block-prefix experiments at 100, 200, 400, 600, "
        "800, and all blocks to separate parameter instability caused by the Vecchia approximation from "
        "instability caused by covariance-family misspecification. Compared Matérn and generalized Cauchy "
        "models on July 2023-2025 data using likelihood paths and parameter convergence.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Covariance model selection:",
    )
    add_bullet(
        doc,
        "Scale-resolved model diagnostics: Eigendecomposed small conditional covariance blocks, "
        "projected residuals into whitened eigen-coordinates, adjusted expected increments for fitted-mean "
        "leverage, and accumulated squared scores by conditional eigenvalue to identify the scales at which "
        "a fitted covariance model systematically under- or overstates residual variation.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Scale-resolved model diagnostics:",
    )
    add_bullet(
        doc,
        "Missing-data spectral validation: Derived how zero-imputation, structured observation masks, "
        "and resampling operators alter the expected periodogram and the pseudo-true variance, range, and "
        "nugget parameters of Debiased Whittle estimation. Developed a corrected-periodogram route for "
        "reducing the resulting scale and noise-floor bias.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Missing-data spectral validation:",
    )
    add_bullet(
        doc,
        "Multivariate frequency diagnostics: Constructed an eight-variate Fourier vector for hourly fields, "
        "computed the mask-aware finite-sample expected 8x8 cross-periodogram, applied Cholesky whitening, "
        "and profiled residual power by norm, latitude, longitude, and diagonal frequency. Validated the "
        "expected cross-periodogram over 1,000 simulations with MAD 0.00648.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Multivariate frequency diagnostics:",
    )
    add_bullet(
        doc,
        "Physical validation: Used asymmetric one-hour cross-variograms and two-dimensional "
        "empirical ridge locations to test whether fitted advection vectors align with observed movement. "
        "Added near-minimum counts and relative-gap measures so flat or multimodal ridges are flagged as "
        "ambiguous rather than over-interpreted.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Physical validation:",
    )
    add_bullet(
        doc,
        "Research engineering: Implemented reusable PyTorch likelihood engines, GLS mean profiling, "
        "L-BFGS optimization, CPU/GPU fallbacks, chunked conditional diagnostics, simulation generators "
        "with circulant embedding, and restartable HPC pipelines with cached fit artifacts.",
        num_id,
        8.9,
        after=0.8,
        line=1.02,
        bold_lead="Research engineering:",
    )

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)

    add_section_heading(doc, "Selected Research Workstreams and Software")
    add_labeled_paragraph(
        doc,
        "Missing-Data Spectral Diagnostics. ",
        "Derived random-mask, deterministic-mask, and resampling-operator effects on "
        "the expected periodogram; characterized pseudo-true covariance parameters and developed a two-step "
        "bias-correction strategy.",
        8.9,
        after=1.0,
        line=1.03,
    )
    add_labeled_paragraph(
        doc,
        "Cross-Frequency Periodogram Covariance. ",
        "Derived exact Gaussian cross-frequency covariance under tapers and "
        "time-varying observation masks, with smooth, full-convolution, and structured-shrinkage "
        "estimators for finite-sample uncertainty.",
        8.9,
        after=1.0,
        line=1.03,
    )
    add_labeled_paragraph(
        doc,
        "Vecchia Conditional-Eigen Diagnostic Suite. ",
        "Reusable PyTorch research software for blockwise whitening, leverage-adjusted score accumulation, "
        "scale-ordered residual diagnostics, parameter-mismatch experiments, and full-eigen sanity checks.",
        8.9,
        after=1.0,
        line=1.03,
    )
    add_labeled_paragraph(
        doc,
        "Advection-Ridge and Cross-Variogram Diagnostics. ",
        "Three-model comparison framework for July 2022-2025 that aligns empirical cross-variogram minima "
        "with fitted transport vectors and reports ridge ambiguity before interpreting direction or speed.",
        8.9,
        after=1.1,
        line=1.03,
    )

    add_section_heading(doc, "Industry Experience")
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Quantitative Finance / Risk Modeling",
        "Summer 2026",
        width,
        9.0,
        after=0.3,
    )
    add_bullet(
        doc,
        "Designed a sequential anomaly-detection framework for binary risk indicators by modeling cumulative "
        "evidence as a random walk and using Brownian-motion approximations to quantify stopping-time uncertainty.",
        num_id,
        8.85,
        after=0.6,
        line=1.02,
    )
    add_bullet(
        doc,
        "Structured simulation-based comparisons of fixed-sample and sequential rules around detection delay, "
        "false-positive/false-negative control, and the value of earlier decisions.",
        num_id,
        8.85,
        after=0.7,
        line=1.02,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.0,
        after=0.3,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pricing pipeline for mid-market property risk, including feature "
        "processing, GLM benchmarking, validation, and performance diagnostics over 2.46M+ records on AWS EC2.",
        num_id,
        8.85,
        after=0.6,
        line=1.02,
    )
    add_bullet(
        doc,
        "Used model diagnostics and feature attribution to translate predictive patterns into pricing and "
        "risk-segmentation recommendations for business stakeholders.",
        num_id,
        8.85,
        after=0.7,
        line=1.02,
    )

    add_section_heading(doc, "Selected Quantitative Project")
    add_tabbed_line(
        doc,
        "IMC Prosperity Algorithmic Trading Competition - Top 2.39% of Participants",
        "Apr 2025",
        width,
        9.0,
        after=0.3,
    )
    add_bullet(
        doc,
        "Built and evaluated a market-making strategy with inventory-aware quoting, dynamic liquidation "
        "rules, and benchmark-driven performance analysis.",
        num_id,
        8.85,
        after=0.7,
        line=1.02,
    )

    add_section_heading(doc, "Technical Skills")
    add_labeled_paragraph(
        doc,
        "Programming and Systems: ",
        "Python, NumPy, Pandas, PyTorch, SciPy, LightGBM, scikit-learn, R, SQL/MySQL, Linux, Git, AWS EC2, HPC/SLURM",
        8.8,
        after=0.5,
        line=1.02,
    )
    add_labeled_paragraph(
        doc,
        "Statistical Computing: ",
        "MLE, Gaussian processes, Vecchia approximations, Matérn and generalized Cauchy covariance, "
        "spectral/Whittle methods, FFT and circulant embedding, GLS, L-BFGS, Monte Carlo simulation",
        8.8,
        after=0.5,
        line=1.02,
    )
    add_labeled_paragraph(
        doc,
        "Modeling and Validation: ",
        "Stochastic processes, sequential testing, Brownian approximations, eigen diagnostics, cross-validation, "
        "GLM/GBM, model monitoring, missing-data diagnostics, directional semivariograms",
        8.8,
        after=0.5,
        line=1.02,
    )

    add_section_heading(doc, "Coursework and Languages")
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Linear Algebra, Probability Theory, Stochastic Processes, Statistical Computing, "
        "Advanced Theory of Statistics I-II, Data Structures and Algorithms, Microeconomics, Econometrics",
        8.75,
        after=0.4,
        line=1.02,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.75,
        after=0,
        line=1.02,
    )

    out = FINAL / "Joonwon_Lee_Quantitative_Research_CV.docx"
    doc.save(out)
    return out


if __name__ == "__main__":
    print(build_finance_resume())
    print(build_wells_fargo_resume())
    print(build_morgan_stanley_resume())
    print(build_balyasny_resume())
    print(build_goldman_sachs_resume())
    print(build_microsoft_resume())
    print(build_google_youtube_resume())
    print(build_google_ads_metrics_resume())
    print(build_google_data_scientist_resume())
    print(build_valley_bank_resume())
    print(build_blackrock_resume())
    print(build_blackrock_sae_resume())
    print(build_upstart_resume())
    print(build_akuna_capital_resume())
    print(build_jpmorgan_spg_qtr_resume())
    print(build_goldman_core_planning_resume())
    print(build_walleye_single_stock_vol_resume())
    print(build_sentilink_resume())
    print(build_teza_technologies_resume())
    print(build_freddie_mac_resume())
    print(build_radix_trading_resume())
    print(build_3red_partners_resume())
    print(build_openai_safety_resume())
    print(build_capital_one_model_risk_resume())
    print(build_capital_one_principal_quant_modeler_resume())
    print(build_wells_fargo_qa_2027_resume())
    print(build_imc_quantitative_researcher_2027_resume())
    print(build_virtu_quantitative_strategist_resume())
    print(build_barclays_electronic_trading_resume())
    print(build_cubist_entry_level_quantitative_researcher_resume())
    print(build_citadel_quantitative_researcher_resume())
    print(build_two_sigma_quantitative_researcher_resume())
    print(build_tower_2027_from_two_sigma_resume())
    print(build_google_student_researcher_phd_resume())
    print(build_new_york_life_resume())
    print(build_voleon_data_scientist_resume())
    print(build_state_street_credit_risk_resume())
    print(build_liberty_mutual_data_scientist_resume())
    print(build_research_cv())
