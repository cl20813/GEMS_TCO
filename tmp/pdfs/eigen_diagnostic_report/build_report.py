from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
WORK = ROOT / "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic"
TMP = ROOT / "tmp/pdfs/eigen_diagnostic_report"
OUT = ROOT / "output/pdf/matrix_free_lanczos_full_spectrum_diagnostic_140352.pdf"

FULL_DIR = WORK / "real_full_vecchia_precision_lanczos_20240703_090326"
MAXMIN_DIR = WORK / "real_maxmin400_exact_vs_subset_vecchia_4way_20240703_090326"
ROBUST_DIR = WORK / "maxmin_vs_contiguous_4way_robustness_20240703_090326"

FULL_FIG = FULL_DIR / "full_vecchia_precision_eigen_diagnostic.png"
FOURWAY_FIG = MAXMIN_DIR / "exact_vecchia_four_way_diagnostic.png"
ROBUST_FIG = ROBUST_DIR / "maxmin_vs_contiguous_robustness.png"

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
TOP = 17 * mm
BOTTOM = 16 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT

NAVY = HexColor("#12304A")
BLUE = HexColor("#1E6688")
TEAL = HexColor("#2C887D")
PALE_BLUE = HexColor("#EAF3F7")
PALE_TEAL = HexColor("#E7F3F0")
PALE_GOLD = HexColor("#FFF4D7")
GOLD = HexColor("#D69A2D")
RED = HexColor("#A64545")
PALE_RED = HexColor("#FBECEC")
INK = HexColor("#15242E")
MUTED = HexColor("#526670")
LIGHT = HexColor("#D7E0E5")
VERY_LIGHT = HexColor("#F5F8FA")


def register_fonts() -> None:
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    if font_path is None:
        raise FileNotFoundError("No Korean-capable font found")
    pdfmetrics.registerFont(TTFont("Korean", font_path))
    pdfmetrics.registerFontFamily(
        "Korean", normal="Korean", bold="Korean", italic="Korean", boldItalic="Korean"
    )


register_fonts()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="KBody",
        fontName="Korean",
        fontSize=9.2,
        leading=14.0,
        textColor=INK,
        alignment=TA_JUSTIFY,
        wordWrap="CJK",
        spaceAfter=5.5,
    )
)
styles.add(
    ParagraphStyle(
        name="KBodySmall",
        parent=styles["KBody"],
        fontSize=8.0,
        leading=11.5,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="KCaption",
        parent=styles["KBody"],
        fontSize=7.7,
        leading=10.7,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceBefore=3,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="KH1",
        fontName="Korean",
        fontSize=17,
        leading=22,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=9,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="KH2",
        fontName="Korean",
        fontSize=12.2,
        leading=16,
        textColor=BLUE,
        spaceBefore=7,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="KH3",
        fontName="Korean",
        fontSize=9.8,
        leading=13,
        textColor=NAVY,
        spaceBefore=5,
        spaceAfter=3,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="KTitle",
        fontName="Korean",
        fontSize=25,
        leading=34,
        textColor=NAVY,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="KSubtitle",
        fontName="Korean",
        fontSize=12,
        leading=18,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="KEquation",
        fontName="Korean",
        fontSize=11.2,
        leading=17,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="KBullet",
        parent=styles["KBody"],
        leftIndent=13,
        firstLineIndent=-8,
        bulletIndent=0,
        spaceAfter=3,
    )
)
styles.add(
    ParagraphStyle(
        name="KQuote",
        parent=styles["KBody"],
        fontSize=11,
        leading=17,
        textColor=NAVY,
        leftIndent=8,
        rightIndent=8,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="KFooter",
        fontName="Korean",
        fontSize=6.8,
        leading=8,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="KTable",
        fontName="Korean",
        fontSize=7.7,
        leading=10.2,
        textColor=INK,
        wordWrap="CJK",
    )
)
styles.add(
    ParagraphStyle(
        name="KTableHead",
        fontName="Korean",
        fontSize=7.8,
        leading=10.2,
        textColor=colors.white,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
)


def P(text: str, style: str = "KBody") -> Paragraph:
    return Paragraph(text, styles[style])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>•</bullet>{text}", styles["KBullet"])


def rule(color=LIGHT, thickness=0.7, space_before=2, space_after=6):
    return HRFlowable(
        width="100%",
        thickness=thickness,
        color=color,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )


def callout(title: str, body: str, bg=PALE_BLUE, border=BLUE) -> Table:
    cell = [P(title, "KH3"), P(body, "KBody")]
    t = Table([[cell]], colWidths=[CONTENT_W - 4 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def equation(lines: list[str] | str) -> Table:
    if isinstance(lines, str):
        lines = [lines]
    content = [P(line, "KEquation") for line in lines]
    t = Table([[content]], colWidths=[CONTENT_W - 12 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), VERY_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def make_table(rows, widths, header=True, font_size=7.7, aligns=None, repeat_rows=1):
    formatted = []
    for i, row in enumerate(rows):
        style_name = "KTableHead" if header and i == 0 else "KTable"
        formatted.append([x if hasattr(x, "wrap") else P(str(x), style_name) for x in row])
    t = Table(formatted, colWidths=widths, repeatRows=repeat_rows if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, LIGHT),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
        ]
        start = 1
    else:
        start = 0
    for r in range(start, len(rows)):
        if (r - start) % 2 == 1:
            cmds.append(("BACKGROUND", (0, r), (-1, r), VERY_LIGHT))
    if aligns:
        for col, align in enumerate(aligns):
            cmds.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    t.setStyle(TableStyle(cmds))
    return t


def fit_image(path: Path, max_w=CONTENT_W, max_h=110 * mm) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def pipeline_diagram() -> Drawing:
    d = Drawing(CONTENT_W, 78)
    box_w, box_h, gap = 91, 39, 14
    xs = [0, box_w + gap, 2 * (box_w + gap), 3 * (box_w + gap), 4 * (box_w + gap)]
    labels = [
        ("Vecchia fit", "adapted 4/3/2"),
        ("Sparse B", "210.96 MiB"),
        ("Operator", "v -> B'Bv"),
        ("Lanczos / SLQ", "m=512, s=12"),
        ("Diagnostic", "F(t), G(t), bands"),
    ]
    colors_fill = [PALE_TEAL, PALE_TEAL, PALE_BLUE, PALE_BLUE, PALE_GOLD]
    for i, x in enumerate(xs):
        d.add(Rect(x, 25, box_w, box_h, rx=5, ry=5, fillColor=colors_fill[i], strokeColor=BLUE, strokeWidth=0.7))
        d.add(String(x + box_w / 2, 48, labels[i][0], fontName="Korean", fontSize=7.4, textAnchor="middle", fillColor=NAVY))
        d.add(String(x + box_w / 2, 34, labels[i][1], fontName="Korean", fontSize=6.6, textAnchor="middle", fillColor=MUTED))
        if i < len(xs) - 1:
            x0 = x + box_w
            x1 = xs[i + 1]
            d.add(Line(x0 + 2, 44, x1 - 3, 44, strokeColor=BLUE, strokeWidth=1.1))
            d.add(Line(x1 - 7, 48, x1 - 3, 44, strokeColor=BLUE, strokeWidth=1.1))
            d.add(Line(x1 - 7, 40, x1 - 3, 44, strokeColor=BLUE, strokeWidth=1.1))
    d.add(String(0, 8, "Dense K, dense Ω, full Q를 한 번도 만들지 않는다", fontName="Korean", fontSize=7.5, fillColor=RED))
    return d


def memory_chart() -> Drawing:
    # Log-scaled lengths because the dense/full-eigen objects are hundreds of times larger.
    labels = ["Sparse B", "Lanczos basis", "Dense K", "Dense K + Q"]
    mib = [210.96, 548.25, 146.7666 * 1024, 2 * 146.7666 * 1024]
    d = Drawing(CONTENT_W, 112)
    x0, y0, max_w = 92, 15, CONTENT_W - 150
    max_log = math.log10(max(mib))
    fills = [TEAL, BLUE, GOLD, RED]
    for i, (lab, val) in enumerate(zip(labels, mib)):
        y = 88 - i * 24
        length = max_w * math.log10(max(val, 1)) / max_log
        d.add(String(0, y + 2, lab, fontName="Korean", fontSize=7.7, fillColor=INK))
        d.add(Rect(x0, y, length, 10, fillColor=fills[i], strokeColor=None))
        val_text = f"{val:.0f} MiB" if val < 1024 else f"{val/1024:.1f} GiB"
        d.add(String(x0 + length + 5, y + 1, val_text, fontName="Korean", fontSize=7.2, fillColor=INK))
    d.add(String(x0, 2, "막대 길이는 log scale; K+Q는 eigensolver workspace 제외 최소치", fontName="Korean", fontSize=6.5, fillColor=MUTED))
    return d


def timing_chart() -> Drawing:
    labels = ["Residual Lanczos", "SLQ (12 probes)", "Other", "Total"]
    vals = [9.4526, 110.9068, 147.8402 - 9.4526 - 110.9068, 147.8402]
    d = Drawing(CONTENT_W, 125)
    chart = HorizontalBarChart()
    chart.x = 112
    chart.y = 20
    chart.height = 84
    chart.width = CONTENT_W - 155
    chart.data = [vals]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Korean"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 160
    chart.valueAxis.valueStep = 40
    chart.valueAxis.labels.fontName = "Korean"
    chart.valueAxis.labels.fontSize = 6.5
    chart.bars[0].fillColor = BLUE
    chart.bars[0].strokeColor = None
    d.add(chart)
    d.add(String(112, 5, "seconds", fontName="Korean", fontSize=6.5, fillColor=MUTED))
    return d


def band_chart() -> Drawing:
    csv_path = FULL_DIR / "full_precision_equal_mode_bands.csv"
    vals = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            vals.append(float(row["standardized_energy_per_mode"]))
    d = Drawing(CONTENT_W, 170)
    x0, y0, cw, ch = 40, 33, CONTENT_W - 56, 115
    ymin, ymax = 0.76, 1.17
    d.add(Line(x0, y0, x0, y0 + ch, strokeColor=MUTED, strokeWidth=0.6))
    d.add(Line(x0, y0, x0 + cw, y0, strokeColor=MUTED, strokeWidth=0.6))
    for tick in [0.8, 0.9, 1.0, 1.1]:
        yy = y0 + (tick - ymin) / (ymax - ymin) * ch
        d.add(Line(x0, yy, x0 + cw, yy, strokeColor=LIGHT if tick != 1.0 else NAVY, strokeWidth=0.5 if tick != 1.0 else 1.0))
        d.add(String(x0 - 6, yy - 2, f"{tick:.1f}", fontName="Korean", fontSize=6.4, fillColor=MUTED, textAnchor="end"))
    bw = cw / len(vals)
    for i, v in enumerate(vals):
        xx = x0 + i * bw + 1
        yy0 = y0 + (1.0 - ymin) / (ymax - ymin) * ch
        yyv = y0 + (v - ymin) / (ymax - ymin) * ch
        fill = TEAL if v >= 1.0 else BLUE
        d.add(Rect(xx, min(yy0, yyv), bw - 2, abs(yyv - yy0), fillColor=fill, strokeColor=None))
        d.add(String(xx + (bw - 2) / 2, y0 - 11, str(i + 1), fontName="Korean", fontSize=5.6, fillColor=MUTED, textAnchor="middle"))
    d.add(String(x0 + cw / 2, 6, "covariance eigenscale band: smooth / large-variance  ->  rough / small-variance", fontName="Korean", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    d.add(String(3, y0 + ch / 2, "energy / mode", fontName="Korean", fontSize=6.6, fillColor=MUTED))
    return d


def validation_chart() -> Drawing:
    rows = []
    with (MAXMIN_DIR / "E2_V2_lanczos_accuracy.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["reorthogonalization"] == "full":
                rows.append(row)
    orders = [64, 128, 256, 512]
    exact = {int(r["lanczos_steps"]): float(r["energy_per_n_rmse"]) for r in rows if r["operator"] == "exact_precision"}
    vec = {int(r["lanczos_steps"]): float(r["energy_per_n_rmse"]) for r in rows if r["operator"] == "subset_vecchia_precision"}
    d = Drawing(CONTENT_W, 176)
    x0, y0, cw, ch = 48, 30, CONTENT_W - 80, 120
    ymin, ymax = 0.0, 0.0072
    d.add(Line(x0, y0, x0, y0 + ch, strokeColor=MUTED, strokeWidth=0.7))
    d.add(Line(x0, y0, x0 + cw, y0, strokeColor=MUTED, strokeWidth=0.7))
    for tick in [0.0, 0.002, 0.004, 0.006]:
        yy = y0 + (tick - ymin) / (ymax - ymin) * ch
        d.add(Line(x0, yy, x0 + cw, yy, strokeColor=LIGHT, strokeWidth=0.5))
        d.add(String(x0 - 7, yy - 2, f"{tick:.3f}", fontName="Korean", fontSize=6.2, fillColor=MUTED, textAnchor="end"))
    xs = []
    for i, m in enumerate(orders):
        xx = x0 + i * cw / (len(orders) - 1)
        xs.append(xx)
        d.add(String(xx, y0 - 13, str(m), fontName="Korean", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    for series, color in [(exact, BLUE), (vec, TEAL)]:
        pts = []
        for xx, m in zip(xs, orders):
            yy = y0 + (series[m] - ymin) / (ymax - ymin) * ch
            pts.append((xx, yy))
        for a, b in zip(pts[:-1], pts[1:]):
            d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=1.8))
        for xx, yy in pts:
            d.add(Rect(xx - 2, yy - 2, 4, 4, fillColor=color, strokeColor=color))
    d.add(String(x0 + cw / 2, 5, "Lanczos steps m", fontName="Korean", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    d.add(String(1, y0 + ch / 2, "energy/n RMSE", fontName="Korean", fontSize=6.5, fillColor=MUTED))
    d.add(Rect(x0 + 6, y0 + ch - 10, 9, 4, fillColor=BLUE, strokeColor=None))
    d.add(String(x0 + 20, y0 + ch - 12, "Exact precision: E2-E1", fontName="Korean", fontSize=6.7, fillColor=INK))
    d.add(Rect(x0 + 145, y0 + ch - 10, 9, 4, fillColor=TEAL, strokeColor=None))
    d.add(String(x0 + 159, y0 + ch - 12, "Vecchia precision: V2-V1", fontName="Korean", fontSize=6.7, fillColor=INK))
    return d


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Matrix-free Lanczos full-spectrum eigen diagnostic for n=140,352",
            author="GEMS TCO space-time model diagnostic",
            subject="Large-scale Vecchia precision eigen-residual diagnostic",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="main")
        self.addPageTemplates(PageTemplate(id="normal", frames=frame, onPage=self._on_page))

    def _on_page(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LIGHT)
            canvas.setLineWidth(0.45)
            canvas.line(LEFT, PAGE_H - 10.5 * mm, PAGE_W - RIGHT, PAGE_H - 10.5 * mm)
            canvas.setFont("Korean", 6.7)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, PAGE_H - 8.5 * mm, "MATRIX-FREE LANCZOS EIGEN DIAGNOSTIC")
            canvas.drawRightString(PAGE_W - RIGHT, 8.2 * mm, f"{doc.page}")
            canvas.drawString(LEFT, 8.2 * mm, "GEMS TCO space-time model | technical note")
        canvas.restoreState()


def add_cover(story):
    story.append(Spacer(1, 18 * mm))
    story.append(P("기술 메모", "KSubtitle"))
    story.append(Spacer(1, 4 * mm))
    story.append(P("Full eigendecomposition 없이 구현한<br/>140,352차원 eigen-residual diagnostic", "KTitle"))
    story.append(Spacer(1, 6 * mm))
    story.append(rule(BLUE, 2.2, 0, 8))
    story.append(P("Matrix-free Lanczos + stochastic Lanczos quadrature + sparse Vecchia precision", "KSubtitle"))
    story.append(Spacer(1, 12 * mm))
    story.append(
        callout(
            "한 문장 결론",
            "명목상 18,000개 공간점 × 8시점의 문제에서 모든 고유벡터를 저장하지 않고도, "
            "fitted Vecchia precision의 <i>전체 스펙트럼에 걸친 누적 residual energy와 eigenscale별 energy</i>를 "
            "계산했다. 실제 유효 관측치 140,352개에서 전체 workflow는 147.84초였고, "
            "3,200차원 기준 문제에서 같은 연산자의 full eigen 결과와 cumulative energy/n RMSE 약 8.3×10<super>-4</super>로 일치했다.",
            bg=PALE_GOLD,
            border=GOLD,
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(pipeline_diagram())
    story.append(Spacer(1, 9 * mm))
    cover_rows = [
        ["분석 대상", "2024-07-03 GEMS TCO, adapted lag 4/3/2 Vecchia fit, batch size 64"],
        ["명목상 크기", "약 18,000 × 8 = 144,000"],
        ["실제 유효 크기", "n = 140,352 (결측 제거 후)"],
        ["대규모 설정", "Lanczos m = 512, Rademacher SLQ probes = 12, no reorthogonalization"],
        ["문서 목적", "방법의 수학적 정당성, 계산 가능성, 검증 범위, 남은 한계를 교수님께 명확히 설명"],
    ]
    t = make_table(cover_rows, [34 * mm, CONTENT_W - 34 * mm], header=False, font_size=8)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE_BLUE), ("TEXTCOLOR", (0, 0), (0, -1), NAVY)]))
    story.append(t)
    story.append(Spacer(1, 8 * mm))
    story.append(P("작성일: 2026-09-04 · 재현 가능한 기존 산출물과 실행 로그를 기준으로 정리", "KCaption"))
    story.append(PageBreak())


def add_exec_summary(story):
    story.append(P("1. 먼저 결론: 계산 장벽은 무엇이 해결되었는가", "KH1"))
    story.append(
        P(
            "원래 diagnostic의 목적은 고유벡터 140,352개를 개별적으로 열람하는 것이 아니다. "
            "관측 residual energy가 covariance의 큰 고유값(매끄럽고 큰 분산), 중간 고유값, 작은 고유값(거칠고 작은 분산) 영역에 "
            "어떻게 배분되는지를 보는 것이다. 이 목적은 full eigendecomposition의 출력물 <i>Q</i> 전체가 아니라 "
            "두 개의 spectral cumulative measure만 있으면 달성된다."
        )
    )
    story.append(
        equation(
            [
                "F<sub>Ω</sub>(t) = n<super>-1</super> tr 1{Ω ≤ t}",
                "G<sub>Ω,r</sub>(t) = n<super>-1</super> r<super>T</super> Ω 1{Ω ≤ t} r,   Ω = K<super>-1</super>",
            ]
        )
    )
    story.append(P("Matrix-free Lanczos와 SLQ는 <i>Q</i>를 만들지 않고도 바로 이 두 양을 근사한다. 따라서 다음 세 가지가 분리되어야 한다."))
    rows = [
        ["질문", "현재 답", "근거"],
        ["full eigendecomposition 없이 동일한 aggregate diagnostic을 계산할 수 있는가?", "예", "n=3,200에서 같은 연산자 full eigen과 Lanczos 비교"],
        ["n≈140,000에서 시간·메모리상 실행 가능한가?", "예", "전체 n=140,352, 147.84초, sparse B 210.96 MiB"],
        ["모든 eigenpair를 얻었는가?", "아니오", "필요한 spectral measure만 계산"],
        ["Vecchia 결과가 계산 불가능한 exact-K diagnostic과 동일한가?", "아직 아님", "이것은 별도의 approximation 질문"],
    ]
    story.append(make_table(rows, [68 * mm, 23 * mm, CONTENT_W - 91 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "교수님께 사용할 가장 정확한 표현",
            "“We compute the full-spectrum aggregate eigen-residual diagnostic of the fitted Vecchia precision matrix, "
            "matrix-free, without forming the dense covariance, dense precision, or the full eigenvector matrix.” "
            "이 표현은 계산한 대상과 계산하지 않은 대상을 동시에 정확히 밝힌다.",
            bg=PALE_TEAL,
            border=TEAL,
        )
    )
    story.append(P("왜 이것이 단순한 우회가 아니라 원래 diagnostic의 직접 계산인지, 다음 절부터 순서대로 보인다."))
    story.append(PageBreak())


def add_original_diagnostic(story):
    story.append(P("2. 작은 n에서 하던 full eigen diagnostic", "KH1"))
    story.append(P("작은 문제에서는 fitted covariance를 다음처럼 고유분해한다."))
    story.append(equation("K = Q Λ Q<super>T</super>,    λ<sub>1</sub> ≥ ··· ≥ λ<sub>n</sub> > 0"))
    story.append(P("GLS residual을 r이라 하면 각 covariance eigenmode의 standardized residual energy는"))
    story.append(equation("z<sub>j</sub><super>2</super> = (q<sub>j</sub><super>T</super>r)<super>2</super> / λ<sub>j</sub>"))
    story.append(P("이다. 올바르게 규정된 Gaussian model과 알려진 parameter 아래에서는 각 mode의 기대 energy가 1이다. 실제 진단에서는 개별 z<sub>j</sub><super>2</super>의 큰 변동을 그대로 읽기보다 누적곡선 또는 eigenscale band 평균을 사용한다."))
    story.append(P("해석의 방향", "KH2"))
    rows = [
        ["covariance scale", "precision scale", "대표적 해석", "energy/mode"],
        ["큰 λ", "작은 μ=1/λ", "large-variance, smooth/global mode", "1보다 크면 해당 scale의 residual 변동이 과다"],
        ["중간 λ", "중간 μ", "intermediate structure", "1 근처면 평균적으로 model과 일치"],
        ["작은 λ", "큰 μ", "small-variance, rough/local mode", "1보다 작으면 해당 scale의 residual 변동이 과소"],
    ]
    story.append(make_table(rows, [32 * mm, 32 * mm, 54 * mm, CONTENT_W - 118 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(P("누적곡선은 ‘앞에서부터 energy가 얼마나 쌓이는가’를, band 평균은 ‘어느 eigenscale 구간에서 기준 1을 벗어나는가’를 보여준다. 따라서 개별 eigenvector가 아니라 <i>스펙트럼에 따른 energy 배분</i>이 진단의 본체다."))
    story.append(P("18,000 × 8에서 full eigen이 막히는 이유", "KH2"))
    n = 140_352
    dense_gib = n * n * 8 / 2**30
    basis_mib = n * 512 * 8 / 2**20
    rows2 = [
        ["객체", "크기/저장량", "비고"],
        ["Dense K", f"{dense_gib:.2f} GiB", "float64 한 장만; actual n=140,352"],
        ["Dense K + Q", f"최소 {2*dense_gib:.2f} GiB", "eigensolver workspace와 복사본 제외"],
        ["Sparse B", "210.96 MiB", "18,387,361 nonzeros"],
        ["Lanczos basis (m=512)", f"약 {basis_mib:.2f} MiB", "현재 구현의 n×m float64 basis"],
    ]
    story.append(make_table(rows2, [41 * mm, 39 * mm, CONTENT_W - 80 * mm]))
    story.append(memory_chart())
    story.append(P("메모리뿐 아니라 dense symmetric eigendecomposition은 O(n<super>3</super>) 연산이다. n=140,352에서는 dense matrix를 저장할 수 있더라도 full eigen 계산 자체가 실용적이지 않다. 필요한 질문을 spectral measure로 바꾸는 것이 핵심이다."))
    story.append(PageBreak())


def add_spectral_reformulation(story):
    story.append(P("3. covariance eigen 문제를 precision spectral measure로 바꾸기", "KH1"))
    story.append(P("Ω=K<super>-1</super>이면 K와 Ω는 같은 고유벡터를 공유하고 μ<sub>j</sub>=1/λ<sub>j</sub>이다. covariance 고유값을 큰 순서로 보는 것은 precision 고유값을 작은 순서로 보는 것과 정확히 같다."))
    story.append(equation("K q<sub>j</sub> = λ<sub>j</sub>q<sub>j</sub>   ⇔   Ω q<sub>j</sub> = μ<sub>j</sub>q<sub>j</sub>,   μ<sub>j</sub>=λ<sub>j</sub><super>-1</super>"))
    story.append(P("threshold t 이하 precision eigenspace에 대하여"))
    story.append(
        equation(
            [
                "F<sub>Ω</sub>(t) = n<super>-1</super> Σ<sub>j</sub> 1{μ<sub>j</sub>≤t}",
                "G<sub>Ω,r</sub>(t) = n<super>-1</super> Σ<sub>j: μj≤t</sub> μ<sub>j</sub>(q<sub>j</sub><super>T</super>r)<super>2</super>",
            ]
        )
    )
    story.append(P("가 된다. G의 한 항은 μ<sub>j</sub>(q<sub>j</sub><super>T</super>r)<super>2</super>=(q<sub>j</sub><super>T</super>r)<super>2</super>/λ<sub>j</sub>=z<sub>j</sub><super>2</super>이므로, 이는 원래 full eigen diagnostic과 동일한 누적 energy다."))
    story.append(P("곡선과 band의 통계적 기준", "KH2"))
    story.append(P("parameter를 알고 있고 r∼N(0,Ω<super>-1</super>)이면 q<sub>j</sub><super>T</super>r∼N(0,μ<sub>j</sub><super>-1</super>)이므로 각 standardized energy의 기대값은 1이다. 따라서"))
    story.append(equation("E[G<sub>Ω,r</sub>(t)] = F<sub>Ω</sub>(t)"))
    story.append(P("이다. x축을 mode fraction F, y축을 cumulative energy/n G로 그리면 45도 선이 기준이다. mode fraction 구간 [a,b]의 band energy/mode는"))
    story.append(equation("E<sub>band</sub> = [G(t<sub>b</sub>)-G(t<sub>a</sub>)] / [F(t<sub>b</sub>)-F(t<sub>a</sub>)]"))
    story.append(P("로 계산하며 기준값은 1이다. 이 두 함수만 알면 원래 원했던 low-middle-high eigenscale 진단이 완성된다."))
    story.append(
        callout(
            "핵심 등가성",
            "Full eigen은 F와 G를 계산하는 한 가지 구현일 뿐이다. Lanczos/SLQ는 같은 spectral measure를 "
            "matrix-vector product만으로 계산하는 다른 구현이다. 따라서 ‘full spectrum을 본다’와 ‘full eigendecomposition을 한다’는 동일한 문장이 아니다.",
            bg=PALE_GOLD,
            border=GOLD,
        )
    )
    story.append(PageBreak())


def add_vecchia_operator(story):
    story.append(P("4. fitted Vecchia model이 sparse precision operator를 주는 이유", "KH1"))
    story.append(P("Vecchia likelihood는 각 관측 또는 block의 조건부분포를 제한된 과거 이웃에 조건화한다 [1-4]. 단일 관측 표기로 쓰면"))
    story.append(equation("y<sub>i</sub> | y<sub>N(i)</sub> ∼ N(a<sub>i</sub><super>T</super>y<sub>N(i)</sub>, d<sub>i</sub>)"))
    story.append(P("이고, 정규화된 conditional innovation은"))
    story.append(equation("e<sub>i</sub> = {y<sub>i</sub> - a<sub>i</sub><super>T</super>y<sub>N(i)</sub>} / √d<sub>i</sub>"))
    story.append(P("이다. 모든 innovation을 쌓으면 e=By가 된다. B의 각 행은 target과 소수의 conditioning neighbor에만 nonzero를 가지므로 희소하다. Vecchia joint density의 quadratic form은"))
    story.append(equation("||Br||<super>2</super> = r<super>T</super>B<super>T</super>Br,    Ω<super>~</super> = B<super>T</super>B"))
    story.append(P("로 쓸 수 있다. 그러나 Ω<super>~</super>를 명시적으로 만들 필요도 없다. 임의의 벡터 v에 대한 precision matvec를"))
    story.append(equation("v  ↦  Ω<super>~</super>v = B<super>T</super>(Bv)"))
    story.append(P("로 계산하면 된다. 이것이 Lanczos가 요구하는 유일한 연산이다."))
    story.append(P("이번 full-data operator의 실제 구조", "KH2"))
    rows = [
        ["항목", "값"],
        ["Vecchia geometry", "adapted directional corridor, lag 4/3/2"],
        ["fit batch size", "64 (저장된 fit 재사용, 이 실행에서 refit 없음)"],
        ["B shape", "140,352 × 140,352"],
        ["nonzeros", "18,387,361"],
        ["행당 nonzero", "평균 131.01, 중앙값 145, 최대 160"],
        ["density", "0.00093343"],
        ["CSR storage", "221,209,744 bytes = 210.96 MiB"],
    ]
    story.append(make_table(rows, [52 * mm, CONTENT_W - 52 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "구현 identity check",
            "Native Vecchia 코드의 residual quadratic form은 140,352.46559954, sparse B 계산은 "
            "140,352.46559641이었다. 상대오차는 2.23×10<super>-11</super>이며, reconstructed NLL과 저장된 NLL의 차이도 "
            "-1.11×10<super>-11</super>이다. 즉 sparse operator는 fitted model의 quadratic form을 수치적으로 재현한다.",
            bg=PALE_TEAL,
            border=TEAL,
        )
    )
    story.append(PageBreak())


def add_lanczos(story):
    story.append(P("5. Lanczos와 SLQ가 F(t), G(t)를 계산하는 방식", "KH1"))
    story.append(P("Lanczos는 symmetric operator Ω<super>~</super>와 시작벡터 q<sub>1</sub>만으로 Krylov subspace를 만든다."))
    story.append(equation("K<sub>m</sub>(Ω<super>~</super>,q<sub>1</sub>) = span{q<sub>1</sub>, Ω<super>~</super>q<sub>1</sub>, …, (Ω<super>~</super>)<super>m-1</super>q<sub>1</sub>}"))
    story.append(P("m번의 recurrence 후 큰 Ω<super>~</super> 대신 작은 m×m tridiagonal T<sub>m</sub>을 얻는다. T<sub>m</sub>의 eigenvalues는 quadrature nodes, 첫 성분 제곱은 spectral weights가 된다 [5-8]."))
    story.append(P("A. Residual-started Lanczos로 G 계산", "KH2"))
    story.append(P("q<sub>1</sub>=r/||r||로 시작하고 f<sub>t</sub>(x)=x1{x≤t}라 두면"))
    story.append(equation("r<super>T</super>f<sub>t</sub>(Ω<super>~</super>)r ≈ ||r||<super>2</super> e<sub>1</sub><super>T</super>f<sub>t</sub>(T<sub>m</sub>)e<sub>1</sub>"))
    story.append(P("이다. 이것이 G<sub>Ω~,r</sub>(t)의 numerator를 준다. residual은 한 개이므로 Lanczos run 한 번이면 된다."))
    story.append(P("B. Stochastic Lanczos quadrature로 F 계산", "KH2"))
    story.append(P("trace는 Rademacher probe g<sub>s</sub>를 사용한 Hutchinson identity로 바꾼다."))
    story.append(equation("tr h(Ω<super>~</super>) = E[g<super>T</super>h(Ω<super>~</super>)g] ≈ S<super>-1</super>Σ<sub>s=1</sub><super>S</super> g<sub>s</sub><super>T</super>h(Ω<super>~</super>)g<sub>s</sub>"))
    story.append(P("각 probe에 Lanczos를 적용하고 h<sub>t</sub>(x)=1{x≤t}를 넣으면 F<sub>Ω~</sub>(t)가 나온다. 이번 full run에서는 S=12, m=512를 사용했다."))
    story.append(P("실행 순서", "KH2"))
    steps = [
        "1) 저장된 fitted parameter와 GLS residual r을 읽는다.",
        "2) adapted lag 4/3/2 conditional coefficient로 sparse whitening matrix B를 만든다.",
        "3) 함수 A(v)=B<super>T</super>(Bv)를 정의한다. Dense Ω<super>~</super>는 만들지 않는다.",
        "4) r-start Lanczos로 T<sub>m,r</sub>을 만들고 G(t)를 계산한다.",
        "5) 12개 Rademacher-start Lanczos로 F(t)와 Monte Carlo standard error를 계산한다.",
        "6) F를 covariance mode fraction으로 사용해 cumulative curve와 20개 equal-mode band를 만든다.",
        "7) m=64,128,256,512를 비교해 Krylov truncation convergence를 확인한다.",
    ]
    for s in steps:
        story.append(bullet(s))
    story.append(pipeline_diagram())
    story.append(PageBreak())


def add_complexity(story):
    story.append(P("6. 계산복잡도와 실제 실행 비용", "KH1"))
    story.append(P("Dense full eigen의 병목을 ‘행렬 저장 + O(n<super>3</super>) factorization’에서 ‘희소 matvec의 반복’으로 바꾼다. nnz(B)를 B의 nonzero 수라 하면 Ω<super>~</super>v 한 번은 Bv와 B<super>T</super>w 두 번의 sparse multiply이므로 O(nnz(B))이다."))
    rows = [
        ["방법", "주요 저장", "주요 연산", "이번 문제"],
        ["Dense full eigen", "O(n²): K와 Q", "O(n³)", "K만 146.77 GiB; 비실용적"],
        ["Residual Lanczos", "O(nm) basis + B", "O(m·nnz(B))", "m=512, 9.45 s"],
        ["SLQ", "probe별 Lanczos workspace + B", "O(Sm·nnz(B))", "S=12, 110.91 s"],
        ["전체 workflow", "dense K/Q/Ω 없음", "13개 Lanczos run + 준비", "147.84 s"],
    ]
    story.append(make_table(rows, [37 * mm, 43 * mm, 42 * mm, CONTENT_W - 122 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(timing_chart())
    story.append(P("Timing breakdown", "KH2"))
    rows2 = [
        ["구간", "seconds"],
        ["data load", "0.149"],
        ["Vecchia precompute", "1.399"],
        ["GLS beta", "1.139"],
        ["native quadratic check", "1.153"],
        ["sparse B build", "1.890"],
        ["full residual Lanczos", "9.453"],
        ["full SLQ, 12 probes", "110.907"],
        ["workflow total", "147.840"],
    ]
    story.append(make_table(rows2, [CONTENT_W * 0.7, CONTENT_W * 0.3], aligns=["LEFT", "RIGHT"]))
    story.append(Spacer(1, 4 * mm))
    story.append(P("실행 환경은 macOS 15.7.9, arm64, 14 logical CPUs, Python 3.12.3이었다. 시간은 특정 하드웨어와 구현에 의존하지만, 이 실행은 적어도 ‘전체 자료에서 현실적인 wall-clock으로 가능하다’는 직접 증거다."))
    story.append(PageBreak())


def add_validation(story):
    story.append(P("7. 작은 문제에서 full eigen과 직접 맞춰 본 검증", "KH1"))
    story.append(P("n=3,200 = 400개 max-min 공간점 × 8시점 subset에서는 dense eigendecomposition이 가능하다. 이를 이용해 계산 대상과 근사 오차를 분리한 4-way 비교를 했다."))
    rows = [
        ["기호", "operator", "계산 방식", "역할"],
        ["E1", "exact covariance/precision", "full eigen", "exact operator 기준값"],
        ["E2", "E1과 동일", "precision Lanczos + SLQ", "Lanczos 수치오차 검증"],
        ["V1", "subset-specific Vecchia precision", "full eigen", "Vecchia operator 기준값"],
        ["V2", "V1과 동일", "precision Lanczos + SLQ", "Lanczos 수치오차 검증"],
    ]
    story.append(make_table(rows, [17 * mm, 50 * mm, 48 * mm, CONTENT_W - 115 * mm]))
    story.append(Spacer(1, 4 * mm))
    rows2 = [
        ["같은 연산자 비교, m=512", "energy/n RMSE", "spectral CDF RMSE"],
        ["E2 - E1: exact precision", "0.000831", "0.001511"],
        ["V2 - V1: Vecchia precision", "0.000822", "0.001285"],
    ]
    story.append(make_table(rows2, [79 * mm, 42 * mm, CONTENT_W - 121 * mm], aligns=["LEFT", "RIGHT", "RIGHT"]))
    story.append(validation_chart())
    story.append(P("그림 1. Lanczos order가 증가할수록 같은 operator의 cumulative energy/n RMSE가 감소한다. m=512에서는 exact precision과 Vecchia precision 모두 약 8×10<super>-4</super>이다.", "KCaption"))
    story.append(P("Full reorthogonalization과 no reorthogonalization도 m=512에서 각각 exact precision energy/n RMSE 0.000831과 0.000843이었다. 대규모 run은 메모리/시간을 고려해 no reorthogonalization을 사용했다."))
    story.append(
        callout(
            "검증의 올바른 결론",
            "E2-E1과 V2-V1은 Lanczos/SLQ가 <i>주어진 operator의 aggregate spectrum</i>을 얼마나 잘 계산하는지 검증한다. "
            "E1-V1은 operator 자체가 exact인지 Vecchia인지의 차이이며, Lanczos 정확도와 혼동하면 안 된다.",
            bg=PALE_GOLD,
            border=GOLD,
        )
    )
    story.append(PageBreak())


def add_fourway_figure(story):
    story.append(P("8. 4-way 검증 그림 읽는 법", "KH1"))
    story.append(fit_image(FOURWAY_FIG, max_h=108 * mm))
    story.append(P("그림 2. n=3,200 max-min subset의 E1-E2-V1-V2 비교. 위쪽은 누적 spectral diagnostic, 아래쪽은 band/오차 요약이다. 기존 재현 산출물을 그대로 삽입했다.", "KCaption"))
    story.append(P("왼쪽 두 비교가 핵심적인 numerical validation이다. E2가 E1과, V2가 V1과 가까우므로 matrix-free 경로는 같은 operator의 full eigen diagnostic을 재현한다. 반면 E1과 V1의 차이는 sparse max-min subset에서 새로 만든 Vecchia approximation의 차이다."))
    story.append(P("Hard band에 대한 주의", "KH2"))
    story.append(P("indicator 1{x≤t}는 불연속 함수라 Lanczos quadrature에서 가장 어려운 대상이다. m=512에서도 20개 hard-band energy/mode RMSE는 E2-E1 0.0421, V2-V1 0.0277이었다. 전체 자료의 5% band가 알려진 parameter 아래 갖는 이상적 표준편차 √(2/7018)≈0.0169보다 크다."))
    story.append(P("따라서 cumulative curve의 numerical validation은 강하지만, 현재 hard 20-band의 작은 굴곡과 최대편차 D는 탐색적으로 해석해야 한다. 다음 단계에서 smooth overlapping filters가 중요한 이유다."))
    story.append(
        callout(
            "Endpoint가 1인 것만으로는 검증이 아니다",
            "전체 endpoint r<super>T</super>Ω<super>~</super>r/n은 Lanczos가 선형함수를 매우 정확히 적분하고 fitted scale도 이를 1 근처로 맞추기 때문에 쉽게 일치한다. "
            "검증력은 endpoint가 아니라 threshold 전 범위의 curve RMSE와 order/probe convergence에서 나온다.",
            bg=PALE_RED,
            border=RED,
        )
    )
    story.append(PageBreak())


def add_full_results(story):
    story.append(P("9. n=140,352 전체 자료에서 실제로 얻은 diagnostic", "KH1"))
    story.append(fit_image(FULL_FIG, max_h=56 * mm))
    story.append(P("그림 3. Full-data matrix-free Vecchia precision diagnostic, m=512, SLQ probes=12. Dense K, Ω, Q는 생성하지 않았다.", "KCaption"))
    story.append(band_chart())
    story.append(P("그림 4. 20개 equal-mode band의 standardized energy/mode. 왼쪽은 large covariance/smooth scale, 오른쪽은 small covariance/rough scale이다. 가로선 1이 fitted model의 알려진-parameter 기준이다.", "KCaption"))
    story.append(P("관측된 broad pattern", "KH2"))
    story.append(bullet("첫 두 smooth/large-covariance bands: 0.811, 0.865."))
    story.append(bullet("중간의 여러 bands는 1보다 크고, 최대는 band 9의 1.132."))
    story.append(bullet("마지막 세 rough/small-covariance bands: 0.871, 0.866, 0.891."))
    story.append(bullet("전체 평균 standardized energy는 1.000003. fitted scale 때문에 endpoint 자체보다 비균일한 scale 배분이 정보다."))
    story.append(P("m=256에서 m=512로 올렸을 때 cumulative energy/n RMSE는 0.001095, spectral CDF RMSE는 0.000723이었다. 넓은 low-high-low pattern은 두 order에서 보였지만 hard threshold 기반 shape D는 8.718에서 6.323으로 아직 민감했다."))
    story.append(
        callout(
            "해석은 ‘어느 scale에서 어긋나는가’까지",
            "이 그림은 fitted pipeline이 residual variation을 어느 eigenscale에 과다/과소 배분하는지 보여준다. 그러나 이것만으로 mean misspecification, covariance-family misspecification, "
            "Vecchia neighbor approximation, parameter estimation 중 어느 것이 원인인지 단정하지 않는다.",
            bg=PALE_TEAL,
            border=TEAL,
        )
    )
    story.append(PageBreak())


def add_vecchia_target(story):
    story.append(P("10. 왜 E1-V1 일치는 원래 목적의 필수 관문이 아닌가", "KH1"))
    story.append(P("우리가 실제 fitting, likelihood, prediction과 uncertainty 계산에 사용하는 operational model이 fitted Vecchia model이라면, 그 model의 precision Ω<super>~</super>=B<super>T</super>B를 진단하는 것은 그 자체로 정당한 목표다."))
    story.append(P("알려진 parameter 아래에서 r∼N(0,Ω<super>~</super><super>-1</super>)이면"))
    story.append(equation("E[r<super>T</super> Ω<super>~</super> 1{Ω<super>~</super>≤t} r] = tr 1{Ω<super>~</super>≤t}"))
    story.append(P("이므로 Vecchia의 eigenbasis에서 energy/mode=1을 보는 diagnostic은 수학적으로 완결된다. exact K와 Ω<super>~</super>가 다를 수 있다는 사실은 diagnostic의 무효를 뜻하지 않는다. 진단 대상이 ‘exact GP’가 아니라 ‘실제로 사용한 fitted Vecchia pipeline’임을 뜻한다."))
    rows = [
        ["주장하려는 연구 목표", "E1-V1 일치 필요?", "설명"],
        ["fitted Vecchia model이 어느 eigenscale에서 data를 설명하지 못하는지 진단", "아니오", "Ω<super>~</super> 자체가 target"],
        ["large n에서 eigen-residual diagnostic을 가능하게 함", "아니오", "same-operator validation + full run이면 충분"],
        ["Vecchia curve를 infeasible exact full-K curve와 동일하다고 주장", "예", "approximation error를 별도 검증해야 함"],
        ["covariance-family failure와 neighbor approximation failure를 완전히 분리", "예", "추가 설계/비교 필요"],
    ]
    story.append(make_table(rows, [76 * mm, 26 * mm, CONTENT_W - 102 * mm]))
    story.append(P("Subset 결과가 알려 준 추가 정보", "KH2"))
    story.append(P("희박한 global max-min subset에서 E1-V1 cumulative curve/n RMSE는 0.01165로 같은-operator Lanczos 오차 약 0.00083보다 약 14배 컸다. 그러나 이 subset의 Vecchia B는 행당 평균 9.1 nonzeros에 불과해 full-data 평균 131과 전혀 다르다."))
    story.append(P("반면 conditioning density가 높은 contiguous 20×20 subset에서는 행당 평균 nonzero가 102.2로 늘고 E1-V1 curve RMSE가 0.00156으로 7.45배 줄었다. 이는 sparse max-min 결과를 full-data Vecchia approximation error로 그대로 옮기면 안 된다는 증거다."))
    story.append(fit_image(ROBUST_FIG, max_h=72 * mm))
    story.append(P("그림 5. Max-min과 contiguous subset의 robustness 비교. 이 비교는 exact-vs-Vecchia 주장 범위를 정교화하지만 matrix-free computational claim의 전제는 아니다.", "KCaption"))
    story.append(PageBreak())


def add_claims(story):
    story.append(P("11. 지금 강하게 말할 수 있는 것과 아직 말하면 안 되는 것", "KH1"))
    good = [
        "Full eigendecomposition 없이 fitted Vecchia precision의 full-spectrum aggregate eigen diagnostic을 계산했다.",
        "Residual-started Lanczos가 cumulative standardized energy G를, SLQ가 mode-count CDF F를 계산한다.",
        "n=3,200에서 같은 operator의 full eigen과 matrix-free 결과가 cumulative energy/n RMSE 약 8×10<super>-4</super>로 일치했다.",
        "n=140,352에서 sparse B<super>T</super>B matvec만으로 전체 workflow를 147.84초에 실행했다.",
        "fitted Vecchia operational model의 scale-dependent lack of fit을 보는 diagnostic으로 수학적으로 정당하다.",
    ]
    bad = [
        "140,352개의 개별 eigenvalues와 eigenvectors를 모두 계산했다.",
        "현재 full-data curve가 infeasible exact full-K eigen diagnostic과 동일함을 증명했다.",
        "현재 20개 hard band의 작은 차이와 shape D가 모두 numerical error보다 크고 통계적으로 유의하다.",
        "곡선의 이상이 covariance family 때문인지, Vecchia approximation 때문인지, mean/parameter estimation 때문인지 이미 분리했다.",
        "fitted parameter를 무시한 45도 기준만으로 정식 goodness-of-fit p-value를 얻었다.",
    ]
    cols = []
    for title, items, bg, border in [
        ("방어 가능한 주장", good, PALE_TEAL, TEAL),
        ("아직 제한해야 할 주장", bad, PALE_RED, RED),
    ]:
        content = [P(title, "KH2")] + [bullet(x) for x in items]
        t = Table([[content]], colWidths=[(CONTENT_W - 7 * mm) / 2])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, border), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        cols.append(t)
    outer = Table([cols], colWidths=[(CONTENT_W - 5 * mm) / 2] * 2, hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 5 * mm), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    story.append(outer)
    story.append(Spacer(1, 7 * mm))
    story.append(P("교수님과의 논의에서 예상되는 질문", "KH2"))
    qa = [
        ["질문", "짧은 답"],
        ["‘Full eigenanalysis’라고 불러도 되나?", "Full eigendecomposition이 아니라 full-spectrum aggregate eigen diagnostic이라고 부르는 것이 정확하다."],
        ["왜 exact K가 아닌가?", "실제 fitting/prediction에 사용한 operational Vecchia model 자체를 진단한다. Exact-K equivalence는 더 강한 별도 주장이다."],
        ["SLQ randomness는 충분한가?", "현재 12 probes는 feasibility run이다. 32/64/128 probe 및 seed 반복으로 Monte Carlo error를 고정해야 한다."],
        ["Band가 유의한가?", "아직 아니다. 불연속 hard filter error와 fitted-parameter uncertainty를 bootstrap으로 보정해야 한다."],
        ["그럼 성과는 무엇인가?", "이전에는 불가능했던 14만 차원 scale-resolved model diagnostic을 현실적인 operator cost로 계산 가능하게 만든 것이다."],
    ]
    story.append(make_table(qa, [57 * mm, CONTENT_W - 57 * mm]))
    story.append(PageBreak())


def add_next_steps(story):
    story.append(P("12. 정식 diagnostic으로 완성하기 위한 다음 단계", "KH1"))
    story.append(P("계산 가능성은 확보되었다. 이제 수치적·통계적 calibration을 강화하면 방법론의 설득력이 크게 높아진다."))
    rows = [
        ["우선순위", "작업", "목적", "판정 기준 예시"],
        ["1", "Smooth overlapping spectral filters", "hard indicator의 Gibbs/threshold 민감도 완화", "filter bandwidth를 바꿔도 broad pattern 유지"],
        ["2", "m = 512, 768, 1024", "Krylov truncation convergence", "curve/filter-energy 차이가 목표 tolerance 이하"],
        ["3", "probes = 32, 64, 128 + seed repeats", "SLQ Monte Carlo error", "probe SE와 seed variability 보고"],
        ["4", "Parametric bootstrap with refitting", "β와 covariance parameter estimation leverage 반영", "simulated envelopes와 global statistic calibration"],
        ["5", "Targeted Ritz-vector localization", "이상 scale을 공간·시간 pattern으로 번역", "전체 Q가 아닌 selected modes만 추출"],
        ["6", "Exact/Vecchia controlled simulations", "family failure와 approximation failure 분리", "truth-known scenarios에서 power/type-I error"],
    ]
    story.append(make_table(rows, [15 * mm, 48 * mm, 61 * mm, CONTENT_W - 124 * mm]))
    story.append(Spacer(1, 5 * mm))
    story.append(P("제안하는 최종 보고 구조", "KH2"))
    story.append(bullet("Main figure: smooth-filter energy profile with bootstrap envelope."))
    story.append(bullet("Numerical appendix: m/probe/seed convergence and same-operator n=3,200 validation."))
    story.append(bullet("Approximation appendix: max-min vs contiguous subset comparison, with conditioning density."))
    story.append(bullet("Interpretation appendix: selected Ritz vectors 또는 physical contrasts와의 연결."))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "추천 연구 메시지",
            "“We replace an infeasible O(n<super>3</super>) eigendecomposition by a matrix-free spectral diagnostic whose cost is governed by sparse Vecchia precision matvecs. "
            "The method reproduces full-eigen aggregate diagnostics on tractable problems and scales to all 140,352 observations. "
            "Its target is the fitted operational model; statistical calibration is handled separately by smooth filters and refitted bootstrap.”",
            bg=PALE_GOLD,
            border=GOLD,
        )
    )
    story.append(PageBreak())


def add_repro(story):
    story.append(P("부록 A. 재현 정보", "KH1"))
    story.append(P("분석 설정", "KH2"))
    rows = [
        ["항목", "값"],
        ["data day", "2024-07-03"],
        ["potential / valid observations", "145,008 / 140,352"],
        ["fit geometry", "adapted directional corridor, lag 4/3/2"],
        ["fit batch size", "64"],
        ["stored NLL", "1.37780259886663"],
        ["fitted sigma^2 / nugget", "13.072900596 / 1.482197171"],
        ["ranges (lat, lon, time)", "0.165402933, 0.193869398, 1.368325460"],
        ["advection (lat, lon)", "-0.073055191, -0.282999728"],
        ["full-data Lanczos", "m=512, 12 Rademacher probes, no reorthogonalization"],
        ["random seed", "20260903"],
    ]
    story.append(make_table(rows, [58 * mm, CONTENT_W - 58 * mm]))
    story.append(P("핵심 재현 파일", "KH2"))
    paths = [
        "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/real_full_vecchia_precision_lanczos_090326.py",
        "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/vecchia_sparse_precision_operator_090326.py",
        "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/real_maxmin400_exact_vs_subset_vecchia_4way_090326.py",
        "Exercises/st_model/day/amarel_simulation/space_time/interaction_diagnostic/compare_maxmin_contiguous_4way_090326.py",
    ]
    for p in paths:
        story.append(P(p, "KBodySmall"))
    story.append(P("핵심 결과 디렉터리", "KH2"))
    dirs = [
        str(FULL_DIR.relative_to(ROOT)),
        str(MAXMIN_DIR.relative_to(ROOT)),
        str((WORK / "real_contiguous20x20_exact_vs_subset_vecchia_4way_20240703_090326").relative_to(ROOT)),
        str(ROBUST_DIR.relative_to(ROOT)),
    ]
    for p in dirs:
        story.append(P(p, "KBodySmall"))
    story.append(Spacer(1, 3 * mm))
    story.append(P("재현성 메모", "KH2"))
    story.append(bullet("Full-data run은 저장된 fitted parameter를 재사용했으며 이 workflow 자체에서는 model refit을 하지 않았다."))
    story.append(bullet("4-way subset validation은 E1/E2/V1/V2 모두 같은 full-data GLS residual과 parameter를 사용했다."))
    story.append(bullet("Global max-min subset용 Vecchia는 selected observations에서 새로 구성했다. Full precision의 principal submatrix를 marginal precision처럼 해석하지 않았다."))
    story.append(bullet("Full-data B identity와 NLL reconstruction을 별도로 확인해 operator construction error를 배제했다."))
    story.append(PageBreak())


def add_references(story):
    story.append(P("부록 B. 참고문헌", "KH1"))
    refs = [
        "[1] Vecchia, A. V. (1988). Estimation and Model Identification for Continuous Spatial Processes. "
        "<i>Journal of the Royal Statistical Society: Series B</i>, 50(2), 297-312. "
        "<link href='https://doi.org/10.1111/j.2517-6161.1988.tb01729.x' color='#1E6688'>doi:10.1111/j.2517-6161.1988.tb01729.x</link>",
        "[2] Stein, M. L., Chi, Z., & Welty, L. J. (2004). Approximating likelihoods for large spatial data sets. "
        "<i>Journal of the Royal Statistical Society: Series B</i>, 66(2), 275-296. "
        "<link href='https://academic.oup.com/jrsssb/article/66/2/275/7098431' color='#1E6688'>article page</link>",
        "[3] Guinness, J. (2018). Permutation and Grouping Methods for Sharpening Gaussian Process Approximations. "
        "<i>Technometrics</i>, 60(4), 415-429. "
        "<link href='https://doi.org/10.1080/00401706.2018.1437476' color='#1E6688'>doi:10.1080/00401706.2018.1437476</link>",
        "[4] Katzfuss, M., & Guinness, J. (2021). A General Framework for Vecchia Approximations of Gaussian Processes. "
        "<i>Statistical Science</i>, 36(1), 124-141. "
        "<link href='https://doi.org/10.1214/19-STS755' color='#1E6688'>doi:10.1214/19-STS755</link>",
        "[5] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via Stochastic Lanczos Quadrature. "
        "<i>SIAM Journal on Matrix Analysis and Applications</i>, 38(4), 1075-1099. "
        "<link href='https://doi.org/10.1137/16M1104974' color='#1E6688'>doi:10.1137/16M1104974</link>",
        "[6] Chen, T., Trogdon, T., & Ubaru, S. (2025). Randomized Matrix-Free Quadrature: Unified and Uniform Bounds for "
        "Stochastic Lanczos Quadrature and the Kernel Polynomial Method. <i>SIAM Journal on Scientific Computing</i>, 47(3), A1733-A1757. "
        "<link href='https://doi.org/10.1137/23M1600414' color='#1E6688'>doi:10.1137/23M1600414</link>",
        "[7] Golub, G. H., & Meurant, G. (2009). <i>Matrices, Moments and Quadrature with Applications</i>. Princeton University Press.",
        "[8] Simon, H. D. (1984). The Lanczos algorithm with partial reorthogonalization. <i>Mathematics of Computation</i>, 42, 115-142.",
    ]
    for ref in refs:
        story.append(P(ref, "KBodySmall"))
        story.append(Spacer(1, 2 * mm))
    story.append(Spacer(1, 5 * mm))
    story.append(rule(BLUE, 1.0, 0, 7))
    story.append(P("마지막 요약", "KH2"))
    story.append(P("이 작업의 핵심 성과는 140,352차원 covariance의 full eigenvectors를 억지로 계산한 것이 아니다. 원래 diagnostic에 필요한 spectral measure를 정확히 식별하고, fitted Vecchia precision이 제공하는 sparse operator와 Lanczos/SLQ를 결합해 같은 질문을 dense matrix 없이 계산한 것이다. 작은 문제의 full eigen 비교는 계산 경로를 검증하고, 전체 자료 실행은 scalability를 입증한다. Exact-K equivalence와 formal goodness-of-fit calibration은 그 다음 단계의 독립된 연구 질문이다."))
    story.append(Spacer(1, 8 * mm))
    story.append(P("END OF TECHNICAL NOTE", "KCaption"))


def build():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for p in [FULL_FIG, FOURWAY_FIG, ROBUST_FIG]:
        if not p.exists():
            raise FileNotFoundError(p)

    story = []
    add_cover(story)
    add_exec_summary(story)
    add_original_diagnostic(story)
    add_spectral_reformulation(story)
    add_vecchia_operator(story)
    add_lanczos(story)
    add_complexity(story)
    add_validation(story)
    add_fourway_figure(story)
    add_full_results(story)
    add_vecchia_target(story)
    add_claims(story)
    add_next_steps(story)
    add_repro(story)
    add_references(story)

    doc = ReportDocTemplate(str(OUT))
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    build()
