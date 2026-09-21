from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp" / "pdfs" / "lanczos_guide_assets"
OUT = ROOT / "output" / "pdf" / "lanczos_spatial_spectral_diagnostic_guide.pdf"
TMP.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
MATH_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT = "AppleGothic"
pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)

NAVY = colors.HexColor("#14263D")
BLUE = colors.HexColor("#2F6BFF")
TEAL = colors.HexColor("#0C8F8F")
AMBER = colors.HexColor("#F2A93B")
RED = colors.HexColor("#C94C4C")
INK = colors.HexColor("#202B38")
MUTED = colors.HexColor("#5E6B78")
PALE_BLUE = colors.HexColor("#EDF3FF")
PALE_TEAL = colors.HexColor("#EAF8F6")
PALE_AMBER = colors.HexColor("#FFF5E3")
PALE_RED = colors.HexColor("#FDEEEE")
LINE = colors.HexColor("#D8E0E8")
WHITE = colors.white

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


styles = getSampleStyleSheet()
base = dict(fontName=FONT, wordWrap="CJK")

S = {
    "cover_kicker": ParagraphStyle(
        "cover_kicker", **base, fontSize=10, leading=14, textColor=TEAL, spaceAfter=10
    ),
    "cover_title": ParagraphStyle(
        "cover_title", **base, fontSize=28, leading=38, textColor=NAVY, spaceAfter=14
    ),
    "cover_subtitle": ParagraphStyle(
        "cover_subtitle", **base, fontSize=13, leading=21, textColor=MUTED, spaceAfter=18
    ),
    "h1": ParagraphStyle(
        "h1", **base, fontSize=20, leading=28, textColor=NAVY, spaceBefore=3, spaceAfter=9
    ),
    "h2": ParagraphStyle(
        "h2", **base, fontSize=14, leading=20, textColor=NAVY, spaceBefore=11, spaceAfter=6
    ),
    "h3": ParagraphStyle(
        "h3", **base, fontSize=11.5, leading=17, textColor=TEAL, spaceBefore=7, spaceAfter=4
    ),
    "body": ParagraphStyle(
        "body", **base, fontSize=9.8, leading=16.2, textColor=INK, spaceAfter=6
    ),
    "small": ParagraphStyle(
        "small", **base, fontSize=8.3, leading=13, textColor=MUTED, spaceAfter=4
    ),
    "bullet": ParagraphStyle(
        "bullet", **base, fontSize=9.5, leading=15.5, textColor=INK, leftIndent=12, firstLineIndent=-8, bulletIndent=2, spaceAfter=3
    ),
    "number": ParagraphStyle(
        "number", **base, fontSize=9.5, leading=15.5, textColor=INK, leftIndent=16, firstLineIndent=-12, spaceAfter=4
    ),
    "callout": ParagraphStyle(
        "callout", **base, fontSize=10.2, leading=17, textColor=NAVY, alignment=TA_LEFT
    ),
    "caption": ParagraphStyle(
        "caption", **base, fontSize=8, leading=12, textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8
    ),
    "toc": ParagraphStyle(
        "toc", **base, fontSize=10, leading=17, textColor=INK, leftIndent=4, spaceAfter=2
    ),
    "table": ParagraphStyle(
        "table", **base, fontSize=8.2, leading=12, textColor=INK
    ),
    "table_head": ParagraphStyle(
        "table_head", **base, fontSize=8.4, leading=12, textColor=WHITE, alignment=TA_CENTER
    ),
    "formula_label": ParagraphStyle(
        "formula_label", **base, fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER
    ),
    "ref": ParagraphStyle(
        "ref", **base, fontSize=8.1, leading=12.8, textColor=MUTED, leftIndent=11, firstLineIndent=-11, spaceAfter=4
    ),
}


class Rule(Flowable):
    def __init__(self, width=CONTENT_W, color=LINE, thickness=0.7, space=6):
        super().__init__()
        self.width = width
        self.height = space
        self.color = color
        self.thickness = thickness

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.height / 2, self.width, self.height / 2)


def callout(text: str, bg=PALE_BLUE, border=BLUE, padding=9) -> Table:
    t = Table([[P(text, S["callout"])]], colWidths=[CONTENT_W])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), padding),
                ("RIGHTPADDING", (0, 0), (-1, -1), padding),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    return t


def info_table(rows, widths, header=True, aligns=None, font_size=8.2):
    cooked = []
    for i, row in enumerate(rows):
        cooked.append(
            [
                P(str(cell), S["table_head"] if header and i == 0 else S["table"])
                for cell in row
            ]
        )
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        if len(rows) > 1:
            commands += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")])]
    else:
        commands += [("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")])]
    if aligns:
        for col, align in enumerate(aligns):
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), align))
    t.setStyle(TableStyle(commands))
    return t


DISPLAY_MAP = {
    r"K=Q\Lambda Q^{\mathsf T},\qquad \Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)": "K = QΛQ^T,     Λ = diag(λ[1], …, λ[n])",
    r"z_j=\frac{q_j^{\mathsf T}r}{\sqrt{\lambda_j}},\qquad z_j^2=\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}": "z[j] = q[j]^T r / √λ[j],     z[j]^2 = (q[j]^T r)^2 / λ[j]",
    r"C_k=\sum_{j=1}^{k}z_j^2=\sum_{j=1}^{k}\frac{(q_j^{\mathsf T}r)^2}{\lambda_j},\qquad \mathbb E(C_k)\approx k": "C[k] = Σ(j=1,…,k) z[j]^2 = Σ(j=1,…,k) (q[j]^T r)^2/λ[j],     E(C[k]) ≈ k",
    r"\Omega=B^{\mathsf T}B\quad\Longrightarrow\quad \Omega v=B^{\mathsf T}(Bv)": "Ω = B^T B     =>     Ωv = B^T(Bv)",
    r"Kq_j=\lambda_jq_j\quad\Longrightarrow\quad \Omega q_j=\mu_jq_j,\qquad \mu_j=\lambda_j^{-1}": "Kq[j] = λ[j]q[j]     =>     Ωq[j] = μ[j]q[j],     μ[j] = λ[j]^(-1)",
    r"\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}=\mu_j(q_j^{\mathsf T}r)^2": "(q[j]^T r)^2 / λ[j] = μ[j](q[j]^T r)^2",
    r"\mathcal K_m(\Omega,r)=\mathrm{span}\{r,\Omega r,\Omega^2r,\ldots,\Omega^{m-1}r\}": "Krylov_m(Ω,r) = span{r, Ωr, Ω^2r, …, Ω^(m-1)r}",
    r"T_m=Q_m^{\mathsf T}\Omega Q_m,\qquad \Omega Q_m=Q_mT_m+\beta_mq_{m+1}e_m^{\mathsf T}": "T_m = Q_m^T ΩQ_m,     ΩQ_m = Q_m T_m + β_m q_(m+1) e_m^T",
    r"T_m=\begin{bmatrix}\alpha_1&\beta_1&&\\\beta_1&\alpha_2&\ddots&\\&\ddots&\ddots&\beta_{m-1}\\&&\beta_{m-1}&\alpha_m\end{bmatrix}": "T_m = tridiag(β_1,…,β_(m-1) ; α_1,…,α_m ; β_1,…,β_(m-1))",
    r"r^{\mathsf T}f(\Omega)r\;\approx\;\lVert r\rVert^2e_1^{\mathsf T}f(T_m)e_1": "r^T f(Ω)r  ≈  ||r||^2 e_1^T f(T_m)e_1",
    r"\lVert r\rVert^2e_1^{\mathsf T}f(T_m)e_1=\sum_{\ell=1}^{m}\underbrace{\lVert r\rVert^2U_{1\ell}^2}_{w_\ell}\,f(\theta_\ell)": "||r||^2 e_1^T f(T_m)e_1 = Σ(l=1,…,m) w_l f(θ_l),     w_l = ||r||^2 U_(1l)^2",
    r"f_t(\mu)=\mu\,\mathbf 1(\mu\le t)": "f_t(μ) = μ · 1(μ ≤ t)",
    r"r^{\mathsf T}f_t(\Omega)r=\sum_{\mu_j\le t}\mu_j(q_j^{\mathsf T}r)^2=\sum_{\lambda_j\ge 1/t}\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}": "r^T f_t(Ω)r = Σ(μ[j]≤t) μ[j](q[j]^T r)^2 = Σ(λ[j]≥1/t) (q[j]^T r)^2/λ[j]",
    r"G_r(t)=\frac{1}{n}r^{\mathsf T}\Omega\mathbf 1(\Omega\le t)r=\frac{1}{n}\sum_{\mu_j\le t}\mu_j(q_j^{\mathsf T}r)^2": "G_r(t) = n^(-1) r^T Ω1(Ω≤t)r = n^(-1) Σ(μ[j]≤t) μ[j](q[j]^T r)^2",
    r"F_\Omega(t)=\frac{1}{n}\operatorname{tr}\mathbf 1(\Omega\le t)=\frac{1}{n}\sum_{j=1}^{n}\mathbf 1(\mu_j\le t)": "F_Ω(t) = n^(-1) tr 1(Ω≤t) = n^(-1) Σ(j=1,…,n) 1(μ[j]≤t)",
    r"\mathbb E_g[g^{\mathsf T}f(\Omega)g]=\operatorname{tr}f(\Omega)": "E_g[g^T f(Ω)g] = tr f(Ω)",
    r"\operatorname{tr}f(\Omega)\approx\frac{1}{s}\sum_{i=1}^{s}\lVert g_i\rVert^2e_1^{\mathsf T}f(T_m^{(i)})e_1": "tr f(Ω) ≈ s^(-1) Σ(i=1,…,s) ||g[i]||^2 e_1^T f(T_m^(i))e_1",
    r"R_B=\frac{1}{|B|}\sum_{j\in B}z_j^2": "R_B = |B|^(-1) Σ(j∈B) z[j]^2",
    r"\boxed{\text{sparse Vecchia precision}+\text{matrix-free matvec}+\text{Lanczos/SLQ}}": "sparse Vecchia precision  +  matrix-free matvec  +  Lanczos/SLQ",
    r"z_j^2=\mu_j(q_j^{\mathsf T}r)^2=(q_j^{\mathsf T}r)^2/\lambda_j": "z[j]^2 = μ[j](q[j]^T r)^2 = (q[j]^T r)^2/λ[j]",
    r"\mathbb E[G_r(t)\mid\Omega]\approx F_\Omega(t)": "E[G_r(t) | Ω] ≈ F_Ω(t)",
    r"G_r(\infty)=n^{-1}r^{\mathsf T}\Omega r": "G_r(∞) = n^(-1) r^T Ωr",
    r"\mathrm{Cost}_{\mathrm{residual}}\approx O(m\,\mathrm{nnz}(B))": "Cost(residual) ≈ O(m · nnz(B))",
    r"\mathrm{Cost}_{\mathrm{SLQ}}\approx O(s\,m\,\mathrm{nnz}(B))": "Cost(SLQ) ≈ O(s · m · nnz(B))",
}


def equation(tex: str, width=150 * mm, fontsize=17, label=None):
    key = str(abs(hash(("ascii-index-v3", tex, width, fontsize))))
    path = TMP / f"eq_{key}.png"
    if not path.exists():
        display = DISPLAY_MAP.get(tex, tex)
        font = ImageFont.truetype(MATH_FONT_PATH, int(fontsize * 2.8))
        probe = PILImage.new("RGBA", (10, 10), (255, 255, 255, 0))
        pdraw = ImageDraw.Draw(probe)
        bbox = pdraw.textbbox((0, 0), display, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        canvas = PILImage.new("RGBA", (max(120, tw + 44), max(70, th + 32)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((canvas.width / 2, canvas.height / 2), display, font=font, fill="#14263D", anchor="mm")
        canvas.save(path)
    with PILImage.open(path) as im:
        ratio = im.height / im.width
    img = Image(str(path), width=width, height=width * ratio)
    if label:
        return [img, P(label, S["formula_label"]), Spacer(1, 3)]
    return [img, Spacer(1, 3)]


def make_figures():
    body = ImageFont.truetype(FONT_PATH, 34)
    small = ImageFont.truetype(FONT_PATH, 27)
    label = ImageFont.truetype(FONT_PATH, 31)

    def center_text(draw, xy, text, font, fill):
        draw.text(xy, text, font=font, fill=fill, anchor="mm")

    # Figure 1: computational pipelines
    im = PILImage.new("RGB", (2200, 840), "white")
    d = ImageDraw.Draw(im)
    d.text((50, 55), "FULL EIGEN", font=label, fill="#C94C4C")
    d.text((50, 455), "MATRIX-FREE LANCZOS", font=label, fill="#0C8F8F")
    xs = [50, 610, 1170, 1730]
    texts1 = ["dense K 또는 Ω 생성", "모든 n개 eigenpair", "mode별 z^2 계산", "누적·band 집계"]
    texts2 = ["v → B^T(Bv)", "m-step Krylov basis", "작은 T_m 분석", "집계량 직접 근사"]
    for y, texts, fc, ec in [(145, texts1, "#FDEEEE", "#C94C4C"), (545, texts2, "#EAF8F6", "#0C8F8F")]:
        for i, (x, txt) in enumerate(zip(xs, texts)):
            d.rounded_rectangle((x, y, x+420, y+145), radius=18, fill=fc, outline=ec, width=4)
            center_text(d, (x+210, y+72), txt, body, "#202B38")
            if i < 3:
                d.line((x+432, y+72, xs[i+1]-18, y+72), fill=ec, width=5)
                d.polygon([(xs[i+1]-18, y+72), (xs[i+1]-36, y+60), (xs[i+1]-36, y+84)], fill=ec)
    im.save(TMP / "pipeline.png")

    # Figure 2: eigenvalue order reversal
    im = PILImage.new("RGB", (1900, 800), "white")
    d = ImageDraw.Draw(im)
    d.text((65, 235), "K", font=ImageFont.truetype(FONT_PATH, 52), fill="#2F6BFF", anchor="mm")
    d.text((65, 560), "Ω", font=ImageFont.truetype(FONT_PATH, 52), fill="#0C8F8F", anchor="mm")
    d.text((160, 95), "큰 covariance eigenvalue λ", font=small, fill="#2F6BFF")
    d.text((1740, 95), "작은 λ", font=small, fill="#2F6BFF", anchor="ra")
    d.text((160, 710), "작은 precision eigenvalue μ = 1/λ", font=small, fill="#0C8F8F")
    d.text((1740, 710), "큰 μ", font=small, fill="#0C8F8F", anchor="ra")
    positions = [190 + i*190 for i in range(9)]
    sizes = [60, 55, 50, 45, 40, 35, 30, 25, 20]
    for i, (x, radius) in enumerate(zip(positions, sizes)):
        d.ellipse((x-radius, 235-radius, x+radius, 235+radius), fill="#6B91FF", outline="white", width=3)
        xr = positions[-1-i]
        rr = sizes[-1-i]
        d.ellipse((xr-rr, 560-rr, xr+rr, 560+rr), fill="#52B7AF", outline="white", width=3)
        d.line((x, 300, xr, 495), fill="#C6CFD8", width=2)
    center_text(d, (950, 400), "같은 eigenvector · 역순의 eigenvalue", body, "#5E6B78")
    im.save(TMP / "spectrum_mapping.png")

    # Figure 3: conceptual cumulative diagnostic
    im = PILImage.new("RGB", (1800, 980), "white")
    d = ImageDraw.Draw(im)
    left, top, right, bottom = 210, 80, 1700, 800
    d.line((left, bottom, right, bottom), fill="#202B38", width=4)
    d.line((left, bottom, left, top), fill="#202B38", width=4)
    for k in range(1, 5):
        y = bottom - k*(bottom-top)/5
        d.line((left, y, right, y), fill="#E2E8EE", width=2)
    def pt(x, y):
        return (left + x*(right-left), bottom - y*(bottom-top))
    ideal, surplus, deficit = [], [], []
    prev_s, prev_d = 0.0, 0.0
    for i in range(201):
        x = i/200
        s = x + 0.85*math.exp(-((x-0.22)/0.17)**2)*x*(1-x)
        de = x - 0.70*math.exp(-((x-0.78)/0.18)**2)*x*(1-x)
        prev_s = max(prev_s, min(1.0, max(0.0, s)))
        prev_d = max(prev_d, min(1.0, max(0.0, de)))
        ideal.append(pt(x, x)); surplus.append(pt(x, prev_s)); deficit.append(pt(x, prev_d))
    d.line(ideal, fill="#14263D", width=6)
    d.line(surplus, fill="#C94C4C", width=6)
    d.line(deficit, fill="#2F6BFF", width=6)
    d.text(((left+right)/2, 900), "누적 mode 비율  FΩ(t)", font=body, fill="#202B38", anchor="mm")
    d.text((55, (top+bottom)/2), "누적 residual energy  Gᵣ(t)", font=small, fill="#202B38", anchor="mm")
    legend = [("#14263D", "기대선: G ≈ F"), ("#C94C4C", "large-scale energy 과다 예시"), ("#2F6BFF", "small-scale energy 부족 예시")]
    for i, (color, text) in enumerate(legend):
        yy = 130 + i*58
        d.line((260, yy, 340, yy), fill=color, width=6)
        d.text((365, yy), text, font=small, fill="#202B38", anchor="lm")
    im.save(TMP / "diagnostic_curve.png")


class GuideDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="Matrix-free Lanczos/SLQ 공간 스펙트럴 진단",
            author="OpenAI Codex",
            subject="Vecchia precision operator를 이용한 scalable spatial residual spectral diagnostic",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="normal")
        self.addPageTemplates(PageTemplate(id="body", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(LEFT, PAGE_H - 11 * mm, PAGE_W - RIGHT, PAGE_H - 11 * mm)
            canvas.setFont(FONT, 7.5)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, PAGE_H - 8.2 * mm, "MATRIX-FREE LANCZOS / SLQ · SPATIAL SPECTRAL DIAGNOSTIC")
            canvas.drawRightString(PAGE_W - RIGHT, 9 * mm, f"{page - 1}")
        canvas.restoreState()


def section(story, num, title, lead=None):
    block = [P(f"{num}. {title}", S["h1"]), Rule()]
    if lead:
        block.append(P(lead, S["body"]))
    story.append(KeepTogether(block))


def bullet(story, text):
    story.append(P(f"• {text}", S["bullet"]))


def numbered(story, n, text):
    story.append(P(f"{n}. {text}", S["number"]))


def add_equation(story, tex, width=150 * mm, fontsize=17, label=None):
    story.extend(equation(tex, width, fontsize, label))


def build_story():
    make_figures()
    story = []

    # Cover
    story += [Spacer(1, 23 * mm)]
    story.append(P("TECHNICAL GUIDE · SPATIAL STATISTICS", S["cover_kicker"]))
    story.append(P("Matrix-free Lanczos와 SLQ를 이용한<br/>대규모 공간 스펙트럴 진단", S["cover_title"]))
    story.append(P("Full eigendecomposition 없이 Vecchia precision operator에서<br/>mode 비율과 residual energy를 추정하는 원리, 계산량, 해석 및 검증", S["cover_subtitle"]))
    story.append(Spacer(1, 7 * mm))
    story.append(callout(
        "<b>핵심 한 문장</b><br/>모든 eigenvalue와 eigenvector를 구하지 않고, "
        "<font color='#0C8F8F'>v → Ωv = B^T(Bv)</font>만 반복하여 진단에 필요한 aggregate spectral quantity를 직접 근사한다.",
        bg=PALE_TEAL,
        border=TEAL,
        padding=12,
    ))
    story.append(Spacer(1, 14 * mm))
    cover_rows = [
        [P("대상 규모", S["small"]), P("n = 140,352", S["body"])],
        [P("Lanczos 차수", S["small"]), P("m = 512 (전체 차원의 약 0.365%)", S["body"])],
        [P("Precision factor", S["small"]), P("Ω = B^T B,  nnz(B) ≈ 18.4 million", S["body"])],
        [P("핵심 산출물", S["small"]), P("mode CDF FΩ(t), residual-energy CDF Gr(t), band energy, shape discrepancy", S["body"])],
    ]
    ct = Table(cover_rows, colWidths=[39 * mm, 112 * mm])
    ct.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(ct)
    story.append(Spacer(1, 17 * mm))
    story.append(P("Prepared from the supplied technical note · 2026-09-05", S["small"]))
    story.append(PageBreak())

    # Contents and notation
    story.append(P("문서 구성", S["h1"]))
    story.append(Rule())
    toc = [
        "1. 문제와 결론", "2. Full eigen 진단", "3. Matrix-free precision operator",
        "4. Covariance와 precision spectrum", "5. Lanczos 압축 원리", "6. Quadratic-form spectral approximation",
        "7. Residual Lanczos와 SLQ", "8. Aggregate spectral quantities", "9. 계산복잡도와 실제 규모",
        "10. 작은 subset에서 full eigen이 빠른 이유", "11. 해석 방법", "12. 한계와 수치적 주의점",
        "13. 구현 및 검증 체크리스트", "14. 최종 요약", "부록 A. 기호와 공식", "부록 B. 참고 문헌",
    ]
    for item in toc:
        story.append(P(item, S["toc"]))
    story.append(Spacer(1, 7 * mm))
    story.append(P("먼저 구분할 용어", S["h2"]))
    story.append(info_table([
        ["용어", "이 문서에서의 의미"],
        ["Lanczos algorithm", "대칭행렬을 Krylov subspace 위의 작은 삼대각행렬로 투영하는 반복 알고리즘"],
        ["Lanczos eigensolver", "작은 투영행렬의 Ritz eigenpair를 통해 원행렬의 일부 eigenpair를 근사하는 방식"],
        ["Lanczos quadrature", "eigenpair 자체보다 v^T f(Ω)v 같은 spectral integral을 직접 근사하는 방식"],
        ["SLQ", "random probe와 Lanczos quadrature를 결합해 tr f(Ω)를 추정하는 방식"],
        ["Lanczos approximation", "감마함수 근사를 뜻하기도 하므로 본 맥락에서는 피하는 표현"],
    ], [38*mm, 123*mm]))
    story.append(PageBreak())

    # Section 1
    section(story, 1, "문제와 결론", "관측치 수가 14만을 넘는 공간모형에서는 dense covariance 또는 precision matrix의 full eigendecomposition이 저장 단계부터 현실적이지 않다. 그러나 진단 목표가 개별 eigenvector가 아니라 넓은 eigenscale별 누적 energy라면, 전체 분해는 필요하지 않다.")
    story.append(Image(str(TMP / "pipeline.png"), width=CONTENT_W, height=CONTENT_W * 4.2 / 11))
    story.append(P("그림 1. Full eigen은 모든 mode를 계산한 뒤 집계한다. Lanczos/SLQ는 필요한 집계량을 operator action으로 직접 근사한다.", S["caption"]))
    story.append(callout(
        "<b>결론</b><br/>현재 방법의 계산 구조는 dense <i>O(n^3)</i> eigendecomposition이 아니라, "
        "sparse matvec을 반복하는 <i>O(m · nnz(B))</i> 구조다. SLQ probe가 s개면 중심 비용은 <i>O(s · m · nnz(B))</i>다.",
        bg=PALE_BLUE, border=BLUE
    ))
    story.append(P("이 절약이 가능한 이유는 Lanczos 하나가 아니라 다음 세 요소의 결합이다.", S["body"]))
    bullet(story, "Vecchia 모형이 제공하는 sparse precision factor B")
    bullet(story, "Ω를 만들지 않고 Ωv = B^T(Bv)만 계산하는 matrix-free operator")
    bullet(story, "모든 eigenpair 대신 aggregate spectral target만 근사하는 Lanczos/SLQ")

    # Section 2
    section(story, 2, "Full eigen 진단은 무엇을 계산하는가")
    story.append(P("공분산행렬 K가 symmetric positive definite이고 다음과 같이 분해된다고 하자.", S["body"]))
    add_equation(story, r"K=Q\Lambda Q^{\mathsf T},\qquad \Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)")
    story.append(P("여기서 q[j]는 j번째 spatial eigenmode이고 λ[j]는 해당 mode의 분산이다. residual r를 q[j] 방향으로 투영한 뒤 그 mode의 표준편차로 나누면 standardized coordinate가 된다.", S["body"]))
    add_equation(story, r"z_j=\frac{q_j^{\mathsf T}r}{\sqrt{\lambda_j}},\qquad z_j^2=\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}")
    story.append(P("모형이 맞고 평균구조 및 모수 추정의 영향을 잠시 무시하면 q[j]^T r은 N(0, λ[j])이므로 z[j]는 N(0,1), z[j]^2의 기대값은 1이다. 따라서 큰 covariance eigenvalue부터 정렬한 누적량은 다음과 같다.", S["body"]))
    add_equation(story, r"C_k=\sum_{j=1}^{k}z_j^2=\sum_{j=1}^{k}\frac{(q_j^{\mathsf T}r)^2}{\lambda_j},\qquad \mathbb E(C_k)\approx k")
    story.append(P("C[k]/k가 1보다 크면 해당 누적 band에 기대보다 많은 standardized residual energy가 있고, 1보다 작으면 적다. 이 방식은 해석이 직접적이지만 모든 λ[j]와 q[j]를 요구한다.", S["body"]))
    story.append(P("Dense full decomposition의 병목", S["h2"]))
    story.append(info_table([
        ["항목", "복잡도", "n = 140,352에서의 규모"],
        ["행렬 저장", "O(n^2)", "19,698,683,904개 원소"],
        ["Float64 한 행렬", "8n^2 bytes", "약 157.6 GB = 146.8 GiB"],
        ["Full symmetric eigen", "O(n^3)", "n^3 ≈ 2.765 × 10^15 규모"],
        ["Eigenvector Q 저장", "O(n^2)", "K와 별도로 다시 약 157.6 GB"],
    ], [39*mm, 35*mm, 87*mm]))
    story.append(P("실제 eigensolver의 상수와 병렬화 수준에 따라 시간은 달라지지만, 이 규모에서는 메모리 요구량만으로도 dense full eigen 전략을 배제할 수 있다.", S["small"]))

    # Section 3
    section(story, 3, "Matrix-free precision operator")
    story.append(P("Matrix-free는 수학적 행렬 Ω가 존재하지 않는다는 뜻이 아니다. Ω의 모든 원소를 배열로 만들거나 저장하지 않고, 임의의 vector v에 대한 Ωv를 반환하는 함수만 제공한다는 뜻이다.", S["body"]))
    add_equation(story, r"\Omega=B^{\mathsf T}B\quad\Longrightarrow\quad \Omega v=B^{\mathsf T}(Bv)")
    story.append(P("Vecchia factor B가 sparse하면 Bv와 B^T u 모두 sparse matrix-vector multiplication으로 처리된다. 따라서 한 번의 Ωv 계산은 B의 nonzero를 대략 두 차례 통과하는 구조다.", S["body"]))
    story.append(info_table([
        ["하지 않는 계산", "실제로 수행하는 계산"],
        ["dense K 생성", "sparse B 저장"],
        ["dense Ω 생성", "Bv와 B^T u"],
        ["Ω^(-1) 계산", "operator Ωv 호출"],
        ["모든 eigenpair 계산", "m차 Krylov projection"],
    ], [80.5*mm, 80.5*mm]))
    story.append(Spacer(1, 5))
    story.append(callout(
        "<b>정확성과 근사의 위치</b><br/>B가 정의하는 fitted Vecchia precision Ω = B^T B에 대한 Ωv는 정확하다. "
        "Lanczos가 근사하는 것은 이 operator의 전체 spectral information이다. 다만 Ω 자체가 원래 exact covariance model의 inverse를 Vecchia 방식으로 근사한 것이라면, 그 모형 근사오차는 별도로 존재한다.",
        bg=PALE_AMBER, border=AMBER
    ))

    # Section 4
    section(story, 4, "Covariance와 precision spectrum은 어떻게 연결되는가")
    story.append(P("Ω = K^(-1)이거나, 더 정확히는 Ω가 fitted approximate covariance K-tilde의 inverse라고 하자. K와 Ω는 eigenvector를 공유하며 eigenvalue는 역수 관계다.", S["body"]))
    add_equation(story, r"Kq_j=\lambda_jq_j\quad\Longrightarrow\quad \Omega q_j=\mu_jq_j,\qquad \mu_j=\lambda_j^{-1}")
    story.append(Image(str(TMP / "spectrum_mapping.png"), width=150*mm, height=150*mm*4.4/10.5))
    story.append(P("그림 2. 큰 covariance variance mode는 작은 precision eigenvalue mode에 해당한다. 정렬 순서가 뒤집힌다.", S["caption"]))
    add_equation(story, r"\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}=\mu_j(q_j^{\mathsf T}r)^2")
    story.append(P("따라서 K에서 큰 λ부터 보던 기존 진단은 Ω에서 작은 μ부터 보면 동일하다. 이 대응은 low-precision spectrum을 large-scale covariance variation으로 해석하게 해 준다. 단, spatial scale과 eigenvalue의 관계는 covariance kernel과 sampling geometry에 따라 달라지므로 eigenvalue 크기를 물리적 파장과 일대일로 동일시하면 안 된다.", S["body"]))

    # Section 5
    section(story, 5, "Lanczos가 큰 행렬을 작은 문제로 압축하는 원리")
    story.append(P("Residual 방향의 spectral information이 목표라면 시작 vector를 q[1] = r/||r||로 둔다. Lanczos는 Ω를 반복 적용해서 Krylov subspace를 생성한다.", S["body"]))
    add_equation(story, r"\mathcal K_m(\Omega,r)=\mathrm{span}\{r,\Omega r,\Omega^2r,\ldots,\Omega^{m-1}r\}")
    story.append(P("이 공간의 orthonormal basis Q_m = [q[1],…,q[m]]를 구성하면, Ω는 Q_m 위에서 작은 symmetric tridiagonal matrix T_m으로 표현된다.", S["body"]))
    add_equation(story, r"T_m=Q_m^{\mathsf T}\Omega Q_m,\qquad \Omega Q_m=Q_mT_m+\beta_mq_{m+1}e_m^{\mathsf T}")
    story.append(P("한 단계의 short recurrence는 다음과 같다.", S["h2"]))
    numbered(story, 1, "w ← Ωq[j] = B^T(Bq[j])를 계산한다.")
    numbered(story, 2, "α[j] ← q[j]^T w를 계산하고 w ← w - α[j]q[j] - β[j-1]q[j-1]로 직교화한다.")
    numbered(story, 3, "β[j] ← ||w||, q[j+1] ← w/β[j]로 다음 basis vector를 만든다.")
    add_equation(story, r"T_m=\begin{bmatrix}\alpha_1&\beta_1&&\\\beta_1&\alpha_2&\ddots&\\&\ddots&\ddots&\beta_{m-1}\\&&\beta_{m-1}&\alpha_m\end{bmatrix}")
    story.append(callout(
        "n = 140,352, m = 512이면 full matrix가 아니라 512 × 512의 T_m을 분석한다. m/n ≈ 0.003648, 즉 전체 차원의 약 0.365%다.",
        bg=PALE_TEAL, border=TEAL
    ))

    # Section 6
    section(story, 6, "Quadratic-form spectral approximation", "Lanczos의 핵심은 일부 eigenpair를 얻는 데만 있지 않다. 적절한 scalar function f에 대해 다음 quadratic form을 직접 근사할 수 있다.")
    add_equation(story, r"r^{\mathsf T}f(\Omega)r\;\approx\;\lVert r\rVert^2e_1^{\mathsf T}f(T_m)e_1")
    story.append(P("작은 행렬을 T_m = UΘU^T로 분해하면 근사는 quadrature node θ[l]과 weight ||r||^2(U[1,l])^2의 가중합이 된다.", S["body"]))
    add_equation(story, r"\lVert r\rVert^2e_1^{\mathsf T}f(T_m)e_1=\sum_{\ell=1}^{m}\underbrace{\lVert r\rVert^2U_{1\ell}^2}_{w_\ell}\,f(\theta_\ell)")
    story.append(P("큰 covariance eigenvalue에 해당하는 μ ≤ t 영역의 누적 standardized energy를 원하면 다음 spectral filter를 선택한다.", S["body"]))
    add_equation(story, r"f_t(\mu)=\mu\,\mathbf 1(\mu\le t)")
    add_equation(story, r"r^{\mathsf T}f_t(\Omega)r=\sum_{\mu_j\le t}\mu_j(q_j^{\mathsf T}r)^2=\sum_{\lambda_j\ge 1/t}\frac{(q_j^{\mathsf T}r)^2}{\lambda_j}")
    story.append(P("즉 full eigen 방식은 모든 mode를 찾은 뒤 합하지만, Lanczos quadrature는 그 합에 해당하는 spectral integral을 작은 T_m에서 바로 계산한다.", S["body"]))

    # Section 7
    section(story, 7, "Residual-started Lanczos와 SLQ의 역할")
    story.append(P("누적 진단곡선에는 서로 다른 두 종류의 정보가 필요하다. 하나는 residual energy이고, 다른 하나는 threshold 안에 들어온 전체 eigenmode의 비율이다.", S["body"]))
    story.append(P("7.1 Residual energy measure", S["h2"]))
    add_equation(story, r"G_r(t)=\frac{1}{n}r^{\mathsf T}\Omega\mathbf 1(\Omega\le t)r=\frac{1}{n}\sum_{\mu_j\le t}\mu_j(q_j^{\mathsf T}r)^2")
    story.append(P("r로 시작한 Lanczos 한 번은 r가 실제로 얼마나 각 spectral region에 투영되는지를 추적한다. 따라서 Gr(t)는 observed residual에 종속된다.", S["body"]))
    story.append(P("7.2 Mode-count measure", S["h2"]))
    add_equation(story, r"F_\Omega(t)=\frac{1}{n}\operatorname{tr}\mathbf 1(\Omega\le t)=\frac{1}{n}\sum_{j=1}^{n}\mathbf 1(\mu_j\le t)")
    story.append(P("Residual vector 하나로는 전체 mode 수를 알 수 없다. E[gg^T] = I를 만족하는 Rademacher 또는 Gaussian random probe g를 사용하면 Hutchinson identity가 성립한다.", S["body"]))
    add_equation(story, r"\mathbb E_g[g^{\mathsf T}f(\Omega)g]=\operatorname{tr}f(\Omega)")
    story.append(P("각 probe마다 Lanczos quadrature를 수행하고 평균하는 것이 stochastic Lanczos quadrature, 즉 SLQ다.", S["body"]))
    add_equation(story, r"\operatorname{tr}f(\Omega)\approx\frac{1}{s}\sum_{i=1}^{s}\lVert g_i\rVert^2e_1^{\mathsf T}f(T_m^{(i)})e_1")
    story.append(info_table([
        ["계산", "시작 vector", "추정 대상", "무엇에 의존하는가"],
        ["Residual Lanczos", "r / ||r||", "Gr(t)", "관측 residual"],
        ["SLQ", "random probes", "FΩ(t)", "operator spectrum"],
    ], [35*mm, 36*mm, 35*mm, 55*mm]))
    story.append(Spacer(1, 5))
    story.append(callout(
        "<b>모형이 맞을 때</b><br/>E[μ[j](q[j]^T r)^2 | Ω] = 1이므로 E[Gr(t) | Ω] ≈ FΩ(t). "
        "따라서 두 누적곡선의 차이는 특정 eigenscale에 residual energy가 과다하거나 부족한지를 보여준다.",
        bg=PALE_BLUE, border=BLUE
    ))

    # Section 8
    section(story, 8, "Diagnostic 중 무엇이 aggregate spectral quantity인가")
    story.append(P("Aggregate spectral quantity는 특정 j번째 mode의 값이 아니라 여러 mode 위에서 합계, 평균, 비율 또는 곡선 형태로 요약한 값이다.", S["body"]))
    story.append(info_table([
        ["진단량", "종류", "의미", "Lanczos/SLQ 적합성"],
        ["λ[j], μ[j]", "개별", "j번째 eigenvalue", "extreme 일부만 근사 가능"],
        ["q[j]", "개별", "j번째 eigenvector", "모든 vector 복원에는 부적합"],
        ["q[j]^T r", "개별", "mode projection", "개별 전체값에는 부적합"],
        ["z[j]^2", "개별", "mode별 standardized energy", "개별 전체값에는 부적합"],
        ["C[k] 또는 C[k]/k", "aggregate", "첫 k개 mode의 합 또는 평균", "매우 적합"],
        ["band energy", "aggregate", "특정 spectral band의 평균 energy", "넓은 band에 적합"],
        ["FΩ(t)", "aggregate", "누적 mode 비율", "SLQ target"],
        ["Gr(t)", "aggregate", "누적 residual energy", "residual Lanczos target"],
        ["r^T Ωr", "global", "전체 standardized energy", "직접 matvec으로도 계산"],
        ["Dshape", "aggregate의 요약", "두 누적곡선의 전체적 차이", "곡선 근사 후 계산"],
    ], [29*mm, 24*mm, 62*mm, 46*mm]))
    story.append(Spacer(1, 6))
    story.append(P("예를 들어 large-covariance 쪽 첫 5%에서 C[k]/k = 2.34라면, 이는 한 eigenvector의 값이 2.34라는 뜻이 아니다. 해당 수천 개 mode에서 standardized residual energy의 평균이 2.34라는 뜻이다.", S["body"]))
    add_equation(story, r"R_B=\frac{1}{|B|}\sum_{j\in B}z_j^2")
    story.append(P("원문에 제시된 예시 band 값 large 0.81, middle 1.13, small 0.87 역시 각각 특정 mode가 아니라 band 전체의 평균 energy다. 원자료와 산출물을 함께 검증하지 않은 값이므로 이 문서에서는 계산 예시로만 취급한다.", S["small"]))

    # Section 9
    section(story, 9, "계산복잡도와 실제 규모")
    story.append(info_table([
        ["방법", "주요 시간", "주요 메모리", "전제"],
        ["Dense full eigen", "O(n^3)", "O(n^2)", "dense matrix 및 모든 eigenvector"],
        ["Dense Lanczos", "O(mn^2) 중심", "O(nm) 또는 더 작음", "Ωv가 dense O(n^2)"],
        ["Sparse matrix-free Lanczos", "O(m · nnz(B)) 중심", "B + basis", "Ωv = B^T(Bv)"],
        ["s-probe SLQ", "O(s · m · nnz(B)) 중심", "B + 재사용 basis/workspace", "probe run을 순차 실행 가능"],
    ], [41*mm, 43*mm, 39*mm, 38*mm]))
    story.append(P("Residual run 1회와 s = 12개의 SLQ probe를 사용하면 최대 13 × 512회의 precision matvec이 수행된다. 각 precision matvec은 B와 B^T를 각각 한 번 적용한다. nnz(B) ≈ 18.4M이므로 비용이 작지는 않지만 dense n^3 연산과 성격이 완전히 다르며 sparse memory bandwidth와 병렬화가 성능을 좌우한다.", S["body"]))
    story.append(P("메모리에서 놓치기 쉬운 항", S["h2"]))
    bullet(story, "Q_m 전체를 Float64로 저장하면 n × m × 8 bytes다. 현재 수치에서는 약 0.575 GB, 즉 0.535 GiB다.")
    bullet(story, "Full reorthogonalization은 대략 O(nm^2)의 추가 연산과 Q_m 저장을 요구할 수 있다.")
    bullet(story, "순수 short recurrence는 몇 개 vector만 유지할 수 있지만, Ritz vector 복원이나 재직교화가 필요하면 basis 저장량이 증가한다.")
    bullet(story, "T_m은 작다. 512 × 512 dense Float64도 약 2 MiB이고, tridiagonal만 저장하면 훨씬 작다.")
    story.append(callout(
        "<b>중요</b><br/>Lanczos만 사용한다고 자동으로 scalable한 것이 아니다. Ωv가 dense O(n^2)라면 m번 반복도 매우 비싸다. "
        "현재 방법이 가능한 핵심은 Ω = B^T B에서 B가 sparse하다는 사실이다.",
        bg=PALE_AMBER, border=AMBER
    ))

    # Section 10
    section(story, 10, "왜 n = 3,200 subset에서는 full eigen이 더 빨랐는가")
    story.append(info_table([
        ["측정 항목", "원문 실행시간"],
        ["dense K 생성", "약 0.17초"],
        ["full eigen", "약 3.07초"],
        ["Lanczos/SLQ 전체", "약 49.6초"],
    ], [90*mm, 71*mm]))
    story.append(P("이는 모순이 아니다. n = 3,200에서는 dense matrix가 약 81.9 MB여서 메모리에 들어가고, LAPACK의 symmetric eigensolver가 고도로 최적화되어 있다. 반면 benchmark의 Lanczos/SLQ는 residual run, 32개 random probe, probe별 최대 512 iterations, 반복 matvec과 orthogonalization을 수행했다.", S["body"]))
    story.append(P("따라서 subset 실험의 목적은 속도 우위를 보이는 것이 아니라 다음을 검증하는 것이다.", S["body"]))
    bullet(story, "동일한 subset에서 full eigen curve와 Lanczos/SLQ curve가 충분히 가까운가?")
    bullet(story, "m과 probe 수를 늘릴 때 broad-band 진단이 안정화되는가?")
    bullet(story, "근사오차가 실제 진단 결론을 바꾸지 않는가?")
    story.append(callout(
        "작은 n에서는 full eigen이 더 빠를 수 있다. 큰 n에서 Lanczos의 장점은 더 빠른 상수항이 아니라, dense 저장과 n^3 scaling을 피한다는 점이다.",
        bg=PALE_TEAL, border=TEAL
    ))

    # Section 11
    section(story, 11, "누적곡선과 band energy를 해석하는 방법")
    story.append(Image(str(TMP / "diagnostic_curve.png"), width=146*mm, height=146*mm*5.2/9.8))
    story.append(P("그림 3. 개념도. 실제 데이터 결과가 아니라 FΩ와 Gr의 관계를 설명하기 위한 예시다.", S["caption"]))
    story.append(P("가로축을 FΩ(t), 세로축을 Gr(t)로 두면 이상적인 기준은 대각선이다. 특정 구간에서 Gr가 F보다 빠르게 증가하면 그 spectral band의 residual energy가 기대보다 크다는 뜻이다.", S["body"]))
    story.append(info_table([
        ["관측 패턴", "통계적 의미", "가능한 모형 해석"],
        ["large-covariance band에서 Gr > F", "smooth/large-scale mode의 residual energy 과다", "mean structure, long-range dependence, large-scale random effect 부족"],
        ["middle band에서 Gr > F", "중간 scale mismatch", "range 또는 covariance shape 부적합 가능"],
        ["small-covariance band에서 Gr > F", "rough/local mode의 residual energy 과다", "nugget, measurement error, local dependence 부족 가능"],
        ["Gr < F", "해당 band의 energy 부족", "over-smoothing, variance 과대, fitted-parameter 효과 가능"],
        ["전체 Gr(∞) ≠ 1", "global standardized energy mismatch", "전체 variance calibration 또는 df 조정 점검"],
    ], [43*mm, 55*mm, 63*mm]))
    story.append(Spacer(1, 5))
    story.append(P("위 해석은 진단적 가설이지 단독으로 원인을 식별하는 증거는 아니다. mean misspecification, parameter estimation, non-Gaussian tails, outlier, ordering-dependent Vecchia approximation도 유사한 패턴을 만들 수 있다.", S["small"]))

    # Section 12
    section(story, 12, "Lanczos/SLQ가 해주는 것, 해주지 않는 것, 주의점")
    story.append(info_table([
        ["잘하는 것", "일반적으로 보장하지 않는 것"],
        ["전체 eigenvalue CDF의 근사", "모든 n개 eigenvalue의 정확한 목록"],
        ["넓은 low/middle/high band의 mode 비율", "매우 좁은 hard band의 정확한 membership"],
        ["누적 및 band residual energy", "모든 z[j]^2의 정확한 개별 순위"],
        ["몇 개의 extreme Ritz values", "모든 eigenvector의 정확한 공간 패턴"],
        ["matrix function quadratic forms", "m ≪ n일 때 full spectral reconstruction"],
    ], [80.5*mm, 80.5*mm]))
    story.append(P("12.1 Indicator function은 매끄럽지 않다", S["h2"]))
    story.append(P("f_t(μ) = μ1(μ ≤ t) 또는 1(μ ≤ t)는 threshold에서 불연속이다. Lanczos quadrature는 log, inverse 같은 smooth function보다 hard step에서 느리게 수렴할 수 있다. 그러므로 매우 좁은 band나 개별 eigenvalue 근처의 jump는 불안정할 수 있다.", S["body"]))
    bullet(story, "넓은 quantile band를 사용하고 threshold grid를 지나치게 촘촘하게 해석하지 않는다.")
    bullet(story, "m을 증가시킨 convergence study를 수행한다.")
    bullet(story, "필요하면 step function을 smooth transition으로 바꾸거나 spectral density smoothing을 사용한다.")
    story.append(P("12.2 Finite precision과 orthogonality loss", S["h2"]))
    story.append(P("Exact arithmetic의 Lanczos vector는 orthogonal하지만 floating-point에서는 이미 수렴한 Ritz direction이 다시 나타나는 ghost eigenvalue 문제가 생길 수 있다. full 또는 selective/partial reorthogonalization, residual norm monitoring이 필요하다.", S["body"]))
    story.append(P("12.3 Random-probe uncertainty", S["h2"]))
    story.append(P("SLQ는 stochastic estimator이므로 probe seed와 수 s에 따라 흔들린다. probe 간 분산 또는 여러 seed 반복으로 Monte Carlo standard error를 보고하고, 진단곡선에는 uncertainty band를 함께 표시하는 것이 좋다.", S["body"]))
    story.append(P("12.4 Fitted residual은 독립 N(0,K)가 아니다", S["h2"]))
    story.append(P("고정효과와 covariance parameter를 같은 데이터에서 추정하면 residual의 유효 자유도가 감소하고 eigenmode 간 분포가 단순 chi-square(1)과 달라질 수 있다. 특히 mean design X와 강하게 겹치는 low-frequency mode는 영향을 많이 받는다.", S["body"]))
    bullet(story, "가능하면 generalized residual 또는 mean-space projection을 명시한다.")
    bullet(story, "기대선 Gr = F만 쓰기보다 fitted model에서 parametric bootstrap envelope를 만든다.")
    bullet(story, "Vecchia ordering과 neighbor size를 바꿔 구조적 민감도를 확인한다.")

    # Section 13
    section(story, 13, "권장 구현 및 검증 체크리스트")
    story.append(P("A. Operator 검증", S["h2"]))
    for txt in [
        "작은 subset에서 explicit Ω와 matrix-free B^T(Bv)의 결과가 numerical tolerance 안에서 일치하는지 확인한다.",
        "대칭성 검사: u^T Ωv와 v^T Ωu가 일치하는지 random vectors로 확인한다.",
        "positive definiteness 또는 near-null direction을 점검하고 breakdown 처리 규칙을 기록한다.",
    ]:
        bullet(story, txt)
    story.append(P("B. Lanczos convergence", S["h2"]))
    for txt in [
        "m = 128, 256, 512 등으로 늘리며 FΩ, Gr, band averages가 안정화되는지 비교한다.",
        "Ritz residual과 orthogonality loss를 모니터링한다.",
        "hard threshold 주변의 결과를 단일 점이 아니라 band 평균으로 해석한다.",
    ]:
        bullet(story, txt)
    story.append(P("C. SLQ uncertainty", S["h2"]))
    for txt in [
        "probe 수 s를 증가시키며 curve와 주요 scalar summary의 Monte Carlo error를 기록한다.",
        "Rademacher probe, random seed, normalization convention을 명시한다.",
        "probe별 결과를 저장해 pointwise standard error 또는 bootstrap band를 계산한다.",
    ]:
        bullet(story, txt)
    story.append(P("D. Full-eigen benchmark", S["h2"]))
    for txt in [
        "n ≈ 3,200처럼 exact eigen이 가능한 subset에서 동일한 sorting과 normalization을 사용한다.",
        "curve sup error, integrated absolute error, band-energy error를 수치로 비교한다.",
        "속도보다 진단 결론의 재현성과 approximation bias를 평가한다.",
    ]:
        bullet(story, txt)
    story.append(P("E. Statistical calibration", S["h2"]))
    for txt in [
        "fitted model에서 residual을 반복 모의하여 null envelope를 구축한다.",
        "mean estimation, covariance estimation, Vecchia approximation을 모두 simulation pipeline에 포함한다.",
        "band 경계와 tuning parameters를 결과를 본 뒤 선택했다면 multiplicity 또는 탐색 편향을 설명한다.",
    ]:
        bullet(story, txt)

    # Section 14
    section(story, 14, "최종 요약")
    story.append(callout(
        "<b>Full eigen</b><br/>K = QΛQ^T를 통해 모든 eigenvalue와 eigenvector를 구한다.<br/>"
        "시간 O(n^3), 메모리 O(n^2). n = 140,352에서는 dense storage부터 비현실적이다.",
        bg=PALE_RED, border=RED
    ))
    story.append(Spacer(1, 4))
    story.append(callout(
        "<b>Matrix-free Lanczos / SLQ</b><br/>v → Ωv = B^T(Bv)를 반복하고, m × m인 T_m만 분석한다.<br/>"
        "Residual-started Lanczos는 Gr(t), random-probe SLQ는 FΩ(t)를 근사한다.",
        bg=PALE_TEAL, border=TEAL
    ))
    story.append(Spacer(1, 4))
    story.append(P("이 진단은 14만 개 mode 각각을 복원하는 대신, eigenscale별 mode 비율과 standardized residual energy의 총량을 비교한다. 따라서 target 자체가 aggregate quantity이며 Lanczos quadrature와 구조적으로 잘 맞는다.", S["body"]))
    add_equation(story, r"\boxed{\text{sparse Vecchia precision}+\text{matrix-free matvec}+\text{Lanczos/SLQ}}", width=145*mm, fontsize=16)
    story.append(P("이 조합은 dense O(n^3) 문제를 반복적인 sparse operator 계산으로 바꾼다. 정확성은 m, probe 수, orthogonalization, threshold smoothing 및 fitted-model calibration으로 검증해야 한다.", S["body"]))

    # Appendix A
    story.append(PageBreak())
    story.append(P("부록 A. 기호와 핵심 공식", S["h1"]))
    story.append(Rule())
    story.append(info_table([
        ["기호", "정의"],
        ["n", "관측치 또는 spatial degree of freedom 수"],
        ["K 또는 K-tilde", "covariance matrix 또는 fitted approximate covariance"],
        ["Ω", "precision operator; Ω = K-tilde^(-1) = B^T B"],
        ["B", "sparse Vecchia precision factor"],
        ["λ[j]", "K의 j번째 covariance eigenvalue"],
        ["μ[j]", "Ω의 j번째 precision eigenvalue; μ[j] = 1/λ[j]"],
        ["r", "분석 대상 residual vector"],
        ["m", "Lanczos iterations / Krylov dimension"],
        ["s", "SLQ random probe 수"],
        ["Q_m", "Krylov subspace의 orthonormal basis"],
        ["T_m", "Q_m^T ΩQ_m인 m × m tridiagonal projection"],
        ["FΩ(t)", "μ ≤ t인 eigenmode의 누적 비율"],
        ["Gr(t)", "μ ≤ t인 mode의 누적 standardized residual energy"],
    ], [35*mm, 126*mm]))
    story.append(Spacer(1, 8))
    for tex in [
        r"z_j^2=\mu_j(q_j^{\mathsf T}r)^2=(q_j^{\mathsf T}r)^2/\lambda_j",
        r"\mathbb E[G_r(t)\mid\Omega]\approx F_\Omega(t)",
        r"G_r(\infty)=n^{-1}r^{\mathsf T}\Omega r",
        r"\mathrm{Cost}_{\mathrm{residual}}\approx O(m\,\mathrm{nnz}(B))",
        r"\mathrm{Cost}_{\mathrm{SLQ}}\approx O(s\,m\,\mathrm{nnz}(B))",
    ]:
        add_equation(story, tex, width=145*mm, fontsize=15)

    # Appendix B
    story.append(PageBreak())
    story.append(P("부록 B. 관련 참고 문헌", S["h1"]))
    story.append(Rule())
    refs = [
        "1. Lanczos, C. (1950). An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. Journal of Research of the National Bureau of Standards, 45, 255-282.",
        "2. Golub, G. H., & Van Loan, C. F. Matrix Computations. Krylov subspace와 symmetric Lanczos의 표준 참고서.",
        "3. Ubaru, S., Chen, J., & Saad, Y. (2017). Fast estimation of tr(f(A)) via stochastic Lanczos quadrature. SIAM Journal on Matrix Analysis and Applications, 38(4), 1075-1099. DOI: 10.1137/16M1104974.",
        "4. Dong, K., Eriksson, D., Nickisch, H., Bindel, D., & Wilson, A. G. (2017). Scalable log determinants for Gaussian process kernel learning. NeurIPS 30. Spatial precipitation, log-Gaussian Cox process, Chicago crime 사례 포함.",
        "5. Pleiss, G., Gardner, J., Weinberger, K., & Wilson, A. G. (2018). Constant-time predictive distributions for Gaussian processes. ICML. Lanczos-based LOVE predictive covariance approximation.",
        "6. Genton, M. G. (2007). Separable approximations of space-time covariance matrices. Environmetrics, 18, 681-695. SVD Lanczos를 이용한 Kronecker covariance approximation.",
        "7. Gyger, T., Furrer, R., & Sigrist, F. (2026). Iterative methods for full-scale Gaussian process approximations for large spatial data. SIAM/ASA Journal on Uncertainty Quantification. DOI: 10.1137/25M1731320.",
    ]
    for ref in refs:
        story.append(P(ref, S["ref"]))
    story.append(Spacer(1, 6))
    story.append(P("문서 범위", S["h2"]))
    story.append(P("이 문서는 사용자가 제공한 기술 메모를 기반으로 원리와 계산구조를 재정리한 설명자료다. 원문의 실행시간과 band 값은 해당 실행 로그 및 원자료를 독립적으로 재분석한 결과가 아니므로, empirical result로 인용할 때는 원 코드와 산출물의 재현성 검증이 필요하다.", S["small"]))

    return story


if __name__ == "__main__":
    doc = GuideDocTemplate(str(OUT))
    doc.build(build_story())
    print(OUT)
