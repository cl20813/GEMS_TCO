from pathlib import Path
from zipfile import ZipFile

from lxml import etree


ROOT = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/career_materials")
SOURCE = ROOT / "barclays_electronic_trading/Joonwon_Lee_Barclays_Electronic_Trading_Resume.docx"
OUTPUT = ROOT / "smbc_systematic_rates/Joonwon_Lee_SMBC_Systematic_Rates_Quant_Resume.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DC = "http://purl.org/dc/elements/1.1/"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
NS = {"w": W, "dc": DC, "cp": CP}


def set_paragraph_runs(paragraph, values):
    runs = paragraph.findall(".//w:r", NS)
    if len(runs) < len(values):
        raise ValueError(f"Paragraph has {len(runs)} runs but {len(values)} values were supplied")
    for index, run in enumerate(runs):
        texts = run.findall(".//w:t", NS)
        if not texts:
            continue
        value = values[index] if index < len(values) else ""
        texts[0].text = value
        if value.startswith(" ") or value.endswith(" "):
            texts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        else:
            texts[0].attrib.pop("{http://www.w3.org/XML/1998/namespace}space", None)
        for extra in texts[1:]:
            extra.text = ""


def set_core(core_bytes):
    root = etree.fromstring(core_bytes)
    values = {
        "dc:title": "Joonwon Lee - SMBC Systematic Rates Quantitative Researcher Resume",
        "dc:subject": "Application for Quantitative Researcher / Strategist - Systematic Trading, Rates, Associate",
        "cp:keywords": (
            "systematic trading, quantitative research, time series, stochastic processes, simulation, "
            "backtesting, market making, Python, C++ integration, statistical diagnostics"
        ),
    }
    for xpath, value in values.items():
        node = root.find(xpath, NS)
        if node is None:
            prefix, local = xpath.split(":")
            node = etree.SubElement(root, f"{{{NS[prefix]}}}{local}")
        node.text = value
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(SOURCE, "r") as source_zip:
        infos = source_zip.infolist()
        parts = {info.filename: source_zip.read(info.filename) for info in infos}

    document = etree.fromstring(parts["word/document.xml"])
    body = document.find("w:body", NS)
    paragraphs = body.findall("w:p", NS)
    if len(paragraphs) != 26:
        raise ValueError(f"Unexpected template paragraph count: {len(paragraphs)}")

    set_paragraph_runs(
        paragraphs[3],
        [
            "Statistics Ph.D. candidate and computational statistician specializing in scalable likelihood inference, "
            "stochastic and time-dependent modeling, and statistical diagnostics for dependent data. Builds reproducible "
            "Python/PyTorch research systems, integrates C++/pybind11 for performance-critical computation, and applies "
            "rigorous hypothesis testing and simulation to evaluate quantitative models under uncertainty."
        ],
    )
    set_paragraph_runs(
        paragraphs[8],
        [
            "Programming and Research Systems: ",
            "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, R, SQL, Linux, "
            "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, backtesting, and restartable research pipelines",
        ],
    )
    set_paragraph_runs(
        paragraphs[9],
        [
            "Quantitative Methods: ",
            "Probability and stochastic processes, time-series and spatiotemporal modeling, hypothesis testing, Monte "
            "Carlo simulation, MLE/GLS, numerical optimization, spectral and autocorrelation analysis, regression and "
            "machine learning, model validation, and performance evaluation",
        ],
    )
    set_paragraph_runs(
        paragraphs[12],
        [
            "Quantified stopping-time uncertainty and early-decision risk in a sequential monitoring framework using "
            "Brownian-motion approximation, dynamic-programming probability propagation, and Monte Carlo simulation."
        ],
    )
    set_paragraph_runs(
        paragraphs[13],
        [
            "Developed and validated a Sequential Probability Ratio Test with explicit hypotheses and likelihood-ratio "
            "boundaries; evaluated boundary-crossing probabilities, stopping behavior, and Type I/II error."
        ],
    )
    set_paragraph_runs(
        paragraphs[15],
        [
            "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, feature "
            "processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2; presented "
            "findings as pricing and risk-segmentation recommendations to business stakeholders."
        ],
    )
    set_paragraph_runs(paragraphs[16], ["QUANTITATIVE RESEARCH AND SYSTEMATIC TRADING"])
    set_paragraph_runs(
        paragraphs[18],
        [
            "Scalable Inference: ",
            "Developed an advection-aware Vecchia likelihood approximation for nonseparable space-time covariance "
            "structures, using ordered local conditional models for likelihood-based parameter estimation and uncertainty "
            "quantification with up to 145,008 observations per day.",
        ],
    )
    set_paragraph_runs(
        paragraphs[19],
        [
            "Stochastic Diagnostics: ",
            "Developed statistical diagnostics for dependent stochastic processes that localize where a fitted model "
            "fails across scale and frequency, distinguishing persistent dependence from high-frequency noise, missingness, "
            "acquisition artifacts, and covariance misspecification.",
        ],
    )
    set_paragraph_runs(
        paragraphs[20],
        [
            "Research Systems:",
            " Built reusable Python/PyTorch workflows for data-quality filtering, time-dependent alignment, optimization, "
            "simulation, and restartable CPU/GPU and HPC execution; integrated performance-critical C++ ordering routines "
            "through pybind11.",
        ],
    )
    set_paragraph_runs(
        paragraphs[22],
        [
            "Built and backtested an inventory-aware market-making strategy around empirically calibrated reference values; "
            "tracked position-limit saturation over a rolling 10-step window and triggered more aggressive "
            "inventory-reducing quotes to manage persistent exposure."
        ],
    )

    parts["word/document.xml"] = etree.tostring(
        document, xml_declaration=True, encoding="UTF-8", standalone="yes"
    )
    parts["docProps/core.xml"] = set_core(parts["docProps/core.xml"])

    with ZipFile(OUTPUT, "w") as output_zip:
        for info in infos:
            output_zip.writestr(info, parts[info.filename])

    print(OUTPUT)


if __name__ == "__main__":
    main()
