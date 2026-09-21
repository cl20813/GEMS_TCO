from pathlib import Path
from zipfile import ZipFile
from lxml import etree


ROOT = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/career_materials")
SOURCE = ROOT / "two_sigma/Joonwon_Lee_Two_Sigma_Quantitative_Researcher_Resume.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DC = "http://purl.org/dc/elements/1.1/"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
NS = {"w": W, "dc": DC, "cp": CP}


def body_paragraphs(root):
    body = root.find("w:body", NS)
    return body, body.findall("w:p", NS)


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


def move_imc_before_dissertation(body, paragraphs):
    dissertation = paragraphs[17]
    for element in (paragraphs[21], paragraphs[22]):
        body.remove(element)
        body.insert(body.index(dissertation), element)


def set_core(core_bytes, title, subject, keywords):
    root = etree.fromstring(core_bytes)
    for xpath, value in (
        ("dc:title", title),
        ("dc:subject", subject),
        ("cp:keywords", keywords),
    ):
        node = root.find(xpath, NS)
        if node is None:
            prefix, local = xpath.split(":")
            namespace = NS[prefix]
            node = etree.SubElement(root, f"{{{namespace}}}{local}")
        node.text = value
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def build(output, summary, programming, methods, title, subject, keywords):
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(SOURCE, "r") as source_zip:
        parts = {info.filename: source_zip.read(info.filename) for info in source_zip.infolist()}
        infos = source_zip.infolist()

    document = etree.fromstring(parts["word/document.xml"])
    body, paragraphs = body_paragraphs(document)
    if len(paragraphs) != 26:
        raise ValueError(f"Unexpected template paragraph count: {len(paragraphs)}")

    set_paragraph_runs(paragraphs[3], [summary])
    set_paragraph_runs(paragraphs[8], ["Programming and Research Systems: ", programming])
    set_paragraph_runs(paragraphs[9], ["Quantitative Methods: ", methods])
    set_paragraph_runs(paragraphs[12], [
        "Developed a sequential risk-monitoring framework for binary indicators, using Brownian-motion "
        "approximation, dynamic-programming probability propagation, and simulation to quantify stopping-time "
        "uncertainty and early-decision tradeoffs."
    ])
    set_paragraph_runs(paragraphs[13], [
        "Specified and validated a Sequential Probability Ratio Test with explicit hypotheses and likelihood-ratio "
        "boundaries; evaluated boundary-crossing probabilities, stopping behavior, and Type I/II error."
    ])
    set_paragraph_runs(paragraphs[15], [
        "Developed an end-to-end LightGBM pricing pipeline across 2.46M+ records, including data preparation, "
        "feature processing, GLM benchmarking, model selection, validation, and performance diagnostics on AWS EC2; "
        "presented findings as pricing and risk-segmentation recommendations to business stakeholders."
    ])
    set_paragraph_runs(paragraphs[16], ["QUANTITATIVE RESEARCH AND TRADING"])
    set_paragraph_runs(paragraphs[18], [
        "Scalable Inference: ",
        "Developed an advection-aware Vecchia likelihood approximation for nonseparable space-time covariance "
        "structures, using ordered local conditional models for likelihood-based parameter estimation and "
        "uncertainty quantification with up to 145,008 observations per day."
    ])
    set_paragraph_runs(paragraphs[19], [
        "Stochastic Diagnostics: ",
        "Developed statistical diagnostics for dependent stochastic processes that localize where a fitted model "
        "fails across scale and frequency, distinguishing persistent dependence from high-frequency noise, "
        "missingness, acquisition artifacts, and covariance misspecification."
    ])
    set_paragraph_runs(paragraphs[20], [
        "Research Systems:",
        " Built reusable Python/PyTorch workflows for data-quality filtering, time-dependent alignment, optimization, "
        "simulation, and restartable CPU/GPU and HPC execution; integrated performance-critical C++ ordering routines "
        "through pybind11."
    ])
    set_paragraph_runs(paragraphs[22], [
        "Built and backtested an inventory-aware market-making strategy around empirically calibrated reference "
        "values; tracked position-limit saturation over a rolling 10-step window and triggered more aggressive "
        "inventory-reducing quotes to manage persistent exposure."
    ])
    move_imc_before_dissertation(body, paragraphs)

    parts["word/document.xml"] = etree.tostring(
        document, xml_declaration=True, encoding="UTF-8", standalone="yes"
    )
    parts["docProps/core.xml"] = set_core(parts["docProps/core.xml"], title, subject, keywords)

    with ZipFile(output, "w") as output_zip:
        for info in infos:
            output_zip.writestr(info, parts[info.filename])


def main():
    build(
        ROOT / "maven_securities/Joonwon_Lee_Maven_Quant_Researcher_Resume.docx",
        "Statistics Ph.D. candidate and computational statistician specializing in scalable likelihood inference, "
        "stochastic modeling, and statistical diagnostics for dependent data. Builds reproducible Python/PyTorch "
        "research systems, integrates C++/pybind11 for performance-critical computation, and applies rigorous "
        "hypothesis testing, simulation, and empirical evaluation to algorithmic research.",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, R, SQL, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, backtesting, and restartable research pipelines",
        "Probability and stochastic processes, hypothesis testing, Monte Carlo simulation, MLE/GLS, numerical "
        "optimization, time-series and spatiotemporal modeling, spectral and autocorrelation analysis, regression "
        "and machine learning, backtesting, model validation, and performance evaluation",
        "Joonwon Lee - Maven Securities Graduate Quant Researcher Resume",
        "Application for Graduate Quant Researcher 2027 - Chicago",
        "quantitative research, market making, algorithm design, pricing models, stochastic processes, simulation, "
        "Python, C++ integration, statistical diagnostics",
    )

    build(
        ROOT / "old_mission/Joonwon_Lee_Old_Mission_Quant_Researcher_Resume.docx",
        "Statistics Ph.D. candidate and computational statistician specializing in scalable likelihood inference, "
        "stochastic-process modeling, numerical optimization, and statistical diagnostics for dependent data. Builds "
        "reproducible Python/PyTorch research systems, integrates C++/pybind11 for performance-critical computation, "
        "and evaluates model fit, uncertainty, and misspecification through simulation and empirical diagnostics.",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, R, SQL, Linux, "
        "Git, AWS EC2, HPC/SLURM; CPU/GPU computing, backtesting, and restartable research pipelines",
        "Stochastic-process and time-series modeling, Monte Carlo simulation, MLE/GLS, numerical optimization, "
        "spectral and autocorrelation analysis, hypothesis testing, model diagnostics and validation, backtesting, "
        "regression, and machine learning",
        "Joonwon Lee - Old Mission Quantitative Researcher Resume",
        "Application for Quantitative Researcher Ph.D. - 2027 Graduate Program",
        "quantitative research, stochastic modeling, model calibration, model validation, market making, Python, "
        "C++ integration, simulation, statistical diagnostics",
    )


if __name__ == "__main__":
    main()
