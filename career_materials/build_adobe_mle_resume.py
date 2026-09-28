from pathlib import Path

from docx import Document

from build_resume_cv import (
    add_body_paragraph,
    add_bullet,
    add_bullet_numbering,
    add_labeled_paragraph,
    add_name_header,
    add_section_heading,
    add_tabbed_line,
    configure_styles,
    set_cell_free_document_defaults,
    set_core_properties,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "adobe_mle_2027"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_adobe_mle_resume() -> Path:
    doc = Document()
    width = set_cell_free_document_defaults(
        doc,
        margin_x=0.58,
        margin_top=0.46,
        margin_bottom=0.46,
    )
    configure_styles(doc, body_size=9.25, body_line=1.0)
    num_id = add_bullet_numbering(doc, left_twips=315, hanging_twips=175)
    set_core_properties(
        doc,
        "Joonwon Lee Adobe 2027 Machine Learning Engineer Resume",
        "Application for 2027 University Graduate Machine Learning Engineer at Adobe",
        "computational statistics, machine learning, predictive modeling, causal inference, "
        "large-scale data, Python, PyTorch, LightGBM, model validation, statistical computing",
    )

    add_name_header(doc, compact=True)

    add_section_heading(doc, "Summary", compact=True)
    add_body_paragraph(
        doc,
        "Computational statistics Ph.D. candidate with experience developing scalable statistical and "
        "machine learning workflows for large, complex datasets. Builds reproducible Python/PyTorch and "
        "LightGBM pipelines for predictive modeling, simulation, optimization, and model validation; combines "
        "rigorous statistical evaluation with clear recommendations for business stakeholders.",
        9.35,
        after=0.7,
        line=1.0,
    )

    add_section_heading(doc, "Education", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. in Statistics, Rutgers University, Piscataway, NJ",
        "Expected May 2027  |  GPA: 3.8/4.0",
        width,
        9.25,
        after=0.25,
        keep_with_next=False,
    )
    add_tabbed_line(
        doc,
        "M.S. in Statistics, University of Minnesota, Minneapolis, MN",
        "Sep 2021  |  GPA: 3.7/4.0",
        width,
        9.25,
        after=0.25,
        keep_with_next=False,
    )

    add_section_heading(doc, "Technical Skills", compact=True)
    add_labeled_paragraph(
        doc,
        "Programming and Computing: ",
        "Python (NumPy, Pandas, SciPy, PyTorch, LightGBM, scikit-learn), C++/pybind11 integration, "
        "Linux, Git, AWS EC2, HPC/SLURM, CPU/GPU computing; SQL/MySQL (working familiarity)",
        9.05,
        after=0.28,
        line=1.0,
    )
    add_labeled_paragraph(
        doc,
        "Machine Learning and Statistics: ",
        "Predictive modeling, GLM/GBM, feature attribution, causal inference, fixed-effects regression, "
        "time-series and spatio-temporal modeling, hypothesis testing, Monte Carlo simulation, numerical "
        "optimization, model selection and validation, uncertainty quantification",
        9.05,
        after=0.35,
        line=1.0,
    )

    add_section_heading(doc, "Experience", compact=True)
    add_tabbed_line(
        doc,
        "JPMorgan Chase - Summer Quantitative Analytics Associate, Model Risk Governance and Review",
        "Summer 2026",
        width,
        9.1,
        after=0.25,
    )
    add_bullet(
        doc,
        "Developed and validated a Sequential Probability Ratio Test for binary risk indicators, "
        "translating a monitoring question into explicit hypotheses, likelihood-ratio boundaries, and "
        "statistically controlled decisions.",
        num_id,
        9.0,
        after=0.38,
        line=1.0,
    )
    add_bullet(
        doc,
        "Designed simulation and dynamic-programming evaluations of Type I/II error, detection delay, "
        "stopping-time uncertainty, and early-decision tradeoffs; documented model behavior and limitations.",
        num_id,
        9.0,
        after=0.5,
        line=1.0,
    )
    add_tabbed_line(
        doc,
        "Travelers - Data Science Leadership Program, Business Insurance Pricing Team",
        "Summer 2025",
        width,
        9.1,
        after=0.25,
    )
    add_bullet(
        doc,
        "Developed an end-to-end LightGBM pipeline across 2.46M+ property-risk records, including data "
        "preparation, feature processing, GLM benchmarking, predictive-performance evaluation, and model "
        "validation on AWS EC2.",
        num_id,
        9.0,
        after=0.38,
        line=1.0,
    )
    add_bullet(
        doc,
        "Analyzed model performance and feature-attribution patterns and presented pricing and "
        "risk-segmentation recommendations to business stakeholders.",
        num_id,
        9.0,
        after=0.5,
        line=1.0,
    )

    add_section_heading(doc, "Selected Research", compact=True)
    add_tabbed_line(
        doc,
        "Ph.D. Dissertation Research - Scalable Statistical Computing and Model Diagnostics",
        "Sep 2024 - Present",
        width,
        9.1,
        after=0.25,
    )
    add_bullet(
        doc,
        "Scalable Modeling: Developed an advection-aware Vecchia likelihood approximation for nonseparable "
        "space-time Gaussian processes, using ordered local conditioning for parameter estimation and "
        "uncertainty quantification with up to 145,008 observations per day.",
        num_id,
        8.95,
        after=0.35,
        line=1.0,
        bold_lead="Scalable Modeling:",
    )
    add_bullet(
        doc,
        "Data and Model Diagnostics: Built quality-filtering, missingness, time-dependent alignment, and "
        "irregular-grid matching workflows; developed scale- and frequency-resolved diagnostics to localize "
        "model misspecification and acquisition artifacts.",
        num_id,
        8.95,
        after=0.35,
        line=1.0,
        bold_lead="Data and Model Diagnostics:",
    )
    add_bullet(
        doc,
        "Research Engineering: Implemented reusable Python/PyTorch workflows with L-BFGS optimization, "
        "CPU/GPU and restartable HPC execution, and performance-critical C++ routines exposed through pybind11.",
        num_id,
        8.95,
        after=0.5,
        line=1.0,
        bold_lead="Research Engineering:",
    )
    add_tabbed_line(
        doc,
        "Longitudinal Observational Study - EITC and Household Labor Supply",
        "2021",
        width,
        9.1,
        after=0.25,
    )
    add_bullet(
        doc,
        "Estimated the effect of the Earned Income Tax Credit on household labor supply using fixed-effects "
        "regression and propensity-score matching with observed household demographic and economic covariates.",
        num_id,
        8.95,
        after=0.45,
        line=1.0,
    )

    add_section_heading(doc, "Additional", compact=True)
    add_labeled_paragraph(
        doc,
        "Coursework: ",
        "Machine Learning, Data Structures and Algorithms, Probability Theory, Statistical Computing, "
        "Stochastic Processes, Advanced Theory of Statistics I-II, Linear Algebra, Econometrics",
        8.85,
        after=0.18,
        line=1.0,
    )
    add_labeled_paragraph(
        doc,
        "Languages: ",
        "Korean (Native); English (Fluent); Chinese (Basic)",
        8.85,
        after=0,
        line=1.0,
    )

    output = OUTPUT_DIR / "Joonwon_Lee_Adobe_Machine_Learning_Engineer_2027_Resume.docx"
    doc.save(output)
    return output


if __name__ == "__main__":
    print(build_adobe_mle_resume())
