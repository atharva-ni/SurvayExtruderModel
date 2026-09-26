"""
Shared text preprocessing and survey/research cue patterns.
Used identically by training, classification and evaluation.
"""

import re

MAX_LENGTH = 384  # tokens; covers title + abstract for nearly all papers

# Keyword filter from the paper (Section III-A-3), applied to the title
PAPER_TITLE_KEYWORDS = re.compile(
    r"\b(?:survey|surveys|review|reviews|overview|taxonomy|comparative study)\b",
    re.IGNORECASE,
)

# Broader survey terms in titles (used as a feature by the learned hybrid)
TITLE_SURVEY_TERMS = re.compile(
    r"\b(?:survey|surveys|review|reviews|tutorial|overview|taxonomy|state[- ]of[- ]the[- ]art|"
    r"systematic mapping|primer|roadmap|comparative study|literature)\b",
    re.IGNORECASE,
)

# Survey phrasing in abstracts
ABSTRACT_SURVEY_CUES = re.compile(
    r"\b(?:this (?:survey|review|tutorial|article reviews|paper reviews|paper surveys)|we (?:survey|review)|"
    r"(?:comprehensive|systematic|extensive|holistic|thorough) (?:overview|survey|review|literature review)|"
    r"we provide an overview|provides? an overview|literature review|open (?:research )?(?:issues|challenges)|"
    r"future research directions|we (?:summarize|categorize|classify) (?:the )?existing)\b",
    re.IGNORECASE,
)

# Original-research phrasing in abstracts
ABSTRACT_RESEARCH_CUES = re.compile(
    r"\b(?:we propose|we develop|we design|we derive|we formulate|is proposed|are proposed|"
    r"our (?:proposed|approach|method|scheme|framework|algorithm|results)|the proposed|"
    r"(?:experimental|simulation|numerical) results|outperforms?|closed-form)\b",
    re.IGNORECASE,
)

# Explicit self-description as a survey (used by the magazine rule)
EXPLICIT_SURVEY_CUES = re.compile(
    r"\b(?:this (?:survey|review|tutorial|article reviews|paper reviews|paper surveys)|we (?:survey|review)|"
    r"(?:comprehensive|systematic|extensive|holistic|thorough|contemporary) (?:overview|survey|review|tutorial)|"
    r"(?:provide|present|give)s? (?:an|a brief|a comprehensive) (?:overview|survey|review|tutorial)|"
    r"literature review)\b",
    re.IGNORECASE,
)

# Magazines: short articles that often mix an overview with a case study
MAGAZINE_VENUE = re.compile(
    r"magazine|^ieee wireless communications$|^ieee network$|^getmobile|^ieee intelligent systems$|"
    r"^communications of the acm$|^ieee internet computing$|^ieee pervasive computing$|^ieee multimedia$|"
    r"^ieee micro$|^computer$|^ieee potentials$",
    re.IGNORECASE,
)

# Publication types (OpenAlex type / Semantic Scholar publicationTypes) that are not papers
NON_PAPER_TYPES = re.compile(
    r"\b(?:book|book-chapter|booksection|editorial|erratum|paratext|peer-review|retraction|"
    r"dissertation|supplementary-materials|news|letterandcomments|lettersandcomments|"
    r"dataset|reference-entry|conference-abstract)\b",
    re.IGNORECASE,
)

# Front matter that is neither a survey nor research
NON_PAPER_TITLE = re.compile(
    r"\b(?:guest editorial|editorial|erratum|corrigendum|masthead|table of contents|"
    r"front cover|in memoriam|call for papers)\b",
    re.IGNORECASE,
)


def clean_text(text) -> str:
    """Remove markup, LaTeX and formatting artifacts; collapse whitespace."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = str(text)
    text = re.sub(r"<tex-math.*?</tex-math>", " ", text, flags=re.DOTALL)  # JATS math
    text = re.sub(r"<[^>]+>", " ", text)                                # HTML / JATS tags
    text = re.sub(r"\$[^$]*\$", " ", text)                             # inline LaTeX math
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})?", " ", text)          # LaTeX commands
    text = re.sub(r"[{}\[\]]", " ", text)                              # brackets
    text = re.sub(r"^\s*abstract[:.\s-]*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def paper_text(title, abstract) -> str:
    """Model input: cleaned title and abstract (the tokenizer lowercases)."""
    title, abstract = clean_text(title), clean_text(abstract)
    return f"{title}. {abstract}" if abstract else title
