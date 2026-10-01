import math

from text_utils import (
    clean_text, paper_text, PAPER_TITLE_KEYWORDS, MAGAZINE_VENUE, NON_PAPER_TYPES,
)


def test_clean_text_removes_markup_and_latex():
    raw = ("Abstract: We study <jats:italic>NOMA</jats:italic> with $n = 128$ users "
           "<inline-formula><tex-math>d < 22</tex-math></inline-formula> and \\textbf{gains} [1].")
    cleaned = clean_text(raw)
    assert "<" not in cleaned and "$" not in cleaned and "\\" not in cleaned and "[" not in cleaned
    assert cleaned.startswith("We study NOMA")
    assert "  " not in cleaned


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(math.nan) == ""
    assert clean_text(123) == "123"


def test_paper_text_joins_title_and_abstract():
    assert paper_text("A Title", "Some abstract.") == "A Title. Some abstract."
    assert paper_text("A Title", None) == "A Title"


def test_paper_title_keywords():
    assert PAPER_TITLE_KEYWORDS.search("A Survey on Blockchain")
    assert PAPER_TITLE_KEYWORDS.search("Network Slicing: A Comparative Study")
    assert not PAPER_TITLE_KEYWORDS.search("Reviewer Assignment with Matching Theory")
    assert not PAPER_TITLE_KEYWORDS.search("Optimal Power Allocation for NOMA")


def test_magazine_venue():
    for venue in ("IEEE Communications Magazine", "IEEE Network", "IEEE wireless communications"):
        assert MAGAZINE_VENUE.search(venue)
    for venue in ("IEEE Wireless Communications Letters", "IEEE Transactions on Wireless Communications"):
        assert not MAGAZINE_VENUE.search(venue)


def test_non_paper_types():
    for t in ("book", "book-chapter", "Book", "BookSection", "JournalArticle; Editorial", "dataset"):
        assert NON_PAPER_TYPES.search(t), t
    for t in ("article", "JournalArticle", "JournalArticle; Review", "conference-paper", "preprint"):
        assert not NON_PAPER_TYPES.search(t), t
