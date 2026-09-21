from pydantic import BaseModel

from ailura.graph.signal import (
    RATING_REVIEW,
    conviction_to_rating,
    extract_rating,
    parse_rating,
    rating_to_action,
    rating_to_weight,
)
from ailura.llm.structured import extract_json_block, parse_into


def test_extract_rating_labeled_and_ambiguous():
    assert extract_rating("**Rating:** Overweight\nSome prose about buy and sell") == "Overweight"
    assert extract_rating("Final Rating - sell") == "Sell"
    assert extract_rating("we could buy or we could sell") is None
    assert parse_rating("nothing here") == RATING_REVIEW
    assert extract_rating("Hold") == "Hold"


def test_rating_maps():
    assert rating_to_action("Overweight") == "BUY" and rating_to_action("Underweight") == "SELL"
    assert rating_to_weight("Sell") == 0.0 and rating_to_weight("Sell", allow_short=True) == -1.0
    assert conviction_to_rating(0.9) == "Buy" and conviction_to_rating(-0.3) == "Underweight"


class M(BaseModel):
    a: int
    b: str


def test_parse_into_tolerates_fences_and_prose():
    assert parse_into(M, 'Sure! ```json\n{"a": 1, "b": "x"}\n```').a == 1
    assert parse_into(M, 'text {"a": 2, "b": "y}"} trailing').b == "y}"
    assert extract_json_block("no json") is None
