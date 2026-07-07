import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rps_game import resolve, Match


def test_resolve_truth_table():
    assert resolve("rock", "scissors") == "a"
    assert resolve("scissors", "rock") == "b"
    assert resolve("scissors", "paper") == "a"
    assert resolve("paper", "scissors") == "b"
    assert resolve("paper", "rock") == "a"
    assert resolve("rock", "paper") == "b"


def test_resolve_ties():
    for m in ("rock", "paper", "scissors"):
        assert resolve(m, m) == "tie"


def test_resolve_whiffs():
    assert resolve("whiff", "rock") == "b"       # a rate -> b gagne
    assert resolve("paper", "whiff") == "a"       # b rate -> a gagne
    assert resolve("whiff", "whiff") == "tie"     # double raté -> egalite


def test_match_life_loss_and_tie():
    m = Match()
    assert m.lives == {"a": 3, "b": 3}
    r = m.play_round("rock", "scissors")           # a gagne
    assert r["winner"] == "a"
    assert r["lives"] == {"a": 3, "b": 2}
    assert r["over"] is False and r["match_winner"] is None
    r = m.play_round("rock", "rock")               # egalite
    assert r["winner"] == "tie"
    assert r["lives"] == {"a": 3, "b": 2}


def test_match_ends_at_zero():
    m = Match()
    for _ in range(2):
        m.play_round("rock", "scissors")           # b: 3 -> 1
    r = m.play_round("rock", "scissors")           # b: 1 -> 0
    assert r["lives"] == {"a": 3, "b": 0}
    assert r["over"] is True
    assert r["match_winner"] == "a"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nOK — {len(fns)} tests")


if __name__ == "__main__":
    _run_all()
