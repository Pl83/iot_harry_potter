#!/usr/bin/env python3
"""Logique pure du duel pierre-feuille-ciseaux (sans reseau, testable seule)."""

MOVES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


def resolve(move_a, move_b):
    """Renvoie 'a', 'b' ou 'tie'. 'whiff' (rate) perd face a un coup valide ;
    deux ratés = egalite ; coups identiques = egalite."""
    a_valid = move_a in MOVES
    b_valid = move_b in MOVES
    if not a_valid and not b_valid:
        return "tie"
    if not a_valid:
        return "b"
    if not b_valid:
        return "a"
    if move_a == move_b:
        return "tie"
    return "a" if BEATS[move_a] == move_b else "b"


class Match:
    """Etat d'un match : vies des deux joueurs, resolution round par round."""

    def __init__(self, lives=3):
        self.lives = {"a": lives, "b": lives}

    def play_round(self, move_a, move_b):
        winner = resolve(move_a, move_b)
        if winner == "a":
            self.lives["b"] -= 1
        elif winner == "b":
            self.lives["a"] -= 1
        over = self.lives["a"] <= 0 or self.lives["b"] <= 0
        match_winner = None
        if over:
            match_winner = "a" if self.lives["a"] > 0 else "b"
        return {
            "winner": winner,
            "lives": dict(self.lives),
            "over": over,
            "match_winner": match_winner,
        }
