#!/usr/bin/env python3
"""Serveur arbitre du duel pierre-feuille-ciseaux (aiohttp : HTTP statique + WS).

Sert la page a `/`, les modules JS, et un WebSocket `/ws`. Detient toute la
verite de jeu : timing du decompte, collecte des coups, resolution, vies.

Usage:
    python scripts/rps_server.py      # ecoute 0.0.0.0:8000
"""
import asyncio
import socket
from pathlib import Path

from aiohttp import web

from rps_game import Match

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
STATIC_FILES = {
    "/": ("rps-battle.html", "text/html; charset=utf-8"),
    "/rps-battle.html": ("rps-battle.html", "text/html; charset=utf-8"),
    "/inference.js": ("inference.js", "application/javascript"),
    "/rps-moves.js": ("rps-moves.js", "application/javascript"),
    "/movement_cnn_weights.js": ("movement_cnn_weights.js", "application/javascript"),
}


class Player:
    def __init__(self, ws, role):
        self.ws = ws
        self.role = role          # "a" ou "b"
        self.move = None          # coup du round courant
        self.conf = None          # confiance du coup courant
        self.rematch = False

    async def send(self, **payload):
        if not self.ws.closed:
            await self.ws.send_json(payload)


class GameServer:
    """Un serveur = une partie a la fois (2 joueurs)."""

    def __init__(self, countdown_step=1.0, collect_timeout=2.5):
        self.players = {}                  # role -> Player
        self.match = None
        self.countdown_step = countdown_step
        self.collect_timeout = collect_timeout
        self._round_task = None
        self._moves_event = asyncio.Event()

    def _other(self, role):
        return "b" if role == "a" else "a"

    async def add_player(self, ws):
        if len(self.players) >= 2:
            await ws.send_json({"type": "full"})
            return None
        role = "a" if "a" not in self.players else "b"
        player = Player(ws, role)
        self.players[role] = player
        await player.send(type="joined", role=role.upper())
        if len(self.players) == 1:
            await player.send(type="waiting")
        elif len(self.players) == 2:
            await self._start_match()
        return player

    async def remove_player(self, player):
        self.players.pop(player.role, None)
        if self._round_task and not self._round_task.done():
            self._round_task.cancel()
        other = self.players.get(self._other(player.role))
        if other:
            await other.send(type="opponentLeft")
        self.match = None

    async def _start_match(self):
        self.match = Match()
        for role, p in list(self.players.items()):
            p.rematch = False
            await p.send(type="start", lives={"you": 3, "opp": 3})
        self._round_task = asyncio.create_task(self._run_rounds())

    async def _run_rounds(self):
        try:
            while self.match is not None and len(self.players) == 2:
                for role, p in list(self.players.items()):
                    p.move = None
                    p.conf = None
                self._moves_event.clear()

                for n in (3, 2, 1):
                    await self._broadcast(type="countdown", n=n)
                    await asyncio.sleep(self.countdown_step)
                await self._broadcast(type="go")

                try:
                    await asyncio.wait_for(self._moves_event.wait(),
                                           timeout=self.collect_timeout)
                except asyncio.TimeoutError:
                    pass  # coups manquants -> whiff ci-dessous

                move_a = self.players["a"].move or "whiff"
                move_b = self.players["b"].move or "whiff"
                result = self.match.play_round(move_a, move_b)
                await self._send_round(move_a, move_b, result)

                if result["over"]:
                    await self._send_match_over(result["match_winner"])
                    if not await self._await_rematch():
                        return
                    self.match = Match()
                    for p in list(self.players.values()):
                        p.rematch = False
                        await p.send(type="start", lives={"you": 3, "opp": 3})
                else:
                    await asyncio.sleep(self.countdown_step)
        except asyncio.CancelledError:
            return

    async def _await_rematch(self):
        """Attend que les 2 joueurs demandent la revanche. False si abandon."""
        while len(self.players) == 2 and not all(p.rematch for p in self.players.values()):
            await asyncio.sleep(0.05)
        return len(self.players) == 2 and all(p.rematch for p in self.players.values())

    def record_move(self, player, move, conf=None):
        player.move = move if move in ("rock", "paper", "scissors") else "whiff"
        player.conf = conf
        if all(p.move is not None for p in self.players.values()) and len(self.players) == 2:
            self._moves_event.set()

    async def _broadcast(self, **payload):
        for p in list(self.players.values()):
            await p.send(**payload)

    async def _send_round(self, move_a, move_b, result):
        moves = {"a": move_a, "b": move_b}
        for role, p in list(self.players.items()):
            other = self._other(role)
            if result["winner"] == "tie":
                winner = "tie"
            elif result["winner"] == role:
                winner = "you"
            else:
                winner = "opp"
            await p.send(
                type="round",
                you={"move": moves[role], "conf": self.players[role].conf},
                opp={"move": moves[other], "conf": self.players[other].conf},
                winner=winner,
                lives={"you": result["lives"][role], "opp": result["lives"][other]},
            )

    async def _send_match_over(self, match_winner):
        for role, p in list(self.players.items()):
            await p.send(type="matchOver",
                         winner="you" if match_winner == role else "opp")


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    server = request.app["game"]
    player = None
    async for msg in ws:
        if msg.type != web.WSMsgType.TEXT:
            continue
        try:
            data = msg.json()
        except Exception:
            continue
        t = data.get("type")
        if t == "join" and player is None:
            player = await server.add_player(ws)
            if player is None:
                await ws.close()
                return ws
        elif t == "move" and player is not None:
            server.record_move(player, data.get("move"), data.get("conf"))
        elif t == "rematch" and player is not None:
            player.rematch = True
    if player is not None:
        await server.remove_player(player)
    return ws


def _static_handler(filename, content_type, web_dir):
    async def handler(request):
        path = web_dir / filename
        if not path.exists():
            return web.Response(status=404, text=f"{filename} introuvable")
        return web.Response(body=path.read_bytes(), content_type=content_type.split(";")[0],
                            charset="utf-8" if "charset" in content_type else None)
    return handler


def make_app(web_dir=None, countdown_step=1.0, collect_timeout=2.5):
    effective_web_dir = Path(web_dir) if web_dir is not None else WEB_DIR
    app = web.Application()
    app["game"] = GameServer(countdown_step=countdown_step, collect_timeout=collect_timeout)
    for route, (filename, ctype) in STATIC_FILES.items():
        app.router.add_get(route, _static_handler(filename, ctype, effective_web_dir))
    app.router.add_get("/ws", ws_handler)
    return app


def _local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    app = make_app()
    ip = _local_ip()
    print("=" * 56)
    print("  DUEL PIERRE-FEUILLE-CISEAUX — serveur arbitre")
    print("=" * 56)
    print(f"  Hote  : http://{ip}:8000/")
    print(f"  Local : http://127.0.0.1:8000/")
    print("  Les deux joueurs ouvrent l'URL Hote, connectent leur carte.")
    print("=" * 56)
    web.run_app(app, host="0.0.0.0", port=8000, print=None)


if __name__ == "__main__":
    main()
