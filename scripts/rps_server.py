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
        self.name = None          # pseudo saisi au join
        self.ready = False        # a appuye sur "Pret"
        self.move = None          # coup du round courant
        self.conf = None          # confiance du coup courant

    async def send(self, **payload):
        if not self.ws.closed:
            await self.ws.send_json(payload)


class GameServer:
    """Un serveur = une partie a la fois (2 joueurs).

    Cycle de vie : salon (lobby) -> match -> salon. Le match ne demarre que
    lorsque les 2 joueurs sont presents ET tous deux `ready`. La revanche
    reutilise ce meme portique : fin de match -> ready remis a False -> les
    joueurs reppuient sur "Rejouer" -> nouveau match.
    """

    def __init__(self, countdown_step=1.0, collect_timeout=2.5):
        self.players = {}                  # role -> Player
        self.match = None
        self.countdown_step = countdown_step
        self.collect_timeout = collect_timeout
        self._lobby_task = None
        self._moves_event = asyncio.Event()

    def _other(self, role):
        return "b" if role == "a" else "a"

    async def add_player(self, ws, name=None):
        if len(self.players) >= 2:
            await ws.send_json({"type": "full"})
            return None
        role = "a" if "a" not in self.players else "b"
        player = Player(ws, role)
        player.name = (name or "").strip() or f"Joueur {role.upper()}"
        self.players[role] = player
        await player.send(type="joined", role=role.upper(), name=player.name)
        await self._broadcast_lobby()
        if len(self.players) == 2 and (self._lobby_task is None or self._lobby_task.done()):
            self._lobby_task = asyncio.create_task(self._lobby_loop())
        return player

    async def set_ready(self, player):
        player.ready = True
        await self._broadcast_lobby()

    async def remove_player(self, player):
        self.players.pop(player.role, None)
        if self._lobby_task and not self._lobby_task.done():
            self._lobby_task.cancel()
        self._lobby_task = None
        self.match = None
        other = self.players.get(self._other(player.role))
        if other:
            other.ready = False
            await other.send(type="opponentLeft")
            await self._broadcast_lobby()

    async def _broadcast_lobby(self):
        for role, p in list(self.players.items()):
            opp = self.players.get(self._other(role))
            await p.send(
                type="lobby",
                you={"name": p.name, "ready": p.ready},
                opp={"name": opp.name, "ready": opp.ready} if opp else None,
            )

    async def _lobby_loop(self):
        """Superviseur : attend que les 2 joueurs soient prets, joue un match,
        puis remet le salon a zero pour une eventuelle revanche."""
        try:
            while len(self.players) == 2:
                while len(self.players) == 2 and not all(p.ready for p in self.players.values()):
                    await asyncio.sleep(0.05)
                if len(self.players) != 2:
                    return
                await self._start_match()
                await self._run_rounds()
                if len(self.players) != 2:
                    return
                for p in self.players.values():
                    p.ready = False
                await self._broadcast_lobby()
        except asyncio.CancelledError:
            return

    async def _start_match(self):
        self.match = Match()
        for role, p in list(self.players.items()):
            opp = self.players.get(self._other(role))
            await p.send(
                type="start",
                you={"name": p.name},
                opp={"name": opp.name} if opp else None,
                lives={"you": 3, "opp": 3},
            )

    async def _run_rounds(self):
        """Joue UN match jusqu'a matchOver (ou retour si un joueur part)."""
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

                if len(self.players) != 2:
                    return  # un joueur est parti pendant la collecte
                move_a = self.players["a"].move or "whiff"
                move_b = self.players["b"].move or "whiff"
                result = self.match.play_round(move_a, move_b)
                await self._send_round(move_a, move_b, result)

                if result["over"]:
                    await self._send_match_over(result["match_winner"])
                    return
                await asyncio.sleep(self.countdown_step)
        except asyncio.CancelledError:
            return

    def record_move(self, player, move, conf=None):
        player.move = move if move in ("rock", "paper", "scissors") else "whiff"
        player.conf = conf
        if all(p.move is not None for p in self.players.values()) and len(self.players) == 2:
            self._moves_event.set()

    async def _broadcast(self, **payload):
        for p in list(self.players.values()):
            await p.send(**payload)

    async def _send_round(self, move_a, move_b, result):
        snapshot = list(self.players.items())
        moves = {"a": move_a, "b": move_b}
        confs = {role: p.conf for role, p in snapshot}
        names = {role: p.name for role, p in snapshot}
        for role, p in snapshot:
            other = self._other(role)
            if result["winner"] == "tie":
                winner = "tie"
            elif result["winner"] == role:
                winner = "you"
            else:
                winner = "opp"
            await p.send(
                type="round",
                you={"move": moves[role], "conf": confs.get(role), "name": names.get(role)},
                opp={"move": moves[other], "conf": confs.get(other), "name": names.get(other)},
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
            player = await server.add_player(ws, data.get("name"))
            if player is None:
                await ws.close()
                return ws
        elif t == "ready" and player is not None:
            await server.set_ready(player)
        elif t == "move" and player is not None:
            server.record_move(player, data.get("move"), data.get("conf"))
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
