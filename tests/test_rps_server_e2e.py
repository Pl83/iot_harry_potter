import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aiohttp import web, ClientSession
from rps_server import make_app, GameServer, Player
from rps_game import Match


async def _recv_until(ws, wanted):
    """Lit les messages jusqu'a en trouver un de type `wanted`, le renvoie."""
    while True:
        msg = await ws.receive_json()
        if msg["type"] == wanted:
            return msg


async def _play_one_round(a, b, move_a, move_b):
    """Attend go, envoie les coups, renvoie les deux messages round."""
    await _recv_until(a, "go")
    await _recv_until(b, "go")
    await a.send_json({"type": "move", "move": move_a, "conf": 0.9})
    await b.send_json({"type": "move", "move": move_b, "conf": 0.9})
    ra = await _recv_until(a, "round")
    rb = await _recv_until(b, "round")
    return ra, rb


async def _scenario():
    app = make_app(countdown_step=0.02, collect_timeout=0.5)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8799)
    await site.start()
    try:
        async with ClientSession() as sa, ClientSession() as sb:
            a = await sa.ws_connect("http://127.0.0.1:8799/ws")
            b = await sb.ws_connect("http://127.0.0.1:8799/ws")
            await a.send_json({"type": "join"})
            await b.send_json({"type": "join"})

            sa_start = await _recv_until(a, "start")
            await _recv_until(b, "start")
            assert sa_start["lives"] == {"you": 3, "opp": 3}

            # a gagne 3 fois (rock > scissors) -> b tombe a 0
            for _ in range(3):
                ra, rb = await _play_one_round(a, b, "rock", "scissors")
                assert ra["winner"] == "you"
                assert rb["winner"] == "opp"

            ma = await _recv_until(a, "matchOver")
            mb = await _recv_until(b, "matchOver")
            assert ma["winner"] == "you"
            assert mb["winner"] == "opp"
            await a.close()
            await b.close()
        print("PASS e2e: match complet, a gagne 3-0")
    finally:
        await runner.cleanup()


async def _scenario_full():
    """Une 3e connexion recoit `full`."""
    app = make_app(countdown_step=0.02, collect_timeout=0.5)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8800)
    await site.start()
    try:
        async with ClientSession() as sa, ClientSession() as sb, ClientSession() as sc:
            a = await sa.ws_connect("http://127.0.0.1:8800/ws")
            b = await sb.ws_connect("http://127.0.0.1:8800/ws")
            await a.send_json({"type": "join"})
            await b.send_json({"type": "join"})
            await _recv_until(a, "joined")
            await _recv_until(b, "joined")
            c = await sc.ws_connect("http://127.0.0.1:8800/ws")
            await c.send_json({"type": "join"})
            msg = await c.receive_json()
            assert msg["type"] == "full", msg
            await a.close(); await b.close(); await c.close()
        print("PASS e2e: 3e joueur rejete (full)")
    finally:
        await runner.cleanup()


async def _scenario_disconnect():
    """Deconnexion en cours de match -> l'autre joueur recoit opponentLeft,
    et le serveur ne crashe pas (regression: mutation de dict pendant iteration)."""
    app = make_app(countdown_step=0.02, collect_timeout=0.5)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8801)
    await site.start()
    try:
        async with ClientSession() as sa, ClientSession() as sb:
            a = await sa.ws_connect("http://127.0.0.1:8801/ws")
            b = await sb.ws_connect("http://127.0.0.1:8801/ws")
            await a.send_json({"type": "join"})
            await b.send_json({"type": "join"})

            await _recv_until(a, "start")
            await _recv_until(b, "start")

            # au moins un round complet pour que la boucle de round soit
            # activement en train de diffuser (countdown/go/round)
            await _play_one_round(a, b, "rock", "scissors")

            # coupe A en plein match, avant/autour du prochain decompte
            await a.close()

            mb = await _recv_until(b, "opponentLeft")
            assert mb["type"] == "opponentLeft"

            await b.close()
        print("PASS e2e: deconnexion en cours de match -> opponentLeft")
    finally:
        await runner.cleanup()


class FakeWS:
    def __init__(self, on_send=None):
        self.closed = False
        self._on_send = on_send

    async def send_json(self, payload):
        if self._on_send is not None:
            self._on_send(payload)


async def _scenario_send_round_race():
    """Deconnexion PENDANT la diffusion de _send_round (mutation de dict en
    cours d'iteration) -> ne doit PAS lever KeyError (regression Fix A)."""
    server = GameServer(countdown_step=0.02, collect_timeout=0.5)

    def on_a_send(payload):
        # simule la deconnexion de B pendant que _send_round diffuse a A
        server.players.pop("b", None)

    ws_a = FakeWS(on_send=on_a_send)
    ws_b = FakeWS()
    player_a = Player(ws_a, "a")
    player_b = Player(ws_b, "b")
    player_a.conf = 0.9
    player_b.conf = 0.8
    server.players["a"] = player_a
    server.players["b"] = player_b

    server.match = Match()
    result = server.match.play_round("rock", "scissors")

    try:
        await server._send_round("rock", "scissors", result)
    except Exception as exc:
        raise AssertionError(
            f"_send_round a leve {exc!r} suite a une deconnexion en cours de broadcast"
        ) from exc

    print("PASS e2e: _send_round survit a une deconnexion en cours de broadcast")


if __name__ == "__main__":
    asyncio.run(_scenario())
    asyncio.run(_scenario_full())
    asyncio.run(_scenario_disconnect())
    asyncio.run(_scenario_send_round_race())
    print("\nOK — e2e serveur")
