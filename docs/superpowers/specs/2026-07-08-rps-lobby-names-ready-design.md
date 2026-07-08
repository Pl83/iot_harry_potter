# RPS Duel — Lobby, Names, Ready-up, Rematch

Date: 2026-07-08
Status: Approved

## Problem

The pierre-feuille-ciseaux duel auto-starts the moment a second player joins, has
no player names, and its rematch is polled inside the round loop. Requested changes:

1. Players enter a **name** when joining; names shown throughout the UI.
2. Match starts only when **both players press "Prêt"** — no auto-start.
3. **Rematch** is a full reset back to the ready screen, keeping names.
4. The **Connect card** button locks once connected ("Carte connectée ✓").

Pure game logic (`rps_game.py`: `Match`, `resolve`) is correct and stays untouched.

## Architecture

Replace the *auto-start + in-loop rematch* lifecycle with an explicit
**lobby → match → lobby** cycle driven by a per-player `ready` flag.

### `Player` (rps_server.py)
Add `name` (str) and `ready` (bool) fields.

### Match lifecycle
- `add_player` no longer calls `_start_match`. It registers the socket and, on
  `join`, records the name.
- A `_lobby_loop` task (started once 2 players are present, or persistent):
  waits until *2 players present AND both `ready`* → runs `_run_rounds` for **one**
  match to completion → resets both `ready=false` → broadcasts `lobby` → loops.
- `_run_rounds` plays a single match (delete the internal `_await_rematch` rematch
  branch). `matchOver` returns control to `_lobby_loop`.

## Protocol (`/ws`)

Client → server:
| msg     | payload            | when                                          |
|---------|--------------------|-----------------------------------------------|
| `join`  | `{name}`           | first **Prêt** press; name then locked        |
| `ready` | —                  | every Prêt / Rejouer press (sets ready=true)  |
| `move`  | `{move, conf}`     | on `go` (unchanged)                            |

Server → client:
| msg          | payload                                             | meaning                    |
|--------------|-----------------------------------------------------|----------------------------|
| `joined`     | `{role, name}`                                      | you're registered          |
| `lobby`      | `{you:{name,ready}, opp:{name,ready}\|null}`         | lobby / rematch screen      |
| `start`      | `{you:{name}, opp:{name}, lives}`                   | both ready → match begins  |
| `round`      | as before + `you.name` / `opp.name` carried through | reveal shows real names    |
| `matchOver`  | `{winner}` (+ ready reset server-side)              | match ended                |
| `full`, `opponentLeft`, `countdown`, `go` | unchanged              |                            |

**Start rule:** match begins only when 2 players present AND both `ready`.
**Rematch:** on `matchOver` server resets `ready=false`, keeps names; **Rejouer**
re-sends `ready` → same gate. The old separate `rematch` message is removed.

## Client (`rps-battle.html`)

- **Lobby panel:** name `<input>` + Connect button + **Prêt** button (enabled when
  name is non-empty).
- Connect card is independent of join; on success the button disables and its label
  becomes **"Carte connectée ✓"**.
- First **Prêt** → send `join{name}` + `ready`, lock the name input, button shows
  "En attente de l'adversaire…".
- Names replace the static *Toi / Adversaire* labels in the lives bar and the reveal.
- **matchOver** → show **Rejouer** button → sends `ready`, returns to waiting state
  (names kept from the still-joined players).

### Accepted tradeoff
Because `join` fires on the first Prêt, the opponent's name is not shown in the
pre-match lobby (only "En attente de l'adversaire…"). Names appear at match start
and on the post-match rematch screen. This keeps the lobby to a single button.

## Testing

- `tests/test_rps_game.py` — unchanged, stays green.
- `tests/test_rps_server_e2e.py` — updated for the new handshake: two clients must
  `join`+`ready` before `start`; add a rematch-reset case (matchOver → both `ready`
  again → new `start`).

## Cleanup
Remove the temporary `[ws]` debug logging added during the "no server running"
investigation.
