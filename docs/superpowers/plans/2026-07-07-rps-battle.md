# Duel Pierre-Feuille-Ciseaux par gestes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deux PC sur le même WiFi s'affrontent à pierre-feuille-ciseaux, le coup de chaque joueur étant un geste reconnu en direct par le `MovementCNN`, arbitré par un serveur Python autoritaire.

**Architecture:** Serveur aiohttp (HTTP statique + WebSocket) sur une des deux machines, détenteur de toute la vérité de jeu (vies, timing du décompte, résolution). La logique de jeu pure est isolée dans `scripts/rps_game.py` (testable sans réseau). Le forward-pass CNN est extrait dans `web/inference.js`, partagé par le moniteur existant et la nouvelle page `web/rps-battle.html`.

**Tech Stack:** Python 3.11, aiohttp 3.11 (serveur), JS pur navigateur (Web Serial + WebSocket), Node.js (tests des modules JS), torch (référence de parité).

## Global Constraints

- Classes du modèle, ordre exact : `["circle", "horizontal", "static", "vertical"]` (labels circle=0, horizontal=1, static=2, vertical=3).
- Mapping geste → coup : `vertical`=rock, `horizontal`=paper, `circle`=scissors, `static`=whiff.
- Seuil de confiance pour valider un coup : `0.6` (softmax de la classe prédite) ; en-dessous → whiff.
- Coups sur le fil (protocole) en anglais : `"rock" | "paper" | "scissors" | "whiff"`. Affichage en français.
- Vies initiales : `3` par joueur. Premier à 0 perd.
- Serveur : écoute `0.0.0.0:8000`, sert la page à `/` et les JS à `/inference.js`, `/rps-moves.js`, `/movement_cnn_weights.js`, WebSocket à `/ws`.
- Fenêtre CNN : 100 derniers échantillons accéléromètre (`s.a`, PAS le gyroscope), z-score par axe (ddof=0).
- Fichiers créés côté web sans build : chargés par `<script src>` (fonctionne en HTTP servi par aiohttp).
- Timeout de collecte d'un coup après `go` : 2,5 s → whiff.

---

### Task 1: Extraire le moteur d'inférence dans `web/inference.js`

**Files:**
- Create: `web/inference.js`
- Create: `tests/test_web_inference.mjs`
- Modify: `web/mpu6050-monitor.html` (bloc lignes ~586-659 et l'appel ligne ~708)

**Interfaces:**
- Consumes: `window.MODEL_WEIGHTS` (défini par `web/movement_cnn_weights.js`, déjà généré) avec `.layers`, `.classes`, `.seq_len`.
- Produces: `window.MovementModel = { forward(x), softmax(z), classes, seqLen, ready }`
  - `forward(x)` : `x` = tableau de 3 `Float64Array(100)` → `Float64Array` de logits (longueur = nb classes).
  - `softmax(z)` : tableau/Float64Array → tableau de probabilités.
  - `classes` : `string[]`, `seqLen` : `number`, `ready` : `boolean`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_inference.mjs` (Node ESM). Il charge les poids + le module et vérifie la parité sur une entrée déterministe (valeurs de référence PyTorch déjà mesurées : logits ≈ `4.0416 0.9332 -5.4571 -2.0294`, argmax 0).

```js
import fs from 'fs';

// Shim navigateur : les fichiers font `window.X = ...`
global.window = {};
const load = p => (0, eval)(fs.readFileSync(p, 'utf8'));
load('web/movement_cnn_weights.js');
load('web/inference.js');
const M = global.window.MovementModel;

// Entrée déterministe : x[c][t] = sin(0.1*t + c)
const x = [0, 1, 2].map(c => {
  const a = new Float64Array(100);
  for (let t = 0; t < 100; t++) a[t] = Math.sin(0.1 * t + c);
  return a;
});
const logits = Array.from(M.forward(x));
const expected = [4.0416, 0.9332, -5.4571, -2.0294];
let maxDiff = 0;
for (let i = 0; i < 4; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - expected[i]));
const argmax = logits.indexOf(Math.max(...logits));

const probs = M.softmax(logits);
const sum = probs.reduce((a, b) => a + b, 0);

let ok = true;
function check(name, cond) { console.log((cond ? 'PASS' : 'FAIL') + ' ' + name); if (!cond) ok = false; }
check('module chargé', !!M && M.ready === true);
check('classes exposées', M.classes.length === 4 && M.classes[0] === 'circle');
check('parité logits (diff < 1e-2)', maxDiff < 1e-2);
check('argmax = 0 (circle)', argmax === 0);
check('softmax somme à 1', Math.abs(sum - 1) < 1e-9);
process.exit(ok ? 0 : 1);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tests/test_web_inference.mjs`
Expected: FAIL — `web/inference.js` n'existe pas encore (erreur de lecture de fichier ou `MovementModel` undefined).

- [ ] **Step 3: Create `web/inference.js`**

```js
// Forward-pass CNN (MovementCNN) en JS pur, partagé par le moniteur et le duel.
// Nécessite window.MODEL_WEIGHTS (chargé avant, via movement_cnn_weights.js).
(function () {
  const MODEL = window.MODEL_WEIGHTS || null;

  function fwdConv(x, L) {
    const W = L.weight, b = L.bias, pad = L.pad;
    const inC = x.length, len = x[0].length;
    const outC = W.length, k = W[0][0].length;
    const padded = x.map(ch => { const p = new Float64Array(len + 2 * pad); p.set(ch, pad); return p; });
    const lout = len + 2 * pad - k + 1;
    const out = [];
    for (let oc = 0; oc < outC; oc++) {
      const acc = new Float64Array(lout);
      for (let ic = 0; ic < inC; ic++) {
        const wk = W[oc][ic], pc = padded[ic];
        for (let kk = 0; kk < k; kk++) {
          const wv = wk[kk];
          for (let i = 0; i < lout; i++) acc[i] += wv * pc[i + kk];
        }
      }
      const bo = b[oc];
      for (let i = 0; i < lout; i++) acc[i] += bo;
      out.push(acc);
    }
    return out;
  }
  function fwdBN(x, L) {
    const w = L.weight, b = L.bias, m = L.mean, v = L.var, eps = L.eps;
    return x.map((ch, c) => {
      const scale = w[c] / Math.sqrt(v[c] + eps), shift = b[c] - m[c] * scale;
      const o = new Float64Array(ch.length);
      for (let i = 0; i < ch.length; i++) o[i] = ch[i] * scale + shift;
      return o;
    });
  }
  function fwdReLU(x) {
    return x.map(ch => { const o = new Float64Array(ch.length); for (let i = 0; i < ch.length; i++) o[i] = ch[i] > 0 ? ch[i] : 0; return o; });
  }
  function fwdMaxPool(x, L) {
    const s = L.size;
    return x.map(ch => {
      const lout = Math.floor(ch.length / s), o = new Float64Array(lout);
      for (let i = 0; i < lout; i++) { let mx = -Infinity; for (let j = 0; j < s; j++) { const val = ch[i * s + j]; if (val > mx) mx = val; } o[i] = mx; }
      return o;
    });
  }
  function fwdAvgPool(x) {
    return x.map(ch => { let s = 0; for (let i = 0; i < ch.length; i++) s += ch[i]; return new Float64Array([s / ch.length]); });
  }
  function fwdLinear(x, L) {
    const flat = []; for (const ch of x) for (const v of ch) flat.push(v);
    const W = L.weight, b = L.bias, out = new Float64Array(W.length);
    for (let o = 0; o < W.length; o++) { let s = b[o]; const wr = W[o]; for (let i = 0; i < flat.length; i++) s += wr[i] * flat[i]; out[o] = s; }
    return out;
  }
  function forward(x) {
    let a = x;
    for (const L of MODEL.layers) {
      switch (L.type) {
        case 'conv': a = fwdConv(a, L); break;
        case 'bn': a = fwdBN(a, L); break;
        case 'relu': a = fwdReLU(a); break;
        case 'maxpool': a = fwdMaxPool(a, L); break;
        case 'avgpool': a = fwdAvgPool(a); break;
        case 'linear': return fwdLinear(a, L);
      }
    }
    return a;
  }
  function softmax(z) {
    let mx = -Infinity; for (const v of z) if (v > mx) mx = v;
    const e = Array.from(z, v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }

  window.MovementModel = {
    ready: !!MODEL,
    classes: MODEL ? MODEL.classes : [],
    seqLen: MODEL ? MODEL.seq_len : 100,
    forward: forward,
    softmax: softmax,
  };
})();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node tests/test_web_inference.mjs`
Expected: 5 lignes PASS, exit 0.

- [ ] **Step 5: Refactor `web/mpu6050-monitor.html` pour consommer le module**

Dans `web/mpu6050-monitor.html` :

1. Après la balise `<script src="movement_cnn_weights.js"></script>`, ajouter sur la ligne suivante :
```html
<script src="inference.js"></script>
```

2. Dans le `<script>` principal, section `// ---------- Inférence live` : **supprimer** les 8 fonctions inline `fwdConv`, `fwdBN`, `fwdReLU`, `fwdMaxPool`, `fwdAvgPool`, `fwdLinear`, `forward`, `softmax` (bloc actuel des lignes ~590-659). **Conserver** les deux lignes :
```js
const MODEL = window.MODEL_WEIGHTS || null;
const INFER_WIN = MODEL ? MODEL.seq_len : 100;
```

3. Remplacer l'appel unique dans `runInference()` :
```js
  const probs = softmax(Array.from(forward(buildInput())));
```
par :
```js
  const probs = MovementModel.softmax(MovementModel.forward(buildInput()));
```

- [ ] **Step 6: Vérifier que le moniteur ne référence plus les fonctions supprimées**

Run: `grep -nE "function (forward|softmax|fwd)" web/mpu6050-monitor.html`
Expected: aucune sortie (les définitions ont migré).
Run: `grep -n "MovementModel" web/mpu6050-monitor.html`
Expected: la ligne de `runInference()` utilisant `MovementModel.softmax(...)`.

- [ ] **Step 7: Commit**

```bash
git add web/inference.js tests/test_web_inference.mjs web/mpu6050-monitor.html
git commit -m "refactor: extract CNN forward-pass into shared web/inference.js"
```

---

### Task 2: Mapping geste → coup dans `web/rps-moves.js`

**Files:**
- Create: `web/rps-moves.js`
- Create: `tests/test_rps_moves.mjs`

**Interfaces:**
- Produces: `window.RPS = { GESTURE_TO_MOVE, THRESHOLD, classifyMove(className, conf) }`
  - `GESTURE_TO_MOVE` : `{circle:"scissors", horizontal:"paper", static:null, vertical:"rock"}`.
  - `THRESHOLD` : `0.6`.
  - `classifyMove(className, conf)` → `"rock" | "paper" | "scissors" | "whiff"`. Renvoie `"whiff"` si `conf < THRESHOLD`, si `className` est `"static"`, ou si `className` est inconnu/nul.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rps_moves.mjs`:

```js
import fs from 'fs';
global.window = {};
(0, eval)(fs.readFileSync('web/rps-moves.js', 'utf8'));
const RPS = global.window.RPS;

let ok = true;
function check(name, cond) { console.log((cond ? 'PASS' : 'FAIL') + ' ' + name); if (!cond) ok = false; }

check('vertical + haute conf → rock', RPS.classifyMove('vertical', 0.9) === 'rock');
check('horizontal → paper', RPS.classifyMove('horizontal', 0.9) === 'paper');
check('circle → scissors', RPS.classifyMove('circle', 0.9) === 'scissors');
check('static → whiff', RPS.classifyMove('static', 0.99) === 'whiff');
check('confiance sous seuil → whiff', RPS.classifyMove('vertical', 0.4) === 'whiff');
check('classe inconnue → whiff', RPS.classifyMove('bogus', 0.9) === 'whiff');
check('seuil exact 0.6 accepté', RPS.classifyMove('vertical', 0.6) === 'rock');
process.exit(ok ? 0 : 1);
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node tests/test_rps_moves.mjs`
Expected: FAIL — `web/rps-moves.js` absent.

- [ ] **Step 3: Create `web/rps-moves.js`**

```js
// Mapping geste (classe CNN) → coup pierre-feuille-ciseaux. Pur, testable en Node.
(function () {
  const GESTURE_TO_MOVE = { circle: 'scissors', horizontal: 'paper', static: null, vertical: 'rock' };
  const THRESHOLD = 0.6;

  function classifyMove(className, conf) {
    if (typeof conf !== 'number' || conf < THRESHOLD) return 'whiff';
    const move = GESTURE_TO_MOVE[className];
    return move ? move : 'whiff';
  }

  window.RPS = { GESTURE_TO_MOVE: GESTURE_TO_MOVE, THRESHOLD: THRESHOLD, classifyMove: classifyMove };
})();
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node tests/test_rps_moves.mjs`
Expected: 7 lignes PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add web/rps-moves.js tests/test_rps_moves.mjs
git commit -m "feat: gesture-to-move mapping for RPS (web/rps-moves.js)"
```

---

### Task 3: Logique de jeu pure dans `scripts/rps_game.py`

**Files:**
- Create: `scripts/rps_game.py`
- Create: `tests/test_rps_game.py`

**Interfaces:**
- Produces:
  - `MOVES = ("rock", "paper", "scissors")`, `BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}`.
  - `resolve(move_a, move_b) -> "a" | "b" | "tie"` : applique les règles, gère les whiffs.
  - `class Match(lives=3)` avec attribut `lives = {"a": int, "b": int}` et méthode
    `play_round(move_a, move_b) -> dict` avec clés `winner` (`"a"|"b"|"tie"`), `lives` (`{"a":int,"b":int}`), `over` (bool), `match_winner` (`"a"|"b"|None`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_rps_game.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_rps_game.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'rps_game'`.

- [ ] **Step 3: Create `scripts/rps_game.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_rps_game.py`
Expected: 5 lignes PASS + `OK — 5 tests`.

- [ ] **Step 5: Commit**

```bash
git add scripts/rps_game.py tests/test_rps_game.py
git commit -m "feat: pure RPS game logic (resolve + Match)"
```

---

### Task 4: Serveur aiohttp arbitre dans `scripts/rps_server.py`

**Files:**
- Create: `scripts/rps_server.py`
- Create: `tests/test_rps_server_e2e.py`

**Interfaces:**
- Consumes: `rps_game.Match`, `rps_game.resolve`.
- Produces:
  - `make_app(web_dir=None, countdown_step=1.0, collect_timeout=2.5) -> aiohttp.web.Application`
    — fabrique l'app (routes HTTP + `/ws`) et attache un `GameServer` partagé. Les délais sont paramétrables pour accélérer les tests.
  - Comportement WebSocket conforme au protocole du spec (messages `joined`, `waiting`, `start`, `countdown`, `go`, `round`, `matchOver`, `opponentLeft`, `full` ; entrants `join`, `move`, `rematch`).
  - `main()` : lance `web.run_app(make_app(), host="0.0.0.0", port=8000)` et imprime les URLs (IP locale).

**Interfaces (détail messages) :** identique au spec `docs/superpowers/specs/2026-07-07-rps-battle-design.md` section « Protocole WebSocket ». `round` est personnalisé par destinataire : `{type:"round", you:{move,conf}, opp:{move,conf}, winner:"you"|"opp"|"tie", lives:{you,opp}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rps_server_e2e.py`. Il démarre l'app sur un port éphémère avec des délais courts, connecte deux clients, joue un match scripté et vérifie la séquence + le vainqueur.

```python
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aiohttp import web, ClientSession
from rps_server import make_app


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
            c = await sc.ws_connect("http://127.0.0.1:8800/ws")
            await c.send_json({"type": "join"})
            msg = await c.receive_json()
            assert msg["type"] == "full", msg
            await a.close(); await b.close(); await c.close()
        print("PASS e2e: 3e joueur rejete (full)")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(_scenario())
    asyncio.run(_scenario_full())
    print("\nOK — e2e serveur")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_rps_server_e2e.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'rps_server'`.

- [ ] **Step 3: Create `scripts/rps_server.py`**

```python
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
        for role, p in self.players.items():
            p.rematch = False
            await p.send(type="start", lives={"you": 3, "opp": 3})
        self._round_task = asyncio.create_task(self._run_rounds())

    async def _run_rounds(self):
        try:
            while self.match is not None and len(self.players) == 2:
                for role, p in self.players.items():
                    p.move = None
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
                    for p in self.players.values():
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

    def record_move(self, player, move):
        player.move = move if move in ("rock", "paper", "scissors") else "whiff"
        if all(p.move is not None for p in self.players.values()) and len(self.players) == 2:
            self._moves_event.set()

    async def _broadcast(self, **payload):
        for p in list(self.players.values()):
            await p.send(**payload)

    async def _send_round(self, move_a, move_b, result):
        moves = {"a": move_a, "b": move_b}
        for role, p in self.players.items():
            other = self._other(role)
            if result["winner"] == "tie":
                winner = "tie"
            elif result["winner"] == role:
                winner = "you"
            else:
                winner = "opp"
            await p.send(
                type="round",
                you={"move": moves[role], "conf": None},
                opp={"move": moves[other], "conf": None},
                winner=winner,
                lives={"you": result["lives"][role], "opp": result["lives"][other]},
            )

    async def _send_match_over(self, match_winner):
        for role, p in self.players.items():
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
            server.record_move(player, data.get("move"))
        elif t == "rematch" and player is not None:
            player.rematch = True
    if player is not None:
        await server.remove_player(player)
    return ws


def _static_handler(filename, content_type):
    async def handler(request):
        path = WEB_DIR / filename
        if not path.exists():
            return web.Response(status=404, text=f"{filename} introuvable")
        return web.Response(body=path.read_bytes(), content_type=content_type.split(";")[0],
                            charset="utf-8" if "charset" in content_type else None)
    return handler


def make_app(web_dir=None, countdown_step=1.0, collect_timeout=2.5):
    app = web.Application()
    app["game"] = GameServer(countdown_step=countdown_step, collect_timeout=collect_timeout)
    for route, (filename, ctype) in STATIC_FILES.items():
        app.router.add_get(route, _static_handler(filename, ctype))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_rps_server_e2e.py`
Expected: `PASS e2e: match complet, a gagne 3-0`, `PASS e2e: 3e joueur rejete (full)`, `OK — e2e serveur`.

- [ ] **Step 5: Commit**

```bash
git add scripts/rps_server.py tests/test_rps_server_e2e.py
git commit -m "feat: aiohttp RPS referee server (HTTP static + WebSocket state machine)"
```

---

### Task 5: Page de duel `web/rps-battle.html`

**Files:**
- Create: `web/rps-battle.html`

**Interfaces:**
- Consumes: `MovementModel.forward/softmax/classes/seqLen` (Task 1), `RPS.classifyMove` (Task 2), le serveur WS (Task 4) via `ws://<host>/ws`.
- Produces: page jouable. Pas de test automatisé (DOM + Web Serial + WS) ; vérification manuelle scriptée + contrôle de syntaxe.

Cette page réutilise le code Web Serial du moniteur (lecture 6 axes, `samples[]`) et le buildInput z-score. Elle capture la prédiction à `go` et pilote l'UI selon les messages serveur.

- [ ] **Step 1: Create `web/rps-battle.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duel — Pierre Feuille Ciseaux (gestes)</title>
<style>
  :root { --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --muted:#898781;
          --border:rgba(11,11,11,.10); --accent:#2a78d6; --win:#1baf7a; --lose:#d03b3b; }
  @media (prefers-color-scheme: dark) {
    :root { --page:#0d0d0d; --surface:#1a1a19; --text:#fff; --muted:#898781;
            --border:rgba(255,255,255,.10); --accent:#3987e5; --win:#199e70; --lose:#e5564f; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--page); color:var(--text); padding:24px;
         font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:20px; }
  h1 { font-size:18px; margin:0; font-weight:650; }
  button { font:inherit; font-size:14px; font-weight:600; padding:8px 18px; border-radius:8px;
           cursor:pointer; border:1px solid var(--border); background:var(--accent); color:#fff; }
  button.secondary { background:var(--surface); color:var(--text); }
  button:disabled { opacity:.45; cursor:default; }
  #status { font-size:13px; color:var(--muted); }
  .panel { background:var(--surface); border:1px solid var(--border); border-radius:12px;
           padding:20px; margin-bottom:20px; }
  .lives { display:flex; justify-content:space-between; gap:20px; font-size:15px; font-weight:650; }
  .hearts { letter-spacing:2px; font-size:20px; }
  .countdown { text-align:center; font-size:64px; font-weight:800; min-height:80px; }
  .reveal { display:flex; justify-content:space-around; align-items:center; text-align:center; gap:12px; }
  .reveal .move { font-size:40px; font-weight:800; text-transform:capitalize; }
  .reveal .who { font-size:13px; color:var(--muted); }
  .verdict { text-align:center; font-size:26px; font-weight:800; margin-top:8px; min-height:34px; }
  .verdict.win { color:var(--win); } .verdict.lose { color:var(--lose); }
  .hint { font-size:12px; color:var(--muted); margin-top:8px; }
  .mapping { font-size:13px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Duel — Pierre / Feuille / Ciseaux par gestes</h1>
  <button id="btnConnect">Connecter la carte</button>
  <span id="status">Connecte ta carte, l'adversaire fait de même. Le match démarre à deux.</span>
</header>

<section class="panel">
  <div class="lives">
    <span>Toi : <span id="livesYou" class="hearts">— — —</span></span>
    <span><span id="livesOpp" class="hearts">— — —</span> : Adversaire</span>
  </div>
</section>

<section class="panel">
  <div id="countdown" class="countdown">—</div>
  <div id="reveal" class="reveal" hidden>
    <div><div id="moveYou" class="move">?</div><div class="who">toi</div></div>
    <div style="font-size:24px;color:var(--muted)">vs</div>
    <div><div id="moveOpp" class="move">?</div><div class="who">adversaire</div></div>
  </div>
  <div id="verdict" class="verdict"></div>
  <button id="btnRematch" class="secondary" hidden>Revanche</button>
</section>

<p class="mapping">Gestes : <strong>vertical</strong> = pierre · <strong>horizontal</strong> = feuille · <strong>circle</strong> = ciseaux. Exécute ton geste <strong>pendant le décompte</strong> — le modèle lit les 2 dernières secondes de l'accéléromètre.</p>

<script src="movement_cnn_weights.js"></script>
<script src="inference.js"></script>
<script src="rps-moves.js"></script>
<script>
"use strict";
const MOVE_FR = { rock:"pierre", paper:"feuille", scissors:"ciseaux", whiff:"raté" };
const MAX_SAMPLES = 500;
const samples = [];

// ---------- Web Serial (identique au moniteur, accel + gyro) ----------
const btnConnect = document.getElementById('btnConnect');
const statusEl = document.getElementById('status');
let port = null, reader = null, keepReading = false;
function setStatus(m) { statusEl.textContent = m; }

async function connect() {
  if (!('serial' in navigator)) { setStatus('Web Serial indisponible — Chrome ou Edge.'); return; }
  try { port = await navigator.serial.requestPort(); await port.open({ baudRate: 115200 }); }
  catch (e) { if (e.name !== 'NotFoundError') setStatus('Ouverture impossible : ' + e.message); port = null; return; }
  btnConnect.disabled = true;
  setStatus('Carte connectée. En attente de l\'adversaire / du décompte.');
  keepReading = true; readLoop();
}
async function readLoop() {
  const dec = new TextDecoder(); let carry = '';
  try {
    while (keepReading && port.readable) {
      reader = port.readable.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          carry += dec.decode(value, { stream: true });
          const lines = carry.split('\n'); carry = lines.pop();
          for (const line of lines) handleLine(line);
        }
      } finally { reader.releaseLock(); reader = null; }
    }
  } catch (e) { if (keepReading) setStatus('Liaison perdue : ' + e.message); }
}
function handleLine(line) {
  const p = line.trim().split(/\s+/);
  if (p.length !== 6) return;
  const v = p.map(Number);
  if (v.some(x => !Number.isFinite(x))) return;
  samples.push({ a: [v[0], v[1], v[2]], g: [v[3], v[4], v[5]] });
  if (samples.length > MAX_SAMPLES) samples.shift();
}
btnConnect.addEventListener('click', connect);

// Mode démo : #demo injecte un geste 'vertical' synthétique (secousse sur ax).
if (location.hash === '#demo') {
  setStatus('Mode démo — geste vertical synthétique.');
  setInterval(() => {
    const t = performance.now() / 1000;
    samples.push({ a: [3 * Math.sin(t * 6), 0.2 * Math.cos(t), 9.81], g: [0, 0, 0] });
    if (samples.length > MAX_SAMPLES) samples.shift();
  }, 20);
}

// ---------- Capture du geste (à GO) ----------
const WIN = MovementModel.ready ? MovementModel.seqLen : 100;
function currentMove() {
  if (samples.length < WIN) return { move: "whiff", conf: 0 };
  const win = samples.slice(-WIN);
  const ch = [new Float64Array(WIN), new Float64Array(WIN), new Float64Array(WIN)];
  for (let i = 0; i < WIN; i++) { const a = win[i].a; ch[0][i]=a[0]; ch[1][i]=a[1]; ch[2][i]=a[2]; }
  for (let c = 0; c < 3; c++) {
    let mean = 0; for (let i=0;i<WIN;i++) mean += ch[c][i]; mean /= WIN;
    let vv = 0; for (let i=0;i<WIN;i++){ const d=ch[c][i]-mean; vv += d*d; } vv /= WIN;
    const sd = Math.sqrt(vv) || 1e-8;
    for (let i=0;i<WIN;i++) ch[c][i] = (ch[c][i]-mean)/sd;
  }
  const probs = MovementModel.softmax(MovementModel.forward(ch));
  let best = 0; for (let i=1;i<probs.length;i++) if (probs[i]>probs[best]) best=i;
  const className = MovementModel.classes[best];
  return { move: RPS.classifyMove(className, probs[best]), conf: probs[best] };
}

// ---------- UI ----------
const el = id => document.getElementById(id);
function hearts(n) { return "❤".repeat(Math.max(0, n)) + "·".repeat(Math.max(0, 3 - n)); }
function showLives(you, opp) { el('livesYou').textContent = hearts(you); el('livesOpp').textContent = hearts(opp); }
function resetRoundUI() { el('reveal').hidden = true; el('verdict').textContent = ''; el('verdict').className = 'verdict'; }

// ---------- WebSocket ----------
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
ws.addEventListener('open', () => ws.send(JSON.stringify({ type: 'join' })));
ws.addEventListener('close', () => setStatus('Déconnecté du serveur.'));
ws.addEventListener('message', ev => {
  const m = JSON.parse(ev.data);
  switch (m.type) {
    case 'joined': setStatus('Rejoint comme joueur ' + m.role + '.'); break;
    case 'waiting': setStatus('En attente de l\'adversaire…'); break;
    case 'full': setStatus('Partie pleine. Réessaie plus tard.'); break;
    case 'start':
      showLives(m.lives.you, m.lives.opp); resetRoundUI();
      el('btnRematch').hidden = true; el('countdown').textContent = '—';
      setStatus('Match ! Prépare ton geste.');
      break;
    case 'countdown':
      resetRoundUI(); el('countdown').textContent = m.n; break;
    case 'go':
      el('countdown').textContent = 'GO';
      const mv = currentMove();
      ws.send(JSON.stringify({ type: 'move', move: mv.move, conf: mv.conf }));
      break;
    case 'round':
      el('reveal').hidden = false;
      el('moveYou').textContent = MOVE_FR[m.you.move];
      el('moveOpp').textContent = MOVE_FR[m.opp.move];
      showLives(m.lives.you, m.lives.opp);
      const v = el('verdict');
      if (m.winner === 'you') { v.textContent = 'Round gagné !'; v.className = 'verdict win'; }
      else if (m.winner === 'opp') { v.textContent = 'Round perdu.'; v.className = 'verdict lose'; }
      else { v.textContent = 'Égalité.'; v.className = 'verdict'; }
      break;
    case 'matchOver':
      el('countdown').textContent = m.winner === 'you' ? '🏆' : '💀';
      el('verdict').textContent = m.winner === 'you' ? 'VICTOIRE' : 'DÉFAITE';
      el('verdict').className = 'verdict ' + (m.winner === 'you' ? 'win' : 'lose');
      el('btnRematch').hidden = false;
      break;
    case 'opponentLeft':
      setStatus('Adversaire parti. En attente d\'un nouveau.'); resetRoundUI();
      el('countdown').textContent = '—';
      break;
  }
});
el('btnRematch').addEventListener('click', () => {
  ws.send(JSON.stringify({ type: 'rematch' }));
  el('btnRematch').disabled = true;
  setStatus('Revanche demandée — en attente de l\'adversaire.');
});
</script>
</body>
</html>
```

- [ ] **Step 2: Contrôle de syntaxe JS de la page**

Extraire et vérifier le script inline avec Node (sans DOM). Comme le fichier est du HTML, on vérifie juste que le bloc JS parse. Run :
```bash
node --check <(sed -n '/<script>$/,/<\/script>/p' web/rps-battle.html | sed '1d;$d')
```
Expected: aucune erreur (exit 0). Si `node --check` sur substitution de process échoue sous l'environnement, copier le bloc `<script>…</script>` (le dernier, sans `src`) dans un fichier temporaire `.js` et lancer `node --check fichier.js`.

- [ ] **Step 3: Vérification manuelle à deux navigateurs (démo, sans carte)**

1. Lancer le serveur : `python scripts/rps_server.py`.
2. Ouvrir deux onglets sur `http://127.0.0.1:8000/#demo` (le `#demo` injecte un geste synthétique, dispensant de la carte).
3. Vérifier la séquence : les deux passent en « Match ! », un décompte 3-2-1-GO synchronisé s'affiche, puis une révélation des coups, un verdict, et les cœurs diminuent.
4. Laisser jouer jusqu'à `VICTOIRE`/`DÉFAITE`, cliquer **Revanche** dans les deux onglets → une nouvelle partie démarre (3 cœurs).
5. Fermer un onglet en cours de partie → l'autre affiche « Adversaire parti ».

Attendu : la séquence complète se déroule sans erreur console. (Les deux `#demo` envoient le même geste → beaucoup d'égalités ; c'est normal, le but ici est la mécanique, pas le hasard.)

- [ ] **Step 4: Commit**

```bash
git add web/rps-battle.html
git commit -m "feat: RPS duel page (Web Serial + WebSocket client + gesture capture)"
```

---

### Task 6: Documentation & mémoire

**Files:**
- Modify: `readme.md`
- Create: `docs/superpowers/plans/2026-07-07-rps-battle.md` (ce fichier, déjà présent)

- [ ] **Step 1: Ajouter une section au `readme.md`**

Ajouter à la fin de `readme.md` :

```markdown

## Duel Pierre-Feuille-Ciseaux (gestes, WebSocket)
- Lancer sur le PC hôte : `python scripts/rps_server.py`
- Les deux joueurs ouvrent `http://<IP-hôte>:8000/`, connectent leur carte, bougent pendant le décompte.
- Geste → coup : vertical=pierre, horizontal=feuille, circle=ciseaux. 3 vies chacun.
- Tests : `python tests/test_rps_game.py`, `python tests/test_rps_server_e2e.py`, `node tests/test_web_inference.mjs`, `node tests/test_rps_moves.mjs`.
```

- [ ] **Step 2: Commit**

```bash
git add readme.md
git commit -m "docs: RPS duel usage in readme"
```

---

## Notes de vérification (self-review)

- **Couverture du spec** : architecture serveur (T4), page duel (T5), module inférence partagé (T1), mapping geste→coup + seuil 0.6 (T2), règles de résolution + vies (T3), protocole WS complet (T4 + T5), déconnexion/full/whiff/timeout (T4 e2e + serveur), tests unitaires + e2e (T1-T4). Décompte serveur synchronisé (T4). Service HTTP statique un-port (T4).
- **Cohérence des types** : `MovementModel.forward/softmax` (T1) consommés en T5 ; `RPS.classifyMove` (T2) en T5 ; `resolve`/`Match.play_round` (T3) en T4 ; `make_app(countdown_step, collect_timeout)` (T4) en test e2e. Coups anglais sur le fil partout ; `MOVE_FR` mappe l'affichage.
- **Pas de placeholder** : chaque étape porte le code complet.
```
