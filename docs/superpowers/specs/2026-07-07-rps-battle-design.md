# Duel Pierre-Feuille-Ciseaux par gestes — Design

**Date :** 2026-07-07
**Statut :** approuvé, prêt pour le plan d'implémentation

## Objectif

Deux PC sur le même réseau WiFi, chacun relié à sa carte ESP32-S3 + MPU6050,
s'affrontent à pierre-feuille-ciseaux. Le coup de chaque joueur est **un geste**
reconnu en direct par le `MovementCNN` déjà entraîné. Chaque joueur a **3 vies** ;
le premier à 0 perd le match.

Réutilise l'acquis du projet : reconnaissance de gestes live (forward-pass CNN en
JS pur, poids dans `web/movement_cnn_weights.js`), Web Serial, moniteur web.

## Architecture

Serveur Python **autoritaire** (l'arbitre) sur l'une des deux machines. Il sert
aussi la page statique en HTTP, donc le 2ᵉ joueur ouvre simplement
`http://<IP-hôte>:8000/` — rien à copier. Les deux navigateurs se connectent au
même endpoint WebSocket ; le serveur détient toute la vérité de jeu (vies,
timing du décompte, résolution).

```
  PC hôte                                   PC invité
  ┌─────────────────────────┐               ┌──────────────────────┐
  │ rps_server.py (aiohttp)  │  ws:// (LAN)  │ navigateur           │
  │  - sert la page HTTP     │◄─────────────►│  rps-battle.html     │
  │  - WebSocket arbitre     │               │  + carte (Web Serial)│
  │  - machine à états       │               └──────────────────────┘
  └─────────────────────────┘
         ▲  ws:// (localhost)
         │
  ┌──────────────────────┐
  │ navigateur (hôte)    │
  │  rps-battle.html     │
  │  + carte (Web Serial)│
  └──────────────────────┘
```

### Composants

| Fichier | Rôle | Dépend de |
|---|---|---|
| `scripts/rps_server.py` | Serveur aiohttp : HTTP statique + WebSocket arbitre + machine à états | `aiohttp` (3.11 présent) |
| `web/rps-battle.html` | Interface du duel : Web Serial, vies, décompte, révélation, résultat, revanche | `inference.js`, `movement_cnn_weights.js` |
| `web/inference.js` | Forward-pass CNN (conv/bn/relu/pool/linear) + softmax, extrait du moniteur | `window.MODEL_WEIGHTS` |
| `web/mpu6050-monitor.html` | Remaniement léger : charge `inference.js` au lieu de sa copie interne | `inference.js` |

`web/inference.js` expose une petite API : `MovementModel.forward(x)` (x =
tableau de 3 `Float64Array(100)`) → logits, et `MovementModel.softmax(z)`.
Le moniteur et le duel l'utilisent tous deux ; aucune logique dupliquée.

## Règles du jeu

### Mapping geste → coup (constante configurable côté client)

| Geste CNN | Coup |
|---|---|
| `vertical` | pierre |
| `horizontal` | feuille |
| `circle` | ciseaux |
| `static` ou confiance < seuil | **raté** (whiff) |

Seuil de confiance : `0.6` (softmax de la classe prédite). En-dessous, ou si la
carte n'est pas connectée, le coup est un **raté**.

### Résolution d'un round

- **Même coup valide** → égalité : aucune vie perdue, on rejoue le round.
- **Coups valides différents** → règle classique (pierre>ciseaux, ciseaux>feuille,
  feuille>pierre) : le perdant perd **1 vie**.
- **Un raté vs un coup valide** → le rateur perd le round : **−1 vie**.
- **Deux ratés** → aucune vie perdue, on rejoue.
- **Vie à 0** → fin de match ; l'adversaire gagne. Bouton *revanche* remet 3 vies
  aux deux et relance une partie.

## Machine à états serveur

```
ATTENTE ──(2 joueurs)──► START(vies=3,3) ──► ┌─► DÉCOMPTE(3,2,1,GO)
                                             │        │
                                        (vie>0)       ▼
                                             │   COLLECTE(moves, timeout 2.5s)
                                             │        │
                                             │        ▼
                                             └── RÉSOLUTION(maj vies, diffuse)
                                                      │
                                                 (une vie==0)
                                                      ▼
                                                    FIN(winner) ──(revanche des 2)──► START
```

- **DÉCOMPTE** : le serveur diffuse `countdown{n:3}`, `countdown{n:2}`,
  `countdown{n:1}` (~1 s chacun) puis `go`. Le timing est piloté par le serveur →
  les deux clients sont synchronisés.
- **COLLECTE** : à réception de `go`, chaque client lit sa prédiction CNN courante
  (fenêtre glissante des 100 derniers échantillons accéléromètre), la mappe et
  envoie `move{move,conf}`. Le serveur attend les deux (timeout 2,5 s → raté).
- **RÉSOLUTION** : le serveur applique les règles, met à jour les vies, diffuse
  `round{...}`. Si une vie atteint 0 → `matchOver`, sinon nouveau round après une
  courte pause.

Rôles : 1ᵉʳ connecté = joueur A, 2ᵉ = B. 3ᵉ connexion → `full` puis fermeture.

## Protocole WebSocket (JSON)

### Serveur → client

| Message | Charge | Sens |
|---|---|---|
| `joined` | `{role:"A"\|"B"}` | connexion acceptée |
| `waiting` | — | en attente de l'adversaire |
| `full` | — | partie pleine, connexion refusée |
| `start` | `{lives:{you:3,opp:3}}` | début de match |
| `countdown` | `{n:3\|2\|1}` | tic du décompte |
| `go` | — | capturer et envoyer le coup maintenant |
| `round` | `{you:{move,conf}, opp:{move,conf}, winner:"you"\|"opp"\|"tie", lives:{you,opp}}` | résultat du round |
| `matchOver` | `{winner:"you"\|"opp"}` | fin de match |
| `opponentLeft` | — | l'adversaire s'est déconnecté |

### Client → serveur

| Message | Charge | Sens |
|---|---|---|
| `join` | `{name?}` | rejoindre la partie |
| `move` | `{move:"rock"\|"paper"\|"scissors"\|"whiff", conf:number}` | coup capturé à GO |
| `rematch` | — | demande de revanche |

Les identités `you`/`opp` sont calculées par le serveur du point de vue de chaque
destinataire : un même round produit deux messages `round` personnalisés.

## Gestion des erreurs

- **Déconnexion en cours de match** → l'autre reçoit `opponentLeft` et revient en
  `ATTENTE` ; la partie en cours est abandonnée.
- **Carte non connectée / pas de prédiction à GO** → le client envoie
  `move{move:"whiff"}`.
- **Client muet après GO** (timeout 2,5 s) → le serveur compte un raté.
- **3ᵉ visiteur** → `full`, puis fermeture propre de la socket.
- **Coup invalide reçu** (valeur inconnue) → traité comme raté.

## Tests

1. **Résolution (pur Python, sans réseau)** : fonction `resolve(move_a, move_b)`
   testée sur les 9 combinaisons valides + les cas de ratés + la décrémentation
   des vies. Table de vérité complète.
2. **Bout-en-bout (sans navigateur ni carte)** : script de test qui ouvre deux
   clients WebSocket, joue un match complet scripté (coups injectés), et vérifie
   la séquence de messages et le vainqueur final.
3. **Mapping geste → coup** : réutilise le moteur d'inférence déjà prouvé
   (`web/inference.js`, parité JS/PyTorch déjà vérifiée dans node).

## Détails de déploiement

- Lancer sur le PC hôte : `python scripts/rps_server.py` (écoute `0.0.0.0:8000`).
- L'hôte affiche son IP locale au démarrage (ex. `http://192.168.1.42:8000/`).
- Les deux joueurs ouvrent cette URL, connectent leur carte (bouton *Connecter*),
  puis le match démarre dès que les deux sont là.
- La page indique clairement « Prépare ton geste pendant le décompte » : à GO, la
  fenêtre de 2 s doit déjà contenir le mouvement.

## Hors périmètre (YAGNI)

- Pas de comptes, de classement persistant, ni de base de données.
- Pas de plus de 2 joueurs simultanés (une seule partie à la fois).
- Pas de matchmaking : une seule partie par serveur.
- Pas de reconnexion à une partie en cours (déconnexion = abandon).
