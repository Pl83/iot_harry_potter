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
