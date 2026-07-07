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
