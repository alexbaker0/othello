// AlexZero Othello engine: game rules, neural net and MCTS, ported from the
// Python (game.py / network.py / mcts.py / server.py) so the browser needs no
// server. Runs in a Web Worker (worker.js) and in Node for tests.

const ROWS = 8, COLS = 8, N = 64, PASS = 64, NM = 65;   // 64 squares + pass

// ---------------------------------------------------------------- game
// Board: Int8Array(64), index r*8+c. +1 = Black (moves first), -1 = White.

const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

function newBoard() {
  const b = new Int8Array(N);
  b[3 * 8 + 3] = -1; b[4 * 8 + 4] = -1; b[3 * 8 + 4] = 1; b[4 * 8 + 3] = 1;
  return b;
}

// Indices that flip if `player` plays at (r, c); empty if that's not legal.
function flips(b, player, r, c) {
  if (b[r * 8 + c] !== 0) return [];
  const out = [];
  for (const [dr, dc] of DIRS) {
    const line = [];
    let rr = r + dr, cc = c + dc;
    while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === -player) {
      line.push(rr * 8 + cc); rr += dr; cc += dc;
    }
    if (line.length && rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === player) out.push(...line);
  }
  return out;
}

// Legal placements for `player`, or [PASS] if there are none.
function legalMoves(b, player) {
  const out = [];
  for (let i = 0; i < N; i++) if (b[i] === 0 && flips(b, player, i >> 3, i & 7).length) out.push(i);
  return out.length ? out : [PASS];
}

function applyMove(b, player, move) {
  const nb = b.slice();
  if (move !== PASS) {
    const f = flips(nb, player, move >> 3, move & 7);
    if (!f.length) throw new Error(`illegal move ${move} for player ${player}`);
    nb[move] = player;
    for (const i of f) nb[i] = player;
  }
  return nb;
}

function hasMove(b, player) {
  for (let i = 0; i < N; i++) if (b[i] === 0 && flips(b, player, i >> 3, i & 7).length) return true;
  return false;
}

// Outcome from Black's view (1 / -1 / 0), or null while either side can move.
function result(b) {
  if (hasMove(b, 1) || hasMove(b, -1)) return null;
  let d = 0;
  for (let i = 0; i < N; i++) d += b[i];
  return d > 0 ? 1 : d < 0 ? -1 : 0;
}

// ---------------------------------------------------------------- network
// Activations are kept in zero-padded (ROWS+2)x(COLS+2) planes so 3x3 convs
// need no bounds checks.

const PH = ROWS + 2, PW = COLS + 2, P = PH * PW;

class Net {
  constructor(manifest, buf) {
    const all = new Float32Array(buf);
    this.t = {};
    for (const m of manifest.tensors) {
      const size = m.shape.reduce((a, b) => a * b, 1);
      this.t[m.name] = all.subarray(m.offset, m.offset + size);
    }
    this.C = manifest.channels;
    this.blocks = manifest.blocks;
    this.a = new Float32Array(this.C * P);    // current activation
    this.h = new Float32Array(this.C * P);    // block intermediate
    this.tmp = new Float32Array(this.C * P);
    this.col = new Float32Array(N * this.C * 9);   // im2col buffer
  }

  // out = relu(conv3x3(inp) + b [+ res]); inp/out/res are padded planes.
  // im2col then a matrix multiply: each output is a dot product of the
  // channel's weights (torch layout ic,kh,kw = K values) with the K inputs
  // under that pixel. 4 channels x 2 pixels are done together so the eight
  // running sums stay in registers. Needs cout % 4 == 0 (true here).
  conv3(inp, cin, w, b, cout, out, res) {
    const K = cin * 9, col = this.col;
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      let j = (r * COLS + c) * K;
      for (let ic = 0; ic < cin; ic++) {
        const s = ic * P + r * PW + c;
        col[j++] = inp[s];          col[j++] = inp[s + 1];          col[j++] = inp[s + 2];
        col[j++] = inp[s + PW];     col[j++] = inp[s + PW + 1];     col[j++] = inp[s + PW + 2];
        col[j++] = inp[s + 2 * PW]; col[j++] = inp[s + 2 * PW + 1]; col[j++] = inp[s + 2 * PW + 2];
      }
    }
    for (let oc = 0; oc < cout; oc += 4) {
      const w0 = oc * K, w1 = w0 + K, w2 = w1 + K, w3 = w2 + K;
      for (let px = 0; px < N; px += 2) {
        const x0 = px * K, x1 = x0 + K;
        let a00 = 0, a01 = 0, a10 = 0, a11 = 0, a20 = 0, a21 = 0, a30 = 0, a31 = 0;
        for (let j = 0; j < K; j++) {
          const u = col[x0 + j], v = col[x1 + j];
          const p = w[w0 + j], q = w[w1 + j], s = w[w2 + j], t = w[w3 + j];
          a00 += p * u; a01 += p * v; a10 += q * u; a11 += q * v;
          a20 += s * u; a21 += s * v; a30 += t * u; a31 += t * v;
        }
        const sums = [a00, a01, a10, a11, a20, a21, a30, a31];
        for (let q = 0; q < 4; q++) for (let d = 0; d < 2; d++) {
          const pp = px + d, o = (oc + q) * P + ((pp / COLS) | 0) * PW + PW + (pp % COLS) + 1;
          let val = sums[q * 2 + d] + b[oc + q];
          if (res) val += res[o];
          this.tmp[o] = val > 0 ? val : 0;
        }
      }
    }
    // Write via tmp so out may alias res (the residual add reads the old values).
    for (let oc = 0; oc < cout; oc++) {
      for (let r = 1; r <= ROWS; r++) {
        const o = oc * P + r * PW + 1;
        for (let c = 0; c < COLS; c++) out[o + c] = this.tmp[o + c];
      }
    }
  }

  // Returns {logits: Float32Array(65), value} for the position seen by `player`.
  forward(board, player) {
    const t = this.t, C = this.C;
    const x = new Float32Array(2 * P);
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const v = board[r * COLS + c] * player;
      if (v === 1) x[(r + 1) * PW + c + 1] = 1;
      else if (v === -1) x[P + (r + 1) * PW + c + 1] = 1;
    }
    this.conv3(x, 2, t["stem.w"], t["stem.b"], C, this.a, null);
    for (let i = 0; i < this.blocks; i++) {
      this.conv3(this.a, C, t[`res${i}.w1`], t[`res${i}.b1`], C, this.h, null);
      this.conv3(this.h, C, t[`res${i}.w2`], t[`res${i}.b2`], C, this.a, this.a);
    }
    // Heads: 1x1 conv (+relu) -> flatten (channel-major, like torch) -> fc.
    const head = (w, b, nc) => {
      const f = new Float32Array(nc * N);
      for (let oc = 0; oc < nc; oc++) for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
        const pi = (r + 1) * PW + c + 1;
        let v = b[oc];
        for (let ic = 0; ic < C; ic++) v += w[oc * C + ic] * this.a[ic * P + pi];
        f[oc * N + r * COLS + c] = v > 0 ? v : 0;
      }
      return f;
    };
    const fc = (w, b, inp, nout, relu) => {
      const nin = inp.length, out = new Float32Array(nout);
      for (let o = 0; o < nout; o++) {
        let v = b[o];
        for (let i = 0; i < nin; i++) v += w[o * nin + i] * inp[i];
        out[o] = relu && v < 0 ? 0 : v;
      }
      return out;
    };
    const logits = fc(t["p_fc.w"], t["p_fc.b"], head(t["p_conv.w"], t["p_conv.b"], 2), NM, false);
    const h1 = fc(t["v_fc1.w"], t["v_fc1.b"], head(t["v_conv.w"], t["v_conv.b"], 1), 64, true);
    const value = Math.tanh(fc(t["v_fc2.w"], t["v_fc2.b"], h1, 1, false)[0]);
    return { logits, value };
  }

  // Softmax over legal moves -> {probs (length 65), value}.
  predict(board, player, legal) {
    const { logits, value } = this.forward(board, player);
    let mx = -Infinity;
    for (const m of legal) mx = Math.max(mx, logits[m]);
    const probs = new Float64Array(NM);
    let s = 0;
    for (const m of legal) { probs[m] = Math.exp(logits[m] - mx); s += probs[m]; }
    for (const m of legal) probs[m] /= s;
    return { probs, value };
  }
}

// Same network compiled from src/net.c to WebAssembly SIMD (much faster).
// Net above stays as the fallback for browsers without wasm SIMD.
class WasmNet {
  static async create(wasmBytes, weightsBuf) {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return new WasmNet(instance.exports, weightsBuf);
  }
  constructor(ex, weightsBuf) {
    this.ex = ex;
    const mem = () => ex.memory.buffer;
    new Float32Array(mem(), ex.weights(), weightsBuf.byteLength / 4).set(new Float32Array(weightsBuf));
    ex.init();
    this.inp = new Float32Array(mem(), ex.input(), 2 * N);
    this.out = new Float32Array(mem(), ex.output(), NM + 1);
  }
  forward(board, player) {
    const x = this.inp;
    for (let i = 0; i < N; i++) {
      const v = board[i] * player;
      x[i] = v === 1 ? 1 : 0;
      x[N + i] = v === -1 ? 1 : 0;
    }
    this.ex.forward();
    return { logits: this.out.slice(0, NM), value: Math.tanh(this.out[NM]) };
  }
}
WasmNet.prototype.predict = Net.prototype.predict;

// ---------------------------------------------------------------- MCTS (PUCT)
// Same search as mcts.py, without root noise (the server played with
// add_noise=False). Child Q is from the opponent's view, so it is negated.
// Pass is an ordinary action that hands the turn over.

class Node {
  constructor(board, player, prior) {
    this.board = board; this.player = player; this.prior = prior;
    this.N = 0; this.W = 0; this.children = null; this.terminal = undefined;
  }
  q() { return this.N > 0 ? this.W / this.N : 0; }
}

function runMCTS(net, board, player, nSims, cPuct = 1.5) {
  const expand = node => {
    const res = result(node.board);
    if (res !== null) { node.terminal = res; return res * node.player; }
    const legal = legalMoves(node.board, node.player);
    const { probs, value } = net.predict(node.board, node.player, legal);
    node.children = legal.map(m => [m, new Node(applyMove(node.board, node.player, m), -node.player, probs[m])]);
    return value;
  };

  const root = new Node(board, player, 1);
  expand(root);
  for (let s = 0; s < nSims; s++) {
    const path = [root];
    let node = root;
    while (node.children) {
      let total = 0;
      for (const [, ch] of node.children) total += ch.N;
      const sq = total > 0 ? Math.sqrt(total) : 1;
      let best = -1e18, bestCh = null;
      for (const [, ch] of node.children) {
        const score = -ch.q() + cPuct * ch.prior * sq / (1 + ch.N);
        if (score > best) { best = score; bestCh = ch; }
      }
      node = bestCh;
      path.push(node);
    }
    if (node.terminal === undefined) node.terminal = result(node.board);
    const value = node.terminal !== null ? node.terminal * node.player : expand(node);
    for (let i = path.length - 1; i >= 0; i--) {
      const n = path[i];
      n.N += 1;
      n.W += n.player === node.player ? value : -value;
    }
  }
  const counts = new Array(NM).fill(0);
  for (const [m, ch] of root.children || []) counts[m] = ch.N;
  return counts;
}

// ---------------------------------------------------------------- API
// Same contract as server.py's compute(). mode: "ai" (AI plays the side
// opposite `humanChar`), "two" (two humans; only forced passes are played),
// "demo" (AI plays ONE move for whoever is to move).

function compute(net, moves, humanChar, sims, mode) {
  const human = humanChar === "B" ? 1 : -1;
  let board = newBoard(), player = 1;
  for (const m of moves) { board = applyMove(board, player, m); player = -player; }
  moves = [...moves];
  const notes = [];
  let aiEval = null, lastMove = null;

  const autoPass = () => {
    const lm = legalMoves(board, player);
    if (lm.length === 1 && lm[0] === PASS) {
      notes.push(`${player === 1 ? "Black" : "White"} had no move — passed.`);
      board = applyMove(board, player, PASS); player = -player; moves.push(PASS);
      return true;
    }
    return false;
  };
  const aiMove = () => {
    aiEval = net.predict(board, player, legalMoves(board, player)).value;
    const counts = runMCTS(net, board, player, sims);
    lastMove = counts.indexOf(Math.max(...counts));
    board = applyMove(board, player, lastMove); player = -player; moves.push(lastMove);
  };

  if (mode === "two") {
    while (result(board) === null && autoPass()) {}
  } else if (mode === "demo") {
    if (result(board) === null && !autoPass()) aiMove();
  } else {
    while (result(board) === null) {
      if (autoPass()) continue;
      if (player === human) break;
      aiMove();
    }
  }

  const res = result(board);
  const sym = v => (v === 1 ? "B" : v === -1 ? "W" : "");
  const grid = [];
  let nb = 0, nw = 0;
  for (let r = 0; r < 8; r++) {
    grid.push([]);
    for (let c = 0; c < 8; c++) {
      const v = board[r * 8 + c];
      grid[r].push(sym(v));
      if (v === 1) nb++; else if (v === -1) nw++;
    }
  }
  return {
    board: grid,
    moves,
    legal: res === null ? legalMoves(board, player).filter(m => m !== PASS) : [],
    last_move: lastMove,
    eval: aiEval === null ? null : Math.round(aiEval * 100) / 100,
    score: { B: nb, W: nw },
    status: res === null ? null : res === 0 ? "draw" : sym(res),
    turn: player === 1 ? "B" : "W",
    notes,
  };
}

if (typeof module !== "undefined") module.exports = { Net, WasmNet, compute, runMCTS, applyMove, legalMoves, result, newBoard, PASS };
