# AlexZero Othello

An AlphaZero-style agent that learned Othello (Reversi) purely by self-play, playable in the browser with no server.

**Play:** https://alexbaker0.github.io/othello/

## How it runs in the browser

The agent was trained in PyTorch (residual conv net, 64 channels × 5 blocks, 65-way policy (64 squares + pass) + value heads). For the web:

- `export_web.py` (in the training project) folds BatchNorm into the convolutions and writes `weights.bin` + `weights.json`.
- `src/net.c` is the forward pass in C, compiled to WebAssembly SIMD (`net.wasm`), about 2 ms per evaluation.
- `engine.js` has the game rules, a plain-JS fallback network, and the same PUCT Monte Carlo Tree Search as the Python, plus the server's auto-pass and three modes (vs AI, two players, demo). Checked against the Python: net outputs agree to ~1e-5, MCTS visit counts match exactly, and a full 61-move self-play game (with a forced pass) is identical.
- `worker.js` runs the search in a Web Worker so the page stays responsive.

Difficulty = MCTS simulations per move: Easy 100, Medium 300, Hard 800. Trained to 14 wins, 6 losses vs a depth-3 positional alpha-beta player.

## Rebuilding the wasm

```
clang --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -Wl,--no-entry -o net.wasm src/net.c
```
