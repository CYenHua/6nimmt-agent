# Opponent-Modeling Agents for *6 nimmt!*

AI agents for the 4-player imperfect-information card game *6 nimmt!*, built as the final
project of **Foundations of Artificial Intelligence (FAI 2026)**. Every agent runs
single-threaded in pure Python under a **1-second-per-move** time limit.

Over the project I built and benchmarked **30+ agents** spanning heuristic search,
Monte Carlo methods, game-theoretic solvers, and learned policies (REINFORCE, AlphaZero-style
self-play). The main finding: **modeling opponents beats searching deeper.** My best agent
infers each opponent's play style online with Bayesian updating and plans against those beliefs.
It finished ahead of every course-provided baseline, while an ISMCTS agent with a much larger
search budget only placed mid-pack.

📄 **Full write-up:** [`docs/report.pdf`](docs/report.pdf)

---

## The game in one paragraph

There are 104 cards, each worth 1–7 "bull-head" penalty points, and 4 rows on the board. Each
turn, all players pick a card at the same time. The cards are then placed in ascending order onto
the row whose last card is the closest one below them. Placing the 6th card in a row, or playing
a card lower than every row's last card, makes you take that row's penalty points. After 10
turns, a lower total is better. The final score is your **rank** at the table. The game has
**hidden information** (you can't see opponents' hands) and **simultaneous moves**, and it
**isn't zero-sum**: the outcome depends heavily on who you're playing against.

## Final agents

### Agent 2: Bayesian opponent inference + determinized search (`BP13`), the strongest
[`src/players/agents/agent2_belief_pimc.py`](src/players/agents/agent2_belief_pimc.py)

- **Online belief tracking.** Keeps a library of 6 heuristic play-style archetypes (bust-avoiding,
  blocking, safe-low, tight-fit, look-ahead-high, …). After every trick, each opponent's observed
  card updates a per-seat log-likelihood over those archetypes. The likelihood is estimated by
  sampling hands consistent with what has been seen.
- **Belief-weighted Monte Carlo rollouts.** In the early and middle game it uses Perfect
  Information Monte Carlo (PIMC): it samples the unseen cards into opponent hands, samples each
  opponent's policy from its posterior, and rolls the game out to the end.
- **Exact endgame search.** With ≤ 4 cards left, it switches to an exhaustive recursive search
  over sampled determinizations. Its objective is expected final rank, with expected penalty
  points as a small tie-breaker (λ = 0.003).

### Agent 1: Regret matching over a policy-level matrix game (`BP18`)
[`src/players/agents/agent1_regret_matching.py`](src/players/agents/agent1_regret_matching.py)

- **Heuristic policy pool.** Five archetypes (bust-avoiding, look-ahead-high, tight-fit, high,
  low) cover most tactical styles. The prior over them, `[0.30, 0.30, 0.20, 0.12, 0.08]`, was
  set by profiling how the course baselines play.
- **Matrix game per trick.** Rows are my candidate cards and columns are the policy the field
  plays. Each payoff is my expected final rank, estimated with PIMC rollouts to the end of the
  game.
- **Regret matching** (200 iterations) solves the game. `_ADV = 0.5` blends the fixed prior with
  the adversarial, regret-matched field strategy. 0.5 was the best value in controlled tests,
  sitting between a pure best response (0) and full maximin (1).
- Its strength is close to Agent 2's but comes from a **different mechanism**. That makes it a
  decorrelated second entry, which is why it was submitted alongside Agent 2 instead of a
  near-copy of Agent 2.

### Engineering for a 1 s budget in pure Python
- **Flat state.** The board is stored as three 1-D arrays (`row_ends`, `row_lens`, `row_pens`)
  instead of nested lists, which maximizes rollout throughput.
- **Incremental belief updates.** A pointer into the move history makes each turn's update
  O(1) amortized instead of re-parsing the whole history (O(T²) overall).
- **Action-space pruning.** Only heuristic-recommended candidate cards enter the regret-matching
  matrix, giving roughly a 10× speed-up.
- **Asymmetric anytime budgets.** Both agents run determinizations until a wall-clock budget runs
  out, so they never time out. Agent 1 uses 0.70 s because its rollout times vary a lot. Agent 2
  uses 0.90 s because its endgame search time is predictable.

## Results

A 16-seat local round-robin with **2,400 games per seat**, against course-provided baselines
B7–B10 (B10 is the strongest) and random padding. Lower average rank is better (1 = best,
4 = worst).

| Agent | Approach | Avg rank |
|---|---|---|
| **BP13** (final Agent 2) | Determinized PIMC + Bayesian opponent inference | **2.21** |
| BP16 | BP13 + belief-floor robustness hedge | 2.23 |
| **BP18** (final Agent 1) | Regret matching / DUCT-style matrix game | 2.24 |
| BP26 | Opponents modeled as planners | 2.25 |
| BP8 | PIMC with a fixed heuristic field (no opponent model) | 2.30 |
| BP29 | Information-Set MCTS + exact endgame | 2.30 |
| BP31 | AlphaZero-style self-play policy (no test-time search) | 2.33 |
| BP17 | Diverse rollout / look-ahead | 2.36 |
| BP30 | AlphaZero-style distillation of the search agent | 2.36 |
| BP19 | REINFORCE MLP policy (no search) | 2.75 |

**Takeaways**
1. **Opponent inference is the key lever.** BP13 and BP16 are the only agents clearly above the
   strongest baseline.
2. **More search doesn't help by itself.** ISMCTS spends far more compute than a cheap rollout
   and still only places mid-pack.
3. **Learned policies hit their teacher's ceiling.** Distillation (BP30) levels off at the
   teacher's strength. REINFORCE (BP19) only matches a baseline. AlphaZero self-play (BP31)
   matched the search agents' level **with no test-time search**, but didn't surpass them.
4. **Local benchmarks can mislead.** ISMCTS looked competitive locally but fell behind on a
   shared class leaderboard. After that, I based decisions on the leaderboard results.

## Repository layout

```
├── src/
│   ├── engine.py, game_utils.py, tournament_runner.py   # game engine & tournament harness (course-provided)
│   └── players/
│       ├── agents/
│       │   ├── agent1_regret_matching.py     # final Agent 1 (submitted version, BP18 family)
│       │   ├── agent2_belief_pimc.py         # final Agent 2 (submitted version, BP13 family)
│       │   └── experiments/                  # one representative agent per method family
│       │       ├── best_player8.py           #   PIMC, no opponent model
│       │       ├── best_player13.py / 16.py / 26.py   #   PIMC + inference variants
│       │       ├── best_player17.py          #   rollout / look-ahead
│       │       ├── best_player19.py (+ weights)       #   REINFORCE
│       │       ├── best_player29.py          #   ISMCTS
│       │       ├── best_player30.py (+ weights)       #   AlphaZero distillation
│       │       └── best_player31.py          #   AlphaZero self-play (weights embedded)
│       └── baselines/                        # random & human (interactive) players
├── configs/
│   ├── game/          # single-game configs (demo, play against the agents)
│   └── tournament/    # tournament configs
├── docs/report.pdf
├── run_single_game.py
└── run_tournament.py
```

## Running it

Requires Python 3.13.

```bash
pip install -r requirements.txt

# Watch the two final agents play one verbose game against a heuristic agent (BP8) and a random player
python run_single_game.py --config configs/game/demo.json

# Play against the agents yourself in the terminal
python run_single_game.py --config configs/game/play_vs_agents.json

# Small tournament with the final agents (a few minutes)
python run_tournament.py --config configs/tournament/final_agents.json

# Head-to-head of every method family (much longer; each agent uses up to ~1 s per move)
python run_tournament.py --config configs/tournament/all_methods.json
```

Results are written to `results/`. The tournament runner seats players at random tables and
repeats each deal across rotated seats to reduce card-luck variance. See the configs for
options (`num_games_per_player`, `num_workers`, `duplication_mode`, …).

> **Note:** The course's compiled baselines (B1–B10) were distributed as Linux-only binaries and
> aren't included in this repo. Because of that, the exact numbers in the results table can't be
> reproduced here. The configs above pit the agents against each other, the greedy heuristic,
> and random players instead.

## Credits

The game engine, tournament harness (`src/engine.py`, `src/game_utils.py`,
`src/tournament_runner.py`, `run_*.py`), and the random and human players were provided by the
FAI 2026 course staff. All agents in `src/players/agents/` and the report are my own work.
