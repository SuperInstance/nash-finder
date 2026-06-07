# nash-finder

**Nash equilibrium computation for agent strategic interactions.**

When agents share a world, they don't act in isolation. Every decision an agent makes
depends on what every other agent decides — and they all know it. This mutual
interdependence is the domain of **game theory**, and its central solution concept is
the **Nash equilibrium**: a state where no agent can improve its outcome by
unilaterally changing its strategy.

`nash-finder` is a Rust library for computing Nash equilibria in normal-form games.
It provides exact solutions via support enumeration, approximate solutions via
fictitious play and best-response dynamics, and structural tools like iterated
dominance elimination. The only external dependency is `serde`.

## Why This Crate Exists

Multi-agent systems need more than coordination protocols. They need **strategic
reasoning** — the ability to model other agents as rational actors, predict their
behavior, and choose actions that are robust against adversarial play.

Whether you're building:

- **Trading agents** that compete in markets
- **Autonomous vehicles** negotiating intersections
- **RL environments** with multiple learning agents
- **Mechanism design** tools for auction or voting systems
- **Security games** for resource allocation

You need to answer the question: *given this strategic interaction, what will
rational agents actually do?* Nash equilibrium is the gold-standard answer.

## The Metaphor: Agents as Game Theorists

Think of each agent as sitting at a table, cards in hand, studying its opponents.
It doesn't just ask "what's my best move?" — it asks "what's my best move *given
that everyone else is asking the same question about me?*"

This recursive reasoning is what makes game theory hard. Nash's theorem (1950)
guarantees that at least one equilibrium always exists in finite games, but finding
it is computationally challenging — it's PPAD-complete in general.

`nash-finder` tackles this by combining multiple solution techniques:

```
         ┌──────────────┐
         │  NormalForm   │  Define the game
         │    Game       │
         └──────┬───────┘
                │
    ┌───────────┼───────────────┐
    ▼           ▼               ▼
┌────────┐ ┌──────────┐ ┌──────────────┐
│Strategy│ │  Best    │ │   Support    │
│        │ │ Response │ │ Enumeration  │
└────┬───┘ └────┬─────┘ └──────┬───────┘
     │          │               │
     └──────────┼───────────────┘
                ▼
        ┌──────────────┐
        │    Nash      │  Verified equilibria
        │  Equilibrium │  (exact & ε-approximate)
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │    Nash      │  Unified solver
        │    Solver    │  (dominance + support + BR dynamics)
        └──────────────┘
```

## Quick Start

```toml
[dependencies]
nash-finder = "0.1"
```

```rust
use nash_finder::game::NormalFormGame;
use nash_finder::nash::NashSolver;

fn main() {
    // Define Prisoner's Dilemma
    let game = NormalFormGame::prisoners_dilemma();

    // Solve for all Nash equilibria
    let solver = NashSolver::new();
    let result = solver.solve(&game);

    println!("Found {} equilibria", result.count());
    for ne in &result.equilibria {
        println!("{}", ne);
    }
}
```

Output:
```
Found 1 equilibria
Nash Equilibrium (ε = 0.00e+00):
  Player 0: Mix([0.0000, 1.0000]) (payoff: -2.0000)
  Player 1: Mix([0.0000, 1.0000]) (payoff: -2.0000)
  [Exact NE]
```

## Module Reference

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `game` | Define normal-form games | `NormalFormGame` |
| `strategy` | Pure and mixed strategy representations | `Strategy`, `PureStrategy`, `MixedStrategy` |
| `best_response` | Best-response computation | `BestResponse` |
| `support` | Support enumeration (exact NE for 2-player) | `SupportEnumeration` |
| `equilibrium` | Verified equilibrium results | `NashEquilibrium`, `EquilibriumBuilder` |
| `nash` | Unified solver combining all techniques | `NashSolver`, `SolverConfig`, `SolverResult` |

## Defining Games

### 2-Player Bimatrix Games

The most common form. Each cell contains `(row_player_payoff, col_player_payoff)`:

```rust
use nash_finder::game::NormalFormGame;

// Battle of the Sexes
//         Opera    Football
// Opera   (3,2)    (0,0)
// Football (0,0)   (2,3)
let bos = NormalFormGame::bimatrix(vec![
    vec![(3.0, 2.0), (0.0, 0.0)],
    vec![(0.0, 0.0), (2.0, 3.0)],
]);
```

### Preset Games

```rust
use nash_finder::game::NormalFormGame;

let pd = NormalFormGame::prisoners_dilemma();     // (Defect, Defect)
let bos = NormalFormGame::battle_of_the_sexes();   // 3 equilibria
let mp = NormalFormGame::matching_pennies();        // 1 mixed NE
let rps = NormalFormGame::rock_paper_scissors();    // Uniform mixed NE
let coord = NormalFormGame::coordination();          // 2 pure + 1 mixed
let sh = NormalFormGame::stag_hunt();                // Risk-dominance
let chicken = NormalFormGame::chicken();              // Brinksmanship
```

### N-Player Games

```rust
use nash_finder::game::NormalFormGame;

let game = NormalFormGame::n_player(
    vec![2, 2, 2],  // 3 players, 2 strategies each
    vec![
        // Player 0 payoffs (8 profiles: 2×2×2)
        vec![1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0],
        // Player 1 payoffs
        vec![1.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        // Player 2 payoffs
        vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0],
    ],
);
assert_eq!(game.num_players(), 3);
```

## Strategies

```rust
use nash_finder::strategy::{Strategy, PureStrategy, MixedStrategy};

// Pure strategy: always play action 0
let pure = PureStrategy::new(0);
let as_mixed = pure.to_mixed(3);  // [1.0, 0.0, 0.0]

// Mixed strategy: randomize
let mixed = MixedStrategy::new(vec![0.3, 0.7]);
println!("Support: {:?}", mixed.support());    // [0, 1]
println!("Entropy: {:.4}", mixed.entropy());   // 0.8813 bits

// Uniform random over 3 actions
let uniform = MixedStrategy::uniform(3);  // [1/3, 1/3, 1/3]

// Strategy enum
let s = Strategy::mixed(vec![0.25, 0.75]);
println!("{}", s);  // Mix([0.2500, 0.7500])
```

## Best Response

Find the optimal counter-strategy:

```rust
use nash_finder::game::NormalFormGame;
use nash_finder::best_response::BestResponse;
use nash_finder::strategy::MixedStrategy;

let pd = NormalFormGame::prisoners_dilemma();
let opponent_cooperates = MixedStrategy::new(vec![1.0, 0.0]);

let br = BestResponse::find_pure(&pd, 0, &opponent_cooperates);
println!("{}", br);  // BR: action 1 (utility: 0.0000)
assert_eq!(br.actions, vec![1]);  // Defect is best response to Cooperate
```

### Fictitious Play

An iterative learning process that converges to Nash equilibrium in many games:

```rust
use nash_finder::best_response::BestResponse;
use nash_finder::game::NormalFormGame;

let rps = NormalFormGame::rock_paper_scissors();
let (player0_strategy, player1_strategy) = BestResponse::fictitious_play(&rps, 10000);

// Converges to approximately uniform: (1/3, 1/3, 1/3)
println!("Player 0: {}", player0_strategy);
println!("Player 1: {}", player1_strategy);
```

## Support Enumeration

Exact Nash equilibrium computation for 2-player games:

```rust
use nash_finder::game::NormalFormGame;
use nash_finder::support::SupportEnumeration;

let coord = NormalFormGame::coordination();
let equilibria = SupportEnumeration::find_all_equilibria(&coord);

println!("Found {} equilibria:", equilibria.len());
for (i, (s0, s1)) in equilibria.iter().enumerate() {
    println!("  NE {}: {} × {}", i, s0, s1);
}
// Found 3 equilibria:
//   NE 0: Mix([1.0000, 0.0000]) × Mix([1.0000, 0.0000])
//   NE 1: Mix([0.0000, 1.0000]) × Mix([0.0000, 1.0000])
//   NE 2: Mix([0.5000, 0.5000]) × Mix([0.5000, 0.5000])
```

## Nash Equilibrium Verification

```rust
use nash_finder::game::NormalFormGame;
use nash_finder::equilibrium::{NashEquilibrium, EquilibriumBuilder};
use nash_finder::strategy::MixedStrategy;

let rps = NormalFormGame::rock_paper_scissors();
let uniform = MixedStrategy::uniform(3);

// Build and verify
let ne = EquilibriumBuilder::new(&rps)
    .with_strategy(uniform.clone())
    .with_strategy(uniform)
    .build();

match ne {
    Some(eq) => {
        println!("Verified: {}", eq);
        println!("Symmetric: {}", eq.is_symmetric());
        println!("Social welfare: {:.4}", eq.social_welfare());
    }
    None => println!("Not a Nash equilibrium!"),
}
```

## Unified Solver

The `NashSolver` combines all techniques into a single interface:

```rust
use nash_finder::game::NormalFormGame;
use nash_finder::nash::{NashSolver, SolverConfig};

let bos = NormalFormGame::battle_of_the_sexes();

// Default solver
let solver = NashSolver::new();
let result = solver.solve(&bos);

println!("Equilibria: {}", result.count());
println!("Pure: {}", result.pure_equilibria().len());
println!("Mixed: {}", result.mixed_equilibria().len());
println!("Best welfare: {:.4}", result.best_welfare().unwrap().social_welfare());

// Custom configuration
let config = SolverConfig {
    max_iterations: 500,
    epsilon: 1e-6,
    use_dominance: true,
    use_support_enum: true,
    use_fictitious_play: false,
    fp_iterations: 5000,
};
let custom_solver = NashSolver::with_config(config);
let result2 = custom_solver.solve(&bos);
```

## Mathematics

### Normal-Form Games

A normal-form (strategic-form) game is a tuple G = (N, {Sᵢ}, {uᵢ}) where:

- **N** = {1, ..., n} is the set of players
- **Sᵢ** = {sᵢ¹, ..., sᵢᵐⁱ} is player i's strategy set
- **uᵢ**: S₁ × S₂ × ... × Sₙ → ℝ is player i's payoff function

A **pure strategy profile** is s = (s₁, s₂, ..., sₙ) ∈ S₁ × S₂ × ... × Sₙ.

### Mixed Strategies

A **mixed strategy** for player i is a probability distribution σᵢ over Sᵢ:

- σᵢ(sᵢʲ) ≥ 0 for all j
- Σⱼ σᵢ(sᵢʲ) = 1

The expected utility is:

  **uᵢ(σ) = Σ_{s∈S} [Πⱼ σⱼ(sⱼ)] · uᵢ(s)**

### Nash's Theorem (1950)

**Theorem:** Every finite game (finite players, finite strategies) has at least one
Nash equilibrium in mixed strategies.

This means: for *any* normal-form game you define with `nash-finder`, a solution
*always exists*. The challenge is finding it efficiently.

### Support Enumeration

For a 2-player game, a Nash equilibrium (σ₁*, σ₂*) satisfies the **indifference
condition**: every action in a player's support yields the same expected utility
against the opponent's mixture.

Given supports S₁ ⊆ A₁ and S₂ ⊆ A₂ with |S₁| = |S₂| = k:

1. **Indifference:** u₁(a, σ₂) = u₁(b, σ₂) for all a, b ∈ S₁
2. **Normalization:** Σ_{j∈S₂} σ₂(j) = 1
3. **Non-negativity:** σ₂(j) > 0 for j ∈ S₂
4. **No deviation:** u₁(a, σ₂) ≥ u₁(c, σ₂) for c ∉ S₁

`nash-finder` solves these conditions via Gaussian elimination for each candidate
support pair, then verifies the result.

### Epsilon-Equilibrium

An **ε-Nash equilibrium** relaxes exact optimality:

  uᵢ(σᵢ*, σ₋ᵢ*) ≥ uᵢ(σᵢ, σ₋ᵢ*) - ε  for all σᵢ

This is useful for:
- **Approximate computation** in large games
- **Learning algorithms** (fictitious play converges to ε-NE)
- **Bounded rationality** models

### Dominance

A strategy sᵢ is **strictly dominated** by tᵢ if:

  uᵢ(tᵢ, s₋ᵢ) > uᵢ(sᵢ, s₋ᵢ)  for all s₋ᵢ

Iterated elimination of strictly dominated strategies preserves all Nash equilibria
while reducing the game's size. `nash-finder` applies this as a preprocessing step.

## Classic Games Reference

### Prisoner's Dilemma

```
            Cooperate  Defect
Cooperate   (-1, -1)   (-3,  0)
Defect      ( 0, -3)   (-2, -2)
```

**Unique NE:** (Defect, Defect) with payoff (-2, -2). The canonical example of
individual rationality leading to collective suboptimality.

### Battle of the Sexes

```
         Opera    Football
Opera    (3, 2)   (0, 0)
Football (0, 0)   (2, 3)
```

**3 NE:** Two pure (both Opera, both Football) and one mixed. Illustrates
coordination problems with conflicting preferences.

### Matching Pennies

```
        Heads    Tails
Heads   (1, -1)  (-1, 1)
Tails   (-1, 1)  (1, -1)
```

**1 NE:** (½, ½) × (½, ½). A zero-sum game with no pure-strategy equilibrium.

### Rock-Paper-Scissors

```
          Rock      Paper     Scissors
Rock      (0, 0)    (-1, 1)   (1, -1)
Paper     (1, -1)   (0, 0)    (-1, 1)
Scissors  (-1, 1)   (1, -1)   (0, 0)
```

**1 NE:** (⅓, ⅓, ⅓) × (⅓, ⅓, ⅓). The unique equilibrium is fully mixed.

### Stag Hunt

```
        Stag     Hare
Stag    (4, 4)   (0, 3)
Hare    (3, 0)   (3, 3)
```

**3 NE:** (Stag, Stag), (Hare, Hare), and one mixed. Models the tension between
cooperation and safety.

### Chicken (Hawk-Dove)

```
       Swerve    Straight
Swerve   (0, 0)   (-1, 1)
Straight (1, -1)  (-10, -10)
```

**3 NE:** Two pure asymmetric and one mixed. Brinksmanship and escalation.

## Design Decisions

### Zero External Dependencies (except serde)

Game theory is foundational mathematics. It shouldn't pull in a dependency tree
the size of a web framework. `serde` is the sole exception — serialization is
essential for persistence and interop.

### f64 Throughout

Game theory payoffs are inherently continuous. We use `f64` throughout for
numerical stability and simplicity. No generic numeric types.

### 2-Player Focus for Exact Solutions

Support enumeration is the primary exact algorithm, and it's specific to 2-player
games. For N-player games, the solver falls back to fictitious play and best-response
dynamics. This mirrors the state of the art: exact NE computation is tractable for
2 players, PPAD-complete for 3+.

### Verification as a First-Class Concern

Every equilibrium found by support enumeration is independently verified against
the best-response conditions. This catches numerical errors from Gaussian elimination
and ensures the results are mathematically sound.

### Builder Pattern for Equilibria

`EquilibriumBuilder` lets you construct candidate equilibria from external sources
and verify them against the game. Useful for testing hypotheses and integrating
with other solvers.

## Limitations

- **2-player support enumeration** is the primary exact method. N-player games
  use approximate methods only.
- **Correlated equilibria** are not supported (only Nash equilibria).
- **Extensive-form games** (game trees) are not supported — only normal form.
- **No LP/QP solver**: indifference equations are solved via Gaussian elimination,
  which may be less numerically robust than commercial LP solvers for large games.
- **Exponential worst case**: support enumeration is O(C(n,k)²) where n is the
  number of strategies and k the support size. Large games should use fictitious play.

## License

MIT
