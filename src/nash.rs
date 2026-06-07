//! Unified Nash equilibrium solver.
//!
//! This module combines support enumeration, best-response dynamics,
//! fictitious play, and iterated dominance elimination into a single
//! solver interface.
//!
//! # Solver Pipeline
//!
//! 1. **Iterated dominance elimination** — remove strictly dominated strategies
//! 2. **Support enumeration** — find all exact equilibria in the reduced game
//! 3. **Best-response dynamics** — verify and refine via dynamic adjustment
//! 4. **Fictitious play** — find approximate equilibria for larger games

use crate::best_response::BestResponse;
use crate::equilibrium::{EquilibriumBuilder, NashEquilibrium};
use crate::game::NormalFormGame;
use crate::strategy::MixedStrategy;
use crate::support::SupportEnumeration;
use serde::{Deserialize, Serialize};

/// Configuration for the Nash solver.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolverConfig {
    /// Maximum iterations for iterative methods.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub epsilon: f64,
    /// Whether to use iterated dominance elimination.
    pub use_dominance: bool,
    /// Whether to use support enumeration.
    pub use_support_enum: bool,
    /// Whether to use fictitious play for approximation.
    pub use_fictitious_play: bool,
    /// Number of fictitious play iterations.
    pub fp_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            max_iterations: 1000,
            epsilon: 1e-8,
            use_dominance: true,
            use_support_enum: true,
            use_fictitious_play: true,
            fp_iterations: 10000,
        }
    }
}

/// Unified Nash equilibrium solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NashSolver {
    config: SolverConfig,
}

impl NashSolver {
    /// Create a new solver with default configuration.
    pub fn new() -> Self {
        NashSolver {
            config: SolverConfig::default(),
        }
    }

    /// Create a solver with custom configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        NashSolver { config }
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Find all Nash equilibria of a 2-player game.
    ///
    /// Uses support enumeration to find exact equilibria, optionally
    /// preceded by iterated dominance elimination.
    pub fn solve(&self, game: &NormalFormGame) -> SolverResult {
        assert_eq!(
            game.num_players(),
            2,
            "solver currently supports 2-player games"
        );

        let mut dominated_info = DominanceInfo::default();

        // Step 1: Iterated dominance elimination
        if self.config.use_dominance {
            dominated_info = self.iterated_dominance(game);
        }

        // Step 2: Support enumeration
        let mut equilibria = Vec::new();
        if self.config.use_support_enum {
            let found = SupportEnumeration::find_all_equilibria(game);
            for (s0, s1) in found {
                let mixed = vec![s0, s1];
                let mut payoffs = Vec::new();
                let mixed_refs: Vec<Vec<f64>> =
                    mixed.iter().map(|s| s.probabilities.clone()).collect();
                for p in 0..game.num_players() {
                    payoffs.push(game.expected_payoff(p, &mixed_refs));
                }
                equilibria.push(NashEquilibrium::new(mixed, payoffs, 0.0));
            }
        }

        // Step 3: Fictitious play for approximate equilibria
        let mut fp_equilibrium = None;
        if self.config.use_fictitious_play {
            let (fp_s0, fp_s1) = BestResponse::fictitious_play(game, self.config.fp_iterations);

            // Check if FP found something new
            let is_new = !equilibria.iter().any(|ne| {
                ne.strategies.len() == 2 && ne.strategies[0] == fp_s0 && ne.strategies[1] == fp_s1
            });

            if is_new {
                let mixed = vec![fp_s0.clone(), fp_s1.clone()];
                let mixed_refs: Vec<Vec<f64>> =
                    mixed.iter().map(|s| s.probabilities.clone()).collect();
                let mut payoffs = Vec::new();
                for p in 0..game.num_players() {
                    payoffs.push(game.expected_payoff(p, &mixed_refs));
                }

                let ne = NashEquilibrium::new(mixed, payoffs, self.config.epsilon);
                fp_equilibrium = Some(ne);
            }
        }

        SolverResult {
            equilibria,
            dominated: dominated_info,
            fp_equilibrium,
        }
    }

    /// Find just the pure-strategy Nash equilibria.
    pub fn find_pure_equilibria(&self, game: &NormalFormGame) -> Vec<NashEquilibrium> {
        let mut results = Vec::new();
        let n0 = game.num_strategies(0);
        let n1 = game.num_strategies(1);

        for a in 0..n0 {
            for b in 0..n1 {
                let s0 = MixedStrategy::relaxed(
                    (0..n0).map(|i| if i == a { 1.0 } else { 0.0 }).collect(),
                );
                let s1 = MixedStrategy::relaxed(
                    (0..n1).map(|j| if j == b { 1.0 } else { 0.0 }).collect(),
                );

                if let Some(ne) = EquilibriumBuilder::new(game)
                    .with_strategy(s0)
                    .with_strategy(s1)
                    .build()
                {
                    results.push(ne);
                }
            }
        }

        results
    }

    /// Perform iterated strict dominance elimination.
    ///
    /// Repeatedly removes strictly dominated strategies until no more
    /// can be eliminated.
    fn iterated_dominance(&self, game: &NormalFormGame) -> DominanceInfo {
        let mut rounds = 0;
        let mut eliminated: Vec<Vec<usize>> = vec![Vec::new(); game.num_players()];
        let mut changed = true;

        while changed {
            changed = false;
            rounds += 1;

            for (player, elim) in eliminated.iter_mut().enumerate() {
                let dom = game.eliminate_dominated(player);
                if !dom.is_empty() {
                    for d in &dom {
                        if !elim.contains(d) {
                            elim.push(*d);
                            changed = true;
                        }
                    }
                }
            }
        }

        DominanceInfo {
            rounds,
            eliminated,
            converged: !changed,
        }
    }

    /// Solve using best-response dynamics from a given starting point.
    pub fn solve_br_dynamics(
        &self,
        game: &NormalFormGame,
        initial: (MixedStrategy, MixedStrategy),
    ) -> Option<NashEquilibrium> {
        let result = BestResponse::best_response_dynamics(
            game,
            initial,
            self.config.max_iterations,
            self.config.epsilon,
        );

        result.map(|(s0, s1)| {
            let mixed = vec![s0.clone(), s1.clone()];
            let mixed_refs: Vec<Vec<f64>> = mixed.iter().map(|s| s.probabilities.clone()).collect();
            let mut payoffs = Vec::new();
            for p in 0..game.num_players() {
                payoffs.push(game.expected_payoff(p, &mixed_refs));
            }
            NashEquilibrium::new(mixed, payoffs, self.config.epsilon)
        })
    }
}

impl Default for NashSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of the Nash solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    /// Exact Nash equilibria found.
    pub equilibria: Vec<NashEquilibrium>,
    /// Dominance elimination information.
    pub dominated: DominanceInfo,
    /// Approximate equilibrium from fictitious play (if distinct).
    pub fp_equilibrium: Option<NashEquilibrium>,
}

impl SolverResult {
    /// Number of equilibria found.
    pub fn count(&self) -> usize {
        self.equilibria.len()
    }

    /// Get pure-strategy equilibria.
    pub fn pure_equilibria(&self) -> Vec<&NashEquilibrium> {
        self.equilibria.iter().filter(|ne| ne.is_pure()).collect()
    }

    /// Get mixed-strategy equilibria.
    pub fn mixed_equilibria(&self) -> Vec<&NashEquilibrium> {
        self.equilibria.iter().filter(|ne| !ne.is_pure()).collect()
    }

    /// Get the equilibrium with highest social welfare.
    pub fn best_welfare(&self) -> Option<&NashEquilibrium> {
        self.equilibria
            .iter()
            .max_by(|a, b| a.social_welfare().partial_cmp(&b.social_welfare()).unwrap())
    }

    /// Get the equilibrium with lowest social welfare.
    pub fn worst_welfare(&self) -> Option<&NashEquilibrium> {
        self.equilibria
            .iter()
            .min_by(|a, b| a.social_welfare().partial_cmp(&b.social_welfare()).unwrap())
    }
}

/// Information about dominance elimination.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DominanceInfo {
    /// Number of elimination rounds.
    pub rounds: usize,
    /// Eliminated strategies per player.
    pub eliminated: Vec<Vec<usize>>,
    /// Whether elimination converged.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_pd() {
        let solver = NashSolver::new();
        let pd = NormalFormGame::prisoners_dilemma();
        let result = solver.solve(&pd);
        assert_eq!(result.count(), 1);
        assert!(result.equilibria[0].is_pure());
        assert!(result.equilibria[0].strategies[0].probabilities[1] > 0.99);
    }

    #[test]
    fn test_solve_coordination() {
        let solver = NashSolver::new();
        let coord = NormalFormGame::coordination();
        let result = solver.solve(&coord);
        assert_eq!(result.count(), 3);
        assert_eq!(result.pure_equilibria().len(), 2);
        assert_eq!(result.mixed_equilibria().len(), 1);
    }

    #[test]
    fn test_solve_matching_pennies() {
        let solver = NashSolver::new();
        let mp = NormalFormGame::matching_pennies();
        let result = solver.solve(&mp);
        assert_eq!(result.count(), 1);
        assert!(result.equilibria[0].is_symmetric());
    }

    #[test]
    fn test_solve_rps() {
        let solver = NashSolver::new();
        let rps = NormalFormGame::rock_paper_scissors();
        let result = solver.solve(&rps);
        assert_eq!(result.count(), 1);
        let ne = &result.equilibria[0];
        for i in 0..3 {
            assert!((ne.strategies[0].probabilities[i] - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_find_pure_equilibria_pd() {
        let solver = NashSolver::new();
        let pd = NormalFormGame::prisoners_dilemma();
        let pure = solver.find_pure_equilibria(&pd);
        assert_eq!(pure.len(), 1);
    }

    #[test]
    fn test_find_pure_equilibria_coordination() {
        let solver = NashSolver::new();
        let coord = NormalFormGame::coordination();
        let pure = solver.find_pure_equilibria(&coord);
        assert_eq!(pure.len(), 2);
    }

    #[test]
    fn test_best_welfare() {
        let solver = NashSolver::new();
        let coord = NormalFormGame::coordination();
        let result = solver.solve(&coord);
        let best = result.best_welfare().unwrap();
        assert!((best.social_welfare() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_worst_welfare() {
        let solver = NashSolver::new();
        let coord = NormalFormGame::coordination();
        let result = solver.solve(&coord);
        let worst = result.worst_welfare().unwrap();
        // Mixed NE has welfare 1.0 (0.5*1 + 0.5*1 + 0.5*0 + 0.5*0 = 1.0)
        assert!(worst.social_welfare() <= 2.0);
    }

    #[test]
    fn test_dominance_pd() {
        let solver = NashSolver::new();
        let pd = NormalFormGame::prisoners_dilemma();
        let result = solver.solve(&pd);
        assert!(result.dominated.rounds > 0);
        assert!(result.dominated.eliminated[0].contains(&0)); // Cooperate eliminated
    }

    #[test]
    fn test_custom_config() {
        let config = SolverConfig {
            max_iterations: 500,
            epsilon: 1e-6,
            use_dominance: false,
            use_support_enum: true,
            use_fictitious_play: false,
            fp_iterations: 1000,
        };
        let solver = NashSolver::with_config(config);
        let pd = NormalFormGame::prisoners_dilemma();
        let result = solver.solve(&pd);
        assert_eq!(result.count(), 1);
        assert_eq!(result.dominated.rounds, 0); // dominance disabled
    }

    #[test]
    fn test_br_dynamics() {
        let solver = NashSolver::new();
        let coord = NormalFormGame::coordination();
        let init = (MixedStrategy::uniform(2), MixedStrategy::uniform(2));
        let result = solver.solve_br_dynamics(&coord, init);
        assert!(result.is_some());
        assert!(result.unwrap().is_pure());
    }

    #[test]
    fn test_solve_battle_of_sexes() {
        let solver = NashSolver::new();
        let bos = NormalFormGame::battle_of_the_sexes();
        let result = solver.solve(&bos);
        assert_eq!(result.count(), 3);
    }

    #[test]
    fn test_solve_chicken() {
        let solver = NashSolver::new();
        let chicken = NormalFormGame::chicken();
        let result = solver.solve(&chicken);
        assert_eq!(result.count(), 3);
    }

    #[test]
    fn test_solver_default() {
        let solver = NashSolver::default();
        assert_eq!(solver.config().max_iterations, 1000);
    }
}
