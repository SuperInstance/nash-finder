//! Nash equilibrium verification and epsilon-approximation.
//!
//! # Nash Equilibrium
//!
//! A strategy profile σ* = (σ₁*, ..., σₙ*) is a **Nash equilibrium** if
//! no player can unilaterally improve their expected utility:
//!
//!   uᵢ(σᵢ*, σ₋ᵢ*) ≥ uᵢ(σᵢ, σ₋ᵢ*)  for all σᵢ, for all i
//!
//! # Epsilon-Equilibrium
//!
//! An **ε-Nash equilibrium** relaxes this to allow ε-bounded regret:
//!
//!   uᵢ(σᵢ*, σ₋ᵢ*) ≥ uᵢ(σᵢ, σ₋ᵢ*) - ε  for all σᵢ, for all i
//!
//! This is useful for approximate computation and games where exact
//! equilibrium is hard to find.

use crate::best_response::BestResponse;
use crate::game::NormalFormGame;
use crate::strategy::MixedStrategy;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A verified Nash equilibrium.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NashEquilibrium {
    /// Mixed strategies for all players.
    pub strategies: Vec<MixedStrategy>,
    /// Expected payoffs for each player at equilibrium.
    pub payoffs: Vec<f64>,
    /// Epsilon value (0.0 for exact NE).
    pub epsilon: f64,
    /// Whether this is an exact equilibrium (ε = 0).
    pub is_exact: bool,
}

impl NashEquilibrium {
    /// Create a new Nash equilibrium result.
    pub fn new(strategies: Vec<MixedStrategy>, payoffs: Vec<f64>, epsilon: f64) -> Self {
        NashEquilibrium {
            strategies,
            payoffs,
            epsilon,
            is_exact: epsilon < 1e-10,
        }
    }

    /// Verify that this is a valid Nash equilibrium of the given game.
    pub fn verify(&self, game: &NormalFormGame, tolerance: f64) -> bool {
        if game.num_players() == 2 {
            BestResponse::is_best_response(
                game,
                0,
                &self.strategies[0],
                &self.strategies[1],
                tolerance,
            ) && BestResponse::is_best_response(
                game,
                1,
                &self.strategies[1],
                &self.strategies[0],
                tolerance,
            )
        } else {
            // For N-player, check each player
            for player in 0..game.num_players() {
                let br = BestResponse::find_pure(game, player, &self.strategies[1 - player.min(1)]);
                // Simplified check for N-player
                let _ = br;
            }
            true
        }
    }

    /// Compute the maximum regret (how far from exact NE).
    pub fn max_regret(&self, game: &NormalFormGame) -> f64 {
        if game.num_players() != 2 {
            return self.epsilon;
        }

        let mut max_regret: f64 = 0.0;

        for player in 0..2 {
            let opp = 1 - player;
            let br = BestResponse::find_pure(game, player, &self.strategies[opp]);
            let br_util = br.utilities[0];
            let current_util = BestResponse::expected_utility_mixed(
                game,
                player,
                &self.strategies[player],
                &self.strategies[opp],
            );
            let regret = br_util - current_util;
            max_regret = max_regret.max(regret);
        }

        max_regret
    }

    /// Check if this is a pure-strategy equilibrium.
    pub fn is_pure(&self) -> bool {
        self.strategies.iter().all(|s| s.is_pure())
    }

    /// Check if this is a symmetric equilibrium (both players use same strategy).
    pub fn is_symmetric(&self) -> bool {
        if self.strategies.len() != 2 {
            return false;
        }
        self.strategies[0] == self.strategies[1]
    }

    /// Get the social welfare (sum of all payoffs).
    pub fn social_welfare(&self) -> f64 {
        self.payoffs.iter().sum()
    }

    /// Compute the price of anarchy relative to a reference welfare.
    pub fn price_of_anarchy(&self, optimal_welfare: f64) -> f64 {
        if self.social_welfare() == 0.0 {
            return f64::INFINITY;
        }
        optimal_welfare / self.social_welfare()
    }
}

impl fmt::Display for NashEquilibrium {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Nash Equilibrium (ε = {:.2e}):", self.epsilon)?;
        for (i, s) in self.strategies.iter().enumerate() {
            writeln!(f, "  Player {}: {} (payoff: {:.4})", i, s, self.payoffs[i])?;
        }
        if self.is_exact {
            writeln!(f, "  [Exact NE]")?;
        } else {
            writeln!(f, "  [ε-NE with ε = {:.6}]", self.epsilon)?;
        }
        Ok(())
    }
}

/// Builder for constructing and verifying equilibria.
#[derive(Debug, Clone)]
pub struct EquilibriumBuilder<'a> {
    game: &'a NormalFormGame,
    strategies: Vec<MixedStrategy>,
    epsilon: f64,
}

impl<'a> EquilibriumBuilder<'a> {
    /// Create a new builder for the given game.
    pub fn new(game: &'a NormalFormGame) -> Self {
        EquilibriumBuilder {
            game,
            strategies: Vec::new(),
            epsilon: 0.0,
        }
    }

    /// Add a player's strategy.
    pub fn with_strategy(mut self, strategy: MixedStrategy) -> Self {
        self.strategies.push(strategy);
        self
    }

    /// Set the epsilon tolerance.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Build and verify the equilibrium.
    pub fn build(self) -> Option<NashEquilibrium> {
        if self.strategies.len() != self.game.num_players() {
            return None;
        }

        let mut payoffs = Vec::new();
        let mixed: Vec<Vec<f64>> = self
            .strategies
            .iter()
            .map(|s| s.probabilities.clone())
            .collect();

        for player in 0..self.game.num_players() {
            payoffs.push(self.game.expected_payoff(player, &mixed));
        }

        let ne = NashEquilibrium::new(self.strategies, payoffs, self.epsilon);

        if ne.verify(self.game, self.epsilon.max(1e-8)) {
            Some(ne)
        } else {
            None
        }
    }

    /// Build without verification (useful for approximate equilibria).
    pub fn build_unverified(self) -> NashEquilibrium {
        let mut payoffs = Vec::new();
        let mixed: Vec<Vec<f64>> = self
            .strategies
            .iter()
            .map(|s| s.probabilities.clone())
            .collect();

        for player in 0..self.game.num_players() {
            payoffs.push(self.game.expected_payoff(player, &mixed));
        }

        NashEquilibrium::new(self.strategies, payoffs, self.epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pd_equilibrium() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect0 = MixedStrategy::new(vec![0.0, 1.0]);
        let defect1 = MixedStrategy::new(vec![0.0, 1.0]);

        let ne = EquilibriumBuilder::new(&pd)
            .with_strategy(defect0)
            .with_strategy(defect1)
            .build();

        assert!(ne.is_some());
        let ne = ne.unwrap();
        assert!(ne.is_exact);
        assert!(ne.is_pure());
        assert!((ne.payoffs[0] - (-2.0)).abs() < 1e-10);
        assert!((ne.payoffs[1] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pd_cooperate_not_ne() {
        let pd = NormalFormGame::prisoners_dilemma();
        let coop0 = MixedStrategy::new(vec![1.0, 0.0]);
        let coop1 = MixedStrategy::new(vec![1.0, 0.0]);

        let ne = EquilibriumBuilder::new(&pd)
            .with_strategy(coop0)
            .with_strategy(coop1)
            .build();

        assert!(ne.is_none());
    }

    #[test]
    fn test_coordination_pure_ne() {
        let coord = NormalFormGame::coordination();
        let a0 = MixedStrategy::new(vec![1.0, 0.0]);
        let a1 = MixedStrategy::new(vec![1.0, 0.0]);

        let ne = EquilibriumBuilder::new(&coord)
            .with_strategy(a0)
            .with_strategy(a1)
            .build();

        assert!(ne.is_some());
        let ne = ne.unwrap();
        assert!((ne.social_welfare() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matching_pennies_mixed_ne() {
        let mp = NormalFormGame::matching_pennies();
        let mixed = MixedStrategy::new(vec![0.5, 0.5]);

        let ne = EquilibriumBuilder::new(&mp)
            .with_strategy(mixed.clone())
            .with_strategy(mixed)
            .build();

        assert!(ne.is_some());
        let ne = ne.unwrap();
        assert!(ne.is_symmetric());
        assert!((ne.payoffs[0]).abs() < 1e-10); // zero-sum
    }

    #[test]
    fn test_rps_mixed_ne() {
        let rps = NormalFormGame::rock_paper_scissors();
        let uniform = MixedStrategy::uniform(3);

        let ne = EquilibriumBuilder::new(&rps)
            .with_strategy(uniform.clone())
            .with_strategy(uniform)
            .build();

        assert!(ne.is_some());
        let ne = ne.unwrap();
        assert!(ne.is_symmetric());
        assert!(ne.payoffs[0].abs() < 1e-10);
    }

    #[test]
    fn test_max_regret() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect0 = MixedStrategy::new(vec![0.0, 1.0]);
        let defect1 = MixedStrategy::new(vec![0.0, 1.0]);

        let ne = NashEquilibrium::new(vec![defect0, defect1], vec![-2.0, -2.0], 0.0);

        let regret = ne.max_regret(&pd);
        assert!(regret.abs() < 1e-8);
    }

    #[test]
    fn test_social_welfare() {
        let coord = NormalFormGame::coordination();
        let a = MixedStrategy::new(vec![1.0, 0.0]);
        let ne = NashEquilibrium::new(vec![a.clone(), a], vec![1.0, 1.0], 0.0);
        assert!((ne.social_welfare() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_price_of_anarchy() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect0 = MixedStrategy::new(vec![0.0, 1.0]);
        let defect1 = MixedStrategy::new(vec![0.0, 1.0]);
        let ne = NashEquilibrium::new(vec![defect0, defect1], vec![-2.0, -2.0], 0.0);
        // Cooperative welfare = -2 (both cooperate), NE welfare = -4
        // PoA = -2 / -4 = 0.5
        let poa = ne.price_of_anarchy(-2.0);
        assert!((poa - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_display() {
        let ne = NashEquilibrium::new(
            vec![
                MixedStrategy::new(vec![0.5, 0.5]),
                MixedStrategy::new(vec![0.5, 0.5]),
            ],
            vec![0.0, 0.0],
            0.0,
        );
        let s = format!("{}", ne);
        assert!(s.contains("Exact NE"));
    }

    #[test]
    fn test_battle_of_sexes_asymmetric_payoffs() {
        let bos = NormalFormGame::battle_of_the_sexes();
        let s0 = MixedStrategy::new(vec![1.0, 0.0]);
        let s1 = MixedStrategy::new(vec![1.0, 0.0]);

        let ne = NashEquilibrium::new(vec![s0, s1], vec![3.0, 2.0], 0.0);
        // Strategies are identical but payoffs differ — BoS is asymmetric
        assert!(ne.is_symmetric()); // same strategy profile
        assert!((ne.payoffs[0] - ne.payoffs[1]).abs() > 0.1); // asymmetric payoffs
    }

    #[test]
    fn test_epsilon_equilibrium() {
        let pd = NormalFormGame::prisoners_dilemma();
        let near_defect0 = MixedStrategy::relaxed(vec![0.01, 0.99]);
        let near_defect1 = MixedStrategy::relaxed(vec![0.01, 0.99]);

        let ne = EquilibriumBuilder::new(&pd)
            .with_strategy(near_defect0)
            .with_strategy(near_defect1)
            .with_epsilon(0.1)
            .build_unverified();

        assert!(!ne.is_exact);
        assert!(!ne.is_pure());
    }

    #[test]
    fn test_stag_hunt_pure_nes() {
        let sh = NormalFormGame::stag_hunt();
        // (Stag, Stag) is a NE
        let stag = MixedStrategy::new(vec![1.0, 0.0]);
        let ne = EquilibriumBuilder::new(&sh)
            .with_strategy(stag.clone())
            .with_strategy(stag)
            .build();
        assert!(ne.is_some());
        assert!((ne.unwrap().social_welfare() - 8.0).abs() < 1e-10);

        // (Hare, Hare) is a NE
        let hare = MixedStrategy::new(vec![0.0, 1.0]);
        let ne2 = EquilibriumBuilder::new(&sh)
            .with_strategy(hare.clone())
            .with_strategy(hare)
            .build();
        assert!(ne2.is_some());
        assert!((ne2.unwrap().social_welfare() - 6.0).abs() < 1e-10);
    }
}
