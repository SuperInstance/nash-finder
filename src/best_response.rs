//! Best-response computation.
//!
//! Given an opponent's strategy, the **best response** is the strategy
//! (pure or mixed) that maximizes a player's expected utility.
//!
//! # Definition
//!
//! For player i facing opponent strategy σ₋ᵢ, a best response σ*ᵢ satisfies:
//!   uᵢ(σ*ᵢ, σ₋ᵢ) ≥ uᵢ(σᵢ, σ₋ᵢ)  for all σᵢ
//!
//! Since utility is linear in the player's own mixing probabilities, there
//! always exists a pure-strategy best response.

use crate::game::NormalFormGame;
use crate::strategy::MixedStrategy;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Result of a best-response computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BestResponseResult {
    /// The best-response action(s). Multiple if tied.
    pub actions: Vec<usize>,
    /// Expected utility of each best-response action.
    pub utilities: Vec<f64>,
    /// Whether the best response is unique.
    pub is_unique: bool,
}

impl fmt::Display for BestResponseResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_unique {
            write!(
                f,
                "BR: action {} (utility: {:.4})",
                self.actions[0], self.utilities[0]
            )
        } else {
            write!(
                f,
                "BR: actions {:?} (utilities: {:?})",
                self.actions, self.utilities
            )
        }
    }
}

/// Best-response computation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestResponse;

impl BestResponse {
    /// Find the pure best response for `player` against `opponent_mixed`.
    ///
    /// For 2-player games, `opponent_mixed` is the mixed strategy of the
    /// other player.
    ///
    /// Returns all actions that achieve the maximum expected utility.
    pub fn find_pure(
        game: &NormalFormGame,
        player: usize,
        opponent_mixed: &MixedStrategy,
    ) -> BestResponseResult {
        assert_eq!(
            game.num_players(),
            2,
            "find_pure currently supports 2-player games"
        );
        let opp = 1 - player;
        assert_eq!(
            opponent_mixed.probabilities.len(),
            game.num_strategies(opp),
            "opponent mixed strategy size mismatch"
        );

        let n = game.num_strategies(player);
        let mut utilities = Vec::with_capacity(n);

        for action in 0..n {
            let mut eu = 0.0;
            for opp_action in 0..game.num_strategies(opp) {
                let mut profile = vec![0; 2];
                profile[player] = action;
                profile[opp] = opp_action;
                eu += opponent_mixed.probabilities[opp_action] * game.payoff(player, &profile);
            }
            utilities.push(eu);
        }

        let max_util = utilities.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let best_actions: Vec<usize> = utilities
            .iter()
            .enumerate()
            .filter(|&(_, u)| (u - max_util).abs() < 1e-10)
            .map(|(i, _)| i)
            .collect();

        let best_utils: Vec<f64> = best_actions.iter().map(|&a| utilities[a]).collect();

        BestResponseResult {
            is_unique: best_actions.len() == 1,
            actions: best_actions,
            utilities: best_utils,
        }
    }

    /// Compute expected utility for `player` playing `action` against
    /// `opponent_mixed` in a 2-player game.
    pub fn expected_utility(
        game: &NormalFormGame,
        player: usize,
        action: usize,
        opponent_mixed: &MixedStrategy,
    ) -> f64 {
        let opp = 1 - player;
        let mut eu = 0.0;
        for opp_action in 0..game.num_strategies(opp) {
            let mut profile = vec![0; 2];
            profile[player] = action;
            profile[opp] = opp_action;
            eu += opponent_mixed.probabilities[opp_action] * game.payoff(player, &profile);
        }
        eu
    }

    /// Compute expected utility for `player` using a mixed strategy against
    /// `opponent_mixed` in a 2-player game.
    pub fn expected_utility_mixed(
        game: &NormalFormGame,
        player: usize,
        player_mixed: &MixedStrategy,
        opponent_mixed: &MixedStrategy,
    ) -> f64 {
        let opp = 1 - player;
        let mut eu = 0.0;
        for a in 0..game.num_strategies(player) {
            for b in 0..game.num_strategies(opp) {
                let mut profile = vec![0; 2];
                profile[player] = a;
                profile[opp] = b;
                eu += player_mixed.probabilities[a]
                    * opponent_mixed.probabilities[b]
                    * game.payoff(player, &profile);
            }
        }
        eu
    }

    /// Check if `player_mixed` is a best response to `opponent_mixed`.
    ///
    /// A mixed strategy is a best response iff every action in its support
    /// yields the same expected utility, and no other action yields more.
    pub fn is_best_response(
        game: &NormalFormGame,
        player: usize,
        player_mixed: &MixedStrategy,
        opponent_mixed: &MixedStrategy,
        epsilon: f64,
    ) -> bool {
        let br = Self::find_pure(game, player, opponent_mixed);
        let br_utility = br.utilities[0];

        // Every action in the support must achieve the same utility as the BR
        for action in player_mixed.support() {
            let u = Self::expected_utility(game, player, action, opponent_mixed);
            if (u - br_utility).abs() > epsilon {
                return false;
            }
        }

        true
    }

    /// Best-response dynamics: iteratively improve strategies.
    ///
    /// Returns the converged strategy profile or `None` if max iterations
    /// is reached without convergence.
    pub fn best_response_dynamics(
        game: &NormalFormGame,
        initial: (MixedStrategy, MixedStrategy),
        max_iterations: usize,
        convergence_epsilon: f64,
    ) -> Option<(MixedStrategy, MixedStrategy)> {
        let mut s0 = initial.0;
        let mut s1 = initial.1;

        for _ in 0..max_iterations {
            // Player 0 best responds to player 1
            let br0 = Self::find_pure(game, 0, &s1);
            let new_s0 = MixedStrategy::relaxed(
                std::iter::repeat_n(0.0, game.num_strategies(0))
                    .enumerate()
                    .map(|(i, _)| if i == br0.actions[0] { 1.0 } else { 0.0 })
                    .collect(),
            );

            // Player 1 best responds to player 0
            let br1 = Self::find_pure(game, 1, &s0);
            let new_s1 = MixedStrategy::relaxed(
                std::iter::repeat_n(0.0, game.num_strategies(1))
                    .enumerate()
                    .map(|(i, _)| if i == br1.actions[0] { 1.0 } else { 0.0 })
                    .collect(),
            );

            // Check convergence
            let converged = new_s0
                .probabilities
                .iter()
                .zip(s0.probabilities.iter())
                .chain(new_s1.probabilities.iter().zip(s1.probabilities.iter()))
                .all(|(a, b)| (a - b).abs() < convergence_epsilon);

            s0 = new_s0;
            s1 = new_s1;

            if converged {
                return Some((s0, s1));
            }
        }

        None
    }

    /// Fictitious play: smoothed best-response dynamics with mixed strategies.
    ///
    /// Each player plays a best response to the empirical distribution
    /// of the opponent's past plays.
    pub fn fictitious_play(
        game: &NormalFormGame,
        iterations: usize,
    ) -> (MixedStrategy, MixedStrategy) {
        let n0 = game.num_strategies(0);
        let n1 = game.num_strategies(1);
        let mut counts0 = vec![1.0; n0]; // Laplace smoothing
        let mut counts1 = vec![1.0; n1];

        for _ in 0..iterations {
            let emp0 = MixedStrategy::relaxed(counts0.clone());
            let emp1 = MixedStrategy::relaxed(counts1.clone());

            let br0 = Self::find_pure(game, 0, &emp1);
            let br1 = Self::find_pure(game, 1, &emp0);

            counts0[br0.actions[0]] += 1.0;
            counts1[br1.actions[0]] += 1.0;
        }

        (
            MixedStrategy::relaxed(counts0),
            MixedStrategy::relaxed(counts1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_response_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        let coop = MixedStrategy::new(vec![1.0, 0.0]);
        let br = BestResponse::find_pure(&pd, 1, &coop);
        // Against cooperate, best response is defect (action 1)
        assert_eq!(br.actions, vec![1]);
        assert!(br.is_unique);
    }

    #[test]
    fn test_best_response_pd_both_defect() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect = MixedStrategy::new(vec![0.0, 1.0]);
        let br = BestResponse::find_pure(&pd, 0, &defect);
        assert_eq!(br.actions, vec![1]); // Defect is BR to Defect
    }

    #[test]
    fn test_best_response_coordination() {
        let coord = NormalFormGame::coordination();
        let a_mixed = MixedStrategy::new(vec![0.8, 0.2]);
        let br = BestResponse::find_pure(&coord, 1, &a_mixed);
        // Best response to 80% A is A (action 0)
        assert_eq!(br.actions, vec![0]);
    }

    #[test]
    fn test_expected_utility() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect = MixedStrategy::new(vec![0.0, 1.0]);
        let eu = BestResponse::expected_utility(&pd, 0, 0, &defect);
        // Cooperate vs Defect: -3
        assert!((eu - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_expected_utility_mixed() {
        let pd = NormalFormGame::prisoners_dilemma();
        let s0 = MixedStrategy::new(vec![0.5, 0.5]);
        let s1 = MixedStrategy::new(vec![0.5, 0.5]);
        let eu = BestResponse::expected_utility_mixed(&pd, 0, &s0, &s1);
        assert!((eu - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_is_best_response_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        let defect0 = MixedStrategy::new(vec![0.0, 1.0]);
        let defect1 = MixedStrategy::new(vec![0.0, 1.0]);
        assert!(BestResponse::is_best_response(
            &pd, 0, &defect0, &defect1, 1e-8
        ));
    }

    #[test]
    fn test_is_not_best_response_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        let coop = MixedStrategy::new(vec![1.0, 0.0]);
        let defect = MixedStrategy::new(vec![0.0, 1.0]);
        // Cooperate is NOT a BR to Defect
        assert!(!BestResponse::is_best_response(
            &pd, 0, &coop, &defect, 1e-8
        ));
    }

    #[test]
    fn test_best_response_dynamics_convergence() {
        let coord = NormalFormGame::coordination();
        let init = (MixedStrategy::uniform(2), MixedStrategy::uniform(2));
        let result = BestResponse::best_response_dynamics(&coord, init, 100, 1e-10);
        assert!(result.is_some());
    }

    #[test]
    fn test_fictitious_play_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        let (s0, s1) = BestResponse::fictitious_play(&pd, 1000);
        // Should converge to (Defect, Defect)
        assert!(s0.probabilities[1] > 0.9);
        assert!(s1.probabilities[1] > 0.9);
    }

    #[test]
    fn test_fictitious_play_rps() {
        let rps = NormalFormGame::rock_paper_scissors();
        let (s0, s1) = BestResponse::fictitious_play(&rps, 10000);
        // Should converge to approximately uniform
        for p in s0.probabilities.iter() {
            assert!((p - (1.0 / 3.0)).abs() < 0.15, "prob: {}", p);
        }
    }

    #[test]
    fn test_best_response_bos() {
        let bos = NormalFormGame::battle_of_the_sexes();
        let s1 = MixedStrategy::new(vec![1.0, 0.0]);
        let br = BestResponse::find_pure(&bos, 0, &s1);
        // Player 0 best responds to player 1 playing A → A (action 0)
        assert_eq!(br.actions, vec![0]);
    }

    #[test]
    fn test_best_response_matching_pennies() {
        let mp = NormalFormGame::matching_pennies();
        let heads = MixedStrategy::new(vec![1.0, 0.0]);
        let br = BestResponse::find_pure(&mp, 1, &heads);
        // If player 0 plays Heads, player 1 wants to mismatch: Tails (action 1)
        assert_eq!(br.actions, vec![1]);
    }
}
