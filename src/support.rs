//! Support enumeration for 2-player games.
//!
//! # Algorithm
//!
//! For a 2-player game, Nash equilibria have the property that each strategy
//! in a player's support is a best response to the opponent's strategy (and
//! yields equal expected utility).
//!
//! The **support enumeration** algorithm:
//! 1. For each pair of support sets (S₁, S₂) where |S₁| = |S₂| = k
//! 2. Solve the indifference conditions: all actions in S₁ have equal utility
//!    against the opponent's mixture, and vice versa
//! 3. Verify the solution is a valid probability distribution
//! 4. Check that no action outside the support is a profitable deviation
//!
//! # Mathematical Foundation
//!
//! For supports S₁ ⊆ A₁ and S₂ ⊆ A₂, we solve:
//!   u₁(a, σ₂) = u₁(b, σ₂)  for all a, b ∈ S₁
//!   Σ_{j ∈ S₂} σ₂(j) = 1, σ₂(j) > 0
//!
//! And symmetrically for player 2.

use crate::best_response::BestResponse;
use crate::game::NormalFormGame;
use crate::strategy::MixedStrategy;
use serde::{Deserialize, Serialize};

/// Support enumeration solver for 2-player games.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportEnumeration;

impl SupportEnumeration {
    /// Find all Nash equilibria of a 2-player game via support enumeration.
    ///
    /// This enumerates all possible support pairs, solves the linear system
    /// for each, and verifies the result.
    pub fn find_all_equilibria(game: &NormalFormGame) -> Vec<(MixedStrategy, MixedStrategy)> {
        assert_eq!(
            game.num_players(),
            2,
            "support enumeration requires 2-player games"
        );
        let mut equilibria = Vec::new();

        let n0 = game.num_strategies(0);
        let n1 = game.num_strategies(1);

        // Check pure-strategy equilibria first
        for a in 0..n0 {
            for b in 0..n1 {
                let s0 = MixedStrategy::relaxed(
                    (0..n0).map(|i| if i == a { 1.0 } else { 0.0 }).collect(),
                );
                let s1 = MixedStrategy::relaxed(
                    (0..n1).map(|j| if j == b { 1.0 } else { 0.0 }).collect(),
                );
                if Self::verify_equilibrium(game, &s0, &s1, 1e-8) {
                    equilibria.push((s0, s1));
                }
            }
        }

        // Enumerate mixed-strategy equilibria
        let max_support = n0.min(n1);
        for k in 2..=max_support {
            for supp0 in Self::combinations(n0, k) {
                for supp1 in Self::combinations(n1, k) {
                    if let Some((s0, s1)) = Self::solve_support(game, &supp0, &supp1)
                        && Self::verify_equilibrium(game, &s0, &s1, 1e-8)
                    {
                        let is_dup = equilibria
                            .iter()
                            .any(|(existing0, existing1)| existing0 == &s0 && existing1 == &s1);
                        if !is_dup {
                            equilibria.push((s0, s1));
                        }
                    }
                }
            }
        }

        equilibria
    }

    /// Generate all k-combinations of {0, 1, ..., n-1}.
    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        if k == 0 || k > n {
            return vec![];
        }
        let mut result = Vec::new();
        let mut current = Vec::new();
        Self::combinations_helper(n, k, 0, &mut current, &mut result);
        result
    }

    fn combinations_helper(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..n {
            current.push(i);
            Self::combinations_helper(n, k, i + 1, current, result);
            current.pop();
        }
    }

    /// Solve for the mixed strategies given fixed support sets.
    ///
    /// For support S₀ = {a₁, ..., aₖ} and S₁ = {b₁, ..., bₖ}:
    /// - All aᵢ must yield equal expected utility against σ₁
    /// - All bⱼ must yield equal expected utility against σ₀
    /// - Probabilities sum to 1 and are non-negative
    fn solve_support(
        game: &NormalFormGame,
        supp0: &[usize],
        supp1: &[usize],
    ) -> Option<(MixedStrategy, MixedStrategy)> {
        let k = supp0.len();
        if k != supp1.len() || k < 2 {
            return None;
        }

        // Solve for σ₁ (opponent's mixed strategy) that makes player 0 indifferent
        // across all actions in supp0
        //
        // u₀(aᵢ, σ₁) = u₀(a₁, σ₁) for i = 2, ..., k
        // Σ σ₁(j) = 1 for j ∈ supp1
        //
        // This gives us k equations in k unknowns (σ₁ values for j ∈ supp1)

        let sigma1 = Self::solve_indifference(game, 0, supp0, supp1)?;

        // Solve for σ₀ that makes player 1 indifferent
        let sigma0 = Self::solve_indifference(game, 1, supp1, supp0)?;

        // Build full probability vectors
        let n0 = game.num_strategies(0);
        let n1 = game.num_strategies(1);
        let mut p0 = vec![0.0; n0];
        let mut p1 = vec![0.0; n1];

        for (i, &a) in supp0.iter().enumerate() {
            p0[a] = sigma0[i];
        }
        for (j, &b) in supp1.iter().enumerate() {
            p1[b] = sigma1[j];
        }

        // Check non-negativity
        for &p in &p0 {
            if p < -1e-10 {
                return None;
            }
        }
        for &p in &p1 {
            if p < -1e-10 {
                return None;
            }
        }

        // Clamp small negatives
        let p0: Vec<f64> = p0.into_iter().map(|x| x.max(0.0)).collect();
        let p1: Vec<f64> = p1.into_iter().map(|x| x.max(0.0)).collect();

        Some((MixedStrategy::relaxed(p0), MixedStrategy::relaxed(p1)))
    }

    /// Solve the indifference equations for one player.
    ///
    /// Given that `player`'s support is `player_support` and the opponent's
    /// support is `opp_support`, find the opponent's mixing probabilities
    /// that make `player` indifferent across `player_support`.
    fn solve_indifference(
        game: &NormalFormGame,
        player: usize,
        player_support: &[usize],
        opp_support: &[usize],
    ) -> Option<Vec<f64>> {
        let opp = 1 - player;
        let k = opp_support.len();
        let m = player_support.len();

        // Build system: m-1 indifference equations + 1 normalization
        // Variables: σ_opp[j] for j in opp_support
        //
        // For i = 1, ..., m-1:
        //   u_player(player_support[i], σ_opp) = u_player(player_support[0], σ_opp)
        //
        // Σ σ_opp[j] = 1

        let mut aug = vec![vec![0.0; k + 1]; m]; // m equations, k variables + rhs

        // Indifference equations
        for eq in 1..m {
            for (j, &opp_action) in opp_support.iter().enumerate() {
                let mut profile_eq = vec![0; 2];
                profile_eq[player] = player_support[eq];
                profile_eq[opp] = opp_action;
                let u_eq = game.payoff(player, &profile_eq);

                let mut profile0 = vec![0; 2];
                profile0[player] = player_support[0];
                profile0[opp] = opp_action;
                let u0 = game.payoff(player, &profile0);

                aug[eq - 1][j] = u_eq - u0;
            }
            aug[eq - 1][k] = 0.0; // rhs
        }

        // Normalization: Σ σ[j] = 1
        for val in aug[m - 1].iter_mut().take(k) {
            *val = 1.0;
        }
        aug[m - 1][k] = 1.0;

        // Solve via Gaussian elimination
        Self::gauss_solve(&mut aug, k)
    }

    /// Gaussian elimination with partial pivoting.
    #[allow(clippy::needless_range_loop)]
    fn gauss_solve(aug: &mut [Vec<f64>], n_vars: usize) -> Option<Vec<f64>> {
        let n_eqs = aug.len();

        for col in 0..n_vars.min(n_eqs) {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n_eqs {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }

            if max_val < 1e-12 {
                return None; // Singular
            }

            aug.swap(col, max_row);

            // Eliminate below
            for row in (col + 1)..n_eqs {
                let factor = aug[row][col] / aug[col][col];
                for j in col..=n_vars {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back-substitution
        let mut solution = vec![0.0; n_vars];
        for i in (0..n_vars.min(n_eqs)).rev() {
            let mut sum = aug[i][n_vars]; // rhs
            for j in (i + 1)..n_vars {
                sum -= aug[i][j] * solution[j];
            }
            if aug[i][i].abs() < 1e-12 {
                return None;
            }
            solution[i] = sum / aug[i][i];
        }

        Some(solution)
    }

    /// Verify that (s0, s1) is a Nash equilibrium.
    fn verify_equilibrium(
        game: &NormalFormGame,
        s0: &MixedStrategy,
        s1: &MixedStrategy,
        epsilon: f64,
    ) -> bool {
        BestResponse::is_best_response(game, 0, s0, s1, epsilon)
            && BestResponse::is_best_response(game, 1, s1, s0, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prisoners_dilemma_one_ne() {
        let pd = NormalFormGame::prisoners_dilemma();
        let ne = SupportEnumeration::find_all_equilibria(&pd);
        assert_eq!(ne.len(), 1);
        // Should be (Defect, Defect)
        assert!(ne[0].0.probabilities[1] > 0.99);
        assert!(ne[0].1.probabilities[1] > 0.99);
    }

    #[test]
    fn test_coordination_three_ne() {
        let coord = NormalFormGame::coordination();
        let ne = SupportEnumeration::find_all_equilibria(&coord);
        assert_eq!(ne.len(), 3); // 2 pure + 1 mixed
    }

    #[test]
    fn test_battle_of_the_sexes_three_ne() {
        let bos = NormalFormGame::battle_of_the_sexes();
        let ne = SupportEnumeration::find_all_equilibria(&bos);
        assert_eq!(ne.len(), 3); // 2 pure + 1 mixed
    }

    #[test]
    fn test_matching_pennies_one_ne() {
        let mp = NormalFormGame::matching_pennies();
        let ne = SupportEnumeration::find_all_equilibria(&mp);
        assert_eq!(ne.len(), 1);
        // Should be (0.5, 0.5) for both
        assert!((ne[0].0.probabilities[0] - 0.5).abs() < 1e-6);
        assert!((ne[0].1.probabilities[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rock_paper_scissors_one_ne() {
        let rps = NormalFormGame::rock_paper_scissors();
        let ne = SupportEnumeration::find_all_equilibria(&rps);
        assert_eq!(ne.len(), 1);
        // Uniform mixed: (1/3, 1/3, 1/3)
        for i in 0..3 {
            assert!((ne[0].0.probabilities[i] - 1.0 / 3.0).abs() < 1e-6);
            assert!((ne[0].1.probabilities[i] - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_chicken_three_ne() {
        let chicken = NormalFormGame::chicken();
        let ne = SupportEnumeration::find_all_equilibria(&chicken);
        assert_eq!(ne.len(), 3); // 2 pure + 1 mixed
    }

    #[test]
    fn test_stag_hunt_three_ne() {
        let sh = NormalFormGame::stag_hunt();
        let ne = SupportEnumeration::find_all_equilibria(&sh);
        assert_eq!(ne.len(), 3); // (Stag,Stag), (Hare,Hare), mixed
    }

    #[test]
    fn test_combinations() {
        let combos = SupportEnumeration::combinations(4, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6
        assert!(combos.contains(&vec![0, 1]));
        assert!(combos.contains(&vec![2, 3]));
    }

    #[test]
    fn test_gauss_solve_identity() {
        let mut aug = vec![vec![1.0, 0.0, 3.0], vec![0.0, 1.0, 5.0]];
        let sol = SupportEnumeration::gauss_solve(&mut aug, 2);
        assert!(sol.is_some());
        let sol = sol.unwrap();
        assert!((sol[0] - 3.0).abs() < 1e-10);
        assert!((sol[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gauss_solve_singular() {
        let mut aug = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 1.0]];
        let sol = SupportEnumeration::gauss_solve(&mut aug, 2);
        assert!(sol.is_none());
    }

    #[test]
    fn test_coordination_mixed_ne_symmetric() {
        let coord = NormalFormGame::coordination();
        let ne = SupportEnumeration::find_all_equilibria(&coord);
        let mixed_ne: Vec<_> = ne.iter().filter(|(s0, _)| !s0.is_pure()).collect();
        assert_eq!(mixed_ne.len(), 1);
        // In symmetric coordination game, mixed NE is (0.5, 0.5)
        assert!((mixed_ne[0].0.probabilities[0] - 0.5).abs() < 1e-6);
        assert!((mixed_ne[0].1.probabilities[0] - 0.5).abs() < 1e-6);
    }
}
