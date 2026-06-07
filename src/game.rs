//! Normal-form game definitions.
//!
//! A normal-form (strategic-form) game is defined by:
//! - A set of N players
//! - A strategy set for each player
//! - A payoff function mapping strategy profiles to utility vectors
//!
//! This module supports both 2-player bimatrix games and N-player
//! polymatrix games where payoffs decompose into pairwise interactions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A normal-form game with N players, strategy sets, and payoff matrices.
///
/// # Type Parameters
/// In this implementation, payoff data is stored as a flat vector of matrices.
/// For a 2-player game, `payoffs[0]` is the row player's matrix and
/// `payoffs[1]` is the column player's matrix. Each matrix has dimensions
/// `[rows × cols]` stored in row-major order.
///
/// # Examples
/// ```
/// use nash_finder::game::NormalFormGame;
///
/// let pd = NormalFormGame::bimatrix(
///     vec![
///         vec![(-1.0, -1.0), (-3.0, 0.0)],
///         vec![ (0.0, -3.0), (-2.0, -2.0)],
///     ],
/// );
/// assert_eq!(pd.num_players(), 2);
/// assert_eq!(pd.num_strategies(0), 2);
/// ```

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalFormGame {
    /// Number of players.
    num_players: usize,
    /// Number of strategies available to each player.
    strategy_counts: Vec<usize>,
    /// Payoff matrices. For 2-player: `payoffs[player][row * cols + col]`.
    /// For N-player polymatrix: pairwise payoff components.
    payoffs: Vec<Vec<f64>>,
}

impl NormalFormGame {
    /// Create a 2-player bimatrix game.
    ///
    /// The `payoff_matrix` is indexed as `[row_strategy][col_strategy]`,
    /// with each entry being `(row_player_payoff, col_player_payoff)`.
    ///
    /// # Panics
    /// Panics if the matrix is empty or rows have inconsistent lengths.
    pub fn bimatrix(payoff_matrix: Vec<Vec<(f64, f64)>>) -> Self {
        let rows = payoff_matrix.len();
        assert!(rows > 0, "payoff matrix must have at least one row");
        let cols = payoff_matrix[0].len();
        assert!(cols > 0, "payoff matrix must have at least one column");

        for row in &payoff_matrix {
            assert_eq!(
                row.len(),
                cols,
                "all rows must have the same number of columns"
            );
        }

        let mut row_payoffs = Vec::with_capacity(rows * cols);
        let mut col_payoffs = Vec::with_capacity(rows * cols);

        for row in &payoff_matrix {
            for (r_pay, c_pay) in row {
                row_payoffs.push(*r_pay);
                col_payoffs.push(*c_pay);
            }
        }

        NormalFormGame {
            num_players: 2,
            strategy_counts: vec![rows, cols],
            payoffs: vec![row_payoffs, col_payoffs],
        }
    }

    /// Create an N-player game from individual payoff tables.
    ///
    /// Each player has a flat payoff vector indexed by the joint strategy profile.
    /// The profile index is computed in mixed-endian order:
    /// `profile_index = p0_action * (s1 * s2 * ...) + p1_action * (s2 * s3 * ...) + ...`
    ///
    /// # Panics
    /// Panics if lengths are inconsistent.
    pub fn n_player(strategy_counts: Vec<usize>, payoffs: Vec<Vec<f64>>) -> Self {
        let num_players = strategy_counts.len();
        assert_eq!(
            payoffs.len(),
            num_players,
            "must provide payoffs for each player"
        );

        let total_profiles: usize = strategy_counts.iter().product();
        for (i, p) in payoffs.iter().enumerate() {
            assert_eq!(
                p.len(),
                total_profiles,
                "player {} payoffs must have {} entries, got {}",
                i,
                total_profiles,
                p.len()
            );
        }

        NormalFormGame {
            num_players,
            strategy_counts,
            payoffs,
        }
    }

    /// Number of players in the game.
    pub fn num_players(&self) -> usize {
        self.num_players
    }

    /// Number of strategies available to player `player`.
    pub fn num_strategies(&self, player: usize) -> usize {
        self.strategy_counts[player]
    }

    /// Strategy counts for all players.
    pub fn strategy_counts(&self) -> &[usize] {
        &self.strategy_counts
    }

    /// Get the payoff for `player` when the strategy profile is given by
    /// `actions` (one action index per player).
    ///
    /// # Panics
    /// Panics if `actions` has wrong length or out-of-range indices.
    pub fn payoff(&self, player: usize, actions: &[usize]) -> f64 {
        assert_eq!(actions.len(), self.num_players);
        let idx = self.profile_index(actions);
        self.payoffs[player][idx]
    }

    /// Get the payoff matrix slice for a 2-player game.
    ///
    /// Returns the flat payoff vector for `player`.
    pub fn payoff_slice(&self, player: usize) -> &[f64] {
        &self.payoffs[player]
    }

    /// Get the full payoff data (read-only).
    pub fn payoffs(&self) -> &[Vec<f64>] {
        &self.payoffs
    }

    /// Compute the linear index for a strategy profile.
    pub fn profile_index(&self, actions: &[usize]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for p in (0..self.num_players).rev() {
            idx += actions[p] * stride;
            stride *= self.strategy_counts[p];
        }
        idx
    }

    /// Get payoff for player 0 at (row, col) in a 2-player game.
    pub fn row_payoff(&self, row: usize, col: usize) -> f64 {
        let cols = self.strategy_counts[1];
        self.payoffs[0][row * cols + col]
    }

    /// Get payoff for player 1 at (row, col) in a 2-player game.
    pub fn col_payoff(&self, row: usize, col: usize) -> f64 {
        let cols = self.strategy_counts[1];
        self.payoffs[1][row * cols + col]
    }

    /// Iterate over all pure strategy profiles (as action tuples).
    pub fn all_profiles(&self) -> ProfileIter<'_> {
        ProfileIter {
            game: self,
            current: vec![0; self.num_players],
            done: false,
        }
    }

    /// Build a Prisoner's Dilemma game.
    ///
    /// Standard payoffs: (T, R, P, S) with T > R > P > S.
    /// Default: T=0, R=-1, P=-2, S=-3.
    pub fn prisoners_dilemma() -> Self {
        Self::bimatrix(vec![
            vec![(-1.0, -1.0), (-3.0, 0.0)],
            vec![(0.0, -3.0), (-2.0, -2.0)],
        ])
    }

    /// Build a Battle of the Sexes game.
    pub fn battle_of_the_sexes() -> Self {
        Self::bimatrix(vec![
            vec![(3.0, 2.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (2.0, 3.0)],
        ])
    }

    /// Build a Matching Pennies game.
    pub fn matching_pennies() -> Self {
        Self::bimatrix(vec![
            vec![(1.0, -1.0), (-1.0, 1.0)],
            vec![(-1.0, 1.0), (1.0, -1.0)],
        ])
    }

    /// Build a Rock-Paper-Scissors game.
    pub fn rock_paper_scissors() -> Self {
        //       Rock        Paper       Scissors
        // Rock    (0,0)     (-1,1)      (1,-1)
        // Paper   (1,-1)    (0,0)       (-1,1)
        // Scissors(-1,1)    (1,-1)      (0,0)
        Self::bimatrix(vec![
            vec![(0.0, 0.0), (-1.0, 1.0), (1.0, -1.0)],
            vec![(1.0, -1.0), (0.0, 0.0), (-1.0, 1.0)],
            vec![(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)],
        ])
    }

    /// Build a Coordination game with two pure Nash equilibria.
    pub fn coordination() -> Self {
        Self::bimatrix(vec![
            vec![(1.0, 1.0), (0.0, 0.0)],
            vec![(0.0, 0.0), (1.0, 1.0)],
        ])
    }

    /// Build a Stag Hunt game.
    pub fn stag_hunt() -> Self {
        Self::bimatrix(vec![
            vec![(4.0, 4.0), (0.0, 3.0)],
            vec![(3.0, 0.0), (3.0, 3.0)],
        ])
    }

    /// Build a Chicken (Hawk-Dove) game.
    pub fn chicken() -> Self {
        Self::bimatrix(vec![
            vec![(0.0, 0.0), (-1.0, 1.0)],
            vec![(1.0, -1.0), (-10.0, -10.0)],
        ])
    }

    /// Compute expected payoff for `player` given mixed strategies for all players.
    /// `mixed` contains one probability vector per player.
    pub fn expected_payoff(&self, player: usize, mixed: &[Vec<f64>]) -> f64 {
        assert_eq!(mixed.len(), self.num_players);
        let mut total = 0.0;
        for profile in self.all_profiles() {
            let mut prob = 1.0;
            for (p, &action) in profile.iter().enumerate() {
                prob *= mixed[p][action];
            }
            total += prob * self.payoff(player, &profile);
        }
        total
    }

    /// Return a new game with dominated strategies eliminated for the given player.
    ///
    /// A pure strategy `s` is strictly dominated if another pure strategy `t`
    /// yields strictly higher payoff against every opponent profile.
    pub fn eliminate_dominated(&self, player: usize) -> Vec<usize> {
        let n = self.strategy_counts[player];
        let mut dominated = Vec::new();

        'outer: for s in 0..n {
            for t in 0..n {
                if s == t {
                    continue;
                }
                if self.strictly_dominates(player, t, s) {
                    dominated.push(s);
                    continue 'outer;
                }
            }
        }
        dominated
    }

    /// Check if strategy `dominator` strictly dominates strategy `dominated` for `player`.
    pub fn strictly_dominates(&self, player: usize, dominator: usize, dominated: usize) -> bool {
        // Proper implementation for 2-player
        if self.num_players == 2 {
            let opp = 1 - player;
            let opp_strats = self.strategy_counts[opp];
            for opp_action in 0..opp_strats {
                let mut profile_dom = vec![0; 2];
                let mut profile_sub = vec![0; 2];
                profile_dom[player] = dominator;
                profile_dom[opp] = opp_action;
                profile_sub[player] = dominated;
                profile_sub[opp] = opp_action;

                if self.payoff(player, &profile_dom) <= self.payoff(player, &profile_sub) {
                    return false;
                }
            }
            return true;
        }

        // General N-player: iterate all opponent profiles
        let mut profile = vec![0usize; self.num_players];
        Self::check_dominance_recursive(self, player, dominator, dominated, &mut profile, 0)
    }

    fn check_dominance_recursive(
        game: &NormalFormGame,
        player: usize,
        dominator: usize,
        dominated: usize,
        profile: &mut Vec<usize>,
        dim: usize,
    ) -> bool {
        if dim == game.num_players {
            let mut p_dom = profile.clone();
            p_dom[player] = dominator;
            let mut p_sub = profile.clone();
            p_sub[player] = dominated;
            return game.payoff(player, &p_dom) > game.payoff(player, &p_sub);
        }
        if dim == player {
            // Skip this dimension — we set it ourselves
            return Self::check_dominance_recursive(
                game,
                player,
                dominator,
                dominated,
                profile,
                dim + 1,
            );
        }
        for a in 0..game.strategy_counts[dim] {
            profile[dim] = a;
            if !Self::check_dominance_recursive(
                game,
                player,
                dominator,
                dominated,
                profile,
                dim + 1,
            ) {
                return false;
            }
        }
        true
    }
}

/// Iterator over all pure strategy profiles.
pub struct ProfileIter<'a> {
    game: &'a NormalFormGame,
    current: Vec<usize>,
    done: bool,
}

impl<'a> Iterator for ProfileIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let result = self.current.clone();
        // Increment
        for p in (0..self.game.num_players).rev() {
            self.current[p] += 1;
            if self.current[p] < self.game.strategy_counts[p] {
                return Some(result);
            }
            self.current[p] = 0;
        }
        self.done = true;
        Some(result)
    }
}

impl fmt::Display for NormalFormGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.num_players == 2 {
            writeln!(
                f,
                "2-player bimatrix game ({}×{})",
                self.strategy_counts[0], self.strategy_counts[1]
            )?;
            for r in 0..self.strategy_counts[0] {
                for c in 0..self.strategy_counts[1] {
                    write!(
                        f,
                        "({:.1}, {:.1}) ",
                        self.row_payoff(r, c),
                        self.col_payoff(r, c)
                    )?;
                }
                writeln!(f)?;
            }
        } else {
            writeln!(f, "{}-player game", self.num_players)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bimatrix_creation() {
        let g = NormalFormGame::bimatrix(vec![
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(5.0, 6.0), (7.0, 8.0)],
        ]);
        assert_eq!(g.num_players(), 2);
        assert_eq!(g.num_strategies(0), 2);
        assert_eq!(g.num_strategies(1), 2);
    }

    #[test]
    fn test_payoff_access() {
        let g = NormalFormGame::bimatrix(vec![
            vec![(1.0, 2.0), (3.0, 4.0)],
            vec![(5.0, 6.0), (7.0, 8.0)],
        ]);
        assert_eq!(g.row_payoff(0, 0), 1.0);
        assert_eq!(g.col_payoff(0, 0), 2.0);
        assert_eq!(g.row_payoff(1, 1), 7.0);
        assert_eq!(g.col_payoff(1, 1), 8.0);
    }

    #[test]
    fn test_profile_iteration() {
        let g = NormalFormGame::bimatrix(vec![
            vec![(0.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 3.0)],
        ]);
        let profiles: Vec<_> = g.all_profiles().collect();
        assert_eq!(profiles.len(), 4);
        assert_eq!(profiles[0], vec![0, 0]);
        assert_eq!(profiles[3], vec![1, 1]);
    }

    #[test]
    fn test_prisoners_dilemma() {
        let pd = NormalFormGame::prisoners_dilemma();
        assert_eq!(pd.row_payoff(0, 0), -1.0);
        assert_eq!(pd.col_payoff(1, 0), -3.0);
    }

    #[test]
    fn test_n_player_game() {
        // 2-player, 2-strategies each — should be equivalent to bimatrix
        let g = NormalFormGame::n_player(
            vec![2, 2],
            vec![
                vec![-1.0, -3.0, 0.0, -2.0], // row player
                vec![-1.0, 0.0, -3.0, -2.0], // col player
            ],
        );
        assert_eq!(g.num_players(), 2);
        assert_eq!(g.payoff(0, &[0, 0]), -1.0);
        assert_eq!(g.payoff(1, &[1, 0]), -3.0);
    }

    #[test]
    fn test_rock_paper_scissors() {
        let rps = NormalFormGame::rock_paper_scissors();
        assert_eq!(rps.num_strategies(0), 3);
        assert_eq!(rps.num_strategies(1), 3);
        assert_eq!(rps.row_payoff(0, 1), -1.0); // Rock vs Paper
        assert_eq!(rps.col_payoff(2, 1), -1.0); // Scissors vs Paper: col player loses
    }

    #[test]
    fn test_expected_payoff() {
        let pd = NormalFormGame::prisoners_dilemma();
        let mixed = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let ep0 = pd.expected_payoff(0, &mixed);
        // Expected: 0.25*(-1) + 0.25*(-3) + 0.25*(0) + 0.25*(-2) = -1.5
        assert!((ep0 - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_strictly_dominates_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        // In PD, "Defect" (action 1) strictly dominates "Cooperate" (action 0)
        assert!(pd.strictly_dominates(0, 1, 0));
        assert!(!pd.strictly_dominates(0, 0, 1));
    }

    #[test]
    fn test_eliminate_dominated_pd() {
        let pd = NormalFormGame::prisoners_dilemma();
        let dom0 = pd.eliminate_dominated(0);
        assert_eq!(dom0, vec![0]); // Cooperate is dominated
    }

    #[test]
    fn test_display() {
        let g = NormalFormGame::coordination();
        let s = format!("{}", g);
        assert!(s.contains("2-player"));
    }

    #[test]
    #[should_panic]
    fn test_empty_matrix_panics() {
        NormalFormGame::bimatrix(vec![]);
    }

    #[test]
    fn test_matching_pennies_zero_sum() {
        let mp = NormalFormGame::matching_pennies();
        for profile in mp.all_profiles() {
            let sum = mp.payoff(0, &profile) + mp.payoff(1, &profile);
            assert!((sum).abs() < 1e-10, "Matching pennies must be zero-sum");
        }
    }

    #[test]
    fn test_stag_hunt() {
        let sh = NormalFormGame::stag_hunt();
        assert_eq!(sh.row_payoff(0, 0), 4.0);
        assert_eq!(sh.row_payoff(1, 1), 3.0);
    }
}
