//! Strategy representations: pure and mixed.
//!
//! A **pure strategy** selects a single action deterministically.
//! A **mixed strategy** randomizes over actions according to a probability
//! distribution.
//!
//! # Formal Definition
//!
//! For a player with action set A = {a₁, a₂, ..., aₙ}, a mixed strategy
//! σ is a probability distribution: σ(aᵢ) ≥ 0 and Σᵢ σ(aᵢ) = 1.
//!
//! The **support** of σ is supp(σ) = {aᵢ : σ(aᵢ) > 0}.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

/// A pure strategy: play exactly one action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PureStrategy {
    /// Index of the chosen action.
    pub action: usize,
}

impl PureStrategy {
    /// Create a pure strategy that plays `action`.
    pub fn new(action: usize) -> Self {
        PureStrategy { action }
    }

    /// Convert to a mixed strategy (degenerate distribution).
    pub fn to_mixed(&self, num_actions: usize) -> MixedStrategy {
        let mut probs = vec![0.0; num_actions];
        probs[self.action] = 1.0;
        MixedStrategy {
            probabilities: probs,
        }
    }
}

impl fmt::Display for PureStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pure({})", self.action)
    }
}

/// A mixed strategy: a probability distribution over actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedStrategy {
    /// Probability assigned to each action. Must sum to 1.0.
    pub probabilities: Vec<f64>,
}

impl MixedStrategy {
    /// Create a mixed strategy from a probability vector.
    ///
    /// # Panics
    /// Panics if probabilities don't sum to ~1.0 or contain negatives.
    pub fn new(probabilities: Vec<f64>) -> Self {
        let sum: f64 = probabilities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "probabilities must sum to 1.0, got {}",
            sum
        );
        for (i, &p) in probabilities.iter().enumerate() {
            assert!(p >= -1e-10, "probability {} is negative: {}", i, p);
        }
        // Clamp small negatives to zero
        let probabilities = probabilities.into_iter().map(|p| p.max(0.0)).collect();
        MixedStrategy { probabilities }
    }

    /// Create a uniform mixed strategy over `n` actions.
    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        MixedStrategy {
            probabilities: vec![p; n],
        }
    }

    /// Get the support set (indices with non-zero probability).
    pub fn support(&self) -> Vec<usize> {
        self.probabilities
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 1e-12)
            .map(|(i, _)| i)
            .collect()
    }

    /// Size of the support set.
    pub fn support_size(&self) -> usize {
        self.support().len()
    }

    /// Number of actions.
    pub fn num_actions(&self) -> usize {
        self.probabilities.len()
    }

    /// Probability of playing action `i`.
    pub fn prob(&self, i: usize) -> f64 {
        self.probabilities[i]
    }

    /// Check if this is a pure strategy (degenerate mixed).
    pub fn is_pure(&self) -> bool {
        self.probabilities.iter().any(|&p| (p - 1.0).abs() < 1e-10)
    }

    /// Get the pure action if this is a pure strategy.
    pub fn as_pure(&self) -> Option<usize> {
        if self.is_pure() {
            Some(
                self.probabilities
                    .iter()
                    .position(|&p| (p - 1.0).abs() < 1e-10)
                    .unwrap(),
            )
        } else {
            None
        }
    }

    /// Entropy of the mixed strategy (measure of randomness).
    pub fn entropy(&self) -> f64 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Create a relaxed mixed strategy (tolerant of small numerical errors).
    pub fn relaxed(probabilities: Vec<f64>) -> Self {
        let sum: f64 = probabilities.iter().sum();
        let probabilities = if (sum - 1.0).abs() > 1e-10 && sum > 0.0 {
            probabilities.into_iter().map(|p| p / sum).collect()
        } else {
            probabilities.into_iter().map(|p| p.max(0.0)).collect()
        };
        MixedStrategy { probabilities }
    }
}

impl PartialEq for MixedStrategy {
    fn eq(&self, other: &Self) -> bool {
        if self.probabilities.len() != other.probabilities.len() {
            return false;
        }
        self.probabilities
            .iter()
            .zip(other.probabilities.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for MixedStrategy {}

impl Hash for MixedStrategy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash rounded values for consistency
        for p in &self.probabilities {
            ((p * 1e10) as i64).hash(state);
        }
    }
}

impl fmt::Display for MixedStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let probs: Vec<String> = self
            .probabilities
            .iter()
            .map(|p| format!("{:.4}", p))
            .collect();
        write!(f, "Mix([{}])", probs.join(", "))
    }
}

/// Unified strategy type: either pure or mixed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Strategy {
    /// A pure strategy selecting a single action.
    Pure(PureStrategy),
    /// A mixed strategy with probabilities over actions.
    Mixed(MixedStrategy),
}

impl Strategy {
    /// Create a pure strategy.
    pub fn pure(action: usize) -> Self {
        Strategy::Pure(PureStrategy::new(action))
    }

    /// Create a mixed strategy.
    pub fn mixed(probabilities: Vec<f64>) -> Self {
        Strategy::Mixed(MixedStrategy::new(probabilities))
    }

    /// Create a uniform random strategy.
    pub fn uniform(n: usize) -> Self {
        Strategy::Mixed(MixedStrategy::uniform(n))
    }

    /// Get the effective probability vector (size `num_actions`).
    pub fn to_probabilities(&self, num_actions: usize) -> Vec<f64> {
        match self {
            Strategy::Pure(p) => p.to_mixed(num_actions).probabilities,
            Strategy::Mixed(m) => m.probabilities.clone(),
        }
    }

    /// Get support set.
    pub fn support(&self) -> Vec<usize> {
        match self {
            Strategy::Pure(p) => vec![p.action],
            Strategy::Mixed(m) => m.support(),
        }
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Strategy::Pure(p) => write!(f, "{}", p),
            Strategy::Mixed(m) => write!(f, "{}", m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_strategy() {
        let s = PureStrategy::new(2);
        assert_eq!(s.action, 2);
        assert_eq!(format!("{}", s), "Pure(2)");
    }

    #[test]
    fn test_pure_to_mixed() {
        let pure = PureStrategy::new(1);
        let mixed = pure.to_mixed(3);
        assert_eq!(mixed.probabilities, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_mixed_strategy_creation() {
        let m = MixedStrategy::new(vec![0.3, 0.7]);
        assert_eq!(m.num_actions(), 2);
        assert!((m.prob(0) - 0.3).abs() < 1e-10);
        assert!((m.prob(1) - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_support() {
        let m = MixedStrategy::new(vec![0.0, 0.6, 0.4]);
        assert_eq!(m.support(), vec![1, 2]);
        assert_eq!(m.support_size(), 2);
    }

    #[test]
    fn test_mixed_is_pure() {
        let pure_m = MixedStrategy::new(vec![1.0, 0.0]);
        assert!(pure_m.is_pure());
        assert_eq!(pure_m.as_pure(), Some(0));

        let mixed_m = MixedStrategy::new(vec![0.5, 0.5]);
        assert!(!mixed_m.is_pure());
        assert_eq!(mixed_m.as_pure(), None);
    }

    #[test]
    fn test_uniform_strategy() {
        let u = MixedStrategy::uniform(4);
        assert_eq!(u.probabilities.len(), 4);
        for p in &u.probabilities {
            assert!((p - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_entropy() {
        let pure = MixedStrategy::new(vec![1.0]);
        assert!(pure.entropy().abs() < 1e-10);

        let uniform = MixedStrategy::uniform(2);
        assert!((uniform.entropy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_strategy_enum() {
        let s = Strategy::pure(0);
        assert!(matches!(s, Strategy::Pure(_)));

        let m = Strategy::mixed(vec![0.5, 0.5]);
        assert!(matches!(m, Strategy::Mixed(_)));
    }

    #[test]
    fn test_strategy_to_probabilities() {
        let s = Strategy::pure(1);
        let probs = s.to_probabilities(3);
        assert_eq!(probs, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_mixed_equality() {
        let m1 = MixedStrategy::new(vec![0.5, 0.5]);
        let m2 = MixedStrategy::new(vec![0.5, 0.5]);
        assert_eq!(m1, m2);
    }

    #[test]
    #[should_panic]
    fn test_invalid_probabilities() {
        MixedStrategy::new(vec![0.3, 0.3]);
    }

    #[test]
    fn test_relaxed_normalizes() {
        let m = MixedStrategy::relaxed(vec![0.3, 0.3]);
        let sum: f64 = m.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
