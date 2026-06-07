//! # nash-finder
//!
//! Nash equilibrium computation for agent strategic interactions.
//!
//! This crate provides tools for finding Nash equilibria in normal-form games,
//! from classic 2×2 games (Prisoner's Dilemma, Battle of the Sexes) to N-player
//! polymatrix games. It combines support enumeration with best-response dynamics
//! and iterated dominance elimination.
//!
//! ## Module Layout
//!
//! - [`game`] — Normal-form game definitions
//! - [`strategy`] — Pure and mixed strategy representations
//! - [`best_response`] — Best-response computation
//! - [`support`] — Support enumeration for 2-player games
//! - [`equilibrium`] — Verified equilibrium types
//! - [`nash`] — Unified solver combining all techniques

pub mod best_response;
pub mod equilibrium;
pub mod game;
pub mod nash;
pub mod strategy;
pub mod support;

pub use best_response::BestResponse;
pub use equilibrium::NashEquilibrium;
pub use game::NormalFormGame;
pub use nash::NashSolver;
pub use strategy::{MixedStrategy, PureStrategy, Strategy};
pub use support::SupportEnumeration;
