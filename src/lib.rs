//! # lau-noether-agents
//!
//! Noether's theorem for agent systems — every symmetry of the agent dynamics
//! yields a conserved quantity, and every conserved quantity yields a symmetry.
//!
//! Core principle: if the Lagrangian L(q, q̇, t) is invariant under a
//! transformation, then there exists a conserved quantity Q = Σ p_i δq_i.

pub mod lagrangian;
pub mod noether;
pub mod discrete;
pub mod symmetry;
pub mod charge;
pub mod fleet;
pub mod conservation;
pub mod broken;
pub mod hamiltonian;
pub mod cudaclaw;

pub use lagrangian::*;
pub use noether::*;
pub use discrete::*;
pub use symmetry::*;
pub use charge::*;
pub use fleet::*;
pub use conservation::*;
pub use broken::*;
pub use hamiltonian::*;
pub use cudaclaw::*;
