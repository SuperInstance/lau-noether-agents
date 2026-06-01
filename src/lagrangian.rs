//! Lagrangian mechanics for agent systems.
//!
//! The agent state space has generalized coordinates q (positions) and
//! generalized velocities q̇. The Lagrangian is L(q, q̇, t) = T(q, q̇) - V(q),
//! where T is kinetic energy and V is potential energy.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

/// Agent state in generalized coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Generalized coordinates (positions).
    pub q: DVector<f64>,
    /// Generalized velocities.
    pub qdot: DVector<f64>,
}

impl AgentState {
    pub fn new(q: DVector<f64>, qdot: DVector<f64>) -> Self {
        assert_eq!(q.nrows(), qdot.nrows(), "q and qdot must have same dimension");
        Self { q, qdot }
    }

    /// Number of generalized coordinates.
    pub fn dim(&self) -> usize {
        self.q.nrows()
    }

    /// Conjugate momenta p = ∂L/∂q̇ using a mass matrix: p = M q̇.
    pub fn momenta(&self, mass_matrix: &DMatrix<f64>) -> DVector<f64> {
        mass_matrix * &self.qdot
    }
}

/// A Lagrangian system defined by kinetic and potential energy.
pub trait LagrangianSystem: Send + Sync {
    /// Kinetic energy T(q, q̇).
    fn kinetic_energy(&self, state: &AgentState) -> f64;

    /// Potential energy V(q).
    fn potential_energy(&self, state: &AgentState) -> f64;

    /// The Lagrangian L = T - V.
    fn lagrangian(&self, state: &AgentState) -> f64 {
        self.kinetic_energy(state) - self.potential_energy(state)
    }

    /// Generalized force: -∂V/∂q (computed via finite differences).
    fn generalized_force(&self, state: &AgentState) -> DVector<f64> {
        let n = state.dim();
        let eps = 1e-8;
        let mut force = DVector::zeros(n);
        for i in 0..n {
            let mut q_plus = state.q.clone();
            let mut q_minus = state.q.clone();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            let state_plus = AgentState::new(q_plus, state.qdot.clone());
            let state_minus = AgentState::new(q_minus, state.qdot.clone());
            force[i] = -(self.potential_energy(&state_plus) - self.potential_energy(&state_minus)) / (2.0 * eps);
        }
        force
    }

    /// Mass matrix M(q) such that p = M q̇ and T = ½ q̇ᵀ M q̇.
    /// Default: identity matrix.
    fn mass_matrix(&self, state: &AgentState) -> DMatrix<f64> {
        DMatrix::identity(state.dim(), state.dim())
    }
}

/// A simple Lagrangian system with a diagonal mass matrix and a potential function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleLagrangian {
    /// Diagonal mass entries.
    pub masses: DVector<f64>,
}

impl SimpleLagrangian {
    pub fn new(masses: Vec<f64>) -> Self {
        Self {
            masses: DVector::from_vec(masses),
        }
    }

    pub fn uniform(n: usize, mass: f64) -> Self {
        Self {
            masses: DVector::from_element(n, mass),
        }
    }
}

impl LagrangianSystem for SimpleLagrangian {
    fn kinetic_energy(&self, state: &AgentState) -> f64 {
        let mut t = 0.0;
        for i in 0..state.dim() {
            t += 0.5 * self.masses[i] * state.qdot[i] * state.qdot[i];
        }
        t
    }

    fn potential_energy(&self, _state: &AgentState) -> f64 {
        0.0 // Free particle by default
    }

    fn mass_matrix(&self, state: &AgentState) -> DMatrix<f64> {
        DMatrix::from_diagonal(&self.masses.rows(0, state.dim().min(self.masses.nrows())))
    }
}

/// A Lagrangian system with a harmonic potential V = ½ Σ k_i (q_i - q0_i)².
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicLagrangian {
    pub masses: DVector<f64>,
    pub spring_constants: DVector<f64>,
    pub equilibrium: DVector<f64>,
}

impl HarmonicLagrangian {
    pub fn new(masses: Vec<f64>, spring_constants: Vec<f64>, equilibrium: Vec<f64>) -> Self {
        assert_eq!(masses.len(), spring_constants.len());
        assert_eq!(masses.len(), equilibrium.len());
        Self {
            masses: DVector::from_vec(masses),
            spring_constants: DVector::from_vec(spring_constants),
            equilibrium: DVector::from_vec(equilibrium),
        }
    }

    pub fn uniform(n: usize, mass: f64, k: f64) -> Self {
        Self {
            masses: DVector::from_element(n, mass),
            spring_constants: DVector::from_element(n, k),
            equilibrium: DVector::zeros(n),
        }
    }
}

impl LagrangianSystem for HarmonicLagrangian {
    fn kinetic_energy(&self, state: &AgentState) -> f64 {
        let mut t = 0.0;
        for i in 0..state.dim() {
            t += 0.5 * self.masses[i] * state.qdot[i] * state.qdot[i];
        }
        t
    }

    fn potential_energy(&self, state: &AgentState) -> f64 {
        let mut v = 0.0;
        for i in 0..state.dim() {
            let dq = state.q[i] - self.equilibrium[i];
            v += 0.5 * self.spring_constants[i] * dq * dq;
        }
        v
    }

    fn mass_matrix(&self, state: &AgentState) -> DMatrix<f64> {
        DMatrix::from_diagonal(&self.masses.rows(0, state.dim().min(self.masses.nrows())))
    }
}

/// Euler-Lagrange equation residual: d/dt(∂L/∂q̇) - ∂L/∂q = 0.
/// For T = ½ q̇ᵀ M q̇, this gives M q̈ + ∂V/∂q = 0.
pub fn euler_lagrange_residual(
    system: &dyn LagrangianSystem,
    state: &AgentState,
    qddot: &DVector<f64>,
) -> DVector<f64> {
    let m = system.mass_matrix(state);
    let force = system.generalized_force(state);
    &m * qddot - force
}

/// Total energy E = T + V (conserved when L has no explicit time dependence).
pub fn total_energy(system: &dyn LagrangianSystem, state: &AgentState) -> f64 {
    system.kinetic_energy(state) + system.potential_energy(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_creation() {
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![0.5, 0.3]));
        assert_eq!(state.dim(), 2);
        assert_eq!(state.q[0], 1.0);
        assert_eq!(state.qdot[1], 0.3);
    }

    #[test]
    fn test_simple_kinetic_energy() {
        let sys = SimpleLagrangian::uniform(2, 2.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![3.0, 4.0]));
        // T = ½ * 2 * 9 + ½ * 2 * 16 = 9 + 16 = 25
        assert!((sys.kinetic_energy(&state) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_lagrangian_free_particle() {
        let sys = SimpleLagrangian::uniform(1, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![2.0]));
        // L = T - V = 2 - 0 = 2
        assert!((sys.lagrangian(&state) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_potential() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 4.0);
        let state = AgentState::new(DVector::from_vec(vec![3.0]), DVector::from_vec(vec![0.0]));
        // V = ½ * 4 * 9 = 18
        assert!((sys.potential_energy(&state) - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_momenta() {
        let sys = SimpleLagrangian::uniform(2, 3.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![1.0, 2.0]));
        let m = sys.mass_matrix(&state);
        let p = state.momenta(&m);
        assert!((p[0] - 3.0).abs() < 1e-10);
        assert!((p[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_energy() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![2.0]), DVector::from_vec(vec![3.0]));
        // T = ½ * 1 * 9 = 4.5, V = ½ * 1 * 4 = 2, E = 6.5
        assert!((total_energy(&sys, &state) - 6.5).abs() < 1e-10);
    }
}
