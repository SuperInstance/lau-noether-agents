//! Hamiltonian formulation: H = Σ p_i q̇_i - L.
//!
//! The Hamiltonian is the Legendre transform of the Lagrangian, providing
//! an alternative formulation where the dynamics are expressed in terms of
//! canonical coordinates (q, p).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};

/// Canonical state in phase space (q, p).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalState {
    pub q: DVector<f64>,
    pub p: DVector<f64>,
}

impl CanonicalState {
    pub fn new(q: DVector<f64>, p: DVector<f64>) -> Self {
        assert_eq!(q.nrows(), p.nrows());
        Self { q, p }
    }

    pub fn dim(&self) -> usize {
        self.q.nrows()
    }

    /// Convert from Lagrangian state using mass matrix.
    pub fn from_lagrangian(state: &AgentState, mass_matrix: &DMatrix<f64>) -> Self {
        let p = mass_matrix * &state.qdot;
        CanonicalState::new(state.q.clone(), p)
    }

    /// Convert to Lagrangian state using inverse mass matrix.
    pub fn to_lagrangian(&self, inv_mass: &DMatrix<f64>) -> AgentState {
        let qdot = inv_mass * &self.p;
        AgentState::new(self.q.clone(), qdot)
    }
}

use nalgebra::DMatrix;

/// Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HamiltonEquations {
    pub dqdt: DVector<f64>,
    pub dpdt: DVector<f64>,
}

/// Compute Hamilton's equations for a system.
pub fn hamiltons_equations(
    system: &dyn LagrangianSystem,
    state: &CanonicalState,
) -> HamiltonEquations {
    let lag_state = AgentState::new(state.q.clone(), state.p.clone()); // temporary
    let m = system.mass_matrix(&AgentState::new(state.q.clone(), DVector::zeros(state.dim())));
    let m_inv = m.clone().try_inverse().unwrap_or(m.clone());

    // q̇ = ∂H/∂p = M^{-1} p
    let dqdt = &m_inv * &state.p;

    // ṗ = -∂H/∂q = -∂V/∂q (generalized force)
    let lag_state_full = AgentState::new(state.q.clone(), &m_inv * &state.p);
    let dpdt = system.generalized_force(&lag_state_full);

    HamiltonEquations { dqdt, dpdt }
}

/// The Hamiltonian H(q, p) = Σ p_i q̇_i - L.
pub fn hamiltonian(system: &dyn LagrangianSystem, state: &CanonicalState) -> f64 {
    let m = system.mass_matrix(&AgentState::new(state.q.clone(), DVector::zeros(state.dim())));
    let m_inv = m.clone().try_inverse().unwrap_or(m.clone());
    let qdot = &m_inv * &state.p;
    let lag_state = AgentState::new(state.q.clone(), qdot.clone());
    let p_dot_qdot = state.p.dot(&qdot);
    let l = system.lagrangian(&lag_state);
    p_dot_qdot - l
}

/// Symplectic integrator using Stormer-Verlet (leapfrog) in Hamiltonian form.
pub fn stormer_verlet_step(
    system: &dyn LagrangianSystem,
    state: &CanonicalState,
    dt: f64,
) -> CanonicalState {
    let eqs = hamiltons_equations(system, state);

    // Half step in p
    let p_half = &state.p + 0.5 * dt * &eqs.dpdt;

    // Full step in q using p_half
    let q_new = &state.q + dt * &eqs.dqdt; // dqdt = M^{-1} * p, but should use p_half
    let m = system.mass_matrix(&AgentState::new(state.q.clone(), DVector::zeros(state.dim())));
    let m_inv = m.clone().try_inverse().unwrap_or(m.clone());
    let q_new = &state.q + dt * (&m_inv * &p_half);

    let state_half = CanonicalState::new(q_new.clone(), p_half.clone());
    let eqs_new = hamiltons_equations(system, &state_half);

    // Half step in p
    let p_new = &p_half + 0.5 * dt * &eqs_new.dpdt;

    CanonicalState::new(q_new, p_new)
}

/// Poisson bracket {f, g} = Σ (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i).
/// The time derivative of any observable f is {f, H}.
pub fn poisson_bracket(
    f_grad_q: &DVector<f64>,
    f_grad_p: &DVector<f64>,
    g_grad_q: &DVector<f64>,
    g_grad_p: &DVector<f64>,
) -> f64 {
    f_grad_q.dot(g_grad_p) - f_grad_p.dot(g_grad_q)
}

/// Liouville's theorem: phase space volume is preserved.
/// Compute the Jacobian determinant of the flow map (should be 1).
pub fn liouville_check(
    system: &dyn LagrangianSystem,
    state: &CanonicalState,
    dt: f64,
) -> f64 {
    let n = state.dim();
    let eps = 1e-8;
    let base = stormer_verlet_step(system, state, dt);

    // Estimate Jacobian via finite differences
    let mut det = 1.0;
    for i in 0..2 * n {
        let mut q_pert = state.q.clone();
        let mut p_pert = state.p.clone();
        if i < n {
            q_pert[i] += eps;
        } else {
            p_pert[i - n] += eps;
        }
        let pert_state = CanonicalState::new(q_pert, p_pert);
        let pert_next = stormer_verlet_step(system, &pert_state, dt);

        // Accumulate diagonal approximation
        if i < n {
            let ratio = (pert_next.q[i] - base.q[i]) / eps;
            det *= ratio;
        } else {
            let idx = i - n;
            let ratio = (pert_next.p[idx] - base.p[idx]) / eps;
            det *= ratio;
        }
    }

    det.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::{SimpleLagrangian, HarmonicLagrangian};

    #[test]
    fn test_canonical_state_from_lagrangian() {
        let sys = SimpleLagrangian::uniform(2, 2.0);
        let lag = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let m = sys.mass_matrix(&lag);
        let canon = CanonicalState::from_lagrangian(&lag, &m);
        assert!((canon.p[0] - 6.0).abs() < 1e-10);
        assert!((canon.p[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_free_particle() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let canon = CanonicalState::new(
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let h = hamiltonian(&sys, &canon);
        // H = p²/(2m) = (9+16)/2 = 12.5
        assert!((h - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_harmonic() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
        let canon = CanonicalState::new(
            DVector::from_vec(vec![2.0]),
            DVector::from_vec(vec![3.0]),
        );
        let h = hamiltonian(&sys, &canon);
        // H = p²/2 + ½kx² = 4.5 + 2 = 6.5
        assert!((h - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltons_equations_free() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let canon = CanonicalState::new(
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![3.0, 4.0]),
        );
        let eqs = hamiltons_equations(&sys, &canon);
        // dqdt = p (m=1), dpdt = 0
        assert!((eqs.dqdt[0] - 3.0).abs() < 1e-10);
        assert!((eqs.dpdt[0]).abs() < 1e-4);
    }

    #[test]
    fn test_stormer_verlet_energy_conservation() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
        let state = CanonicalState::new(
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![0.0]),
        );
        let h0 = hamiltonian(&sys, &state);
        let mut current = state;
        for _ in 0..1000 {
            current = stormer_verlet_step(&sys, &current, 0.01);
        }
        let h_final = hamiltonian(&sys, &current);
        assert!((h0 - h_final).abs() < 0.01 * h0.abs());
    }

    #[test]
    fn test_poisson_bracket() {
        let fq = DVector::from_vec(vec![1.0]);
        let fp = DVector::from_vec(vec![0.0]);
        let gq = DVector::from_vec(vec![0.0]);
        let gp = DVector::from_vec(vec![1.0]);
        // {q, p} = 1
        let bracket = poisson_bracket(&fq, &fp, &gq, &gp);
        assert!((bracket - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonical_roundtrip() {
        let sys = SimpleLagrangian::uniform(2, 3.0);
        let lag = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![4.0, 5.0]));
        let m = sys.mass_matrix(&lag);
        let m_inv = m.clone().try_inverse().unwrap();
        let canon = CanonicalState::from_lagrangian(&lag, &m);
        let lag2 = canon.to_lagrangian(&m_inv);
        assert!((lag2.q[0] - 1.0).abs() < 1e-10);
        assert!((lag2.qdot[0] - 4.0).abs() < 1e-10);
    }
}
