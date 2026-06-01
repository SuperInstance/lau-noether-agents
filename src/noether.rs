//! Noether's theorem: continuous symmetry → conserved charge.
//!
//! If L is invariant under the infinitesimal transformation q → q + ε δq,
//! then Q = Σ p_i δq_i is conserved.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};

/// An infinitesimal symmetry transformation δq(q).
pub trait SymmetryTransform: Send + Sync {
    /// Name of the symmetry.
    fn name(&self) -> &str;

    /// The infinitesimal variation δq_i(q).
    fn delta_q(&self, state: &AgentState) -> DVector<f64>;

    /// Check if the Lagrangian is invariant under this symmetry (numerically).
    fn verify_invariance(&self, system: &dyn LagrangianSystem, state: &AgentState) -> bool {
        let eps = 1e-6;
        let delta = self.delta_q(state);
        let mut q_shifted = &state.q + eps * &delta;
        let shifted_state = AgentState::new(q_shifted, state.qdot.clone());
        let l_orig = system.lagrangian(state);
        let l_shifted = system.lagrangian(&shifted_state);
        (l_orig - l_shifted).abs() < 1e-4 * (1.0 + l_orig.abs())
    }
}

/// A conserved Noether charge Q = Σ p_i δq_i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoetherCharge {
    /// Name of the charge (derived from symmetry name).
    pub name: String,
    /// The symmetry that generates this charge.
    pub symmetry_name: String,
    /// Numerical value of the charge.
    pub value: f64,
}

impl NoetherCharge {
    /// Compute the Noether charge Q = Σ p_i δq_i for a given symmetry.
    pub fn compute(
        symmetry: &dyn SymmetryTransform,
        system: &dyn LagrangianSystem,
        state: &AgentState,
    ) -> Self {
        let m = system.mass_matrix(state);
        let p = state.momenta(&m);
        let delta = symmetry.delta_q(state);
        let value = p.dot(&delta);
        NoetherCharge {
            name: format!("Q_{}", symmetry.name()),
            symmetry_name: symmetry.name().to_string(),
            value,
        }
    }
}

/// Translation symmetry along coordinate axis i.
pub struct TranslationSymmetry {
    /// Axis index.
    pub axis: usize,
    /// Dimension of the state space.
    pub dim: usize,
}

impl TranslationSymmetry {
    pub fn new(axis: usize, dim: usize) -> Self {
        Self { axis, dim }
    }
}

impl SymmetryTransform for TranslationSymmetry {
    fn name(&self) -> &str {
        "translation"
    }

    fn delta_q(&self, state: &AgentState) -> DVector<f64> {
        let mut delta = DVector::zeros(state.dim());
        if self.axis < state.dim() {
            delta[self.axis] = 1.0;
        }
        delta
    }
}

/// Rotation symmetry in the (i, j) plane.
pub struct RotationSymmetry {
    pub axis_i: usize,
    pub axis_j: usize,
}

impl RotationSymmetry {
    pub fn new(i: usize, j: usize) -> Self {
        Self { axis_i: i, axis_j: j }
    }
}

impl SymmetryTransform for RotationSymmetry {
    fn name(&self) -> &str {
        "rotation"
    }

    fn delta_q(&self, state: &AgentState) -> DVector<f64> {
        let mut delta = DVector::zeros(state.dim());
        if self.axis_i < state.dim() && self.axis_j < state.dim() {
            delta[self.axis_i] = -state.q[self.axis_j];
            delta[self.axis_j] = state.q[self.axis_i];
        }
        delta
    }
}

/// Time translation symmetry → energy conservation.
pub struct TimeTranslationSymmetry;

impl SymmetryTransform for TimeTranslationSymmetry {
    fn name(&self) -> &str {
        "time_translation"
    }

    fn delta_q(&self, state: &AgentState) -> DVector<f64> {
        state.qdot.clone()
    }
}

/// Noether's theorem: check that a symmetry implies a conserved charge.
pub fn noether_verify(
    symmetry: &dyn SymmetryTransform,
    system: &dyn LagrangianSystem,
    trajectory: &[AgentState],
) -> NoetherVerification {
    let invariant = symmetry.verify_invariance(system, &trajectory[0]);
    let charges: Vec<f64> = trajectory
        .iter()
        .map(|s| NoetherCharge::compute(symmetry, system, s).value)
        .collect();
    let charge_conserved = if charges.len() > 1 {
        let mean = charges.iter().sum::<f64>() / charges.len() as f64;
        let max_dev = charges.iter().map(|c| (c - mean).abs()).fold(0.0f64, f64::max);
        max_dev < 1e-4 * (1.0 + mean.abs())
    } else {
        true
    };
    NoetherVerification {
        symmetry_name: symmetry.name().to_string(),
        lagrangian_invariant: invariant,
        charge_conserved,
        charges,
    }
}

/// Result of Noether verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoetherVerification {
    pub symmetry_name: String,
    pub lagrangian_invariant: bool,
    pub charge_conserved: bool,
    pub charges: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::SimpleLagrangian;

    #[test]
    fn test_translation_delta() {
        let sym = TranslationSymmetry::new(0, 2);
        let state = AgentState::new(DVector::from_vec(vec![3.0, 4.0]), DVector::from_vec(vec![1.0, 2.0]));
        let delta = sym.delta_q(&state);
        assert!((delta[0] - 1.0).abs() < 1e-10);
        assert!((delta[1]).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_delta() {
        let sym = RotationSymmetry::new(0, 1);
        let state = AgentState::new(DVector::from_vec(vec![3.0, 4.0]), DVector::from_vec(vec![0.0, 0.0]));
        let delta = sym.delta_q(&state);
        assert!((delta[0] - (-4.0)).abs() < 1e-10);
        assert!((delta[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_charge_translation() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![3.0, 4.0]));
        let sym = TranslationSymmetry::new(0, 2);
        let charge = NoetherCharge::compute(&sym, &sys, &state);
        // Q = p_0 * δq_0 = 1*3*1 = 3
        assert!((charge.value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_charge_rotation() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![3.0, 4.0]), DVector::from_vec(vec![1.0, 2.0]));
        let sym = RotationSymmetry::new(0, 1);
        let charge = NoetherCharge::compute(&sym, &sys, &state);
        // Q = p_0 * (-q_1) + p_1 * q_0 = 1*(-4) + 2*3 = 2 (angular momentum)
        assert!((charge.value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_verify_free_particle() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let q0 = vec![0.0, 0.0];
        let v = vec![3.0, 4.0];
        let dt = 0.1;
        let trajectory: Vec<AgentState> = (0..10)
            .map(|i| {
                let t = i as f64 * dt;
                AgentState::new(
                    DVector::from_vec(vec![q0[0] + v[0] * t, q0[1] + v[1] * t]),
                    DVector::from_vec(v.clone()),
                )
            })
            .collect();
        let sym = TranslationSymmetry::new(0, 2);
        let result = noether_verify(&sym, &sys, &trajectory);
        assert!(result.lagrangian_invariant);
        assert!(result.charge_conserved);
    }

    #[test]
    fn test_time_translation_charge() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0, 0.0]), DVector::from_vec(vec![3.0, 4.0]));
        let sym = TimeTranslationSymmetry;
        let charge = NoetherCharge::compute(&sym, &sys, &state);
        // Q = Σ p_i * q̇_i = 1*3*3 + 1*4*4 = 9 + 16 = 25 = 2T (the energy!)
        assert!((charge.value - 25.0).abs() < 1e-10);
    }
}
