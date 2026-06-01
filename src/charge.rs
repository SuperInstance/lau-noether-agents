//! Conserved charge computation from detected symmetries.
//!
//! Given a set of detected symmetries, compute the corresponding Noether charges
//! and track them through agent trajectories.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};
use crate::noether::{NoetherCharge, SymmetryTransform, TranslationSymmetry, RotationSymmetry, TimeTranslationSymmetry};
use crate::symmetry::{SymmetryKind, GaugeSymmetry};

/// A computed conserved charge with its trajectory values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedChargeTracker {
    pub name: String,
    pub symmetry_kind: SymmetryKind,
    pub values: Vec<f64>,
    pub times: Vec<f64>,
}

impl ConservedChargeTracker {
    /// Build a tracker for a given symmetry kind.
    pub fn new(name: &str, kind: SymmetryKind) -> Self {
        Self {
            name: name.to_string(),
            symmetry_kind: kind,
            values: Vec::new(),
            times: Vec::new(),
        }
    }

    /// Record a charge value at a given time.
    pub fn record(&mut self, time: f64, value: f64) {
        self.times.push(time);
        self.values.push(value);
    }

    /// Compute and record the charge for a state.
    pub fn compute_and_record(
        &mut self,
        system: &dyn LagrangianSystem,
        state: &AgentState,
        time: f64,
    ) {
        let charge = compute_charge_for_kind(&self.symmetry_kind, system, state);
        self.record(time, charge);
    }

    /// Check if the charge is conserved within tolerance.
    pub fn is_conserved(&self, tol: f64) -> bool {
        if self.values.len() <= 1 {
            return true;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        self.values.iter().all(|v| (v - mean).abs() < tol * (1.0 + mean.abs()))
    }

    /// Maximum relative deviation.
    pub fn max_relative_deviation(&self) -> f64 {
        if self.values.len() <= 1 {
            return 0.0;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        if mean.abs() < 1e-15 {
            return self.values.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        }
        self.values.iter().map(|v| (v - mean).abs() / mean.abs()).fold(0.0f64, f64::max)
    }

    /// Mean value.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }
}

/// Compute a Noether charge for a given symmetry kind.
pub fn compute_charge_for_kind(
    kind: &SymmetryKind,
    system: &dyn LagrangianSystem,
    state: &AgentState,
) -> f64 {
    match kind {
        SymmetryKind::Translation { axis } => {
            let sym = TranslationSymmetry::new(*axis, state.dim());
            NoetherCharge::compute(&sym, system, state).value
        }
        SymmetryKind::Rotation { axis_i, axis_j } => {
            let sym = RotationSymmetry::new(*axis_i, *axis_j);
            NoetherCharge::compute(&sym, system, state).value
        }
        SymmetryKind::Gauge { direction } => {
            let sym = GaugeSymmetry::new(direction.clone());
            NoetherCharge::compute(&sym, system, state).value
        }
        SymmetryKind::Scaling => {
            // For scaling, Q = Σ p_i q_i (dilatation charge)
            let m = system.mass_matrix(state);
            let p = state.momenta(&m);
            p.dot(&state.q)
        }
        SymmetryKind::TimeTranslation => {
            let sym = TimeTranslationSymmetry;
            NoetherCharge::compute(&sym, system, state).value
        }
    }
}

/// Compute all Noether charges from a list of symmetry kinds.
pub fn compute_all_charges(
    kinds: &[SymmetryKind],
    system: &dyn LagrangianSystem,
    state: &AgentState,
) -> Vec<NoetherCharge> {
    kinds
        .iter()
        .map(|kind| {
            let val = compute_charge_for_kind(kind, system, state);
            NoetherCharge {
                name: format!("Q_{:?}", kind),
                symmetry_name: format!("{:?}", kind),
                value: val,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::SimpleLagrangian;

    #[test]
    fn test_tracker_conservation() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let mut tracker = ConservedChargeTracker::new(
            "momentum_x",
            SymmetryKind::Translation { axis: 0 },
        );
        let v = vec![3.0, 4.0];
        for i in 0..10 {
            let t = i as f64 * 0.1;
            let state = AgentState::new(
                DVector::from_vec(vec![v[0] * t, v[1] * t]),
                DVector::from_vec(v.clone()),
            );
            tracker.compute_and_record(&sys, &state, t);
        }
        assert!(tracker.is_conserved(1e-8));
        assert!(tracker.max_relative_deviation() < 1e-8);
    }

    #[test]
    fn test_tracker_mean() {
        let mut tracker = ConservedChargeTracker::new(
            "test",
            SymmetryKind::Translation { axis: 0 },
        );
        tracker.record(0.0, 5.0);
        tracker.record(1.0, 5.0);
        tracker.record(2.0, 5.0);
        assert!((tracker.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_all_charges() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let kinds = vec![
            SymmetryKind::Translation { axis: 0 },
            SymmetryKind::Translation { axis: 1 },
            SymmetryKind::Rotation { axis_i: 0, axis_j: 1 },
        ];
        let charges = compute_all_charges(&kinds, &sys, &state);
        assert_eq!(charges.len(), 3);
        // px = m*v_x = 1*3 = 3
        assert!((charges[0].value - 3.0).abs() < 1e-10);
        // py = m*v_y = 1*4 = 4
        assert!((charges[1].value - 4.0).abs() < 1e-10);
        // L = x*py - y*px = 1*4 - 2*3 = -2
        assert!((charges[2].value - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_empty_tracker_conserved() {
        let tracker = ConservedChargeTracker::new("empty", SymmetryKind::Translation { axis: 0 });
        assert!(tracker.is_conserved(1e-8));
    }

    #[test]
    fn test_scaling_charge() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![3.0, 4.0]), DVector::from_vec(vec![1.0, 2.0]));
        let val = compute_charge_for_kind(&SymmetryKind::Scaling, &sys, &state);
        // Q = p·q = 1*3 + 2*4 = 11
        assert!((val - 11.0).abs() < 1e-10);
    }
}
