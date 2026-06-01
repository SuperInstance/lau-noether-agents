//! Broken symmetries → approximate conservation laws.
//!
//! When a symmetry is only approximate (e.g., weak perturbation), the
//! corresponding Noether charge is only approximately conserved. We can
//! quantify the degree of symmetry breaking and the rate of charge drift.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};
use crate::noether::{NoetherCharge, SymmetryTransform};
use crate::charge::ConservedChargeTracker;
use crate::symmetry::SymmetryKind;

/// Degree of symmetry breaking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryBreaking {
    /// Name of the broken symmetry.
    pub symmetry_name: String,
    /// How much the Lagrangian changes under the symmetry transformation.
    pub invariance_violation: f64,
    /// Rate of charge drift per unit time.
    pub charge_drift_rate: f64,
    /// Approximate conservation quality (0 = no conservation, 1 = exact).
    pub conservation_quality: f64,
}

impl SymmetryBreaking {
    /// Analyze symmetry breaking for a given symmetry and trajectory.
    pub fn analyze(
        symmetry: &dyn SymmetryTransform,
        system: &dyn LagrangianSystem,
        trajectory: &[AgentState],
        times: &[f64],
    ) -> Self {
        // Measure Lagrangian variation
        let eps = 1e-6;
        let base_l = system.lagrangian(&trajectory[0]);
        let delta = symmetry.delta_q(&trajectory[0]);
        let q_shifted = &trajectory[0].q + eps * &delta;
        let shifted = AgentState::new(q_shifted, trajectory[0].qdot.clone());
        let l_shifted = system.lagrangian(&shifted);
        let invariance_violation = (base_l - l_shifted).abs() / eps;

        // Measure charge drift
        let charges: Vec<f64> = trajectory
            .iter()
            .map(|s| NoetherCharge::compute(symmetry, system, s).value)
            .collect();

        let charge_drift_rate = if charges.len() > 1 && times.len() > 1 {
            let dt_total = times[times.len() - 1] - times[0];
            if dt_total.abs() > 1e-15 {
                (charges[charges.len() - 1] - charges[0]).abs() / dt_total
            } else {
                0.0
            }
        } else {
            0.0
        };

        let conservation_quality = if charges.len() > 1 {
            let mean = charges.iter().sum::<f64>() / charges.len() as f64;
            let max_dev = charges.iter().map(|c| (c - mean).abs()).fold(0.0f64, f64::max);
            if mean.abs() < 1e-15 {
                if max_dev < 1e-10 { 1.0 } else { 0.0 }
            } else {
                1.0 / (1.0 + max_dev / mean.abs())
            }
        } else {
            1.0
        };

        SymmetryBreaking {
            symmetry_name: symmetry.name().to_string(),
            invariance_violation,
            charge_drift_rate,
            conservation_quality,
        }
    }
}

/// Approximate conservation law from a broken symmetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproximateConservationLaw {
    pub symmetry_name: String,
    pub charge_name: String,
    pub quality: f64,
    pub initial_charge: f64,
    pub final_charge: f64,
    pub max_deviation: f64,
    pub is_approximately_conserved: bool,
}

impl ApproximateConservationLaw {
    /// Check approximate conservation with a quality threshold.
    pub fn from_tracker(tracker: &ConservedChargeTracker, quality_threshold: f64) -> Self {
        let initial = tracker.values.first().copied().unwrap_or(0.0);
        let final_val = tracker.values.last().copied().unwrap_or(0.0);
        let max_deviation = tracker.max_relative_deviation();
        let quality = tracker.is_conserved(quality_threshold) as u8 as f64;

        ApproximateConservationLaw {
            symmetry_name: tracker.name.clone(),
            charge_name: tracker.name.clone(),
            quality,
            initial_charge: initial,
            final_charge: final_val,
            max_deviation,
            is_approximately_conserved: max_deviation < quality_threshold,
        }
    }
}

/// Adiabatic invariant: for slowly broken symmetries, the charge changes slowly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdiabaticInvariant {
    pub charge_name: String,
    pub initial_value: f64,
    pub final_value: f64,
    /// Ratio of timescale of symmetry breaking to dynamical timescale.
    pub adiabatic_parameter: f64,
    pub relative_change: f64,
}

impl AdiabaticInvariant {
    pub fn analyze(
        tracker: &ConservedChargeTracker,
        dynamical_timescale: f64,
    ) -> Self {
        let initial = tracker.values.first().copied().unwrap_or(0.0);
        let final_val = tracker.values.last().copied().unwrap_or(0.0);
        let total_time = if tracker.times.len() > 1 {
            tracker.times[tracker.times.len() - 1] - tracker.times[0]
        } else {
            1.0
        };

        let relative_change = if initial.abs() > 1e-15 {
            (final_val - initial).abs() / initial.abs()
        } else {
            (final_val - initial).abs()
        };

        AdiabaticInvariant {
            charge_name: tracker.name.clone(),
            initial_value: initial,
            final_value: final_val,
            adiabatic_parameter: total_time / dynamical_timescale,
            relative_change,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::{SimpleLagrangian, HarmonicLagrangian};
    use crate::noether::TranslationSymmetry;
    use crate::conservation::simulate_with_conservation;

    #[test]
    fn test_symmetry_broken_harmonic() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0]));
        let result = simulate_with_conservation(
            &sys, &state, 0.01, 100,
            &[SymmetryKind::Translation { axis: 0 }],
        );
        // Harmonic potential breaks translation symmetry
        let breaking = SymmetryBreaking::analyze(
            &TranslationSymmetry::new(0, 1),
            &sys,
            &result.trajectory,
            &result.times,
        );
        assert!(breaking.invariance_violation > 0.0);
        assert!(breaking.conservation_quality < 1.0);
    }

    #[test]
    fn test_symmetry_unbroken_free() {
        let sys = SimpleLagrangian::uniform(1, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![1.0]));
        let result = simulate_with_conservation(
            &sys, &state, 0.01, 100,
            &[SymmetryKind::Translation { axis: 0 }],
        );
        let breaking = SymmetryBreaking::analyze(
            &TranslationSymmetry::new(0, 1),
            &sys,
            &result.trajectory,
            &result.times,
        );
        assert!(breaking.invariance_violation < 1e-4);
        assert!(breaking.conservation_quality > 0.99);
    }

    #[test]
    fn test_approximate_conservation_law() {
        let sys = SimpleLagrangian::uniform(1, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0]));
        let result = simulate_with_conservation(
            &sys, &state, 0.01, 50,
            &[SymmetryKind::Translation { axis: 0 }],
        );
        let law = ApproximateConservationLaw::from_tracker(&result.charge_trackers[0], 1e-4);
        assert!(law.is_approximately_conserved);
    }

    #[test]
    fn test_adiabatic_invariant() {
        let sys = SimpleLagrangian::uniform(1, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0]));
        let result = simulate_with_conservation(
            &sys, &state, 0.01, 50,
            &[SymmetryKind::Translation { axis: 0 }],
        );
        let adiabatic = AdiabaticInvariant::analyze(&result.charge_trackers[0], 1.0);
        assert!(adiabatic.relative_change < 1e-6);
    }

    #[test]
    fn test_charge_drift_rate() {
        let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0]));
        let result = simulate_with_conservation(
            &sys, &state, 0.01, 100,
            &[SymmetryKind::Translation { axis: 0 }],
        );
        let breaking = SymmetryBreaking::analyze(
            &TranslationSymmetry::new(0, 1),
            &sys,
            &result.trajectory,
            &result.times,
        );
        // Momentum is NOT conserved for harmonic oscillator
        assert!(breaking.charge_drift_rate > 0.0);
    }
}
