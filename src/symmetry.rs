//! Symmetry detection in agent dynamics.
//!
//! Detects translation, rotation, and gauge symmetries by checking
//! Lagrangian invariance under infinitesimal transformations.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};
use crate::noether::{SymmetryTransform, TranslationSymmetry, RotationSymmetry};

/// Types of symmetries that can be detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SymmetryKind {
    Translation { axis: usize },
    Rotation { axis_i: usize, axis_j: usize },
    Gauge { direction: DVector<f64> },
    Scaling,
    TimeTranslation,
}

/// A detected symmetry with quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSymmetry {
    pub kind: SymmetryKind,
    pub invariance_error: f64,
    pub confidence: f64,
}

/// Result of symmetry detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryDetectionResult {
    pub detected: Vec<DetectedSymmetry>,
    pub sample_states: Vec<AgentState>,
}

/// Detect symmetries of a Lagrangian system by probing invariance.
pub fn detect_symmetries(
    system: &dyn LagrangianSystem,
    sample_state: &AgentState,
    dim: usize,
) -> Vec<DetectedSymmetry> {
    let mut results = Vec::new();
    let eps = 1e-6;
    let base_l = system.lagrangian(sample_state);

    // Check translation symmetries along each axis
    // Use second-order check: compare error to eps^2 * (second-order scale)
    for axis in 0..dim {
        let sym = TranslationSymmetry::new(axis, dim);
        let delta = sym.delta_q(sample_state);
        let q_shifted = &sample_state.q + eps * &delta;
        let shifted = AgentState::new(q_shifted, sample_state.qdot.clone());
        let l_shifted = system.lagrangian(&shifted);
        // Normalize error by eps — if invariant, error should be O(eps^2)
        let error_normalized = (base_l - l_shifted).abs() / eps;
        let error = (base_l - l_shifted).abs();
        let confidence = 1.0 / (1.0 + error_normalized * 1e3);
        if error_normalized < 1e-2 * (1.0 + base_l.abs()) {
            results.push(DetectedSymmetry {
                kind: SymmetryKind::Translation { axis },
                invariance_error: error,
                confidence,
            });
        }
    }

    // Check rotation symmetries for each pair
    for i in 0..dim {
        for j in (i + 1)..dim {
            let sym = RotationSymmetry::new(i, j);
            let delta = sym.delta_q(sample_state);
            let q_shifted = &sample_state.q + eps * &delta;
            let shifted = AgentState::new(q_shifted, sample_state.qdot.clone());
            let l_shifted = system.lagrangian(&shifted);
            let error_normalized = (base_l - l_shifted).abs() / eps;
            let error = (base_l - l_shifted).abs();
            let confidence = 1.0 / (1.0 + error_normalized * 1e3);
            if error_normalized < 1e-2 * (1.0 + base_l.abs()) {
                results.push(DetectedSymmetry {
                    kind: SymmetryKind::Rotation { axis_i: i, axis_j: j },
                    invariance_error: error,
                    confidence,
                });
            }
        }
    }

    // Check gauge symmetry (along a specific direction)
    let gauge_dir = DVector::from_element(dim, 1.0 / (dim as f64).sqrt());
    {
        let delta = &gauge_dir * eps;
        let q_shifted = &sample_state.q + &delta;
        let shifted = AgentState::new(q_shifted, sample_state.qdot.clone());
        let l_shifted = system.lagrangian(&shifted);
        let error = (base_l - l_shifted).abs();
        if error < 1e-4 * (1.0 + base_l.abs()) {
            results.push(DetectedSymmetry {
                kind: SymmetryKind::Gauge { direction: gauge_dir },
                invariance_error: error,
                confidence: 1.0 / (1.0 + error * 1e6),
            });
        }
    }

    results
}

/// A gauge symmetry transformation.
pub struct GaugeSymmetry {
    pub direction: DVector<f64>,
}

impl GaugeSymmetry {
    pub fn new(direction: DVector<f64>) -> Self {
        Self { direction }
    }
}

impl SymmetryTransform for GaugeSymmetry {
    fn name(&self) -> &str {
        "gauge"
    }

    fn delta_q(&self, _state: &AgentState) -> DVector<f64> {
        self.direction.clone()
    }
}

/// A scaling symmetry transformation.
pub struct ScalingSymmetry;

impl SymmetryTransform for ScalingSymmetry {
    fn name(&self) -> &str {
        "scaling"
    }

    fn delta_q(&self, state: &AgentState) -> DVector<f64> {
        state.q.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrangian::SimpleLagrangian;

    #[test]
    fn test_detect_translation_free_particle() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let detected = detect_symmetries(&sys, &state, 2);
        let translations: Vec<_> = detected.iter()
            .filter(|d| matches!(d.kind, SymmetryKind::Translation { .. }))
            .collect();
        assert!(translations.len() >= 2, "Should detect both translation symmetries, found {}", translations.len());
    }

    #[test]
    fn test_detect_rotation_free_particle() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let detected = detect_symmetries(&sys, &state, 2);
        let rotations: Vec<_> = detected.iter()
            .filter(|d| matches!(d.kind, SymmetryKind::Rotation { .. }))
            .collect();
        assert!(!rotations.is_empty(), "Should detect rotation symmetry");
    }

    #[test]
    fn test_gauge_symmetry() {
        let sys = SimpleLagrangian::uniform(2, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![3.0, 4.0]));
        let gauge = GaugeSymmetry::new(DVector::from_vec(vec![1.0, 0.0]));
        let charge = crate::noether::NoetherCharge::compute(&gauge, &sys, &state);
        assert!((charge.value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_symmetry_delta() {
        let sym = ScalingSymmetry;
        let state = AgentState::new(DVector::from_vec(vec![3.0, 4.0]), DVector::from_vec(vec![1.0, 2.0]));
        let delta = sym.delta_q(&state);
        assert!((delta[0] - 3.0).abs() < 1e-10);
        assert!((delta[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry_detection_result_serialization() {
        let result = SymmetryDetectionResult {
            detected: vec![],
            sample_states: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("detected"));
    }

    #[test]
    fn test_no_translation_with_potential() {
        use crate::lagrangian::HarmonicLagrangian;
        let sys = HarmonicLagrangian::uniform(2, 1.0, 1.0);
        let state = AgentState::new(DVector::from_vec(vec![1.0, 2.0]), DVector::from_vec(vec![0.0, 0.0]));
        let detected = detect_symmetries(&sys, &state, 2);
        let translations: Vec<_> = detected.iter()
            .filter(|d| matches!(d.kind, SymmetryKind::Translation { .. }))
            .collect();
        // Harmonic potential centered at origin breaks translation symmetry
        // But our gauge check might pass - that's okay, we only check translations here
        assert!(translations.is_empty(), "Harmonic potential should break translation symmetry");
    }
}
