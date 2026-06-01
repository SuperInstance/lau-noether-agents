//! Discrete Noether theorem for finite-volume / time-stepping schemes.
//!
//! For a discrete Lagrangian L_d(q_k, q_{k+1}), the discrete Euler-Lagrange
//! equation yields a discrete Noether charge that is exactly conserved.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::AgentState;

/// Discrete Lagrangian: L_d(q_k, q_{k+1}, h) approximates ∫_{t_k}^{t_{k+1}} L dt.
pub trait DiscreteLagrangian: Send + Sync {
    /// Evaluate the discrete Lagrangian.
    fn evaluate(&self, q_k: &DVector<f64>, q_k1: &DVector<f64>, h: f64) -> f64;

    /// ∂L_d/∂q_{k+1} — the discrete momentum at step k+1.
    fn dL_dq_k1(&self, q_k: &DVector<f64>, q_k1: &DVector<f64>, h: f64) -> DVector<f64> {
        let eps = 1e-8;
        let n = q_k1.nrows();
        let mut grad = DVector::zeros(n);
        for i in 0..n {
            let mut q_plus = q_k1.clone();
            let mut q_minus = q_k1.clone();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            grad[i] = (self.evaluate(q_k, &q_plus, h) - self.evaluate(q_k, &q_minus, h)) / (2.0 * eps);
        }
        grad
    }

    /// ∂L_d/∂q_k — the negative discrete momentum at step k.
    fn dL_dq_k(&self, q_k: &DVector<f64>, q_k1: &DVector<f64>, h: f64) -> DVector<f64> {
        let eps = 1e-8;
        let n = q_k.nrows();
        let mut grad = DVector::zeros(n);
        for i in 0..n {
            let mut q_plus = q_k.clone();
            let mut q_minus = q_k.clone();
            q_plus[i] += eps;
            q_minus[i] -= eps;
            grad[i] = (self.evaluate(&q_plus, q_k1, h) - self.evaluate(&q_minus, q_k1, h)) / (2.0 * eps);
        }
        grad
    }
}

/// Trapezoidal discrete Lagrangian: L_d = h/2 * (L(q_k, v_k) + L(q_{k+1}, v_{k+1}))
/// where v_k = (q_{k+1} - q_k) / h.
pub struct TrapezoidalDiscreteLagrangian<F>
where
    F: Fn(&DVector<f64>, &DVector<f64>) -> f64 + Send + Sync,
{
    /// Continuous Lagrangian L(q, q̇).
    pub continuous_lagrangian: F,
}

impl<F> DiscreteLagrangian for TrapezoidalDiscreteLagrangian<F>
where
    F: Fn(&DVector<f64>, &DVector<f64>) -> f64 + Send + Sync,
{
    fn evaluate(&self, q_k: &DVector<f64>, q_k1: &DVector<f64>, h: f64) -> f64 {
        let v_k = (q_k1 - q_k) / h;
        let l_k = (self.continuous_lagrangian)(q_k, &v_k);
        let l_k1 = (self.continuous_lagrangian)(q_k1, &v_k);
        h * 0.5 * (l_k + l_k1)
    }
}

/// Midpoint discrete Lagrangian: L_d = h * L((q_k+q_{k+1})/2, (q_{k+1}-q_k)/h).
pub struct MidpointDiscreteLagrangian<F>
where
    F: Fn(&DVector<f64>, &DVector<f64>) -> f64 + Send + Sync,
{
    pub continuous_lagrangian: F,
}

impl<F> DiscreteLagrangian for MidpointDiscreteLagrangian<F>
where
    F: Fn(&DVector<f64>, &DVector<f64>) -> f64 + Send + Sync,
{
    fn evaluate(&self, q_k: &DVector<f64>, q_k1: &DVector<f64>, h: f64) -> f64 {
        let q_mid = (q_k + q_k1) * 0.5;
        let v = (q_k1 - q_k) / h;
        h * (self.continuous_lagrangian)(&q_mid, &v)
    }
}

/// Discrete Noether charge: Q_d = ∂L_d/∂q_{k+1} · δq_{k+1}.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteNoetherCharge {
    pub name: String,
    pub values: Vec<f64>,
}

impl DiscreteNoetherCharge {
    /// Compute discrete Noether charge along a trajectory.
    pub fn compute_trajectory(
        discrete_lagrangian: &dyn DiscreteLagrangian,
        trajectory_q: &[DVector<f64>],
        h: f64,
        delta_q: &dyn Fn(&DVector<f64>) -> DVector<f64>,
        name: &str,
    ) -> Self {
        let mut values = Vec::new();
        for k in 0..trajectory_q.len() - 1 {
            let p_k1 = discrete_lagrangian.dL_dq_k1(&trajectory_q[k], &trajectory_q[k + 1], h);
            let dq = delta_q(&trajectory_q[k + 1]);
            values.push(p_k1.dot(&dq));
        }
        DiscreteNoetherCharge {
            name: name.to_string(),
            values,
        }
    }

    /// Check if the charge is conserved within tolerance.
    pub fn is_conserved(&self, tol: f64) -> bool {
        if self.values.len() <= 1 {
            return true;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        self.values.iter().all(|v| (v - mean).abs() < tol * (1.0 + mean.abs()))
    }

    /// Maximum deviation from the mean.
    pub fn max_deviation(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        self.values.iter().map(|v| (v - mean).abs()).fold(0.0f64, f64::max)
    }
}

/// Discrete Euler-Lagrange residual: D1L_d(q_k, q_{k+1}) + D2L_d(q_{k-1}, q_k) = 0.
pub fn discrete_euler_lagrange_residual(
    discrete_lagrangian: &dyn DiscreteLagrangian,
    q_prev: &DVector<f64>,
    q_curr: &DVector<f64>,
    q_next: &DVector<f64>,
    h: f64,
) -> DVector<f64> {
    let d1 = discrete_lagrangian.dL_dq_k(q_curr, q_next, h);
    let d2 = discrete_lagrangian.dL_dq_k1(q_prev, q_curr, h);
    &d1 + &d2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_particle_lagrangian(_q: &DVector<f64>, qdot: &DVector<f64>) -> f64 {
        0.5 * qdot.iter().map(|v| v * v).sum::<f64>()
    }

    #[test]
    fn test_trapezoidal_discrete_lagrangian() {
        let dl = TrapezoidalDiscreteLagrangian { continuous_lagrangian: free_particle_lagrangian };
        let q_k = DVector::from_vec(vec![0.0]);
        let q_k1 = DVector::from_vec(vec![1.0]);
        let h = 0.1;
        let val = dl.evaluate(&q_k, &q_k1, h);
        // v = 10, L = 50, L_d = 0.1 * 50 = 5.0
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_discrete_lagrangian() {
        let dl = MidpointDiscreteLagrangian { continuous_lagrangian: free_particle_lagrangian };
        let q_k = DVector::from_vec(vec![0.0]);
        let q_k1 = DVector::from_vec(vec![1.0]);
        let h = 0.1;
        let val = dl.evaluate(&q_k, &q_k1, h);
        // q_mid = 0.5, v = 10, L = 50, L_d = 0.1 * 50 = 5.0
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_noether_charge_conservation() {
        let dl = TrapezoidalDiscreteLagrangian { continuous_lagrangian: free_particle_lagrangian };
        let h = 0.01;
        let v = 3.0;
        let trajectory: Vec<DVector<f64>> = (0..100)
            .map(|i| DVector::from_vec(vec![v * i as f64 * h]))
            .collect();
        let charge = DiscreteNoetherCharge::compute_trajectory(
            &dl,
            &trajectory,
            h,
            &|_| DVector::from_vec(vec![1.0]),
            "momentum",
        );
        assert!(charge.is_conserved(1e-4));
    }

    #[test]
    fn test_discrete_noether_charge_max_deviation() {
        let dl = TrapezoidalDiscreteLagrangian { continuous_lagrangian: free_particle_lagrangian };
        let h = 0.01;
        let trajectory: Vec<DVector<f64>> = (0..50)
            .map(|i| DVector::from_vec(vec![3.0 * i as f64 * h]))
            .collect();
        let charge = DiscreteNoetherCharge::compute_trajectory(
            &dl, &trajectory, h,
            &|_| DVector::from_vec(vec![1.0]), "momentum",
        );
        assert!(charge.max_deviation() < 1e-4);
    }

    #[test]
    fn test_discrete_euler_lagrange() {
        let dl = MidpointDiscreteLagrangian { continuous_lagrangian: free_particle_lagrangian };
        let h = 0.1;
        let q_prev = DVector::from_vec(vec![0.0]);
        let q_curr = DVector::from_vec(vec![0.3]);
        let q_next = DVector::from_vec(vec![0.6]);
        let res = discrete_euler_lagrange_residual(&dl, &q_prev, &q_curr, &q_next, h);
        // Free particle: DEL should be zero for uniform motion
        assert!(res.iter().all(|r| r.abs() < 1e-4));
    }
}
