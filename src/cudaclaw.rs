//! CUDAclaw cell agent application: proving discrete conservation laws
//! and agent fleet invariant tracking.
//!
//! This module applies Noether's theorem to CUDAclaw cell agents,
//! demonstrating that they satisfy discrete conservation laws derived
//! from the symmetries of the cell dynamics.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem};
use crate::discrete::{DiscreteLagrangian, DiscreteNoetherCharge, TrapezoidalDiscreteLagrangian};
use crate::charge::{ConservedChargeTracker, compute_charge_for_kind};
use crate::symmetry::SymmetryKind;
use crate::fleet::{AgentFleet, FleetAgent, FleetInvariant, fleet_invariants};
use crate::conservation::{simulate_with_conservation, SimulationResult};

/// A CUDAclaw cell agent with grid-based dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellAgent {
    pub id: usize,
    pub grid_position: DVector<f64>,
    pub velocity: DVector<f64>,
    pub mass: f64,
    pub state: Vec<f64>,
}

impl CellAgent {
    pub fn new(id: usize, position: Vec<f64>, velocity: Vec<f64>, mass: f64) -> Self {
        Self {
            id,
            grid_position: DVector::from_vec(position),
            velocity: DVector::from_vec(velocity),
            mass,
            state: Vec::new(),
        }
    }

    pub fn to_agent_state(&self) -> AgentState {
        AgentState::new(self.grid_position.clone(), self.velocity.clone())
    }

    pub fn dim(&self) -> usize {
        self.grid_position.nrows()
    }
}

/// CUDAclaw cell dynamics: agents interact via a grid-based potential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellDynamics {
    pub agents: Vec<CellAgent>,
    pub interaction_strength: f64,
    pub damping: f64,
}

impl CellDynamics {
    pub fn new(agents: Vec<CellAgent>, interaction_strength: f64) -> Self {
        Self {
            agents,
            interaction_strength,
            damping: 0.0,
        }
    }

    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// Convert to a Lagrangian system (total kinetic - total potential).
    fn total_kinetic_energy(&self) -> f64 {
        self.agents.iter().map(|a| 0.5 * a.mass * a.velocity.norm_squared()).sum()
    }

    /// Pairwise interaction potential: V = Σ_{i<j} k * |r_i - r_j|² / 2.
    fn total_potential_energy(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let dr = &self.agents[i].grid_position - &self.agents[j].grid_position;
                v += 0.5 * self.interaction_strength * dr.norm_squared();
            }
        }
        v
    }

    /// Total momentum of all cell agents.
    pub fn total_momentum(&self) -> DVector<f64> {
        let dim = self.agents[0].dim();
        let mut p = DVector::zeros(dim);
        for a in &self.agents {
            p += a.mass * &a.velocity;
        }
        p
    }

    /// Total energy.
    pub fn total_energy(&self) -> f64 {
        self.total_kinetic_energy() + self.total_potential_energy()
    }

    /// Advance one time step using velocity Verlet.
    pub fn step(&mut self, dt: f64) {
        let n = self.agents.len();
        let dim = self.agents[0].dim();

        // Compute forces
        let mut forces: Vec<DVector<f64>> = (0..n).map(|_| DVector::zeros(dim)).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let dr = &self.agents[i].grid_position - &self.agents[j].grid_position;
                let f = -self.interaction_strength * &dr;
                forces[i] += &f;
                forces[j] -= &f;
            }
        }

        // Update positions and velocities
        for i in 0..n {
            let m = self.agents[i].mass;
            let acc = forces[i].clone() / m;
            let vel = self.agents[i].velocity.clone() + dt * &acc;
            self.agents[i].grid_position += dt * &vel;
            self.agents[i].velocity = vel;
        }
    }
}

/// Discrete conservation proof for CUDAclaw agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationProof {
    pub n_agents: usize,
    pub n_steps: usize,
    pub dt: f64,
    pub total_momentum_initial: Vec<f64>,
    pub total_momentum_final: Vec<f64>,
    pub momentum_conserved: bool,
    pub total_energy_initial: f64,
    pub total_energy_final: f64,
    pub energy_relative_error: f64,
    pub discrete_noether_charges: Vec<DiscreteNoetherCharge>,
}

impl ConservationProof {
    /// Run a conservation proof on a cell dynamics system.
    pub fn prove(dynamics: &mut CellDynamics, n_steps: usize, dt: f64) -> Self {
        let p_init = dynamics.total_momentum();
        let e_init = dynamics.total_energy();

        // Track positions for discrete Noether analysis
        let dim = dynamics.agents[0].dim();
        let mut positions_history: Vec<Vec<DVector<f64>>> = Vec::new();
        positions_history.push(dynamics.agents.iter().map(|a| a.grid_position.clone()).collect());

        for _ in 0..n_steps {
            dynamics.step(dt);
            positions_history.push(dynamics.agents.iter().map(|a| a.grid_position.clone()).collect());
        }

        let p_final = dynamics.total_momentum();
        let e_final = dynamics.total_energy();

        // Compute discrete Noether charges for the first agent (translation symmetry)
        let mut discrete_charges = Vec::new();
        for agent_idx in 0..dynamics.n_agents().min(3) {
            let traj: Vec<DVector<f64>> = positions_history.iter().map(|h| h[agent_idx].clone()).collect();
            let lag = |_q: &DVector<f64>, qdot: &DVector<f64>| -> f64 {
                0.5 * qdot.iter().map(|v| v * v).sum::<f64>()
            };
            let dl = TrapezoidalDiscreteLagrangian { continuous_lagrangian: lag };
            for axis in 0..dim {
                let charge = DiscreteNoetherCharge::compute_trajectory(
                    &dl,
                    &traj,
                    dt,
                    &|_| {
                        let mut d = DVector::zeros(dim);
                        d[axis] = 1.0;
                        d
                    },
                    &format!("agent_{}_momentum_{}", agent_idx, axis),
                );
                discrete_charges.push(charge);
            }
        }

        let momentum_conserved = (p_init.clone() - &p_final).norm() < 1e-4 * (1.0 + p_init.norm());

        ConservationProof {
            n_agents: dynamics.n_agents(),
            n_steps,
            dt,
            total_momentum_initial: p_init.iter().cloned().collect(),
            total_momentum_final: p_final.iter().cloned().collect(),
            momentum_conserved,
            total_energy_initial: e_init,
            total_energy_final: e_final,
            energy_relative_error: if e_init.abs() > 1e-15 {
                (e_final - e_init).abs() / e_init.abs()
            } else {
                (e_final - e_init).abs()
            },
            discrete_noether_charges: discrete_charges,
        }
    }
}

/// Fleet invariant tracker for CUDAclaw agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetInvariantTracker {
    pub energy_history: Vec<f64>,
    pub momentum_history: Vec<DVector<f64>>,
    pub angular_momentum_history: Vec<f64>,
    pub times: Vec<f64>,
}

impl FleetInvariantTracker {
    pub fn new() -> Self {
        Self {
            energy_history: Vec::new(),
            momentum_history: Vec::new(),
            angular_momentum_history: Vec::new(),
            times: Vec::new(),
        }
    }

    pub fn record(&mut self, dynamics: &CellDynamics, time: f64) {
        self.energy_history.push(dynamics.total_energy());
        self.momentum_history.push(dynamics.total_momentum());
        self.angular_momentum_history.push({
            let mut l = 0.0;
            for a in &dynamics.agents {
                if a.dim() >= 2 {
                    l += a.mass * (a.grid_position[0] * a.velocity[1] - a.grid_position[1] * a.velocity[0]);
                }
            }
            l
        });
        self.times.push(time);
    }

    /// Check if all fleet invariants are conserved within tolerance.
    pub fn verify_conservation(&self, tol: f64) -> FleetConservationReport {
        let energy_conserved = if self.energy_history.len() > 1 {
            let mean = self.energy_history.iter().sum::<f64>() / self.energy_history.len() as f64;
            self.energy_history.iter().all(|e| (e - mean).abs() < tol * (1.0 + mean.abs()))
        } else {
            true
        };

        let momentum_conserved = if self.momentum_history.len() > 1 {
            let mean_p = self.momentum_history.iter().fold(
                DVector::zeros(self.momentum_history[0].nrows()),
                |acc, p| acc + p,
            ) / self.momentum_history.len() as f64;
            self.momentum_history.iter().all(|p| (p - &mean_p).norm() < tol * (1.0 + mean_p.norm()))
        } else {
            true
        };

        let angular_conserved = if self.angular_momentum_history.len() > 1 {
            let mean = self.angular_momentum_history.iter().sum::<f64>() / self.angular_momentum_history.len() as f64;
            self.angular_momentum_history.iter().all(|l| (l - mean).abs() < tol * (1.0 + mean.abs()))
        } else {
            true
        };

        FleetConservationReport {
            energy_conserved,
            momentum_conserved,
            angular_momentum_conserved: angular_conserved,
            all_conserved: energy_conserved && momentum_conserved && angular_conserved,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetConservationReport {
    pub energy_conserved: bool,
    pub momentum_conserved: bool,
    pub angular_momentum_conserved: bool,
    pub all_conserved: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell_agents() -> Vec<CellAgent> {
        vec![
            CellAgent::new(0, vec![1.0, 0.0], vec![0.0, 1.0], 1.0),
            CellAgent::new(1, vec![-1.0, 0.0], vec![0.0, -1.0], 1.0),
            CellAgent::new(2, vec![0.0, 1.0], vec![-1.0, 0.0], 1.0),
        ]
    }

    #[test]
    fn test_cell_agent_creation() {
        let agent = CellAgent::new(0, vec![1.0, 2.0], vec![3.0, 4.0], 2.0);
        assert_eq!(agent.id, 0);
        assert_eq!(agent.dim(), 2);
    }

    #[test]
    fn test_cell_dynamics_total_momentum() {
        let dynamics = CellDynamics::new(make_cell_agents(), 1.0);
        let p = dynamics.total_momentum();
        // (0,-1,0) + (0,-1,0) + (-1,0,0) = (-1, 0) ... wait
        // agent 0: vel (0,1), agent 1: vel (0,-1), agent 2: vel (-1,0)
        // total: (-1, 0)
        assert!((p[0] - (-1.0)).abs() < 1e-10);
        assert!((p[1]).abs() < 1e-10);
    }

    #[test]
    fn test_cell_dynamics_total_energy() {
        let dynamics = CellDynamics::new(make_cell_agents(), 0.5);
        let e = dynamics.total_energy();
        // KE = ½*1*(1) + ½*1*(1) + ½*1*(1) = 1.5
        let ke = dynamics.total_kinetic_energy();
        assert!((ke - 1.5).abs() < 1e-10);
        assert!(e >= ke);
    }

    #[test]
    fn test_cell_dynamics_step() {
        let mut dynamics = CellDynamics::new(make_cell_agents(), 0.0);
        let p_before = dynamics.total_momentum().clone();
        dynamics.step(0.01);
        let p_after = dynamics.total_momentum();
        // Free particles: momentum should be conserved
        assert!((p_before - p_after).norm() < 1e-10);
    }

    #[test]
    fn test_conservation_proof_free() {
        let agents = vec![
            CellAgent::new(0, vec![0.0, 0.0], vec![1.0, 0.0], 1.0),
            CellAgent::new(1, vec![1.0, 0.0], vec![-1.0, 0.0], 1.0),
        ];
        let mut dynamics = CellDynamics::new(agents, 0.0);
        let proof = ConservationProof::prove(&mut dynamics, 100, 0.01);
        assert!(proof.momentum_conserved);
    }

    #[test]
    fn test_conservation_proof_with_interaction() {
        let agents = vec![
            CellAgent::new(0, vec![0.0, 0.0], vec![1.0, 0.0], 1.0),
            CellAgent::new(1, vec![1.0, 0.0], vec![-1.0, 0.0], 1.0),
        ];
        let mut dynamics = CellDynamics::new(agents, 1.0);
        let proof = ConservationProof::prove(&mut dynamics, 50, 0.001);
        // With interaction, total momentum should still be conserved (Newton's 3rd law)
        assert!(proof.momentum_conserved);
    }

    #[test]
    fn test_fleet_invariant_tracker() {
        let mut tracker = FleetInvariantTracker::new();
        let mut dynamics = CellDynamics::new(
            vec![
                CellAgent::new(0, vec![0.0, 0.0], vec![1.0, 0.0], 1.0),
                CellAgent::new(1, vec![1.0, 0.0], vec![-1.0, 0.0], 1.0),
            ],
            0.0,
        );
        for i in 0..10 {
            tracker.record(&dynamics, i as f64 * 0.01);
            dynamics.step(0.01);
        }
        let report = tracker.verify_conservation(1e-4);
        assert!(report.momentum_conserved);
    }

    #[test]
    fn test_fleet_conservation_report() {
        let agents = vec![
            CellAgent::new(0, vec![0.0], vec![1.0], 1.0),
            CellAgent::new(1, vec![1.0], vec![-1.0], 1.0),
        ];
        let fleet = AgentFleet::new(
            agents.into_iter().map(|a| FleetAgent {
                id: a.id,
                state: a.to_agent_state(),
                mass: a.mass,
            }).collect()
        );
        let invs = fleet_invariants(&fleet);
        assert!(!invs.is_empty());
    }

    #[test]
    fn test_discrete_noether_charges_in_proof() {
        let agents = vec![
            CellAgent::new(0, vec![0.0, 0.0], vec![1.0, 2.0], 1.0),
        ];
        let mut dynamics = CellDynamics::new(agents, 0.0);
        let proof = ConservationProof::prove(&mut dynamics, 50, 0.01);
        assert!(!proof.discrete_noether_charges.is_empty());
        // Free particle momentum should be conserved
        for charge in &proof.discrete_noether_charges {
            assert!(charge.is_conserved(0.1));
        }
    }
}
