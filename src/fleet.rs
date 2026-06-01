//! Agent fleet symmetries: permutation symmetry → total fleet charges.
//!
//! When agents are identical, the fleet has permutation symmetry. By Noether's
//! theorem, this yields conserved "total fleet charges" (like total momentum,
//! total angular momentum, total energy).

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::lagrangian::{AgentState, LagrangianSystem, SimpleLagrangian};
use crate::noether::NoetherCharge;

/// An agent in a fleet with its own state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetAgent {
    pub id: usize,
    pub state: AgentState,
    pub mass: f64,
}

/// A fleet of agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFleet {
    pub agents: Vec<FleetAgent>,
}

impl AgentFleet {
    pub fn new(agents: Vec<FleetAgent>) -> Self {
        Self { agents }
    }

    /// Number of agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Total fleet momentum (sum of individual momenta).
    pub fn total_momentum(&self) -> DVector<f64> {
        let dim = self.agents[0].state.dim();
        let mut total = DVector::zeros(dim);
        for agent in &self.agents {
            total += agent.mass * &agent.state.qdot;
        }
        total
    }

    /// Total fleet angular momentum about the origin.
    /// For 2D: L = Σ m_i (x_i v_{yi} - y_i v_{xi}).
    pub fn total_angular_momentum(&self) -> f64 {
        let mut l = 0.0;
        for agent in &self.agents {
            if agent.state.dim() >= 2 {
                l += agent.mass * (agent.state.q[0] * agent.state.qdot[1]
                    - agent.state.q[1] * agent.state.qdot[0]);
            }
        }
        l
    }

    /// Total kinetic energy.
    pub fn total_kinetic_energy(&self) -> f64 {
        self.agents.iter().map(|a| 0.5 * a.mass * a.state.qdot.norm_squared()).sum()
    }

    /// Total energy (kinetic + potential from fleet interactions).
    pub fn total_energy(&self, potential: &dyn Fn(&AgentFleet) -> f64) -> f64 {
        self.total_kinetic_energy() + potential(self)
    }

    /// Center of mass position.
    pub fn center_of_mass(&self) -> DVector<f64> {
        let dim = self.agents[0].state.dim();
        let total_mass: f64 = self.agents.iter().map(|a| a.mass).sum();
        let mut com = DVector::zeros(dim);
        for agent in &self.agents {
            com += (agent.mass / total_mass) * &agent.state.q;
        }
        com
    }

    /// Center of mass velocity.
    pub fn center_of_mass_velocity(&self) -> DVector<f64> {
        let dim = self.agents[0].state.dim();
        let total_mass: f64 = self.agents.iter().map(|a| a.mass).sum();
        let mut v_com = DVector::zeros(dim);
        for agent in &self.agents {
            v_com += (agent.mass / total_mass) * &agent.state.qdot;
        }
        v_com
    }
}

/// Permutation symmetry: swapping two identical agents doesn't change the fleet Lagrangian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationSymmetry {
    pub agent_i: usize,
    pub agent_j: usize,
}

impl PermutationSymmetry {
    pub fn new(i: usize, j: usize) -> Self {
        Self { agent_i: i, agent_j: j }
    }

    /// Check if two agents are identical (same mass).
    pub fn is_valid(&self, fleet: &AgentFleet) -> bool {
        let ai = &fleet.agents[self.agent_i];
        let aj = &fleet.agents[self.agent_j];
        (ai.mass - aj.mass).abs() < 1e-10
    }
}

/// Fleet invariant: a conserved quantity derived from permutation symmetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetInvariant {
    pub name: String,
    pub value: f64,
}

impl FleetInvariant {
    /// Total fleet momentum invariant.
    pub fn total_momentum(fleet: &AgentFleet, axis: usize) -> Self {
        let p = fleet.total_momentum();
        FleetInvariant {
            name: format!("total_momentum_{}", axis),
            value: p[axis],
        }
    }

    /// Total fleet angular momentum invariant.
    pub fn total_angular_momentum(fleet: &AgentFleet) -> Self {
        FleetInvariant {
            name: "total_angular_momentum".to_string(),
            value: fleet.total_angular_momentum(),
        }
    }

    /// Total fleet energy invariant.
    pub fn total_energy(fleet: &AgentFleet, potential: &dyn Fn(&AgentFleet) -> f64) -> Self {
        FleetInvariant {
            name: "total_energy".to_string(),
            value: fleet.total_energy(potential),
        }
    }

    /// Center of mass momentum (should be conserved for isolated fleet).
    pub fn center_of_mass_momentum(fleet: &AgentFleet, axis: usize) -> Self {
        let total_mass: f64 = fleet.agents.iter().map(|a| a.mass).sum();
        let v_com = fleet.center_of_mass_velocity();
        FleetInvariant {
            name: format!("com_momentum_{}", axis),
            value: total_mass * v_com[axis],
        }
    }
}

/// Compute all fleet invariants from permutation symmetry.
pub fn fleet_invariants(fleet: &AgentFleet) -> Vec<FleetInvariant> {
    let mut invariants = Vec::new();
    let dim = fleet.agents[0].state.dim();

    for axis in 0..dim {
        invariants.push(FleetInvariant::total_momentum(fleet, axis));
    }

    if dim >= 2 {
        invariants.push(FleetInvariant::total_angular_momentum(fleet));
    }

    invariants
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fleet() -> AgentFleet {
        AgentFleet::new(vec![
            FleetAgent {
                id: 0,
                state: AgentState::new(
                    DVector::from_vec(vec![1.0, 0.0]),
                    DVector::from_vec(vec![0.0, 1.0]),
                ),
                mass: 1.0,
            },
            FleetAgent {
                id: 1,
                state: AgentState::new(
                    DVector::from_vec(vec![-1.0, 0.0]),
                    DVector::from_vec(vec![0.0, -1.0]),
                ),
                mass: 1.0,
            },
        ])
    }

    #[test]
    fn test_total_momentum() {
        let fleet = make_fleet();
        let p = fleet.total_momentum();
        assert!((p[0]).abs() < 1e-10);
        assert!((p[1]).abs() < 1e-10);
    }

    #[test]
    fn test_total_angular_momentum() {
        let fleet = make_fleet();
        let l = fleet.total_angular_momentum();
        // L = 1*(1*1 - 0*0) + 1*((-1)*(-1) - 0*0) = 1 + 1 = 2
        assert!((l - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_kinetic_energy() {
        let fleet = make_fleet();
        let t = fleet.total_kinetic_energy();
        // T = ½*1*1 + ½*1*1 = 1
        assert!((t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_center_of_mass() {
        let fleet = make_fleet();
        let com = fleet.center_of_mass();
        assert!((com[0]).abs() < 1e-10);
        assert!((com[1]).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_symmetry_valid() {
        let fleet = make_fleet();
        let perm = PermutationSymmetry::new(0, 1);
        assert!(perm.is_valid(&fleet));
    }

    #[test]
    fn test_fleet_invariants() {
        let fleet = make_fleet();
        let invs = fleet_invariants(&fleet);
        assert!(invs.len() >= 3); // 2 momentum + 1 angular
    }

    #[test]
    fn test_total_energy_with_potential() {
        let fleet = make_fleet();
        let potential = |_: &AgentFleet| 5.0;
        let e = fleet.total_energy(&potential);
        assert!((e - 6.0).abs() < 1e-10); // 1.0 KE + 5.0 PE
    }

    #[test]
    fn test_com_momentum_conservation() {
        let fleet = make_fleet();
        let inv = FleetInvariant::center_of_mass_momentum(&fleet, 0);
        // Total momentum is zero, so COM momentum should be zero
        assert!(inv.value.abs() < 1e-10);
    }
}
