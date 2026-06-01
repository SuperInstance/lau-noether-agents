# lau-noether-agents

**Noether's theorem for agent systems** — every symmetry of the dynamics yields a conserved quantity, and every conserved quantity exposes a symmetry.

This crate applies the machinery of Lagrangian and Hamiltonian mechanics to multi-agent fleets. You define a Lagrangian for your agents, detect its symmetries, and the library automatically derives the corresponding conserved charges (momentum, angular momentum, energy, …) and verifies they hold along simulated trajectories.

---

## What This Does

- **Lagrangian mechanics for agents**: define agent state as generalized coordinates `(q, q̇)`, specify kinetic and potential energy, and get equations of motion for free.
- **Symmetry detection**: probe a Lagrangian system for translation, rotation, gauge, and scaling symmetries by checking infinitesimal invariance.
- **Noether charge computation**: for every detected symmetry, compute the conserved charge `Q = Σ pᵢ δqᵢ` and verify it stays constant.
- **Broken symmetry analysis**: quantify how much a perturbation breaks a symmetry, measure charge drift rate, and compute adiabatic invariants.
- **Hamiltonian formulation**: Legendre-transform to `(q, p)` phase space, integrate with Störmer-Verlet, check Liouville's theorem and Poisson brackets.
- **Discrete Noether theorem**: exact conservation for time-stepping schemes via discrete Lagrangians (trapezoidal and midpoint variants).
- **Fleet-level invariants**: permutation symmetry of identical agents → total fleet momentum, angular momentum, and center-of-mass conservation.
- **CUDAclaw application**: cell-agent dynamics with pairwise interaction potentials, full conservation proofs, and discrete Noether charge tracking.

---

## Key Idea

> **Noether's theorem**: If the Lagrangian `L(q, q̇, t)` is invariant under an infinitesimal transformation `q → q + ε δq`, then the quantity `Q = Σᵢ pᵢ δqᵢ` is exactly conserved.

This crate makes that theorem computational. You plug in a Lagrangian; the library finds the symmetries and proves the conservation laws.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-noether-agents = "0.1"
```

Or use directly via git:

```toml
[dependencies]
lau-noether-agents = { git = "https://github.com/SuperInstance/lau-noether-agents" }
```

Requires Rust 2021 edition. Dependencies: [`nalgebra`](https://crates.io/crates/nalgebra) (linear algebra), `serde` + `serde_json` (serialization).

---

## Quick Start

```rust
use lau_noether_agents::*;
use nalgebra::DVector;

// 1. Define a system — free particle in 2D, unit mass
let system = SimpleLagrangian::uniform(2, 1.0);

// 2. Set initial state: position (3, 4), velocity (1, 2)
let state = AgentState::new(
    DVector::from_vec(vec![3.0, 4.0]),
    DVector::from_vec(vec![1.0, 2.0]),
);

// 3. Detect symmetries
let symmetries = detect_symmetries(&system, &state, 2);
println!("Detected {} symmetries", symmetries.len());

// 4. Simulate with conservation tracking
let result = simulate_with_conservation(
    &system, &state, 0.01, 100,
    &symmetries.iter().map(|s| s.kind.clone()).collect::<Vec<_>>(),
);
println!("All charges conserved: {}", result.all_conserved(1e-6));
println!("Max relative deviation: {:.2e}", result.max_relative_deviation());
```

### Fleet of Agents

```rust
let fleet = AgentFleet::new(vec![
    FleetAgent { id: 0, state: /* … */, mass: 1.0 },
    FleetAgent { id: 1, state: /* … */, mass: 1.0 },
]);

// Total fleet invariants (from permutation symmetry)
let invariants = fleet_invariants(&fleet);
for inv in &invariants {
    println!("{}: {:.6}", inv.name, inv.value);
}
```

### Broken Symmetry Analysis

```rust
// Harmonic oscillator: translation symmetry is broken by the potential
let sys = HarmonicLagrangian::uniform(1, 1.0, 1.0);
let state = AgentState::new(DVector::from_vec(vec![1.0]), DVector::from_vec(vec![0.0]));
let result = simulate_with_conservation(&sys, &state, 0.01, 100,
    &[SymmetryKind::Translation { axis: 0 }]);

let breaking = SymmetryBreaking::analyze(
    &TranslationSymmetry::new(0, 1), &sys,
    &result.trajectory, &result.times,
);
println!("Invariance violation: {:.4}", breaking.invariance_violation);
println!("Conservation quality: {:.4}", breaking.conservation_quality);
```

---

## API Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `AgentState` | `lagrangian` | Generalized coordinates `(q, q̇)` |
| `LagrangianSystem` | `lagrangian` | Trait: kinetic energy, potential energy, mass matrix |
| `SimpleLagrangian` | `lagrangian` | Free particle with diagonal mass matrix |
| `HarmonicLagrangian` | `lagrangian` | Harmonic potential `V = ½ k (q - q₀)²` |
| `CanonicalState` | `hamiltonian` | Phase-space state `(q, p)` |

### Symmetry & Noether

| Type | Module | Description |
|------|--------|-------------|
| `SymmetryTransform` | `noether` | Trait: infinitesimal transformation `δq(q)` |
| `TranslationSymmetry` | `noether` | `δq_i = ê_axis` → linear momentum |
| `RotationSymmetry` | `noether` | `δq = (−qⱼ, qᵢ)` → angular momentum |
| `TimeTranslationSymmetry` | `noether` | `δq = q̇` → energy |
| `NoetherCharge` | `noether` | Conserved charge `Q = Σ pᵢ δqᵢ` |
| `detect_symmetries()` | `symmetry` | Auto-detect symmetries of a Lagrangian |
| `noether_verify()` | `noether` | Verify symmetry → charge conservation on a trajectory |

### Simulation & Verification

| Type | Module | Description |
|------|--------|-------------|
| `simulate_with_conservation()` | `conservation` | Symplectic Euler simulation with charge tracking |
| `symplectic_euler_step()` | `conservation` | Single symplectic Euler step |
| `velocity_verlet_step()` | `conservation` | Velocity Verlet integrator |
| `SimulationResult` | `conservation` | Trajectory + charge trackers + conservation summary |
| `ConservedChargeTracker` | `charge` | Track a Noether charge through time |

### Hamiltonian

| Type | Module | Description |
|------|--------|-------------|
| `hamiltonian()` | `hamiltonian` | Compute `H(q, p) = Σ pᵢ q̇ᵢ − L` |
| `hamiltons_equations()` | `hamiltonian` | `q̇ = ∂H/∂p`, `ṗ = −∂H/∂q` |
| `stormer_verlet_step()` | `hamiltonian` | Symplectic Störmer-Verlet integrator |
| `poisson_bracket()` | `hamiltonian` | `{f, g} = Σ (∂f/∂qᵢ ∂g/∂pᵢ − ∂f/∂pᵢ ∂g/∂qᵢ)` |
| `liouville_check()` | `hamiltonian` | Verify phase-space volume preservation |

### Discrete Mechanics

| Type | Module | Description |
|------|--------|-------------|
| `DiscreteLagrangian` | `discrete` | Trait for `L_d(qₖ, qₖ₊₁, h)` |
| `TrapezoidalDiscreteLagrangian` | `discrete` | Trapezoidal quadrature |
| `MidpointDiscreteLagrangian` | `discrete` | Midpoint quadrature |
| `DiscreteNoetherCharge` | `discrete` | Exactly-conserved discrete charge |
| `discrete_euler_lagrange_residual()` | `discrete` | Discrete EL equation residual |

### Fleet & CUDAclaw

| Type | Module | Description |
|------|--------|-------------|
| `AgentFleet` | `fleet` | Collection of agents with fleet-level invariants |
| `FleetInvariant` | `fleet` | Total momentum, angular momentum, energy |
| `PermutationSymmetry` | `fleet` | Swapping identical agents |
| `CellDynamics` | `cudaclaw` | Grid-based agents with pairwise potentials |
| `ConservationProof` | `cudaclaw` | Full discrete conservation proof |
| `FleetInvariantTracker` | `cudaclaw` | Track energy/momentum/angular momentum over time |

---

## How It Works

1. **Define the dynamics** via the `LagrangianSystem` trait — you provide `kinetic_energy`, `potential_energy`, and optionally `mass_matrix`.
2. **Detect symmetries** by perturbing the state along candidate transformations and measuring Lagrangian invariance to `O(ε²)`.
3. **Compute Noether charges** as `Q = p · δq` where `p = M q̇` are conjugate momenta.
4. **Simulate** with a symplectic integrator (symplectic Euler or velocity Verlet) that naturally preserves the geometric structure.
5. **Verify conservation** by tracking charges along the trajectory — exact symmetries yield constant charges to machine precision.

For the Hamiltonian path: Legendre-transform to `(q, p)`, integrate with Störmer-Verlet, and check Liouville's theorem (phase-space volume = 1).

For discrete mechanics: define `L_d(qₖ, qₖ₊₁)` and the discrete Noether charge is *exactly* conserved by the discrete Euler-Lagrange equations, regardless of time-step size.

---

## The Math

### Noether's Theorem (Continuous)

Given a Lagrangian `L(q, q̇)` and an infinitesimal transformation `q → q + ε δq`:

- **If** `δL = 0` (the Lagrangian is invariant)
- **Then** `Q = Σᵢ pᵢ δqᵢ` is conserved: `dQ/dt = 0`

where `pᵢ = ∂L/∂q̇ᵢ = Mᵢⱼ q̇ⱼ` are the conjugate momenta.

| Symmetry | δq | Conserved charge |
|----------|-----|-----------------|
| Translation along axis *i* | `êᵢ` | Linear momentum `pᵢ` |
| Rotation in (*i*, *j*) plane | `(−qⱼ, qᵢ)` | Angular momentum `L = x pᵧ − y pₓ` |
| Time translation | `q̇` | Energy `E = 2T` |
| Scaling | `q` | Dilatation `D = p · q` |
| Gauge (direction *d*) | `d` | Gauge charge `Q = p · d` |

### Euler-Lagrange Equations

```
d/dt (∂L/∂q̇) − ∂L/∂q = 0
```

For `T = ½ q̇ᵀ M q̇` this gives `M q̈ + ∂V/∂q = 0`.

### Discrete Noether Theorem

For a discrete Lagrangian `L_d(qₖ, qₖ₊₁)` approximating `∫ L dt`:

- **Discrete momentum**: `pₖ₊₁ = ∂L_d/∂qₖ₊₁`
- **Discrete EL**: `D₁L_d(qₖ, qₖ₊₁) + D₂L_d(qₖ₋₁, qₖ) = 0`
- **Discrete Noether charge**: `Q_d = pₖ₊₁ · δqₖ₊₁` is exactly conserved.

### Hamiltonian Formulation

Legendre transform: `H(q, p) = p · q̇ − L`

Hamilton's equations: `q̇ = ∂H/∂p`, `ṗ = −∂H/∂q`

Poisson bracket: `{f, g} = Σᵢ (∂f/∂qᵢ ∂g/∂pᵢ − ∂f/∂pᵢ ∂g/∂qᵢ)`

Time evolution: `df/dt = {f, H}` → conserved quantities satisfy `{Q, H} = 0`.

Liouville's theorem: the symplectic flow preserves phase-space volume (`det J = 1`).

---

## License

MIT
