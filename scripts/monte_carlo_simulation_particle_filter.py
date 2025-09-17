"""
monte_carlo_simulation_particle_filter.py

Monte Carlo simulation of the two-observer quantum state tracking task
using a Particle Filter (Bayesian particle approximation) for state estimation.

This script investigates the asymptotic behavior of the average fidelity
between two independent observers (A and B) who are simultaneously estimating
a quantum system state through a series of measurements. The system size
(parameterized by m, the number of bits of memory) is varied to study its
effect on the maximum achievable fidelity F_max(m).

The core result is the relationship: F_max(m) ≈ 1 - α / 2^m

Outputs:
- fidelity_results.csv: Average fidelity vs. measurement number for each m.
- plateau_estimates.csv: Mean plateau fidelity for each m (averaged over trials).
- plateau_estimates_raw.csv: Per-trial plateau fidelity values for bootstrap analysis.
- fidelity_convergence.pdf: Plot of fidelity convergence dynamics.

Usage example:
# From the repository root
python scripts/monte_carlo_simulation_particle_filter.py --m_list 1 2 3 4 5 6 7 --n_max 3000 --trials 1000 --update_mode particle --particles 2048 --measure_protocol same
python monte_carlo_simulation_particle_filter.py --m_list 1 2 3 4 5 6 7 --n_max 3000 --trials 1000 --update_mode particle --particles 2048 --measure_protocol same

# From the scripts directory
python monte_carlo_simulation_particle_filter.py --m_list 1 2 3 4 5 6 7 --n_max 3000 --trials 1000 --update_mode particle --particles 2048 --measure_protocol same

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/operational-quantum-foundations
Citation: DOI:10.5281/zenodo.17139825
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_repo_paths():
    """Determine repository paths for consistent data and figure storage"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Always find repo root by looking for specific markers
    # (like .git directory or setup.py) or go up one level if in scripts
    repo_root = script_dir
    if os.path.basename(script_dir) == 'scripts':
        repo_root = os.path.dirname(script_dir)
    else:
        # Try to find repo root by looking for common markers
        current = script_dir
        while current != os.path.dirname(current):  # while not at root
            if any(os.path.exists(os.path.join(current, marker)) for marker in ['.git', 'setup.py', 'README.md']):
                repo_root = current
                break
            current = os.path.dirname(current)
    
    figures_dir = os.path.join(repo_root, "figures")
    data_dir = os.path.join(repo_root, "data")
    return {
        'repo_root': repo_root,
        'figures_dir': figures_dir,
        'data_dir': data_dir,
        'script_dir': script_dir
    }


def normalize(state: np.ndarray) -> np.ndarray:
    """
    Normalize a quantum state vector to unit norm.
    
    Args:
        state: Complex-valued quantum state vector
        
    Returns:
        Normalized state vector, or original state if norm is zero
    """
    norm = np.linalg.norm(state)
    return state if norm == 0 else state / norm


def bloch_from_state(psi: np.ndarray) -> np.ndarray:
    """
    Convert a two-level quantum state to its Bloch vector representation.
    
    For a qubit state |ψ⟩ = a|0⟩ + b|1⟩, the Bloch vector components are:
    - sx = 2 * Re(a* b) (x-component)  
    - sy = 2 * Im(a* b) (y-component)
    - sz = |a|² - |b|² (z-component)
    
    Args:
        psi: Two-component complex state vector [a, b]
        
    Returns:
        Three-component real Bloch vector [sx, sy, sz]
    """
    a, b = psi[0], psi[1]
    sx = 2.0 * np.real(np.conj(a) * b)  # X Pauli expectation value
    sy = 2.0 * np.imag(np.conj(a) * b)  # Y Pauli expectation value  
    sz = np.abs(a) ** 2 - np.abs(b) ** 2  # Z Pauli expectation value
    return np.array([sx, sy, sz], dtype=float)


def state_from_bloch(vec: np.ndarray) -> np.ndarray:
    """
    Convert a Bloch vector back to a quantum state representation.
    
    Given Bloch vector (x, y, z), constructs the corresponding qubit state
    using spherical coordinates: θ = arccos(z/r), φ = atan2(y, x)
    where r = ||(x, y, z)||.
    
    Args:
        vec: Three-component Bloch vector [x, y, z]
        
    Returns:
        Two-component normalized complex state vector
    """
    x, y, z = vec
    r = math.sqrt(max(0.0, x * x + y * y + z * z))  # Bloch vector magnitude
    
    # Handle degenerate case of zero vector
    if r < 1e-12:
        return normalize(np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=complex))
    
    # Convert to spherical coordinates on the Bloch sphere
    theta = math.acos(max(-1.0, min(1.0, z / r)))  # Polar angle [0, π]
    phi = math.atan2(y, x)  # Azimuthal angle [-π, π]
    
    # Construct state components from spherical coordinates
    a = math.cos(theta / 2.0)  # |0⟩ amplitude
    b = math.sin(theta / 2.0) * complex(math.cos(phi), math.sin(phi))  # |1⟩ amplitude
    
    return normalize(np.array([a, b], dtype=complex))


def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Calculate the quantum fidelity between two pure states.
    
    For pure states |ψ⟩ and |φ⟩, the fidelity is F = |⟨ψ|φ⟩|²,
    which measures the overlap probability between the states.
    
    Args:
        psi: First quantum state vector
        phi: Second quantum state vector
        
    Returns:
        Fidelity value in [0, 1], where 1 indicates identical states
    """
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def measure_in_basis(state: np.ndarray, basis: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Perform a quantum measurement of a state in a given orthonormal basis.
    
    Simulates the probabilistic collapse of the quantum state according to
    the Born rule, where measurement probabilities are |⟨basis_i|state⟩|².
    
    Args:
        state: Quantum state to be measured
        basis: 2×2 matrix with basis vectors as columns
        
    Returns:
        Tuple of (measurement_outcome, collapsed_state)
        - measurement_outcome: 0 or 1 indicating which basis vector was measured
        - collapsed_state: Post-measurement normalized state
    """
    # Calculate measurement probabilities using Born rule
    probs = np.abs(np.conj(basis.T) @ state) ** 2
    probs = probs.flatten()
    probs = probs / probs.sum()  # Ensure normalization
    
    # Sample measurement outcome according to probabilities
    outcome = int(np.random.choice([0, 1], p=probs))
    
    # State collapses to the corresponding basis vector
    collapsed = basis[:, outcome]
    return outcome, normalize(collapsed)


def quantize_bloch_theta_only(bloch_vec: np.ndarray, m_bits: int) -> Tuple[int, np.ndarray]:
    """
    Quantize a Bloch vector using theta-only discretization for memory constraints.
    
    This function implements the memory limitation by discretizing only the polar
    angle θ into 2^m uniform bins, while setting the azimuthal angle φ = 0.
    This represents the finite memory capacity of quantum state estimators.
    
    Args:
        bloch_vec: Three-component Bloch vector [x, y, z]
        m_bits: Number of memory bits determining quantization resolution
        
    Returns:
        Tuple of (quantization_index, quantized_state)
        - quantization_index: Discrete bin index for the polar angle
        - quantized_state: Quantum state corresponding to quantized Bloch vector
    """
    x, y, z = bloch_vec
    r = math.sqrt(x * x + y * y + z * z)
    
    # Extract polar angle from Bloch vector
    theta = 0.0 if r < 1e-12 else math.acos(max(-1.0, min(1.0, z / r)))
    
    # Discretize theta into 2^m uniform bins
    K = 2 ** m_bits  # Total number of quantization levels
    idx = int(min(K - 1, math.floor(theta / math.pi * K)))  # Bin index
    
    # Map back to center of quantization bin
    theta_center = (idx + 0.5) * (math.pi / K)
    
    # Construct quantized Bloch vector (φ = 0, so y = 0)
    xq, yq, zq = math.sin(theta_center), 0.0, math.cos(theta_center)
    bloch_q = np.array([xq, yq, zq], dtype=float)
    
    # Convert back to state representation
    state_q = state_from_bloch(bloch_q)
    return idx, state_q


def random_particles_on_sphere(M: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uniformly distributed random points on the unit sphere (Bloch sphere).
    
    Uses the standard method for uniform spherical sampling:
    - θ is sampled from arccos(uniform(-1,1)) to ensure uniform area distribution
    - φ is sampled uniformly from [-π, π]
    
    Args:
        M: Number of random particles to generate
        
    Returns:
        Tuple of (theta_array, phi_array) containing spherical coordinates
        - theta_array: Polar angles in [0, π]
        - phi_array: Azimuthal angles in [-π, π]
    """
    u, v = np.random.rand(M), np.random.rand(M)
    theta = np.arccos(1 - 2 * u)   # Uniform distribution on [0, π]
    phi = 2 * math.pi * v - math.pi  # Uniform distribution on [-π, π]
    return theta, phi


def bloch_from_theta_phi(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian Bloch vector coordinates.
    
    Standard spherical-to-Cartesian transformation for points on unit sphere:
    x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
    
    Args:
        theta: Polar angle array [0, π]
        phi: Azimuthal angle array [-π, π]
        
    Returns:
        Array of shape (N, 3) containing Bloch vectors [x, y, z]
    """
    x = np.sin(theta) * np.cos(phi)  # X component
    y = np.sin(theta) * np.sin(phi)  # Y component  
    z = np.cos(theta)                # Z component
    return np.stack([x, y, z], axis=1)


def likelihood_of_outcome(particle_bloch: np.ndarray, outcome_state: np.ndarray) -> np.ndarray:
    """
    Calculate measurement likelihood for each particle given observed outcome.
    
    For Bayesian particle filtering, this computes P(measurement|particle_state)
    using the Born rule: likelihood = |⟨outcome_state|particle_state⟩|²
    
    Args:
        particle_bloch: Array of shape (N, 3) containing N Bloch vectors
        outcome_state: Two-component quantum state that was observed
        
    Returns:
        Array of N likelihood values for updating particle weights
    """
    # Convert Bloch vectors back to quantum state representation
    thetas = np.arccos(np.clip(particle_bloch[:, 2], -1.0, 1.0))  # Extract polar angles
    phis = np.arctan2(particle_bloch[:, 1], particle_bloch[:, 0])  # Extract azimuthal angles
    
    # Reconstruct quantum states from spherical coordinates
    a = np.cos(thetas / 2.0)  # |0⟩ amplitudes
    b = np.sin(thetas / 2.0) * (np.cos(phis) + 1j * np.sin(phis))  # |1⟩ amplitudes
    particle_states = np.vstack([a, b]).T  # Shape: (N, 2)
    
    # Calculate overlap probabilities using Born rule
    overlaps = np.abs(np.dot(np.conj(particle_states), outcome_state)) ** 2
    
    # Ensure numerical stability by clipping to avoid zero likelihoods
    return np.clip(overlaps.flatten(), 1e-12, 1.0)


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    """
    Perform systematic resampling for particle filter to combat degeneracy.
    
    Systematic resampling provides lower variance than multinomial resampling
    by using equally-spaced random positions. This helps maintain particle
    diversity in the filter.
    
    Args:
        weights: Normalized particle weights summing to 1
        
    Returns:
        Array of particle indices for resampling
    """
    N = len(weights)
    # Generate systematic sampling positions
    positions = (np.arange(N) + np.random.rand()) / N
    cumulative = np.cumsum(weights)  # Cumulative weight distribution
    
    indexes = np.zeros(N, dtype=int)
    i, j = 0, 0
    
    # Systematic sampling algorithm
    while i < N:
        if positions[i] < cumulative[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    
    return indexes


def particle_filter_update(theta, phi, weights, outcome_state, resample_thresh=0.5):
    """
    Update particle filter state based on new measurement outcome.
    
    Implements the standard particle filter update cycle:
    1. Weight update using measurement likelihood
    2. Resampling when effective sample size drops below threshold
    3. Jitter addition to prevent particle collapse
    
    Args:
        theta: Current particle polar angles
        phi: Current particle azimuthal angles  
        weights: Current particle weights
        outcome_state: Observed measurement outcome state
        resample_thresh: ESS threshold for triggering resampling (default: 0.5)
        
    Returns:
        Tuple of (updated_theta, updated_phi, updated_weights)
    """
    # Convert spherical coordinates to Bloch vectors for likelihood calculation
    bloch = bloch_from_theta_phi(theta, phi)
    
    # Update particle weights using measurement likelihood
    lik = likelihood_of_outcome(bloch, outcome_state)
    weights *= lik
    
    # Normalize weights to maintain probability distribution
    weight_sum = np.sum(weights)
    weights = np.ones_like(weights) / len(weights) if weight_sum <= 0 else weights / weight_sum
    
    # Calculate effective sample size to assess particle degeneracy
    ess = 1.0 / np.sum(weights ** 2)
    M = len(weights)
    
    # Resample if effective sample size drops below threshold
    if ess < resample_thresh * M:
        # Systematic resampling to combat particle degeneracy
        idx = systematic_resample(weights)
        theta, phi = theta[idx], phi[idx]
        weights = np.ones(M) / M  # Reset to uniform weights
        
        # Add small random jitter to prevent particle collapse
        jitter_theta = (np.random.randn(M) * (0.5 * math.pi / M)).clip(-0.01, 0.01)
        jitter_phi = (np.random.randn(M) * (0.5 * 2 * math.pi / M)).clip(-0.02, 0.02)
        
        # Apply jitter while maintaining valid spherical coordinate ranges
        theta = np.clip(theta + jitter_theta, 0.0, math.pi)
        phi = (phi + jitter_phi + math.pi) % (2 * math.pi) - math.pi
    
    return theta, phi, weights


def estimate_state_from_particles(theta, phi, weights) -> np.ndarray:
    """
    Estimate the quantum state from weighted particle distribution.
    
    Computes the weighted average of particle Bloch vectors, then normalizes
    and converts back to quantum state representation. This provides the
    Bayesian point estimate of the current quantum state.
    
    Args:
        theta: Particle polar angles
        phi: Particle azimuthal angles
        weights: Particle weights (normalized)
        
    Returns:
        Estimated quantum state as two-component complex vector
    """
    # Convert particles to Bloch vector representation
    bloch = bloch_from_theta_phi(theta, phi)
    
    # Compute weighted average Bloch vector
    mean_vec = np.sum(bloch * weights.reshape(-1, 1), axis=0)
    
    # Normalize to unit sphere (handle degenerate case)
    norm = np.linalg.norm(mean_vec)
    mean_vec = np.array([1.0, 0.0, 0.0]) if norm < 1e-12 else mean_vec / norm
    
    # Convert back to quantum state representation
    return state_from_bloch(mean_vec)


def run_simulation(m_list, n_max, trials, learning_rate, initial_state,
                   measure_protocol="independent", update_mode="particle",
                   plateau_window=30, outdir=".", particles_count=1024):
    """
    Execute the main Monte Carlo simulation for two-observer quantum state tracking.
    
    This function implements the core simulation loop where two independent observers
    (A and B) simultaneously estimate a quantum state through sequential measurements.
    The simulation studies how memory constraints (parameterized by m) affect the
    asymptotic fidelity between the observers' estimates.
    
    Args:
        m_list: List of memory bit values to simulate
        n_max: Maximum number of measurements per trial
        trials: Number of Monte Carlo trials for statistical averaging
        learning_rate: Learning rate for simple update mode (unused in particle mode)
        initial_state: True initial quantum state being estimated
        measure_protocol: Measurement strategy ("independent", "same", "fixed_X")
        update_mode: State update method ("particle" or "simple")
        plateau_window: Window size for plateau fidelity estimation
        outdir: Output directory for results
        particles_count: Number of particles for particle filter
        
    Returns:
        Tuple of (full_results_df, plateau_summary_df, raw_plateau_df)
    """
    # Define standard Pauli measurement bases
    basis_X = np.array([[1 / math.sqrt(2), 1 / math.sqrt(2)],
                        [1 / math.sqrt(2), -1 / math.sqrt(2)]], dtype=complex)  # σx eigenbasis
    basis_Z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)  # σz eigenbasis (computational)
    basis_Y = np.array([[1 / math.sqrt(2), 1 / math.sqrt(2)],
                        [1j / math.sqrt(2), -1j / math.sqrt(2)]], dtype=complex)  # σy eigenbasis
    bases = [basis_X, basis_Y, basis_Z]
    
    results, raw_plateaus = [], []  # Storage for simulation results
    
    # Main simulation loop over memory values
    for m in m_list:
        fidelity_accum = np.zeros(n_max, dtype=float)  # Accumulator for averaging across trials
        
        # Monte Carlo trials for statistical sampling
        for t in range(trials):
            # Initialize observer states based on update mode
            if update_mode == 'particle':
                # Initialize particle filters for both observers
                theta_A, phi_A = random_particles_on_sphere(particles_count)
                weights_A = np.ones(particles_count) / particles_count
                theta_B, phi_B = random_particles_on_sphere(particles_count)  
                weights_B = np.ones(particles_count) / particles_count
                
                # Initial state estimates from particle distributions
                est_A = estimate_state_from_particles(theta_A, phi_A, weights_A)
                est_B = estimate_state_from_particles(theta_B, phi_B, weights_B)
            else:
                # Simple update mode: start with copies of initial state
                est_A, est_B = normalize(initial_state.copy()), normalize(initial_state.copy())
            
            trial_fidelities = []  # Store fidelities for this trial
            
            # Sequential measurement loop
            for n in range(n_max):
                # Select measurement bases according to protocol
                if measure_protocol == "independent":
                    # Each observer chooses random basis independently
                    basis_A, basis_B = bases[np.random.randint(0, 3)], bases[np.random.randint(0, 3)]
                elif measure_protocol == "same":
                    # Both observers use same randomly chosen basis
                    chosen = bases[np.random.randint(0, 3)]
                    basis_A = basis_B = chosen
                elif measure_protocol == "fixed_X":
                    # Both observers always measure in X basis
                    basis_A = basis_B = basis_X
                else:
                    raise ValueError("Unknown measure_protocol")
                
                # Perform measurements on the true state (independent measurements)
                outcome_A, _ = measure_in_basis(initial_state, basis_A)
                outcome_B, _ = measure_in_basis(initial_state, basis_B)
                
                # Extract measurement outcome eigenstates for updates
                eig_A, eig_B = basis_A[:, outcome_A], basis_B[:, outcome_B]
                
                # Update observer estimates based on their measurements
                if update_mode == 'particle':
                    # Particle filter updates for both observers
                    theta_A, phi_A, weights_A = particle_filter_update(theta_A, phi_A, weights_A, eig_A)
                    est_A = estimate_state_from_particles(theta_A, phi_A, weights_A)
                    _, est_A = quantize_bloch_theta_only(bloch_from_state(est_A), m)  # Apply memory constraint
                    
                    theta_B, phi_B, weights_B = particle_filter_update(theta_B, phi_B, weights_B, eig_B)
                    est_B = estimate_state_from_particles(theta_B, phi_B, weights_B)
                    _, est_B = quantize_bloch_theta_only(bloch_from_state(est_B), m)  # Apply memory constraint
                else:
                    # Simple gradient-based updates (fallback mode)
                    est_A = simple_update_fallback(est_A, eig_A, learning_rate)
                    _, est_A = quantize_bloch_theta_only(bloch_from_state(est_A), m)  # Apply memory constraint
                    
                    est_B = simple_update_fallback(est_B, eig_B, learning_rate)
                    _, est_B = quantize_bloch_theta_only(bloch_from_state(est_B), m)  # Apply memory constraint
                
                # Calculate fidelity between observer estimates
                F = fidelity_pure(est_A, est_B)
                trial_fidelities.append(F)
                fidelity_accum[n] += F  # Accumulate for cross-trial averaging
            
            # Calculate plateau estimate for this trial (average over final window)
            plateau_mean_trial = float(np.mean(trial_fidelities[-plateau_window:]))
            raw_plateaus.append({"m": int(m), "trial": int(t), "plateau_mean": plateau_mean_trial})
        
        # Average fidelities across all trials for this m value
        fidelity_avg = fidelity_accum / trials
        
        # Store results for each measurement step
        for n in range(n_max):
            results.append({"n": n + 1, "m": m, "fidelity_avg": float(fidelity_avg[n])})
    
    # Convert results to DataFrames for analysis and export
    df_all = pd.DataFrame(results)
    df_plateau_raw = pd.DataFrame(raw_plateaus)
    df_plateau = df_plateau_raw.groupby("m")["plateau_mean"].mean().reset_index()
    
    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Export results to CSV files
    df_all.to_csv(os.path.join(outdir, "fidelity_results.csv"), index=False)
    df_plateau.to_csv(os.path.join(outdir, "plateau_estimates.csv"), index=False)
    df_plateau_raw.to_csv(os.path.join(outdir, "plateau_estimates_raw.csv"), index=False)
    
    # Generate convergence dynamics plot
    plt.figure(figsize=(10, 6))
    for m in m_list:
        sub = df_all[df_all["m"] == m]
        plt.plot(sub["n"], sub["fidelity_avg"], label=f"m={m}")
    plt.xlabel("Measurement number n")
    plt.ylabel("Average fidelity between A and B")
    plt.title("Convergence of Average Fidelity for Different System Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fidelity_convergence.pdf"))
    plt.close()
    
    return df_all, df_plateau, df_plateau_raw


def simple_update_fallback(est_state: np.ndarray, measurement_eigenstate: np.ndarray, lr: float) -> np.ndarray:
    """
    Simple gradient-based state update for comparison with particle filter.
    
    This implements a basic learning rule that interpolates between the current
    estimate and the measurement outcome in Bloch space. Used as a fallback
    when particle filtering is disabled.
    
    Args:
        est_state: Current state estimate
        measurement_eigenstate: Measured eigenstate to incorporate
        lr: Learning rate controlling update step size
        
    Returns:
        Updated state estimate after incorporating measurement
    """
    # Convert states to Bloch vector representation for interpolation
    bloch = bloch_from_state(est_state)
    bloch_eig = bloch_from_state(measurement_eigenstate)
    
    # Linear interpolation in Bloch space
    bloch_upd = (1.0 - lr) * bloch + lr * bloch_eig
    
    # Normalize to unit sphere (handle degenerate case)
    norm_b = np.linalg.norm(bloch_upd)
    bloch_upd = np.array([1.0, 0.0, 0.0]) if norm_b < 1e-12 else bloch_upd / norm_b
    
    # Convert back to state representation
    return state_from_bloch(bloch_upd)


def parse_args():
    """
    Parse command-line arguments for simulation configuration.
    
    Returns:
        Parsed arguments namespace containing all simulation parameters
    """
    parser = argparse.ArgumentParser(description="Convergence of Average Fidelity for Different System Sizes")
    
    # Core simulation parameters
    parser.add_argument("--m_list", nargs="+", type=int, default=[1, 2, 3, 4, 6],
                       help="List of memory bit values to simulate")
    parser.add_argument("--n_max", type=int, default=300,
                       help="Maximum number of measurements per trial")
    parser.add_argument("--trials", type=int, default=200,
                       help="Number of Monte Carlo trials for averaging")
    parser.add_argument("--learning_rate", type=float, default=0.18,
                       help="Learning rate for simple update mode")
    
    # Protocol configuration
    parser.add_argument("--measure_protocol", type=str, default="independent",
                        choices=["independent", "same", "fixed_X"],
                        help="Measurement basis selection protocol")
    parser.add_argument("--update_mode", type=str, default="particle", 
                        choices=["simple", "particle"],
                        help="State estimation method")
    
    # Analysis parameters  
    parser.add_argument("--plateau_window", type=int, default=30,
                       help="Window size for plateau fidelity estimation")
    
    # Particle filter specific
    parser.add_argument("--particles", type=int, default=1024,
                       help="Number of particles for particle filter")
    
    return parser.parse_args()


def main():
    """
    Main execution function coordinating the Monte Carlo simulation.
    
    Parses command-line arguments, initializes simulation parameters,
    executes the simulation, and reports results.
    """
    args = parse_args()
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Initialize true quantum state (equal superposition)
    initial_state = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)], dtype=complex)
    
    # Get repository paths
    paths = get_repo_paths()
    args.outdir = paths['figures_dir']
    
    # Execute main simulation
    df_all, df_plateau, df_plateau_raw = run_simulation(
        m_list=args.m_list,
        n_max=args.n_max,
        trials=args.trials,
        learning_rate=args.learning_rate,
        initial_state=initial_state,
        measure_protocol=args.measure_protocol,
        update_mode=args.update_mode,
        plateau_window=args.plateau_window,
        outdir=args.outdir,
        particles_count=args.particles,
    )
    
    # Report results and file locations
    print(f"Results saved: {os.path.join(args.outdir, 'fidelity_results.csv')}")
    print(f"Plateau estimates (mean): {os.path.join(args.outdir, 'plateau_estimates.csv')}")
    print(f"Per-trial plateau values: {os.path.join(args.outdir, 'plateau_estimates_raw.csv')}")
    print(f"Convergence plot saved: {os.path.join(args.outdir, 'fidelity_convergence.pdf')}")
    
    # Display plateau estimates summary
    print("\nPlateau estimates (F_max(m)):")
    print(df_plateau.to_string(index=False))


if __name__ == '__main__':
    main()