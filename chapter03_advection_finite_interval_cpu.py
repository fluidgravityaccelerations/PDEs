#!/usr/bin/env python3
"""Finite-interval advection with prescribed inflow and numerical outflow closures.

The program solves u_t + a u_x = 0 on [0, L] for a > 0.  The left boundary
is prescribed from the characteristic solution, while the right endpoint is
closed numerically when the centered stencil requires it.  It compares frozen,
copy, linear-extrapolation, and discrete outgoing closures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FiniteRunResult:
    x: Array
    numerical: Array
    exact: Array
    times: Array
    residual_history: Array
    error_history: Array
    dt: float
    nsteps: int
    courant: float
    relative_l2_error: float
    residual_energy_ratio: float
    max_residual: float


def compact_cosine_pulse(x: Array, center: float, half_width: float) -> Array:
    """A compact C1 cosine bell, exactly zero outside its support."""
    distance = np.abs(x - center)
    values = np.zeros_like(x, dtype=np.float64)
    inside = distance <= half_width
    values[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside] / half_width))
    return values


def gaussian_pulse(x: Array, center: float, sigma: float) -> Array:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def initial_data(x: Array, case: str, length: float) -> Array:
    if case == "cosine":
        return compact_cosine_pulse(x, center=0.25 * length, half_width=0.10 * length)
    if case == "gaussian":
        return gaussian_pulse(x, center=0.25 * length, sigma=0.045 * length)
    raise ValueError(f"Unknown finite-interval case: {case}")


def inflow_value(time: float, *, case: str, velocity: float, length: float) -> float:
    """Prescribed physical inflow data at x=0.

    The examples use zero inflow.  The arguments are retained to make the
    function easy to replace by a nonzero manufactured boundary signal.
    """
    del time, case, velocity, length
    return 0.0


def exact_finite_solution(
    x: Array,
    time: float,
    *,
    velocity: float,
    length: float,
    case: str,
) -> Array:
    """Characteristic solution with zero inflow for positive velocity."""
    if velocity <= 0.0:
        raise ValueError("The finite-interval example currently assumes velocity > 0")
    feet = x - velocity * time
    exact = np.zeros_like(x, dtype=np.float64)
    from_initial_line = feet >= 0.0
    if np.any(from_initial_line):
        exact[from_initial_line] = initial_data(feet[from_initial_line], case, length)
    return exact


def apply_outflow_closure(
    new: Array,
    old: Array,
    *,
    closure: str,
    courant: float,
) -> None:
    """Update the final point after the interior values have been computed."""
    if closure == "frozen":
        new[-1] = old[-1]
    elif closure == "copy":
        new[-1] = new[-2]
    elif closure == "extrapolation":
        new[-1] = 2.0 * new[-2] - new[-3]
    elif closure == "outgoing":
        q = (1.0 - courant) / (1.0 + courant)
        new[-1] = old[-2] + q * (old[-1] - new[-2])
    else:
        raise ValueError(f"Unknown outflow closure: {closure}")


def one_step_finite(
    old: Array,
    *,
    method: str,
    closure: str,
    courant: float,
    inflow: float,
) -> Array:
    """Advance a one-step method on a finite interval for positive velocity."""
    new = np.empty_like(old)
    new[0] = inflow

    if method == "upwind":
        # The one-sided stencil follows the characteristics and therefore does
        # not require an additional outflow closure.
        new[1:] = old[1:] - courant * (old[1:] - old[:-1])
        return new

    right = old[2:]
    center = old[1:-1]
    left = old[:-2]

    if method == "lax-friedrichs":
        new[1:-1] = 0.5 * (right + left) - 0.5 * courant * (right - left)
    elif method == "lax-wendroff":
        new[1:-1] = (
            center
            - 0.5 * courant * (right - left)
            + 0.5 * courant * courant * (right - 2.0 * center + left)
        )
    elif method == "ftcs":
        new[1:-1] = center - 0.5 * courant * (right - left)
    else:
        raise ValueError(f"Unsupported finite-interval one-step method: {method}")

    apply_outflow_closure(new, old, closure=closure, courant=courant)
    return new


def leapfrog_start(
    initial: Array,
    *,
    startup: str,
    closure: str,
    courant: float,
    inflow: float,
) -> Array:
    if startup == "lax-wendroff":
        return one_step_finite(
            initial,
            method="lax-wendroff",
            closure=closure,
            courant=courant,
            inflow=inflow,
        )
    if startup == "upwind":
        return one_step_finite(
            initial,
            method="upwind",
            closure=closure,
            courant=courant,
            inflow=inflow,
        )
    if startup == "ftcs":
        return one_step_finite(
            initial,
            method="ftcs",
            closure=closure,
            courant=courant,
            inflow=inflow,
        )
    raise ValueError(f"Unknown leapfrog startup: {startup}")


def solve_finite_interval(
    method: str,
    *,
    closure: str = "outgoing",
    nx: int = 400,
    velocity: float = 1.0,
    cfl: float = 0.8,
    final_time: float = 1.10,
    length: float = 1.0,
    case: str = "cosine",
    startup: str = "lax-wendroff",
    save_every: int = 1,
) -> FiniteRunResult:
    """Solve the positive-velocity finite-interval problem."""
    if nx < 8:
        raise ValueError("nx must be at least 8")
    if velocity <= 0.0:
        raise ValueError("The present finite-interval implementation assumes velocity > 0")
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must satisfy 0 < cfl <= 1")
    if save_every < 1:
        raise ValueError("save_every must be positive")

    dx = length / nx
    tentative_dt = cfl * dx / velocity
    nsteps = max(1, int(np.ceil(final_time / tentative_dt)))
    dt = final_time / nsteps
    courant = velocity * dt / dx
    x = np.linspace(0.0, length, nx + 1, dtype=np.float64)

    initial = initial_data(x, case, length)
    initial_norm = max(float(np.linalg.norm(initial)), 1.0e-30)
    residual_history = [float(np.linalg.norm(initial) / initial_norm)]
    error_history = [0.0]
    times = [0.0]

    if method == "leapfrog":
        previous = initial.copy()
        current = leapfrog_start(
            previous,
            startup=startup,
            closure=closure,
            courant=courant,
            inflow=inflow_value(dt, case=case, velocity=velocity, length=length),
        )
        if nsteps >= 1:
            residual_history.append(float(np.linalg.norm(current) / initial_norm))
            exact_now = exact_finite_solution(x, dt, velocity=velocity, length=length, case=case)
            error_history.append(float(np.linalg.norm(current - exact_now) / initial_norm))
            times.append(dt)
        for step in range(1, nsteps):
            next_field = np.empty_like(current)
            next_field[0] = inflow_value(
                (step + 1) * dt,
                case=case,
                velocity=velocity,
                length=length,
            )
            next_field[1:-1] = previous[1:-1] - courant * (
                current[2:] - current[:-2]
            )
            apply_outflow_closure(
                next_field,
                current,
                closure=closure,
                courant=courant,
            )
            previous, current = current, next_field
            if (step + 1) % save_every == 0 or step + 1 == nsteps:
                time_now = (step + 1) * dt
                residual_history.append(float(np.linalg.norm(current) / initial_norm))
                exact_now = exact_finite_solution(x, time_now, velocity=velocity, length=length, case=case)
                error_history.append(float(np.linalg.norm(current - exact_now) / initial_norm))
                times.append(time_now)
        numerical = current
    else:
        old = initial.copy()
        for step in range(nsteps):
            old = one_step_finite(
                old,
                method=method,
                closure=closure,
                courant=courant,
                inflow=inflow_value(
                    (step + 1) * dt,
                    case=case,
                    velocity=velocity,
                    length=length,
                ),
            )
            if (step + 1) % save_every == 0 or step + 1 == nsteps:
                time_now = (step + 1) * dt
                residual_history.append(float(np.linalg.norm(old) / initial_norm))
                exact_now = exact_finite_solution(x, time_now, velocity=velocity, length=length, case=case)
                error_history.append(float(np.linalg.norm(old - exact_now) / initial_norm))
                times.append(time_now)
        numerical = old

    exact = exact_finite_solution(
        x,
        final_time,
        velocity=velocity,
        length=length,
        case=case,
    )
    relative_l2 = float(np.linalg.norm(numerical - exact) / initial_norm)
    residual_ratio = float(np.linalg.norm(numerical) / initial_norm)

    return FiniteRunResult(
        x=x,
        numerical=numerical,
        exact=exact,
        times=np.asarray(times, dtype=np.float64),
        residual_history=np.asarray(residual_history, dtype=np.float64),
        error_history=np.asarray(error_history, dtype=np.float64),
        dt=dt,
        nsteps=nsteps,
        courant=courant,
        relative_l2_error=relative_l2,
        residual_energy_ratio=residual_ratio,
        max_residual=float(np.max(np.abs(numerical))),
    )


def closure_study(
    *,
    nx: int = 400,
    velocity: float = 1.0,
    cfl: float = 0.8,
    final_time: float = 1.10,
    length: float = 1.0,
    method: str = "lax-wendroff",
    case: str = "cosine",
) -> dict[str, FiniteRunResult]:
    return {
        closure: solve_finite_interval(
            method,
            closure=closure,
            nx=nx,
            velocity=velocity,
            cfl=cfl,
            final_time=final_time,
            length=length,
            case=case,
        )
        for closure in ("frozen", "copy", "extrapolation", "outgoing")
    }


def generate_boundary_figure(output: Path) -> dict[str, FiniteRunResult]:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    crossing = closure_study(nx=120, final_time=0.75)
    histories = closure_study(nx=120, final_time=0.90)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2))
    reference = crossing["outgoing"]
    axes[0].plot(reference.x, reference.exact, linewidth=2.2, label="Exact")
    for closure, result in crossing.items():
        axes[0].plot(result.x, result.numerical, label=closure.capitalize())
    axes[0].set_title("Pulse crossing the numerical outflow boundary")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    for closure, result in histories.items():
        axes[1].semilogy(
            result.times,
            np.maximum(result.error_history, 1.0e-16),
            label=closure.capitalize(),
        )
    axes[1].axvline(0.65, linestyle="--", linewidth=1.0, label="Pulse reaches outflow")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\|u_h-u\|_2/\|u_h(0)\|_2$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return histories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["upwind", "lax-friedrichs", "lax-wendroff", "leapfrog", "ftcs"],
        default="lax-wendroff",
    )
    parser.add_argument(
        "--outflow",
        choices=["frozen", "copy", "extrapolation", "outgoing"],
        default="outgoing",
    )
    parser.add_argument(
        "--startup",
        choices=["lax-wendroff", "upwind", "ftcs"],
        default="lax-wendroff",
    )
    parser.add_argument("--case", choices=["cosine", "gaussian"], default="cosine")
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--final-time", type=float, default=1.10)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--closure-study", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.closure_study is not None:
        results = generate_boundary_figure(args.closure_study)
        print("Outflow-closure study:")
        for closure, result in results.items():
            print(
                f"  {closure:13s}: residual L2 ratio = {result.residual_energy_ratio:.6e}, "
                f"max residual = {result.max_residual:.6e}"
            )

    result = solve_finite_interval(
        args.method,
        closure=args.outflow,
        startup=args.startup,
        nx=args.nx,
        velocity=args.velocity,
        cfl=args.cfl,
        final_time=args.final_time,
        length=args.length,
        case=args.case,
    )
    print(f"method                  = {args.method}")
    print(f"outflow closure         = {args.outflow}")
    print(f"case                    = {args.case}")
    print(f"grid cells              = {args.nx}")
    print(f"time steps              = {result.nsteps}")
    print(f"dt                      = {result.dt:.12e}")
    print(f"Courant number          = {result.courant:.12e}")
    print(f"relative L2 error       = {result.relative_l2_error:.12e}")
    print(f"residual L2 ratio       = {result.residual_energy_ratio:.12e}")
    print(f"maximum residual        = {result.max_residual:.12e}")

    if args.plot is not None:
        import matplotlib.pyplot as plt

        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.2, 4.5))
        plt.plot(result.x, result.exact, linewidth=2.0, label="Exact")
        plt.plot(result.x, result.numerical, label=args.outflow)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=180)
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
