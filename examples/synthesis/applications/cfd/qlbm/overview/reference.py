"""Independent classical propagation for the original 32 initial branches."""

import itertools
import math

import torch


def source_branch_words() -> tuple[int, ...]:
    """Enumerate the covering space and evaluate both reflective axes independently."""
    all_words = []
    time_limit = 2
    # Event eight falls at t=2 and includes every nonzero speed.
    instants = sorted(
        {
            index / speed
            for speed in (1, 2, 3)
            for index in range(1, speed * time_limit + 1)
        }
    )
    assert len(instants) == 8
    # The ninth event follows the coincidences at one and two.
    instants.append(7 / 3)
    for uy, ux, initial_y_dir in itertools.product(range(4), range(4), range(2)):
        x, y, vx, vy = 1, 2, 1, initial_y_dir
        for event in instants:
            for speed in (1, 2, 3):
                if not math.isclose(event * speed, round(event * speed), abs_tol=1e-12):
                    continue
                if ux == speed:
                    x = (x + (1 if vx else -1)) % 8
                if uy == speed:
                    y = (y + (1 if vy else -1)) % 8
                if uy == speed and y in (1, 5) and 2 <= x <= 5:
                    vy ^= 1
                if ux == speed and x in (2, 5) and 1 <= y <= 5:
                    vx ^= 1
        all_words.append((x << 9) | (y << 6) | (vx << 5) | (vy << 4) | (ux << 2) | uy)
    return tuple(all_words)


def source_state() -> torch.Tensor:
    """Construct the full 4,096-amplitude independent reference vector."""
    words = source_branch_words()
    assert len(set(words)) == 32
    state = torch.zeros(4096, dtype=torch.complex128)
    state[list(words)] = math.sqrt(1 / 32)
    return state
