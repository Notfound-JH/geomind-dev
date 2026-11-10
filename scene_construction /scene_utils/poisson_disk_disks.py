import random
import math
from typing import List, Tuple

def poisson_disk_disks(a: float, b: float, r: float, 
                       max_points: int, k_tries: int = 10000) -> List[Tuple[float, float]]:
    print("POISSON DISK:", a, b, r, max_points)
    """
    Bridson Poisson-disk sampling in [0,a] x [0,b]
    Ensures disk centers are within [r, a-r] x [r, b-r]
    Min distance between centers = 2*r

    :param a: width
    :param b: height
    :param r: radius of each disk
    :param max_points: desired number of disks
    :param k_tries: attempts per active sample
    :return: list of disk centers [(x,y), ...]
    """
    min_dist = 2 * r
    cell_size = min_dist / math.sqrt(2)

    # Grid resolution
    nx = int(math.ceil(a / cell_size))
    ny = int(math.ceil(b / cell_size))
    grid = [[None] * ny for _ in range(nx)]

    def grid_pos(pt):
        return int(pt[0] // cell_size), int(pt[1] // cell_size)

    def is_valid(pt):
        gx, gy = grid_pos(pt)
        # Check neighbors
        rad = 2  # covers all possible near cells
        for i in range(max(0, gx - rad), min(nx - 1, gx + rad) + 1):
            for j in range(max(0, gy - rad), min(ny - 1, gy + rad) + 1):
                neighbor = grid[i][j]
                if neighbor is not None:
                    dx = neighbor[0] - pt[0]
                    dy = neighbor[1] - pt[1]
                    if dx*dx + dy*dy < min_dist * min_dist:
                        return False
        return True

    samples = []
    active = []

    # Init with one random point
    x0 = random.uniform(r, a - r)
    y0 = random.uniform(r, b - r)
    samples.append((x0, y0))
    active.append((x0, y0))
    gx0, gy0 = grid_pos((x0, y0))
    grid[gx0][gy0] = (x0, y0)

    # Expand
    while active and len(samples) < max_points:
        idx = random.randrange(len(active))
        base = active[idx]
        found_new = False
        for _ in range(k_tries):
            rad = random.uniform(min_dist, 2 * min_dist)
            theta = random.uniform(0, 2*math.pi)
            nxp = base[0] + rad * math.cos(theta)
            nyp = base[1] + rad * math.sin(theta)

            # Boundary check for disk center
            if not (r <= nxp <= a - r and r <= nyp <= b - r):
                continue

            if is_valid((nxp, nyp)):
                samples.append((nxp, nyp))
                active.append((nxp, nyp))
                gx, gy = grid_pos((nxp, nyp))
                grid[gx][gy] = (nxp, nyp)
                found_new = True
                break

        if not found_new:
            active.pop(idx)

    return samples[:max_points]

