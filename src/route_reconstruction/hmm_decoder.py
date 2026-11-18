"""Viterbi decoding over candidate road arcs.

The transition distance follows directed roads between projected candidate
positions. Missing graph paths receive a finite penalty so decoding survives
disconnected regions.
"""

from functools import lru_cache
import heapq

import numpy as np

from src.data_processing.loader import load_trip
from src.data_processing.spatial_utils import prepare_geometry


def network_geometry(trip):
    geometry = prepare_geometry(trip)
    outgoing = [[] for _ in range(len(trip.nodes))]
    for arc_id, (start, end) in enumerate(trip.arcs):
        outgoing[start].append((int(end), float(geometry.length[arc_id])))
    return geometry, outgoing


def decoder(trip, candidates, probabilities, timestamps, block_starts,
            emission_weight=1.0, distance_scale=30.0, disconnect_penalty=8.0,
            max_path_m=1000.0):
    """Return one candidate arc per observation, respecting GPS block starts."""
    geometry, outgoing = network_geometry(trip)
    k = candidates.shape[1]
    start_node, end_node = trip.arcs[:, 0], trip.arcs[:, 1]
    fraction = np.zeros_like(probabilities)
    projected = np.empty((len(candidates), k, 2))
    for index in range(len(candidates)):
        arcs = candidates[index]
        offset = geometry.gps[index] - geometry.start[arcs]
        fraction[index] = np.divide(
            (offset * geometry.vector[arcs]).sum(axis=1),
            geometry.length[arcs] ** 2,
            out=np.zeros(k), where=geometry.length[arcs] > 0,
        ).clip(0, 1)
        projected[index] = geometry.start[arcs] + fraction[index, :, None] * geometry.vector[arcs]

    @lru_cache(maxsize=50000)
    def shortest(start, end):
        if start == end:
            return 0.0
        queue = [(0.0, start)]
        visited = set()
        while queue:
            length, node = heapq.heappop(queue)
            if length > max_path_m:
                break
            if node == end:
                return length
            if node in visited:
                continue
            visited.add(node)
            for neighbor, edge_length in outgoing[node]:
                if neighbor not in visited:
                    heapq.heappush(queue, (length + edge_length, neighbor))
        return np.inf

    block_set = set(block_starts)
    chosen = np.empty(len(candidates), dtype=np.int32)
    back = np.zeros((len(candidates), k), dtype=np.int16)
    safe = np.clip(probabilities, 1e-8, 1)
    emission = emission_weight * np.log(safe / safe.sum(axis=1, keepdims=True))
    score = emission[0]
    previous_start = 0
    for index in range(1, len(candidates)):
        if index in block_set:
            chosen[index - 1] = np.argmax(score)
            for previous in range(index - 2, previous_start - 1, -1):
                chosen[previous] = back[previous + 1, chosen[previous + 1]]
            score = emission[index]
        else:
            gps_step = np.linalg.norm(geometry.gps[index] - geometry.gps[index - 1])
            previous_arcs = candidates[index - 1]
            current_arcs = candidates[index]
            transitions = np.empty((k, k))
            for a, previous_arc in enumerate(previous_arcs):
                for b, current_arc in enumerate(current_arcs):
                    if previous_arc == current_arc and fraction[index, b] >= fraction[index - 1, a]:
                        path = (fraction[index, b] - fraction[index - 1, a]) * geometry.length[current_arc]
                    else:
                        middle = shortest(int(end_node[previous_arc]), int(start_node[current_arc]))
                        path = (
                            (1 - fraction[index - 1, a]) * geometry.length[previous_arc]
                            + middle + fraction[index, b] * geometry.length[current_arc]
                        )
                    if np.isfinite(path):
                        transitions[a, b] = -abs(path - gps_step) / distance_scale
                    else:
                        transitions[a, b] = (
                            -disconnect_penalty
                            - np.linalg.norm(projected[index - 1, a] - projected[index, b])
                            / distance_scale
                        )
            combined = score[:, None] + transitions
            back[index] = combined.argmax(axis=0)
            score = emission[index] + combined.max(axis=0)
        if index in block_set:
            previous_start = index
    chosen[-1] = np.argmax(score)
    for index in range(len(candidates) - 2, previous_start - 1, -1):
        chosen[index] = back[index + 1, chosen[index + 1]]
    return candidates[np.arange(len(candidates)), chosen]
