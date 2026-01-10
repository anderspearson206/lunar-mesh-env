# pathfinding.py
import numpy as np
import heapq

def a_star_search(heightmap, start, goal, max_incline=1.0):
    """
    Computes the shortest path on a grid using A* with slope constraints.
    
    Args:
        heightmap: 2D numpy array of terrain heights.
        start: Tuple (x, y) start coordinates (integers).
        goal: Tuple (x, y) goal coordinates (integers).
        max_incline: Maximum height increase allowed per step (meters).
        
    Returns:
        path: List of (x, y) tuples from start to goal.
        None: If no path is found.
    """
    h, w = heightmap.shape
    start = tuple(start)
    goal = tuple(goal)
    
    # Priority queue: (f_score, current_node)
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Maps
    came_from = {}
    g_score = {start: 0} # Cost from start
    
    # Neighbors: N, S, E, W
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)] 

    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        cx, cy = current
        current_height = heightmap[cy, cx]
        
        for dx, dy in directions:
            neighbor = (cx + dx, cy + dy)
            nx, ny = neighbor
            
            # boundary check
            if 0 <= nx < w and 0 <= ny < h:
                target_height = heightmap[ny, nx]
                height_diff = target_height - current_height
                
                # slope check
                if height_diff > max_incline:
                    continue 
                
                # cost calc
                incline_penalty = max(0, height_diff * 2.0) 
                tentative_g = g_score[current] + 1.0 + incline_penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None 

def heuristic(a, b):
    # manhattan distance for grid movement
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1] 