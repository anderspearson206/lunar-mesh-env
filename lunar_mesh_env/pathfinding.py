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
    # use euclidean distance as heuristic
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def a_star_search_rm(heightmap, radiomap, start, goal, max_incline=1.0, radio_weight=0.0):
    """
    Computes the shortest path using A* with terrain slope and radio signal constraints.
    
    Args:
        heightmap: 2D numpy array of terrain heights.
        radiomap: 2D numpy array of signal strength (e.g., dBm).
        start, goal: (x, y) tuples.
        max_incline: Max allowed height increase per step.
        radio_weight: How much to prioritize signal (higher = stays in coverage more).
    """
    h, w = heightmap.shape
    start, goal = tuple(start), tuple(goal)
    
    # normalize radio map
    rm_min, rm_max = radiomap.min(), radiomap.max()
    if rm_max != rm_min:
        norm_radio_cost = (rm_max - radiomap) / (rm_max - rm_min)
    else:
        norm_radio_cost = np.zeros_like(radiomap)

    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] 

    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        cx, cy = current
        
        for dx, dy in directions:
            neighbor = (cx + dx, cy + dy)
            nx, ny = neighbor
            
            if 0 <= nx < w and 0 <= ny < h:
                # hard slope constraint
                height_diff = heightmap[ny, nx] - heightmap[cy, cx]
                if height_diff > max_incline:
                    continue 
                
                # distance cost
                step_cost = np.sqrt(dx**2 + dy**2)
                
                # incline penalty (can relate to energy later)
                incline_penalty = max(0, height_diff * 2.0) 
                
                # radio cost penalty
                radio_penalty = norm_radio_cost[ny, nx] * radio_weight
                
                tentative_g = g_score[current] + step_cost + incline_penalty + radio_penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None 

