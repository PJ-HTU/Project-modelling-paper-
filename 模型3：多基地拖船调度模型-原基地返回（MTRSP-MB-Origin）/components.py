from src.problems.base.components import BaseSolution, BaseOperator
from typing import List, Dict, Optional, Union


class Solution(BaseSolution):
    """
    Solution representation for MTRSP-MB (Multi-Base).
    
    The solution explicitly stores four key decision variables:
    1. routes: The routing structure (x_ij^k variables)
    2. task_tugboats: Which tugboats serve each task (y_sk variables)
    3. task_start_times: The start time of each task (tau_s variables)
    4. tugboat_base_assignment: Home base for each tugboat
    """
    
    def __init__(self, routes: List[List[int]], task_tugboats: Dict[int, List[int]], 
                 task_start_times: Dict[int, float], tugboat_base_assignment: List[int],
                 num_tasks: int, num_bases: int):
        """
        Initialize MTRSP-MB solution.
        """
        self.routes = routes
        self.task_tugboats = task_tugboats
        self.task_start_times = task_start_times
        self.tugboat_base_assignment = tugboat_base_assignment
        self.num_tasks = num_tasks
        self.num_bases = num_bases
    
    def __str__(self) -> str:
        """String representation showing routes, task assignments, start times, and base assignments."""
        if not self.routes:
            return "Empty solution with no tugboat routes"
        
        result = "MTRSP-MB Solution (Multi-Base):\n"
        result += "="*60 + "\n"
        
        # Show base assignments
        result += "Base Assignments:\n"
        base_tugboats = {}
        for tugboat_id, base_id in enumerate(self.tugboat_base_assignment):
            if base_id not in base_tugboats:
                base_tugboats[base_id] = []
            base_tugboats[base_id].append(tugboat_id)
        
        for base_id in sorted(base_tugboats.keys()):
            tugboat_list = ", ".join([f"Tugboat{t}" for t in base_tugboats[base_id]])
            result += f"  Base {base_id}: [{tugboat_list}]\n"
        
        # Show routes
        result += "\nTugboat Routes:\n"
        for tugboat_id, route in enumerate(self.routes):
            base_id = self.tugboat_base_assignment[tugboat_id]
            dest_id = self.num_tasks - base_id  # Destination node: n - b
            if len(route) > 0:
                route_str = " -> ".join([f"Task{task_id}" for task_id in route])
                result += f"  Tugboat {tugboat_id} (Base {base_id}): {base_id} -> {route_str} -> {dest_id}\n"
            else:
                result += f"  Tugboat {tugboat_id} (Base {base_id}): Not used\n"
        
        # Show task assignments and start times
        result += "\nTask Assignments:\n"
        if self.task_tugboats:
            sorted_tasks = sorted(self.task_tugboats.items())
            for task_id, tugboat_ids in sorted_tasks:
                tugboats_str = ", ".join([f"Tugboat{i}" for i in sorted(tugboat_ids)])
                collab_indicator = " [COLLABORATIVE]" if len(tugboat_ids) > 1 else ""
                start_time = self.task_start_times.get(task_id, None)
                
                # Show which base(s) the tugboats come from
                bases = set([self.tugboat_base_assignment[t] for t in tugboat_ids])
                bases_str = ", ".join([f"Base{b}" for b in sorted(bases)])
                
                if start_time is not None:
                    result += f"  Task {task_id}: Served by [{tugboats_str}] from {bases_str} at {start_time:.2f}h{collab_indicator}\n"
                else:
                    result += f"  Task {task_id}: Served by [{tugboats_str}] from {bases_str} (no start time set){collab_indicator}\n"
        else:
            result += "  No tasks executed\n"
        
        result += "="*60
        return result


# ==================== Unified Constructive Operators ====================

class AppendTaskOperator(BaseOperator):
    """
    Append a task to the end of one or more tugboat routes (supports both single and collaborative tasks).
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int, start_time: float):
        
        # Normalize to list internally for uniform processing
        if isinstance(tugboat_ids, int):
            self.tugboat_ids = [tugboat_ids]
        else:
            self.tugboat_ids = list(tugboat_ids)
        
        self.task_id = task_id
        self.start_time = start_time
    
    def run(self, solution: Solution) -> Solution:
        
        # Create deep copy of routes to avoid modifying the input solution
        new_routes = [route[:] for route in solution.routes]
        
        # Create copy of task_tugboats
        new_task_tugboats = {task_id: tugboats[:] for task_id, tugboats in solution.task_tugboats.items()}
        
        # Create copy of task_start_times
        new_task_start_times = solution.task_start_times.copy()
        
        # CONSISTENCY GUARANTEE: Update both routes and task_tugboats simultaneously
        
        # Step 1: Append the task to all specified tugboat routes
        for tugboat_id in self.tugboat_ids:
            new_routes[tugboat_id].append(self.task_id)
        
        # Step 2: Update task_tugboats to reflect which tugboats serve this task
        if self.task_id in new_task_tugboats:
            # Task already exists, merge tugboat assignments
            existing_tugboats = set(new_task_tugboats[self.task_id])
            new_tugboats = existing_tugboats.union(self.tugboat_ids)
            new_task_tugboats[self.task_id] = sorted(new_tugboats)
        else:
            # New task, create new entry
            new_task_tugboats[self.task_id] = sorted(self.tugboat_ids)
        
        # Step 3: Set start time (if task already exists, keep existing time for consistency)
        if self.task_id not in new_task_start_times:
            new_task_start_times[self.task_id] = self.start_time
        
        # Return a new Solution object with guaranteed consistency
        return Solution(routes=new_routes, task_tugboats=new_task_tugboats,
                       task_start_times=new_task_start_times,
                       tugboat_base_assignment=solution.tugboat_base_assignment,
                       num_tasks=solution.num_tasks, num_bases=solution.num_bases)


class InsertTaskOperator(BaseOperator):
    """
    Insert a task at specific positions in one or more tugboat routes (supports both single and collaborative tasks).
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int, 
                 positions: Union[int, List[int]], start_time: float):
      
        # Normalize to lists internally
        if isinstance(tugboat_ids, int):
            self.tugboat_ids = [tugboat_ids]
            self.positions = [positions] if isinstance(positions, int) else positions
        else:
            self.tugboat_ids = list(tugboat_ids)
            self.positions = [positions] if isinstance(positions, int) else list(positions)
        
        # Validate that lengths match
        if len(self.tugboat_ids) != len(self.positions):
            raise ValueError(f"Length mismatch: {len(self.tugboat_ids)} tugboats but {len(self.positions)} positions")
        
        self.task_id = task_id
        self.start_time = start_time
    
    def run(self, solution: Solution) -> Solution:
        
        # Create deep copy of routes to avoid modifying the input solution
        new_routes = [route[:] for route in solution.routes]
        
        # Create copy of task_tugboats
        new_task_tugboats = {task_id: tugboats[:] for task_id, tugboats in solution.task_tugboats.items()}
        
        # Create copy of task_start_times
        new_task_start_times = solution.task_start_times.copy()
        
        # CONSISTENCY GUARANTEE: Update both routes and task_tugboats simultaneously
        
        # Step 1: Insert the task at specified positions in each tugboat's route
        for tugboat_id, position in zip(self.tugboat_ids, self.positions):
            new_routes[tugboat_id].insert(position, self.task_id)
        
        # Step 2: Update task_tugboats to reflect which tugboats serve this task
        if self.task_id in new_task_tugboats:
            # Task already exists, merge tugboat assignments
            existing_tugboats = set(new_task_tugboats[self.task_id])
            new_tugboats = existing_tugboats.union(self.tugboat_ids)
            new_task_tugboats[self.task_id] = sorted(new_tugboats)
        else:
            # New task, create new entry
            new_task_tugboats[self.task_id] = sorted(self.tugboat_ids)
        
        # Step 3: Set start time (if task already exists, keep existing time for consistency)
        if self.task_id not in new_task_start_times:
            new_task_start_times[self.task_id] = self.start_time
        
        # Return a new Solution object with guaranteed consistency
        return Solution(routes=new_routes, task_tugboats=new_task_tugboats,
                       task_start_times=new_task_start_times,
                       tugboat_base_assignment=solution.tugboat_base_assignment,
                       num_tasks=solution.num_tasks, num_bases=solution.num_bases)