"""
Components for Multi-Tugboat Routing and Scheduling Problem (MTRSP).

This module defines the Solution class and Operators for the MTRSP problem.
Solution represents the routing and scheduling assignments for all tugboats.
Operators provide constructive actions to build solutions step by step.

Mathematical Model Reference:
    Core Decision Variables (stored in Solution):
        - x_ijk: Encoded as routes - tugboat k's route sequence
        - y_sk: Stored as task_tugboats - which tugboats serve task s
        - tau_s: Stored as task_start_times - service start time of each task
    
    Derived Variables (computed from Solution):
        - z_s: Derived from task_tugboats - whether task s is executed (len(task_tugboats[s]) > 0)

Design Philosophy:
    All operators support both single-tugboat and multi-tugboat (collaborative) task assignments.
    Each operator can assign a task to one or more tugboats simultaneously, ensuring consistency.
    Consistency between routes and task_tugboats is maintained by proper operator design.
"""

from src.problems.base.components import BaseSolution, BaseOperator
from typing import List, Dict, Optional, Union


class Solution(BaseSolution):
    """
    Solution representation for MTRSP.
    
    The solution explicitly stores three key decision variables:
    1. routes: The routing structure (x_ijk variables)
    2. task_tugboats: Which tugboats serve each task (y_sk variables)
    3. task_start_times: The start time of each task (tau_s variables)
    
    From these, we can derive:
    - z_s: Whether task s is executed (len(task_tugboats[s]) > 0)
    
    Attributes:
        routes (list[list[int]]): Routes for all tugboats
            - Each route is a list of task IDs (1-indexed) in visiting order
            - Empty list [] means the tugboat is not used
            - Example: [[1, 3, 5], [2, 4], []] for 3 tugboats
            
        task_tugboats (dict[int, list[int]]): Which tugboats serve each task
            - Key: task_id (1-indexed)
            - Value: list of tugboat IDs (0-indexed) serving this task
            - Only contains entries for executed tasks (z_s = 1)
            - Example: {1: [0, 2], 2: [1], 3: [0], 4: [1], 5: [0]}
            - Empty list [] means task is not executed
            - For single-tugboat tasks: list has one element [k]
            - For collaborative tasks: list has multiple elements [k1, k2, ...]
            
        task_start_times (dict[int, float]): Start time for each executed task
            - Key: task_id (1-indexed)
            - Value: start time tau_s in hours
            - Only contains entries for executed tasks (z_s = 1)
            - Example: {1: 2.5, 2: 1.0, 3: 5.2, 4: 3.5, 5: 8.0}
            - For multi-tugboat tasks, all tugboats start serving at the same time
            
        num_tasks (int): Total number of tasks in the problem instance
        
    Consistency Guarantee:
        The solution maintains consistency between routes and task_tugboats:
        - If task s appears in routes[k], then k must be in task_tugboats[s]
        - If k is in task_tugboats[s], then task s must appear in routes[k]
        - This consistency is guaranteed by proper operator implementation
        - All operators MUST update both routes and task_tugboats simultaneously
        
    Example - Single tugboat tasks:
        routes = [
            [1, 3, 5],  # Tugboat 0 serves tasks 1, 3, 5 alone
            [2, 4],     # Tugboat 1 serves tasks 2, 4 alone
            []          # Tugboat 2 is not used
        ]
        task_tugboats = {
            1: [0],     # Task 1 served by tugboat 0
            2: [1],     # Task 2 served by tugboat 1
            3: [0],     # Task 3 served by tugboat 0
            4: [1],     # Task 4 served by tugboat 1
            5: [0]      # Task 5 served by tugboat 0
        }
        task_start_times = {1: 2.5, 2: 1.0, 3: 5.8, 4: 3.2, 5: 9.5}
        
    Example - Mixed single and collaborative tasks:
        routes = [
            [1, 3, 5],  # Tugboat 0 serves tasks 1, 3, 5
            [2, 4],     # Tugboat 1 serves tasks 2, 4
            [1]         # Tugboat 2 also serves task 1 (collaborative)
        ]
        task_tugboats = {
            1: [0, 2],  # Task 1 is collaborative (tugboats 0 and 2)
            2: [1],     # Task 2 is single tugboat
            3: [0],     # Task 3 is single tugboat
            4: [1],     # Task 4 is single tugboat
            5: [0]      # Task 5 is single tugboat
        }
        task_start_times = {
            1: 2.5,   # Task 1 is collaborative (tugboats 0 and 2)
            2: 1.0,   # Task 2 is single tugboat
            3: 5.8,   # Task 3 is single tugboat
            4: 3.2,   # Task 4 is single tugboat
            5: 9.5    # Task 5 is single tugboat
        }
        
    Notes:
        - A task can appear in multiple tugboat routes (collaborative service)
        - For collaborative tasks, all assigned tugboats must start at the same time
        - Start times must respect time windows: a_s <= tau_s <= b_s
        - Start times must respect time propagation constraints
    """
    
    def __init__(self, routes: List[List[int]], task_tugboats: Dict[int, List[int]], 
                 task_start_times: Dict[int, float], num_tasks: int):
        """
        Initialize MTRSP solution.
        
        Args:
            routes: List of routes for each tugboat, where each route is a list of task IDs
            task_tugboats: Dictionary mapping task_id to list of tugboat IDs serving it
            task_start_times: Dictionary mapping task_id to its start time (only for executed tasks)
            num_tasks: Total number of tasks in the problem
            
        Note:
            The caller is responsible for ensuring consistency between routes and task_tugboats.
            All operators in this module guarantee this consistency.
        """
        self.routes = routes
        self.task_tugboats = task_tugboats
        self.task_start_times = task_start_times
        self.num_tasks = num_tasks
    
    def __str__(self) -> str:
        """String representation showing routes, task assignments, and start times."""
        if not self.routes:
            return "Empty solution with no tugboat routes"
        
        result = "MTRSP Solution:\n"
        result += "="*60 + "\n"
        
        # Show routes
        result += "Tugboat Routes:\n"
        for tugboat_id, route in enumerate(self.routes):
            if len(route) > 0:
                route_str = " -> ".join([f"Task{task_id}" for task_id in route])
                result += f"  Tugboat {tugboat_id}: Depot -> {route_str} -> Depot\n"
            else:
                result += f"  Tugboat {tugboat_id}: Not used\n"
        
        # Show task assignments and start times
        result += "\nTask Assignments:\n"
        if self.task_tugboats:
            sorted_tasks = sorted(self.task_tugboats.items())
            for task_id, tugboat_ids in sorted_tasks:
                tugboats_str = ", ".join([f"Tugboat{i}" for i in sorted(tugboat_ids)])
                collab_indicator = " [COLLABORATIVE]" if len(tugboat_ids) > 1 else ""
                start_time = self.task_start_times.get(task_id, None)
                if start_time is not None:
                    result += f"  Task {task_id}: Served by [{tugboats_str}] at {start_time:.2f}h{collab_indicator}\n"
                else:
                    result += f"  Task {task_id}: Served by [{tugboats_str}] (no start time set){collab_indicator}\n"
        else:
            result += "  No tasks executed\n"
        
        result += "="*60
        return result


# ==================== Unified Constructive Operators ====================

class AppendTaskOperator(BaseOperator):
    """
    Append a task to the end of one or more tugboat routes (supports both single and collaborative tasks).
    
    This operator adds a task at the end of specified tugboat routes and sets its start time.
    It handles both single-tugboat assignments and multi-tugboat collaborative assignments uniformly.
    
    The operator guarantees consistency between routes and task_tugboats by updating both simultaneously.
    
    Attributes:
        tugboat_ids (list[int] or int): Tugboat ID(s) to append to
            - If int: Single tugboat assignment (e.g., tugboat_ids=0)
            - If list: Multi-tugboat collaborative assignment (e.g., tugboat_ids=[0, 2, 5])
        task_id (int): The ID of the task to append (1-indexed, 1 to num_tasks)
        start_time (float): The start time for this task (in hours)
    
    Example - Single tugboat:
        Initial: 
            routes = [[1, 2], [3], []]
            task_tugboats = {1: [0], 2: [0], 3: [1]}
        Apply AppendTaskOperator(tugboat_ids=0, task_id=4, start_time=6.0)
        Result: 
            routes = [[1, 2, 4], [3], []]
            task_tugboats = {1: [0], 2: [0], 3: [1], 4: [0]}
        
    Example - Collaborative (multiple tugboats):
        Initial: 
            routes = [[1, 2], [3], []]
            task_tugboats = {1: [0], 2: [0], 3: [1]}
        Apply AppendTaskOperator(tugboat_ids=[0, 2], task_id=4, start_time=6.0)
        Result: 
            routes = [[1, 2, 4], [3], [4]]
            task_tugboats = {1: [0], 2: [0], 3: [1], 4: [0, 2]}
        Both tugboats 0 and 2 now serve task 4 at the same time.
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int, start_time: float):
        """
        Initialize the append operator.
        
        Args:
            tugboat_ids: Single tugboat ID (int) or list of tugboat IDs (list[int])
            task_id: ID of the task to append (1-indexed, 1 to num_tasks)
            start_time: Start time for this task (in hours, must be >= 0)
        """
        # Normalize tugboat_ids to always be a list internally
        if isinstance(tugboat_ids, int):
            self.tugboat_ids = [tugboat_ids]
        else:
            self.tugboat_ids = sorted(tugboat_ids)  # Keep sorted for consistency
        self.task_id = task_id
        self.start_time = start_time
    
    def run(self, solution: Solution) -> Solution:
        """
        Apply the append operation to create a new solution.
        
        This method guarantees consistency by:
        1. Updating routes: appending task to all specified tugboat routes
        2. Updating task_tugboats: recording which tugboats serve this task
        3. Both updates happen atomically in the same operation
        
        Args:
            solution: The current solution to modify
            
        Returns:
            A new Solution object with the task appended to specified tugboat(s)
            
        Note:
            - Does NOT modify the input solution (creates a new one)
            - If task already exists in solution, merges tugboat assignments and keeps existing start time
            - Handles both single and multi-tugboat assignments uniformly
        """
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
                       task_start_times=new_task_start_times, num_tasks=solution.num_tasks)


class InsertTaskOperator(BaseOperator):
    """
    Insert a task at specific positions in one or more tugboat routes (supports both single and collaborative tasks).
    
    This operator inserts a task at specified positions in tugboat routes.
    For collaborative tasks, each tugboat can have a different insertion position in its route.
    
    The operator guarantees consistency between routes and task_tugboats by updating both simultaneously.
    
    Attributes:
        tugboat_ids (list[int] or int): Tugboat ID(s) to insert into
        task_id (int): The ID of the task to insert (1-indexed, 1 to num_tasks)
        positions (list[int] or int): Insertion position(s) for each tugboat
            - If tugboat_ids is int, positions must be int
            - If tugboat_ids is list, positions must be list of same length
        start_time (float): The start time for this task (in hours)
    
    Example - Single tugboat:
        Initial: 
            routes = [[1, 3, 5], [2], []]
            task_tugboats = {1: [0], 3: [0], 5: [0], 2: [1]}
        Apply InsertTaskOperator(tugboat_ids=0, task_id=4, positions=2, start_time=6.0)
        Result: 
            routes = [[1, 3, 4, 5], [2], []]
            task_tugboats = {1: [0], 3: [0], 5: [0], 2: [1], 4: [0]}
        
    Example - Collaborative with different positions:
        Initial: 
            routes = [[1, 3, 5], [2, 7], []]
            task_tugboats = {1: [0], 3: [0], 5: [0], 2: [1], 7: [1]}
        Apply InsertTaskOperator(tugboat_ids=[0, 1], task_id=4, positions=[1, 2], start_time=6.0)
        Result: 
            routes = [[1, 4, 3, 5], [2, 7, 4], []]
            task_tugboats = {1: [0], 3: [0], 5: [0], 2: [1], 7: [1], 4: [0, 1]}
        Task 4 inserted at position 1 in route 0 and position 2 in route 1.
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int, 
                 positions: Union[int, List[int]], start_time: float):
        """
        Initialize the insert operator.
        
        Args:
            tugboat_ids: Single tugboat ID (int) or list of tugboat IDs (list[int])
            task_id: ID of the task to insert (1-indexed, 1 to num_tasks)
            positions: Single position (int) or list of positions (list[int]), one per tugboat
            start_time: Start time for this task (in hours, must be >= 0)
        """
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
        """
        Apply the insert operation to create a new solution.
        
        This method guarantees consistency by:
        1. Updating routes: inserting task at specified positions in tugboat routes
        2. Updating task_tugboats: recording which tugboats serve this task
        3. Both updates happen atomically in the same operation
        
        Args:
            solution: The current solution to modify
            
        Returns:
            A new Solution object with the task inserted at specified positions
            
        Note:
            - Does NOT modify the input solution (creates a new one)
            - If task already exists in solution, merges tugboat assignments and keeps existing start time
            - Each tugboat gets the task inserted at its corresponding position
        """
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
                       task_start_times=new_task_start_times, num_tasks=solution.num_tasks)