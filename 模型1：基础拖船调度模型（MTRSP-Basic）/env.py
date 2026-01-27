import json
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.mtrsp.components import Solution


class Env(BaseEnv):
    """MTRSP environment that stores instance data, current solution, and problem state to support algorithms."""
    
    def __init__(self, data_name: str, **kwargs):
        """Initialize MTRSP environment.
        
        Args:
            data_name: Name/path of the data file
            **kwargs: Additional arguments passed to BaseEnv
        """
        super().__init__(data_name, "mtrsp")
        
        # Maximum steps to construct a complete solution
        self.construction_steps = self.instance_data["num_tasks"]
        
        # Key item for solution comparison - total cost (fuel + penalty)
        self.key_item = "total_cost"
        
        # Comparison function: lower cost is better
        self.compare = lambda x, y: y - x
    
    @property
    def is_complete_solution(self) -> bool:
        return True
    
    def load_data(self, data_path: str) -> dict:
        """Load tugboat scheduling instance data from JSON file."""
        import json
        
        with open(data_path, 'r') as f:
            instance = json.load(f)
        
        tasks = instance['tasks']
        tugboats = instance['tugboats']
        num_tasks = len(tasks)
        num_tugboats = len(tugboats)
        
        return {
            "num_tasks": num_tasks,
            "num_tugboats": num_tugboats,
            "task_max_tugs": [tasks[str(s)]['num_tugs_needed'] for s in range(1, num_tasks+1)],
            "task_min_horsepower": [tasks[str(s)]['min_horsepower'] for s in range(1, num_tasks+1)],
            "task_time_window_lower": [tasks[str(s)]['time_window'][0] for s in range(1, num_tasks+1)],
            "task_time_window_upper": [tasks[str(s)]['time_window'][1] for s in range(1, num_tasks+1)],
            "task_service_time": [tasks[str(s)]['service_time'] for s in range(1, num_tasks+1)],
            "tugboat_horsepower": [tugboats[str(k)]['horsepower'] for k in range(1, num_tugboats+1)],
            "tugboat_fuel_capacity": [tugboats[str(k)]['fuel_capacity'] for k in range(1, num_tugboats+1)],
            "tugboat_alpha": [tugboats[str(k)]['alpha'] for k in range(1, num_tugboats+1)],
            "tugboat_beta": [tugboats[str(k)]['beta'] for k in range(1, num_tugboats+1)],
            "time_matrix": instance['time_matrix'],
            "big_M": instance['metadata']['M'],
            "planning_horizon": instance['metadata']['T_max'],
            "penalty_weight": instance['metadata']['W']
        }
    
    def init_solution(self) -> Solution:
        """Initialize an empty solution.
        
        Returns:
            Solution with empty routes for all tugboats, no task assignments, and no task start times
        """
        num_tugboats = self.instance_data["num_tugboats"]
        num_tasks = self.instance_data["num_tasks"]
        
        # Create empty routes for each tugboat
        empty_routes = [[] for _ in range(num_tugboats)]
        
        # Create empty task tugboats dictionary
        empty_task_tugboats = {}
        
        # Create empty task start times dictionary
        empty_task_start_times = {}
        
        return Solution(
            routes=empty_routes, 
            task_tugboats=empty_task_tugboats,
            task_start_times=empty_task_start_times, 
            num_tasks=num_tasks
        )
    
    def get_key_value(self, solution: Solution = None) -> float:
        """Calculate the objective function value: total fuel consumption + penalty for unexecuted tasks.
        
        Objective = Total Fuel Consumption + penalty_weight  Number of Unexecuted Tasks
        
        Args:
            solution: Solution to evaluate. If None, uses current_solution.
            
        Returns:
            Total cost (fuel + penalty)
        """
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        num_tugboats = self.instance_data["num_tugboats"]
        time_matrix = self.instance_data["time_matrix"]
        task_service_time = self.instance_data["task_service_time"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        penalty_weight = self.instance_data["penalty_weight"]
        
        # Get task start times from solution
        start_times = solution.task_start_times
        
        # Track which tasks are executed
        executed_tasks = set()
        for route in solution.routes:
            executed_tasks.update(route)
        
        # Compute fuel consumption
        total_fuel = 0.0
        
        for tugboat_id, route in enumerate(solution.routes):
            if len(route) == 0:
                continue
            
            tugboat_hp = tugboat_horsepower[tugboat_id]
            alpha = tugboat_alpha[tugboat_id]
            beta = tugboat_beta[tugboat_id]
            
            current_time = 0.0
            current_node = 0  # Depot
            
            for task_id in route:
                # Travel from current_node to task_id
                travel_time = time_matrix[f"{current_node}_{task_id}"]
                travel_fuel = beta * tugboat_hp * travel_time
                total_fuel += travel_fuel
                
                # Get task start time from solution
                task_start = start_times.get(task_id, 0.0)
                
                # Service the task
                service_fuel = alpha * tugboat_hp * task_service_time[task_id - 1]
                total_fuel += service_fuel
                
                # Update current position and time
                current_time = task_start + task_service_time[task_id - 1]
                current_node = task_id
            
            # Return to depot (end depot is num_tasks + 1)
            return_travel_time = time_matrix[f"{current_node}_{num_tasks + 1}"]
            return_fuel = beta * tugboat_hp * return_travel_time
            total_fuel += return_fuel
        
        # Calculate penalty for unexecuted tasks
        all_tasks = set(range(1, num_tasks + 1))
        unexecuted_tasks = all_tasks - executed_tasks
        penalty = penalty_weight * len(unexecuted_tasks)
        
        # Total cost
        total_cost = total_fuel + penalty
        
        return total_cost
    
    def validation_solution(self, solution: Solution = None) -> bool:
        """Validate if the solution satisfies all constraints.
        
        Checks:
        1. Task nodes validity (1 to num_tasks)
        2. Task execution constraints:
           - At most task_max_tugs tugboats per task
           - Total horsepower >= task_min_horsepower
           - Service starts within time window
           - Service finishes within planning horizon
        3. Tugboat constraints:
           - No overlapping tasks for same tugboat
           - Fuel capacity not exceeded
        
        Args:
            solution: Solution to validate. If None, uses current_solution.
            
        Returns:
            True if solution is valid, False otherwise
        """
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        num_tugboats = self.instance_data["num_tugboats"]
        time_matrix = self.instance_data["time_matrix"]
        task_service_time = self.instance_data["task_service_time"]
        task_time_window_lower = self.instance_data["task_time_window_lower"]
        task_time_window_upper = self.instance_data["task_time_window_upper"]
        task_max_tugs = self.instance_data["task_max_tugs"]
        task_min_horsepower = self.instance_data["task_min_horsepower"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_fuel_capacity = self.instance_data["tugboat_fuel_capacity"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        planning_horizon = self.instance_data["planning_horizon"]
        
        # 1. Check task node validity
        for route in solution.routes:
            for task_id in route:
                if not (1 <= task_id <= num_tasks):
                    return False
        
        # Identify executed tasks
        executed_tasks = set()
        for route in solution.routes:
            executed_tasks.update(route)
        
        # Build task assignments: task_id -> list of tugboat_ids
        task_assignments = {}
        for task_id in executed_tasks:
            task_assignments[task_id] = []
            for tugboat_id, route in enumerate(solution.routes):
                if task_id in route:
                    task_assignments[task_id].append(tugboat_id)
        
        # 2. Check task execution constraints
        for task_id in executed_tasks:
            tugboats_assigned = task_assignments[task_id]
            
            # Check: at most task_max_tugs tugboats
            if len(tugboats_assigned) > task_max_tugs[task_id - 1]:
                return False
            
            # Check: total horsepower >= task_min_horsepower
            total_hp = sum(tugboat_horsepower[tug_id] for tug_id in tugboats_assigned)
            if total_hp < task_min_horsepower[task_id - 1]:
                return False
        
        # Get start times from solution
        start_times = solution.task_start_times
        
        # Validate start times for executed tasks
        for task_id in executed_tasks:
            # Check task has a start time
            if task_id not in start_times:
                return False
            
            task_start = start_times[task_id]
            
            # Check: service starts within time window
            if task_start < task_time_window_lower[task_id - 1] or \
               task_start > task_time_window_upper[task_id - 1]:
                return False
            
            # Check: service finishes within planning horizon
            finish_time = task_start + task_service_time[task_id - 1]
            if finish_time > planning_horizon:
                return False
        
        # 3. Check tugboat constraints
        for tugboat_id, route in enumerate(solution.routes):
            if len(route) == 0:
                continue
            
            tugboat_hp = tugboat_horsepower[tugboat_id]
            alpha = tugboat_alpha[tugboat_id]
            beta = tugboat_beta[tugboat_id]
            
            # Check no overlapping tasks (time feasibility)
            current_time = 0.0
            current_node = 0
            
            for task_id in route:
                travel_time = time_matrix[f"{current_node}_{task_id}"]
                arrival_time = current_time + travel_time
                task_start = start_times[task_id]
                
                # Tugboat must arrive before or at start time
                if arrival_time > task_start:
                    return False
                
                # Update time to end of service
                current_time = task_start + task_service_time[task_id - 1]
                current_node = task_id
            
            # Check fuel capacity
            total_fuel = 0.0
            current_node = 0
            
            for task_id in route:
                # Travel fuel
                travel_time = time_matrix[f"{current_node}_{task_id}"]
                travel_fuel = beta * tugboat_hp * travel_time
                total_fuel += travel_fuel
                
                # Service fuel
                service_fuel = alpha * tugboat_hp * task_service_time[task_id - 1]
                total_fuel += service_fuel
                
                current_node = task_id
            
            # Return to depot fuel
            return_travel_time = time_matrix[f"{current_node}_{num_tasks + 1}"]
            return_fuel = beta * tugboat_hp * return_travel_time
            total_fuel += return_fuel
            
            # Check capacity constraint
            if total_fuel > tugboat_fuel_capacity[tugboat_id]:
                return False
        
        return True

    # ============================================================================
    # SIMPLE HELPER FUNCTIONS (100% CORRECT)
    # ============================================================================
    
    def get_unassigned_tasks(self, solution = None) -> list:
        """Get list of task IDs that have not been assigned to any tugboat route.
        
        Args:
            solution: Current solution. If None, uses current_solution.
            
        Returns:
            List of unassigned task IDs (1-indexed)
        """
        if solution is None:
            solution = self.current_solution
        
        assigned_tasks = set()
        for route in solution.routes:
            assigned_tasks.update(route)
        
        all_tasks = set(range(1, self.instance_data['num_tasks'] + 1))
        unassigned_tasks = list(all_tasks - assigned_tasks)
        
        return unassigned_tasks
    
    def get_travel_time(self, from_node: int, to_node: int) -> float:
        """Get travel time from one node to another using the time matrix.
        
        Args:
            from_node: Source node (0 for depot, 1-num_tasks for tasks)
            to_node: Destination node (1-num_tasks for tasks, num_tasks+1 for depot)
            
        Returns:
            Travel time in hours
        """
        travel_key = f'{from_node}_{to_node}'
        return self.instance_data['time_matrix'].get(travel_key, 0.0)
    
    def get_tugboat_state(self, tugboat_id: int, solution = None) -> dict:
        """Get the current state of a tugboat including its location and time.
        
        Args:
            tugboat_id: Tugboat ID (0-indexed)
            solution: Current solution. If None, uses current_solution.
            
        Returns:
            Dictionary with:
            - current_location: Node ID (0 for depot, 1-num_tasks for tasks)
            - current_time: Time when tugboat finishes its last task (hours)
            - num_tasks_assigned: Number of tasks in tugboat's route
        """
        if solution is None:
            solution = self.current_solution
        
        route = solution.routes[tugboat_id]
        
        if not route:
            return {
                'current_location': 0,
                'current_time': 0.0,
                'num_tasks_assigned': 0
            }
        
        last_task = route[-1]
        last_start_time = solution.task_start_times.get(last_task, 0.0)
        last_service_time = self.instance_data['task_service_time'][last_task - 1]
        current_time = last_start_time + last_service_time
        
        return {
            'current_location': last_task,
            'current_time': current_time,
            'num_tasks_assigned': len(route)
        }
    
    def compute_arrival_time(self, tugboat_id: int, task_id: int, solution = None) -> float:
        """Compute the arrival time of a tugboat at a task entrance location.
        
        Args:
            tugboat_id: Tugboat ID (0-indexed)
            task_id: Task ID (1-indexed)
            solution: Current solution. If None, uses current_solution.
            
        Returns:
            Arrival time at task entrance (hours)
        """
        if solution is None:
            solution = self.current_solution
        
        tugboat_state = self.get_tugboat_state(tugboat_id, solution)
        current_location = tugboat_state['current_location']
        current_time = tugboat_state['current_time']
        
        travel_time = self.get_travel_time(current_location, task_id)
        arrival_time = current_time + travel_time
        
        return arrival_time
    
    def get_tugboat_fuel_used(self, tugboat_id: int, solution = None) -> float:
        """Calculate total fuel consumed by a tugboat based on its current route.
        
        This includes:
        - Travel from depot to first task
        - Service fuel for all tasks
        - Travel between consecutive tasks
        - Return travel from last task to depot
        
        Args:
            tugboat_id: Tugboat ID (0-indexed)
            solution: Current solution. If None, uses current_solution.
            
        Returns:
            Total fuel consumed (kg)
        """
        if solution is None:
            solution = self.current_solution
        
        route = solution.routes[tugboat_id]
        if not route:
            return 0.0
        
        hp = self.instance_data['tugboat_horsepower'][tugboat_id]
        alpha = self.instance_data['tugboat_alpha'][tugboat_id]
        beta = self.instance_data['tugboat_beta'][tugboat_id]
        
        total_fuel = 0.0
        current_node = 0  # Start from depot
        
        for task_id in route:
            # Travel fuel
            travel_time = self.get_travel_time(current_node, task_id)
            total_fuel += beta * hp * travel_time
            
            # Service fuel
            service_time = self.instance_data['task_service_time'][task_id - 1]
            total_fuel += alpha * hp * service_time
            
            current_node = task_id
        
        # Return to depot fuel
        num_tasks = self.instance_data['num_tasks']
        return_travel_time = self.get_travel_time(current_node, num_tasks + 1)
        total_fuel += beta * hp * return_travel_time
        
        return total_fuel
    
    # ============================================================================
    # HELPER FUNCTION REGISTRATION
    # ============================================================================
    
    def helper_function(self) -> dict:
        """Return dictionary of all helper functions for heuristic algorithms.
        
        Only includes simple, verified helper functions.
        """
        return {
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            "get_unassigned_tasks": self.get_unassigned_tasks,
            "get_travel_time": self.get_travel_time,
            "get_tugboat_state": self.get_tugboat_state,
            "compute_arrival_time": self.compute_arrival_time,
            "get_tugboat_fuel_used": self.get_tugboat_fuel_used,      

        }
