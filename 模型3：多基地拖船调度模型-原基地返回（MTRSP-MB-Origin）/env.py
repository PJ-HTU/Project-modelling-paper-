import json
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.mtrsp_mb.components import Solution


class Env(BaseEnv):
    """MTRSP-MB environment for multi-base model with origin return constraint."""
    
    def __init__(self, data_name: str, **kwargs):
        """Initialize MTRSP-MB environment.
        
        Args:
            data_name: Name/path of the data file
            **kwargs: Additional arguments passed to BaseEnv
        """
        super().__init__(data_name, "mtrsp_mb")
        
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
        """Load multi-base tugboat scheduling instance data from JSON file."""
        import json
        
        with open(data_path, 'r') as f:
            instance = json.load(f)
        
        tasks = instance['tasks']
        tugboats = instance['tugboats']
        bases = instance['bases']
        
        num_tasks = len(tasks)
        num_tugboats = len(tugboats)
        num_bases = len(bases)
        
        # Extract base capacity (按基地编号顺序：-1, -2, -3, ...)
        base_capacity = []
        for b in range(1, num_bases + 1):
            base_key = str(-b)
            base_capacity.append(bases[base_key]['capacity_out'])
        
        # Extract tugboat base assignment
        tugboat_base_assignment = []
        for k in range(1, num_tugboats + 1):
            tugboat_base_assignment.append(tugboats[str(k)]['home_base'])
        
        return {
            "num_tasks": num_tasks,
            "num_tugboats": num_tugboats,
            "num_bases": num_bases,
            "task_max_tugs": [tasks[str(s)]['num_tugs_needed'] for s in range(1, num_tasks+1)],
            "task_min_horsepower": [tasks[str(s)]['min_horsepower'] for s in range(1, num_tasks+1)],
            "task_time_window_lower": [tasks[str(s)]['time_window'][0] for s in range(1, num_tasks+1)],
            "task_time_window_upper": [tasks[str(s)]['time_window'][1] for s in range(1, num_tasks+1)],
            "task_service_time": [tasks[str(s)]['service_time'] for s in range(1, num_tasks+1)],
            "tugboat_horsepower": [tugboats[str(k)]['horsepower'] for k in range(1, num_tugboats+1)],
            "tugboat_fuel_capacity": [tugboats[str(k)]['fuel_capacity'] for k in range(1, num_tugboats+1)],
            "tugboat_alpha": [tugboats[str(k)]['alpha'] for k in range(1, num_tugboats+1)],
            "tugboat_beta": [tugboats[str(k)]['beta'] for k in range(1, num_tugboats+1)],
            "tugboat_base_assignment": tugboat_base_assignment,
            "base_capacity": base_capacity,
            "time_matrix": instance['time_matrix'],
            "big_M": instance['metadata']['M'],
            "planning_horizon": instance['metadata']['T_max'],
            "penalty_weight": instance['metadata']['W']
        }
    
    def init_solution(self) -> Solution:
        """Initialize an empty solution with no tasks assigned.
        
        Returns:
            Solution with empty routes, no task assignments, and no start times
        """
        num_tugboats = self.instance_data["num_tugboats"]
        num_tasks = self.instance_data["num_tasks"]
        num_bases = self.instance_data["num_bases"]
        tugboat_base_assignment = self.instance_data["tugboat_base_assignment"]
        
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
            tugboat_base_assignment=tugboat_base_assignment,
            num_tasks=num_tasks,
            num_bases=num_bases
        )
    
    def get_key_value(self, solution = None) -> float:
        """Calculate the objective function value for multi-base model.
        
        Objective = Service Fuel + Transit Fuel + penalty_weight x Number of Unexecuted Tasks
        
        Transit Fuel includes:
        - Z_out: From base origins to first tasks
        - Z_mid: Between tasks
        - Z_in: From last tasks to base destinations
        
        Args:
            solution: Solution to evaluate. If None, uses current_solution.
            
        Returns:
            Total cost (fuel + penalty)
        """
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        num_bases = self.instance_data["num_bases"]
        time_matrix = self.instance_data["time_matrix"]
        task_service_time = self.instance_data["task_service_time"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        tugboat_base_assignment = self.instance_data["tugboat_base_assignment"]
        penalty_weight = self.instance_data["penalty_weight"]
        
        # Track executed tasks
        # ⭐ 修改：只统计任务节点（过滤掉基地和终点）
        executed_tasks = set()
        for route in solution.routes:
            task_nodes = [node for node in route if 1 <= node <= num_tasks]
            executed_tasks.update(task_nodes)
        
        # ========== Service Fuel Consumption ==========
        service_fuel = 0.0
        for task_id, tugboat_ids in solution.task_tugboats.items():
            T_s = task_service_time[task_id - 1]  # Convert to 0-indexed
            
            # Service fuel for each tugboat
            for tugboat_id in tugboat_ids:
                α_k = tugboat_alpha[tugboat_id]
                HP_k = tugboat_horsepower[tugboat_id]
                
                # Z_service = α_k × HP_k × T_s
                service_fuel += α_k * HP_k * T_s
        
        # ========== Transit Fuel Consumption ==========
        transit_fuel = 0.0
        
        for tugboat_id, route in enumerate(solution.routes):
            if len(route) == 0:
                continue
            
            # ⭐ 关键修改：过滤出任务节点（排除基地和终点）
            task_route = [node for node in route if 1 <= node <= num_tasks]
            
            if len(task_route) == 0:
                continue
            
            HP_k = tugboat_horsepower[tugboat_id]
            β_k = tugboat_beta[tugboat_id]
            home_base = tugboat_base_assignment[tugboat_id]  # Negative integer: -1, -2, -3, ...
            
            # Z_out: From base origin to first task
            first_task = task_route[0]
            key_out = f"{home_base}_{first_task}"
            if key_out in time_matrix:
                t_out = time_matrix[key_out]
                transit_fuel += β_k * HP_k * t_out
            
            # Z_mid: Between tasks
            for i in range(len(task_route) - 1):
                from_task = task_route[i]
                to_task = task_route[i + 1]
                key_mid = f"{from_task}_{to_task}"
                if key_mid in time_matrix:
                    t_mid = time_matrix[key_mid]
                    transit_fuel += β_k * HP_k * t_mid
            
            # Z_in: From last task to base destination
            last_task = task_route[-1]
            dest_node = num_tasks - home_base  # Destination: n - b (e.g., n - (-1) = n + 1)
            key_in = f"{last_task}_{dest_node}"
            if key_in in time_matrix:
                t_in = time_matrix[key_in]
                transit_fuel += β_k * HP_k * t_in
        
        # ========== Penalty for Unexecuted Tasks ==========
        num_unexecuted = num_tasks - len(executed_tasks)
        penalty = penalty_weight * num_unexecuted
        
        # ========== Total Cost ==========
        total_cost = service_fuel + transit_fuel + penalty
        
        return total_cost
    
    def validation_solution(self, solution = None) -> bool:
        """Validate solution feasibility for multi-base model.
        
        Checks:
        1. Solution structure consistency (routes <-> task_tugboats)
        2. Task node validity
        3. Task execution constraints (tugboat count, horsepower)
        4. Start times and time windows
        5. Tugboat constraints (time feasibility, fuel capacity)
        6. Base capacity constraints
        7. Origin return constraints (tugboat returns to home base)
        
        Args:
            solution: Solution to validate. If None, uses current_solution.
            
        Returns:
            True if solution is feasible, False otherwise
        """
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        num_tugboats = self.instance_data["num_tugboats"]
        num_bases = self.instance_data["num_bases"]
        task_max_tugs = self.instance_data["task_max_tugs"]
        task_time_window_lower = self.instance_data["task_time_window_lower"]
        task_time_window_upper = self.instance_data["task_time_window_upper"]
        task_min_horsepower = self.instance_data["task_min_horsepower"]
        task_service_time = self.instance_data["task_service_time"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_fuel_capacity = self.instance_data["tugboat_fuel_capacity"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        tugboat_base_assignment = self.instance_data["tugboat_base_assignment"]
        base_capacity = self.instance_data["base_capacity"]
        time_matrix = self.instance_data["time_matrix"]
        planning_horizon = self.instance_data["planning_horizon"]
        
        # ========== 0. Validate Solution Structure Consistency ==========
        # Check: routes <-> task_tugboats consistency
        for task_id, tugboat_ids in solution.task_tugboats.items():
            for tugboat_id in tugboat_ids:
                if tugboat_id >= num_tugboats:
                    return False
                # ⭐ 修改：过滤掉非任务节点
                task_nodes_in_route = [node for node in solution.routes[tugboat_id] if 1 <= node <= num_tasks]
                if task_id not in task_nodes_in_route:
                    return False
        
        for tugboat_id, route in enumerate(solution.routes):
            # ⭐ 修改：只检查任务节点
            task_nodes = [node for node in route if 1 <= node <= num_tasks]
            for task_id in task_nodes:
                if task_id not in solution.task_tugboats:
                    return False
                if tugboat_id not in solution.task_tugboats[task_id]:
                    return False
        
        # ========== 1. Check Task Node Validity ==========
        for route in solution.routes:
            # ⭐ 修改：允许路径中有基地和终点节点，但任务节点必须有效
            task_nodes = [node for node in route if 1 <= node <= num_tasks]
            for task_id in task_nodes:
                if not (1 <= task_id <= num_tasks):
                    return False
        
        # ⭐ 修改：只统计任务节点
        executed_tasks = set()
        for route in solution.routes:
            task_nodes = [node for node in route if 1 <= node <= num_tasks]
            executed_tasks.update(task_nodes)
        
        # ========== 2. Check Task Execution Constraints ==========
        for task_id in executed_tasks:
            tugboats_assigned = solution.task_tugboats[task_id]
            
            # Check: at most task_max_tugs tugboats
            if len(tugboats_assigned) > task_max_tugs[task_id - 1]:
                return False
            
            # Check: total horsepower >= task_min_horsepower
            total_hp = sum(tugboat_horsepower[tug_id] for tug_id in tugboats_assigned)
            if total_hp < task_min_horsepower[task_id - 1]:
                return False
        
        # ========== 3. Validate Start Times and Service Completion ==========
        start_times = solution.task_start_times
        
        for task_id in executed_tasks:
            # Check task has a start time
            if task_id not in start_times:
                return False
            
            task_start = start_times[task_id]
            
            # Check: service starts within time window
            if task_start < task_time_window_lower[task_id - 1] - 1e-6 or \
               task_start > task_time_window_upper[task_id - 1] + 1e-6:
                return False
            
            # Check: service finishes within planning horizon
            service_time = task_service_time[task_id - 1]
            finish_time = task_start + service_time
            if finish_time > planning_horizon + 1e-6:
                return False
        
        # ========== 4. Check Base Capacity Constraints ==========
        # Count tugboats departing from and returning to each base
        base_departures = {b: 0 for b in range(1, num_bases + 1)}
        base_arrivals = {b: 0 for b in range(1, num_bases + 1)}
        
        for tugboat_id, route in enumerate(solution.routes):
            # ⭐ 修改：过滤任务节点
            task_nodes = [node for node in route if 1 <= node <= num_tasks]
            if len(task_nodes) == 0:
                continue
            
            home_base = tugboat_base_assignment[tugboat_id]  # Negative: -1, -2, -3, ...
            base_idx = -home_base  # Convert to positive: 1, 2, 3, ...
            
            base_departures[base_idx] += 1
            base_arrivals[base_idx] += 1
        
        # Check capacity constraints
        for b in range(1, num_bases + 1):
            capacity = base_capacity[b - 1]  # 0-indexed
            
            if base_departures[b] > capacity:
                return False
            
            if base_arrivals[b] > capacity:
                return False
        
        # ========== 5. Check Tugboat Constraints ==========
        for tugboat_id, route in enumerate(solution.routes):
            # ⭐ 修改：过滤任务节点
            task_route = [node for node in route if 1 <= node <= num_tasks]
            
            if len(task_route) == 0:
                continue
            
            HP_k = tugboat_horsepower[tugboat_id]
            α_k = tugboat_alpha[tugboat_id]
            β_k = tugboat_beta[tugboat_id]
            home_base = tugboat_base_assignment[tugboat_id]  # Negative: -1, -2, -3, ...
            
            # Check time feasibility
            current_time = 0.0
            from_node = home_base  # Start from home base (negative number)
            
            for task_id in task_route:
                # Get travel time from current position to task
                key = f"{from_node}_{task_id}"
                
                if key not in time_matrix:
                    return False
                
                travel_time = time_matrix[key]
                
                # ⭐ 关键修改：拖船到达时间
                arrival_time = current_time + travel_time
                
                # 任务开始时间
                task_start = start_times[task_id]
                
                # ⭐ 关键修改：任务必须在拖船到达后才能开始（检查时间传递）
                if task_start < arrival_time - 1e-6:
                    return False
                
                # Update time to end of service
                service_time = task_service_time[task_id - 1]
                current_time = task_start + service_time
                from_node = task_id
            
            # Check return to base is possible within planning horizon
            dest_node = num_tasks - home_base  # Destination: n - b
            key_return = f"{from_node}_{dest_node}"
            
            if key_return not in time_matrix:
                return False
            
            return_time = time_matrix[key_return]
            final_time = current_time + return_time
            
            if final_time > planning_horizon + 1e-6:
                return False
            
            # Check fuel capacity
            total_fuel = 0.0
            from_node = home_base
            
            # Fuel from base to first task
            first_task = task_route[0]
            key_out = f"{from_node}_{first_task}"
            if key_out in time_matrix:
                t_out = time_matrix[key_out]
                total_fuel += β_k * HP_k * t_out
            
            # Fuel for tasks and transit between tasks
            for i, task_id in enumerate(task_route):
                # Service fuel
                service_time = task_service_time[task_id - 1]
                total_fuel += α_k * HP_k * service_time
                
                # Transit to next task (if exists)
                if i < len(task_route) - 1:
                    next_task = task_route[i + 1]
                    key_mid = f"{task_id}_{next_task}"
                    if key_mid in time_matrix:
                        t_mid = time_matrix[key_mid]
                        total_fuel += β_k * HP_k * t_mid
            
            # Fuel from last task to base destination
            last_task = task_route[-1]
            dest_node = num_tasks - home_base
            key_in = f"{last_task}_{dest_node}"
            if key_in in time_matrix:
                t_in = time_matrix[key_in]
                total_fuel += β_k * HP_k * t_in
            
            # Check capacity constraint
            if total_fuel > tugboat_fuel_capacity[tugboat_id] + 1e-6:
                return False
        
        # ========== 6. Check Origin Return Constraint ==========
        # Verify tugboat_base_assignment matches solution structure
        for tugboat_id in range(num_tugboats):
            expected_base = tugboat_base_assignment[tugboat_id]
            actual_base = solution.tugboat_base_assignment[tugboat_id]
            
            if expected_base != actual_base:
                return False
        
        return True

    def helper_function(self) -> dict:
        """Return dictionary of all helper functions for heuristic algorithms.
        
        Includes helper functions specific to multi-base model.
        """
        return {
            # Core functions from env
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
        }