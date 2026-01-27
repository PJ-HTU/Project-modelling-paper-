import json
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.mtrsp_vs.components import Solution


class Env(BaseEnv):
    """MTRSP-VS environment for variable speed model with distance-based routing and speed optimization."""
    
    def __init__(self, data_name: str, **kwargs):
        """Initialize MTRSP-VS environment.
        
        Args:
            data_name: Name/path of the data file
            **kwargs: Additional arguments passed to BaseEnv
        """
        super().__init__(data_name, "mtrsp_vs")
        
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
        """Load tugboat scheduling instance data from JSON file for Model 2 (MTRSP-VS)."""
        import json
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Extract basic dimensions
        num_tasks = data['metadata']['num_tasks']
        num_tugs = data['metadata']['num_tugs']
        num_speeds = 3  # Fixed in Model 2
        
        # Build list-based parameters for solver
        task_max_tugs = [data['tasks'][str(s)]['num_tugs_needed'] for s in range(1, num_tasks+1)]
        task_min_hp = [data['tasks'][str(s)]['min_horsepower'] for s in range(1, num_tasks+1)]
        task_tw_lower = [data['tasks'][str(s)]['time_window'][0] for s in range(1, num_tasks+1)]
        task_tw_upper = [data['tasks'][str(s)]['time_window'][1] for s in range(1, num_tasks+1)]
        task_service_dist = [data['tasks'][str(s)]['service_distance'] for s in range(1, num_tasks+1)]
        
        tugboat_hp = [data['tugboats'][str(k)]['horsepower'] for k in range(1, num_tugs+1)]
        tugboat_fuel = [data['tugboats'][str(k)]['fuel_capacity'] for k in range(1, num_tugs+1)]
        tugboat_alpha = [data['tugboats'][str(k)]['alpha'] for k in range(1, num_tugs+1)]
        tugboat_beta = [data['tugboats'][str(k)]['beta'] for k in range(1, num_tugs+1)]
        
        # Speed parameters
        speed_names = ['slow', 'medium', 'fast']
        speed_values = [data['speed_levels'][name]['speed'] for name in speed_names]
        speed_power_coef = [data['speed_levels'][name]['power_coef'] for name in speed_names]
        
        # Distance matrix (already in correct format)
        distance_matrix = data['distance_matrix']
        
        # Model parameters
        big_M = data['metadata']['M']
        T_max = data['metadata']['T_max']
        penalty_weight = data['metadata']['W']
        
        # Prepare complete instance_data for solver
        instance_data = {
            'num_tasks': num_tasks,
            'num_tugboats': num_tugs,
            'num_speed_levels': num_speeds,
            'task_max_tugs': task_max_tugs,
            'task_min_horsepower': task_min_hp,
            'task_time_window_lower': task_tw_lower,
            'task_time_window_upper': task_tw_upper,
            'task_service_distance': task_service_dist,
            'tugboat_horsepower': tugboat_hp,
            'tugboat_fuel_capacity': tugboat_fuel,
            'tugboat_alpha': tugboat_alpha,
            'tugboat_beta': tugboat_beta,
            'speed_level_names': speed_names,
            'speed_values': speed_values,
            'speed_power_coefficients': speed_power_coef,
            'distance_matrix': distance_matrix,
            'big_M': big_M,
            'planning_horizon': T_max,
            'penalty_weight': penalty_weight
        }
        
        return instance_data
    
    def init_solution(self) -> Solution:
        """Initialize an empty solution with no tasks assigned.
        
        Returns:
            Solution with empty routes, no task assignments, no speeds, and no start times
        """
        num_tugboats = self.instance_data["num_tugboats"]
        num_tasks = self.instance_data["num_tasks"]
        
        # Create empty routes for each tugboat
        empty_routes = [[] for _ in range(num_tugboats)]
        
        # Create empty task tugboats dictionary
        empty_task_tugboats = {}
        
        # Create empty task start times dictionary
        empty_task_start_times = {}
        
        # Create empty task service speeds dictionary
        empty_task_service_speeds = {}
        
        # Create empty transit speeds dictionary
        empty_transit_speeds = {}
        
        return Solution(
            routes=empty_routes, 
            task_tugboats=empty_task_tugboats,
            task_start_times=empty_task_start_times,
            task_service_speeds=empty_task_service_speeds,
            transit_speeds=empty_transit_speeds,
            num_tasks=num_tasks
        )
    
    def get_key_value(self, solution: Solution = None) -> float:
        """Calculate the objective function value with speed-dependent fuel consumption.
        
        Objective = Service Fuel + Transit Fuel + penalty_weight x Number of Unexecuted Tasks
        
        Args:
            solution: Solution to evaluate. If None, uses current_solution.
            
        Returns:
            Total cost (fuel + penalty)
        """
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        distance_matrix = self.instance_data["distance_matrix"]
        task_service_distance = self.instance_data["task_service_distance"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        speed_values = self.instance_data["speed_values"]
        penalty_weight = self.instance_data["penalty_weight"]
        
        v_medium = speed_values[1]  # Medium speed (10.0 knots)
        
        # Track executed tasks
        executed_tasks = set()
        for route in solution.routes:
            executed_tasks.update(route)
        
        # ========== Service Fuel Consumption ==========
        service_fuel = 0.0
        for task_id, tugboat_ids in solution.task_tugboats.items():
            # Get service speed level and distance
            speed_level = solution.task_service_speeds.get(task_id, 1)
            v_L = speed_values[speed_level]
            d_s = task_service_distance[task_id - 1]  # Convert to 0-indexed
            
            # Service fuel for each tugboat
            for tugboat_id in tugboat_ids:
                α_k = tugboat_alpha[tugboat_id]
                HP_k = tugboat_horsepower[tugboat_id]
                
                # ✅ CORRECT: ψ_s^kL = α_k × HP_k × d_s × (v_L² / v_medium³)
                ψ = α_k * HP_k * d_s * (v_L ** 2) / (v_medium ** 3)
                service_fuel += ψ
        
        # ========== Transit Fuel Consumption ==========
        transit_fuel = 0.0
        for tugboat_id, route in enumerate(solution.routes):
            if len(route) == 0:
                continue
            
            HP_k = tugboat_horsepower[tugboat_id]
            β_k = tugboat_beta[tugboat_id]
            
            # Process each arc in the route
            from_node = -1  # Start from depot
            
            for task_id in route:
                # Get transit speed for this arc
                speed_key = (tugboat_id, from_node, task_id)
                speed_level = solution.transit_speeds.get(speed_key, 1)
                v_L = speed_values[speed_level]
                
                # Get distance
                if from_node == -1:
                    # Depot -> task
                    d_ij = distance_matrix[f"0_{task_id}"]
                else:
                    # Task -> task
                    d_ij = distance_matrix[f"{from_node}_{task_id}"]
                
                # ✅ CORRECT: φ_ij^kL = β_k × HP_k × d_ij × (v_L² / v_medium³)
                φ = β_k * HP_k * d_ij * (v_L ** 2) / (v_medium ** 3)
                transit_fuel += φ
                
                from_node = task_id
            
            # Return to depot
            speed_key = (tugboat_id, from_node, num_tasks)
            speed_level = solution.transit_speeds.get(speed_key, 1)
            v_L = speed_values[speed_level]
            
            # Distance: last task -> depot
            d_ij = distance_matrix[f"{from_node}_{num_tasks + 1}"]
            
            φ = β_k * HP_k * d_ij * (v_L ** 2) / (v_medium ** 3)
            transit_fuel += φ
        
        # ========== Penalty for Unexecuted Tasks ==========
        all_tasks = set(range(1, num_tasks + 1))
        unexecuted_tasks = all_tasks - executed_tasks
        penalty = penalty_weight * len(unexecuted_tasks)
        
        # ========== Total Cost ==========
        total_cost = service_fuel + transit_fuel + penalty
        
        return total_cost
    
    def validation_solution(self, solution: Solution = None) -> bool:
        """Validate if the solution satisfies all constraints."""
        if solution is None:
            solution = self.current_solution
        
        # Get instance data
        num_tasks = self.instance_data["num_tasks"]
        num_tugboats = self.instance_data["num_tugboats"]
        distance_matrix = self.instance_data["distance_matrix"]
        task_service_distance = self.instance_data["task_service_distance"]
        task_time_window_lower = self.instance_data["task_time_window_lower"]
        task_time_window_upper = self.instance_data["task_time_window_upper"]
        task_max_tugs = self.instance_data["task_max_tugs"]
        task_min_horsepower = self.instance_data["task_min_horsepower"]
        tugboat_horsepower = self.instance_data["tugboat_horsepower"]
        tugboat_fuel_capacity = self.instance_data["tugboat_fuel_capacity"]
        tugboat_alpha = self.instance_data["tugboat_alpha"]
        tugboat_beta = self.instance_data["tugboat_beta"]
        speed_values = self.instance_data["speed_values"]
        planning_horizon = self.instance_data["planning_horizon"]
        
        v_medium = speed_values[1]
        
        # ========== 0. Validate Solution Structure Consistency ==========
        # Check: routes <-> task_tugboats consistency
        for task_id, tugboat_ids in solution.task_tugboats.items():
            for tugboat_id in tugboat_ids:
                if tugboat_id >= num_tugboats:
                    return False
                if task_id not in solution.routes[tugboat_id]:
                    return False
        
        for tugboat_id, route in enumerate(solution.routes):
            for task_id in route:
                if task_id not in solution.task_tugboats:
                    return False
                if tugboat_id not in solution.task_tugboats[task_id]:
                    return False
        
        # ========== 1. Check Task Node Validity ==========
        for route in solution.routes:
            for task_id in route:
                if not (1 <= task_id <= num_tasks):
                    return False
        
        # Identify executed tasks
        executed_tasks = set()
        for route in solution.routes:
            executed_tasks.update(route)
        
        # ========== 2. Check Speed Level Validity ==========
        for task_id, speed_level in solution.task_service_speeds.items():
            if not (0 <= speed_level <= 2):
                return False
        
        for (tugboat_id, from_node, to_node), speed_level in solution.transit_speeds.items():
            if not (0 <= speed_level <= 2):
                return False
        
        # ========== 3. Check Task Execution Constraints ==========
        for task_id in executed_tasks:
            tugboats_assigned = solution.task_tugboats[task_id]
            
            # Check: at most task_max_tugs tugboats
            if len(tugboats_assigned) > task_max_tugs[task_id - 1]:
                return False
            
            # Check: total horsepower >= task_min_horsepower
            total_hp = sum(tugboat_horsepower[tug_id] for tug_id in tugboats_assigned)
            if total_hp < task_min_horsepower[task_id - 1]:
                return False
        
        # ========== 4. Validate Start Times and Service Completion ==========
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
            
            # Check: task has a service speed
            if task_id not in solution.task_service_speeds:
                return False
            
            # Calculate service duration based on speed
            speed_level = solution.task_service_speeds[task_id]
            v_L = speed_values[speed_level]
            d_s = task_service_distance[task_id - 1]
            service_time = d_s / v_L
            
            # Check: service finishes within planning horizon
            finish_time = task_start + service_time
            if finish_time > planning_horizon + 1e-6:
                return False
        
        # ========== 5. Check Tugboat Constraints ==========
        for tugboat_id, route in enumerate(solution.routes):
            if len(route) == 0:
                continue
            
            HP_k = tugboat_horsepower[tugboat_id]
            α_k = tugboat_alpha[tugboat_id]
            β_k = tugboat_beta[tugboat_id]
            
            # Check time feasibility and arc speed definitions
            current_time = 0.0
            from_node = -1  # Start from depot
            
            for task_id in route:
                # Check transit speed exists
                speed_key = (tugboat_id, from_node, task_id)
                if speed_key not in solution.transit_speeds:
                    return False
                
                speed_level = solution.transit_speeds[speed_key]
                v_L = speed_values[speed_level]
                
                # Get distance
                if from_node == -1:
                    d_ij = distance_matrix[f"0_{task_id}"]
                else:
                    d_ij = distance_matrix[f"{from_node}_{task_id}"]
                
                # Calculate travel time and arrival
                travel_time = d_ij / v_L
                arrival_time = current_time + travel_time
                task_start = start_times[task_id]
                
                # Tugboat must arrive before or at start time
                if arrival_time > task_start + 1e-6:
                    return False
                
                # Calculate service time
                service_speed_level = solution.task_service_speeds[task_id]
                v_service = speed_values[service_speed_level]
                d_s = task_service_distance[task_id - 1]
                service_time = d_s / v_service
                
                # Update time to end of service
                current_time = task_start + service_time
                from_node = task_id
            
            # Check return to depot speed exists
            speed_key = (tugboat_id, from_node, num_tasks)
            if speed_key not in solution.transit_speeds:
                return False
            
            # Check fuel capacity
            total_fuel = 0.0
            from_node = -1
            
            for task_id in route:
                # Transit fuel
                speed_key = (tugboat_id, from_node, task_id)
                speed_level = solution.transit_speeds[speed_key]
                v_L = speed_values[speed_level]
                
                if from_node == -1:
                    d_ij = distance_matrix[f"0_{task_id}"]
                else:
                    d_ij = distance_matrix[f"{from_node}_{task_id}"]
                
                φ = β_k * HP_k * d_ij * (v_L ** 2) / (v_medium ** 3)
                total_fuel += φ
                
                # Service fuel
                service_speed_level = solution.task_service_speeds[task_id]
                v_service = speed_values[service_speed_level]
                d_s = task_service_distance[task_id - 1]
                
                ψ = α_k * HP_k * d_s * (v_service ** 2) / (v_medium ** 3)
                total_fuel += ψ
                
                from_node = task_id
            
            # Return to depot fuel
            speed_key = (tugboat_id, from_node, num_tasks)
            speed_level = solution.transit_speeds[speed_key]
            v_L = speed_values[speed_level]
            d_ij = distance_matrix[f"{from_node}_{num_tasks + 1}"]
            
            φ = β_k * HP_k * d_ij * (v_L ** 2) / (v_medium ** 3)
            total_fuel += φ
            
            # Check capacity constraint
            if total_fuel > tugboat_fuel_capacity[tugboat_id] + 1e-6:
                return False
        
        return True


    def helper_function(self) -> dict:
        """Return dictionary of all helper functions for heuristic algorithms.
        
        Includes helper functions specific to variable speed model.
        """
        return {
        # Core functions from env
        "get_problem_state": self.get_problem_state,
        "validation_solution": self.validation_solution,
    }
