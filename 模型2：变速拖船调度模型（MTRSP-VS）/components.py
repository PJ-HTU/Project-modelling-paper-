from src.problems.base.components import BaseSolution, BaseOperator
from typing import List, Dict, Optional, Union, Tuple


class Solution(BaseSolution):

    def __init__(self, routes: List[List[int]], task_tugboats: Dict[int, List[int]], 
                 task_start_times: Dict[int, float], task_service_speeds: Dict[int, int],
                 transit_speeds: Dict[Tuple[int, int, int], int], num_tasks: int):
        self.routes = routes
        self.task_tugboats = task_tugboats
        self.task_start_times = task_start_times
        self.task_service_speeds = task_service_speeds
        self.transit_speeds = transit_speeds
        self.num_tasks = num_tasks
    
    def __str__(self) -> str:
        """String representation showing routes, task assignments, speeds, and start times."""
        if not self.routes:
            return "Empty solution with no tugboat routes"
        
        # Speed level names for display
        speed_names = {0: "Slow", 1: "Medium", 2: "Fast"}
        
        result = "MTRSP-VS Solution:\n"
        result += "="*80 + "\n"
        
        # Show routes with transit speeds
        result += "Tugboat Routes (with transit speeds):\n"
        for tugboat_id, route in enumerate(self.routes):
            if len(route) > 0:
                route_parts = ["Depot"]
                
                # Build route with transit speeds
                from_node = -1  # Start from depot
                for task_id in route:
                    # Get transit speed for this arc
                    speed_key = (tugboat_id, from_node, task_id)
                    speed_level = self.transit_speeds.get(speed_key, 1)  # Default to medium
                    speed_name = speed_names.get(speed_level, "Unknown")
                    
                    route_parts.append(f"-[{speed_name}]->")
                    route_parts.append(f"Task{task_id}")
                    from_node = task_id
                
                # Return to depot
                speed_key = (tugboat_id, from_node, self.num_tasks)
                speed_level = self.transit_speeds.get(speed_key, 1)
                speed_name = speed_names.get(speed_level, "Unknown")
                route_parts.append(f"-[{speed_name}]->")
                route_parts.append("Depot")
                
                route_str = " ".join(route_parts)
                result += f"  Tugboat {tugboat_id}: {route_str}\n"
            else:
                result += f"  Tugboat {tugboat_id}: Not used\n"
        
        # Show task assignments with service speeds and start times
        result += "\nTask Assignments (with service speeds and start times):\n"
        if self.task_tugboats:
            sorted_tasks = sorted(self.task_tugboats.items())
            for task_id, tugboat_ids in sorted_tasks:
                tugboats_str = ", ".join([f"Tugboat{i}" for i in sorted(tugboat_ids)])
                collab_indicator = " [COLLABORATIVE]" if len(tugboat_ids) > 1 else ""
                
                # Get service speed
                speed_level = self.task_service_speeds.get(task_id, 1)
                speed_name = speed_names.get(speed_level, "Unknown")
                
                # Get start time
                start_time = self.task_start_times.get(task_id, None)
                
                if start_time is not None:
                    result += f"  Task {task_id}: [{tugboats_str}] at {start_time:.2f}h, " \
                             f"Service Speed: {speed_name}{collab_indicator}\n"
                else:
                    result += f"  Task {task_id}: [{tugboats_str}], " \
                             f"Service Speed: {speed_name} (no start time){collab_indicator}\n"
        else:
            result += "  No tasks executed\n"
        
        result += "="*80
        return result


# ==================== Unified Constructive Operators with Speed Selection ====================

class AppendTaskOperator(BaseOperator):
    """
    Append a task to the end of one or more tugboat routes.
    
    Handles both single-tugboat and multi-tugboat collaborative assignments.
    Guarantees consistency across all solution components.
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int, start_time: float,
                 service_speed: int, transit_speeds_to: Union[int, List[int]],
                 transit_speeds_from: Optional[Union[int, List[int]]] = None):
        """
        Args:
            tugboat_ids: Tugboat ID(s) (0-indexed)
            task_id: Task ID (1-indexed)
            start_time: Task start time (hours)
            service_speed: Service speed level (0=slow, 1=medium, 2=fast)
            transit_speeds_to: Transit speed(s) TO this task
            transit_speeds_from: Transit speed(s) FROM this task to depot (optional, default=1)
        """
        # Normalize to lists
        if isinstance(tugboat_ids, int):
            self.tugboat_ids = [tugboat_ids]
        else:
            self.tugboat_ids = sorted(tugboat_ids)
        
        if isinstance(transit_speeds_to, int):
            self.transit_speeds_to = [transit_speeds_to] * len(self.tugboat_ids)
        else:
            self.transit_speeds_to = list(transit_speeds_to)
        
        if transit_speeds_from is None:
            self.transit_speeds_from = [1] * len(self.tugboat_ids)  # Default: medium speed
        elif isinstance(transit_speeds_from, int):
            self.transit_speeds_from = [transit_speeds_from] * len(self.tugboat_ids)
        else:
            self.transit_speeds_from = list(transit_speeds_from)
        
        # Validate
        if len(self.tugboat_ids) != len(self.transit_speeds_to):
            raise ValueError(f"tugboat_ids length {len(self.tugboat_ids)} != transit_speeds_to length {len(self.transit_speeds_to)}")
        if len(self.tugboat_ids) != len(self.transit_speeds_from):
            raise ValueError(f"tugboat_ids length {len(self.tugboat_ids)} != transit_speeds_from length {len(self.transit_speeds_from)}")
        
        self.task_id = task_id
        self.start_time = start_time
        self.service_speed = service_speed
    
    def run(self, solution: 'Solution') -> 'Solution':
        """Apply the append operation."""
        # Deep copy all components
        new_routes = [route[:] for route in solution.routes]
        new_task_tugboats = {tid: tugs[:] for tid, tugs in solution.task_tugboats.items()}
        new_task_start_times = solution.task_start_times.copy()
        new_task_service_speeds = solution.task_service_speeds.copy()
        new_transit_speeds = solution.transit_speeds.copy()
        
        # Update for each tugboat
        for i, tugboat_id in enumerate(self.tugboat_ids):
            # Determine previous node
            if len(new_routes[tugboat_id]) == 0:
                from_node = -1  # From depot
            else:
                from_node = new_routes[tugboat_id][-1]  # From last task
                # Delete old arc: previous_task -> depot
                old_arc = (tugboat_id, from_node, solution.num_tasks)
                if old_arc in new_transit_speeds:
                    del new_transit_speeds[old_arc]
            
            # Append task to route
            new_routes[tugboat_id].append(self.task_id)
            
            # Add arc: from_node -> task_id
            new_transit_speeds[(tugboat_id, from_node, self.task_id)] = self.transit_speeds_to[i]
            
            # Add arc: task_id -> depot
            new_transit_speeds[(tugboat_id, self.task_id, solution.num_tasks)] = self.transit_speeds_from[i]
        
        # Update task_tugboats
        if self.task_id in new_task_tugboats:
            # Collaborative task: add new tugboats
            existing = set(new_task_tugboats[self.task_id])
            combined = existing.union(self.tugboat_ids)
            new_task_tugboats[self.task_id] = sorted(combined)
        else:
            # New task
            new_task_tugboats[self.task_id] = sorted(self.tugboat_ids)
        
        # Set start time
        if self.task_id not in new_task_start_times:
            new_task_start_times[self.task_id] = self.start_time
        else:
            # Validate for collaborative tasks
            if abs(new_task_start_times[self.task_id] - self.start_time) > 1e-6:
                raise ValueError(f"Start time conflict for task {self.task_id}")
        
        # Set service speed
        if self.task_id not in new_task_service_speeds:
            new_task_service_speeds[self.task_id] = self.service_speed
        else:
            # Validate for collaborative tasks
            if new_task_service_speeds[self.task_id] != self.service_speed:
                raise ValueError(f"Service speed conflict for task {self.task_id}")
        
        return Solution(routes=new_routes, task_tugboats=new_task_tugboats,
                       task_start_times=new_task_start_times,
                       task_service_speeds=new_task_service_speeds,
                       transit_speeds=new_transit_speeds, num_tasks=solution.num_tasks)


class InsertTaskOperator(BaseOperator):
    """
    Insert a task at specific positions in one or more tugboat routes.
    
    Handles both single-tugboat and multi-tugboat collaborative assignments.
    Guarantees consistency across all solution components.
    """
    
    def __init__(self, tugboat_ids: Union[int, List[int]], task_id: int,
                 positions: Union[int, List[int]], start_time: float, service_speed: int,
                 transit_speeds_to: Union[int, List[int]], 
                 transit_speeds_from: Union[int, List[int]]):
        """
        Args:
            tugboat_ids: Tugboat ID(s) (0-indexed)
            task_id: Task ID (1-indexed)
            positions: Insertion position(s) in each route
            start_time: Task start time (hours)
            service_speed: Service speed level (0=slow, 1=medium, 2=fast)
            transit_speeds_to: Transit speed(s) TO this task
            transit_speeds_from: Transit speed(s) FROM this task
        """
        # Normalize to lists
        if isinstance(tugboat_ids, int):
            self.tugboat_ids = [tugboat_ids]
        else:
            self.tugboat_ids = list(tugboat_ids)
        
        if isinstance(positions, int):
            self.positions = [positions] * len(self.tugboat_ids)
        else:
            self.positions = list(positions)
        
        if isinstance(transit_speeds_to, int):
            self.transit_speeds_to = [transit_speeds_to] * len(self.tugboat_ids)
        else:
            self.transit_speeds_to = list(transit_speeds_to)
        
        if isinstance(transit_speeds_from, int):
            self.transit_speeds_from = [transit_speeds_from] * len(self.tugboat_ids)
        else:
            self.transit_speeds_from = list(transit_speeds_from)
        
        # Validate
        if len(self.tugboat_ids) != len(self.positions):
            raise ValueError(f"tugboat_ids length {len(self.tugboat_ids)} != positions length {len(self.positions)}")
        if len(self.tugboat_ids) != len(self.transit_speeds_to):
            raise ValueError(f"tugboat_ids length {len(self.tugboat_ids)} != transit_speeds_to length {len(self.transit_speeds_to)}")
        if len(self.tugboat_ids) != len(self.transit_speeds_from):
            raise ValueError(f"tugboat_ids length {len(self.tugboat_ids)} != transit_speeds_from length {len(self.transit_speeds_from)}")
        
        self.task_id = task_id
        self.start_time = start_time
        self.service_speed = service_speed
    
    def run(self, solution: 'Solution') -> 'Solution':
        """Apply the insert operation."""
        # Deep copy all components
        new_routes = [route[:] for route in solution.routes]
        new_task_tugboats = {tid: tugs[:] for tid, tugs in solution.task_tugboats.items()}
        new_task_start_times = solution.task_start_times.copy()
        new_task_service_speeds = solution.task_service_speeds.copy()
        new_transit_speeds = solution.transit_speeds.copy()
        
        # Update for each tugboat
        for i, (tugboat_id, position) in enumerate(zip(self.tugboat_ids, self.positions)):
            # Determine nodes
            if position == 0:
                from_node = -1
            else:
                from_node = new_routes[tugboat_id][position - 1]
            
            if position >= len(new_routes[tugboat_id]):
                to_node = solution.num_tasks  # Depot
            else:
                to_node = new_routes[tugboat_id][position]
            
            # Delete old arc: from_node -> to_node
            old_arc = (tugboat_id, from_node, to_node)
            if old_arc in new_transit_speeds:
                del new_transit_speeds[old_arc]
            
            # Insert task into route
            new_routes[tugboat_id].insert(position, self.task_id)
            
            # Add new arcs: from_node -> task_id -> to_node
            new_transit_speeds[(tugboat_id, from_node, self.task_id)] = self.transit_speeds_to[i]
            new_transit_speeds[(tugboat_id, self.task_id, to_node)] = self.transit_speeds_from[i]
        
        # Update task_tugboats
        if self.task_id in new_task_tugboats:
            # Collaborative task: add new tugboats
            existing = set(new_task_tugboats[self.task_id])
            combined = existing.union(self.tugboat_ids)
            new_task_tugboats[self.task_id] = sorted(combined)
        else:
            # New task
            new_task_tugboats[self.task_id] = sorted(self.tugboat_ids)
        
        # Set start time
        if self.task_id not in new_task_start_times:
            new_task_start_times[self.task_id] = self.start_time
        else:
            # Validate for collaborative tasks
            if abs(new_task_start_times[self.task_id] - self.start_time) > 1e-6:
                raise ValueError(f"Start time conflict for task {self.task_id}")
        
        # Set service speed
        if self.task_id not in new_task_service_speeds:
            new_task_service_speeds[self.task_id] = self.service_speed
        else:
            # Validate for collaborative tasks
            if new_task_service_speeds[self.task_id] != self.service_speed:
                raise ValueError(f"Service speed conflict for task {self.task_id}")
        
        return Solution(routes=new_routes, task_tugboats=new_task_tugboats,
                       task_start_times=new_task_start_times,
                       task_service_speeds=new_task_service_speeds,
                       transit_speeds=new_transit_speeds, num_tasks=solution.num_tasks)