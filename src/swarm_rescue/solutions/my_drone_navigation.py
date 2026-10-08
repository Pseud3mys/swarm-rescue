"""Partial navigation agent using a lidar occupancy grid and A*."""

import math
import weakref
from typing import Optional

import cv2
import numpy as np

from Mapping.lidar_to_grid_map.lidar_to_grid_map import bresenham
from PathPlanning.AStar import a_star
from PathPlanning.AStar.a_star import AStarPlanner

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle


class MyDroneNavigation(DroneAbstract):
    """Navigate toward a fixed goal while building a local 2D map."""

    _debug_agents = weakref.WeakSet()
    _debug_frame = 0
    _debug_window_open = True

    def __init__(
            self,
            identifier: Optional[int] = None,
            misc_data: Optional[MiscData] = None,
            **kwargs,
    ):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

        MyDroneNavigation._debug_agents.add(self)

        self.resolution = 8.0
        self.robot_radius = 12.0
        self.obstacle_stop_distance = 60.0
        self.lidar_hit_margin = max(6.0, self.resolution * 1.5)
        self.goal = np.array((0.0, 0.0))
        self.replan_period = 90
        self.plan_retry_period = 30
        self.next_plan_iteration = 0
        self.max_plan_obstacles = 500
        self.max_frontier_targets = 3
        self.iteration = 0
        self.path_x: list[float] = []
        self.path_y: list[float] = []
        self.path_index = 0
        self.lookahead_points = 3
        self.active_goal = self.goal.copy()
        self.previous_position: Optional[np.ndarray] = None
        self.stuck_frames = 0
        self.recovery_frames = 0
        self.recovery_rotation = 1.0
        self.avoidance_mode = False
        self.avoidance_frames = 0
        self.avoidance_rotation = 1.0
        self.filtered_position: Optional[np.ndarray] = None
        self.filtered_angle: Optional[float] = None
        self.gps_filter_alpha = 0.18
        self.compass_filter_alpha = 0.2

        size_area = self.size_area or (1000.0, 1000.0)
        self.size_area = size_area
        self.grid_width = max(1, int(size_area[0] / self.resolution))
        self.grid_height = max(1, int(size_area[1] / self.resolution))
        self.grid = np.full((self.grid_width, self.grid_height), 0.5)

        a_star.show_animation = False

    def define_message_for_all(self) -> None:
        """Communication is not part of this first navigation milestone."""

    def _world_to_grid(self, point: np.ndarray) -> tuple[int, int]:
        origin = np.asarray(self.size_area) / 2.0
        index = np.floor((point + origin) / self.resolution).astype(int)
        return tuple(np.clip(index, 0, (self.grid_width - 1,
                                        self.grid_height - 1)))

    def _grid_to_world(self, index: tuple[int, int]) -> tuple[float, float]:
        origin = np.asarray(self.size_area) / 2.0
        point = np.asarray(index, dtype=float) * self.resolution
        point += self.resolution / 2.0 - origin
        return float(point[0]), float(point[1])

    def _update_map(self, position: np.ndarray, angle: float) -> None:
        values = self.lidar_values()
        if values is None:
            return

        ray_angles = self.lidar().ray_angles
        max_range = self.lidar().max_range
        hit_threshold = max_range - self.lidar_hit_margin
        start = self._world_to_grid(position)

        for distance, ray_angle in zip(values, ray_angles):
            if not np.isfinite(distance):
                continue

            distance = float(distance)
            clipped_distance = min(distance, float(max_range))
            direction = angle + float(ray_angle)
            endpoint = position + clipped_distance * np.array(
                (math.cos(direction), math.sin(direction)))
            end = self._world_to_grid(endpoint)

            ray = bresenham(start, end)
            for grid_point in ray[:-1]:
                ix, iy = grid_point
                if 0 <= ix < self.grid_width and 0 <= iy < self.grid_height:
                    if self.grid[ix, iy] < 0.8:
                        self.grid[ix, iy] = 0.0

            if self.robot_radius * 0.75 < distance < hit_threshold:
                ix, iy = end
                self.grid[ix, iy] = 1.0

        self.grid[start] = 0.0

    def _path_is_known_free(self, path_x: list[float],
                            path_y: list[float]) -> bool:
        for point_x, point_y in zip(path_x, path_y):
            ix, iy = self._world_to_grid(np.array((point_x, point_y)))
            if self.grid[ix, iy] > 0.2:
                return False
        return True

    def _frontier_goals(self, position: np.ndarray) -> list[np.ndarray]:
        candidates = []
        for ix, iy in np.argwhere(self.grid <= 0.1):
            if ix in (0, self.grid_width - 1) or iy in (0, self.grid_height - 1):
                continue

            neighbours = self.grid[ix - 1:ix + 2, iy - 1:iy + 2]
            if not np.any((neighbours > 0.2) & (neighbours < 0.8)):
                continue

            point = np.array(self._grid_to_world((ix, iy)))
            distance_from_drone = np.linalg.norm(point - position)
            if distance_from_drone < self.resolution * 3.0:
                continue

            score = (np.linalg.norm(point - self.goal) +
                     0.25 * distance_from_drone)
            candidates.append((score, point))

        candidates.sort(key=lambda candidate: candidate[0])
        return [point for _, point in candidates[:self.max_frontier_targets]]

    def _plan(self, position: np.ndarray) -> None:
        self.next_plan_iteration = self.iteration + self.plan_retry_period
        obstacle_indices = np.argwhere(self.grid >= 0.8)
        if len(obstacle_indices) > self.max_plan_obstacles:
            selected = np.linspace(0, len(obstacle_indices) - 1,
                                   self.max_plan_obstacles, dtype=int)
            obstacle_indices = obstacle_indices[selected]
        obstacle_points = [self._grid_to_world(tuple(index))
                           for index in obstacle_indices]

        border_indices = []
        for ix in range(self.grid_width):
            border_indices.extend(((ix, 0), (ix, self.grid_height - 1)))
        for iy in range(self.grid_height):
            border_indices.extend(((0, iy), (self.grid_width - 1, iy)))
        obstacle_points.extend(self._grid_to_world(index)
                               for index in border_indices)

        obstacle_x, obstacle_y = zip(*obstacle_points)
        planner = AStarPlanner(obstacle_x, obstacle_y,
                               self.resolution, self.robot_radius)
        goal_index = self._world_to_grid(self.goal)
        targets = [] if self.grid[goal_index] > 0.1 else [self.goal]
        targets.extend(self._frontier_goals(position))

        for target in targets:
            planned_x, planned_y = planner.planning(
                float(position[0]), float(position[1]),
                float(target[0]), float(target[1]))
            if len(planned_x) < 2:
                continue

            path_start = np.array((planned_x[-1], planned_y[-1]))
            if np.linalg.norm(path_start - position) > self.resolution * 2.0:
                continue
            if not self._path_is_known_free(planned_x, planned_y):
                continue

            self.active_goal = target.copy()
            self.path_x = list(reversed(planned_x))
            self.path_y = list(reversed(planned_y))
            self.path_index = min(1, max(0, len(self.path_x) - 1))
            return

        self.active_goal = self.goal.copy()
        self.path_x, self.path_y = [], []
        self.path_index = 0

    def _avoid_obstacle(self) -> Optional[CommandsDict]:
        values = self.lidar_values()
        if values is None:
            return None

        ray_angles = self.lidar().ray_angles
        front_mask = np.abs(ray_angles) < math.pi / 6.0
        front_values = np.asarray(values)[front_mask]
        front_values = front_values[np.isfinite(front_values)]
        blocked = (len(front_values) > 0 and
               np.min(front_values) < self.obstacle_stop_distance)
        if not blocked:
            if not self.avoidance_mode:
                return None

            self.avoidance_frames += 1
            if self.avoidance_frames >= 12:
                self.avoidance_mode = False
                self.avoidance_frames = 0
                return None

            return {"forward": 0.0, "lateral": 0.0,
                    "rotation": self.avoidance_rotation, "grasper": 0}

        if not self.avoidance_mode:
            self.avoidance_mode = True
            self.avoidance_frames = 0
            self.path_x, self.path_y = [], []
            self.path_index = 0

            left_mask = ((ray_angles >= math.pi / 6.0) &
                         (ray_angles < math.pi / 2.0))
            right_mask = ((ray_angles <= -math.pi / 6.0) &
                          (ray_angles > -math.pi / 2.0))
            left_values = np.asarray(values)[left_mask]
            right_values = np.asarray(values)[right_mask]
            left_values = left_values[np.isfinite(left_values)]
            right_values = right_values[np.isfinite(right_values)]
            left_clearance = (np.mean(left_values)
                              if len(left_values) else 0.0)
            right_clearance = (np.mean(right_values)
                               if len(right_values) else 0.0)
            self.avoidance_rotation = (
                -1.0 if left_clearance > right_clearance else 1.0)

        self.avoidance_frames += 1
        return {"forward": 0.0, "lateral": 0.0,
                "rotation": self.avoidance_rotation, "grasper": 0}

    @staticmethod
    def _debug_world_to_pixel(agent, point: np.ndarray,
                              panel_size: tuple[int, int]) -> tuple[int, int]:
        width, height = panel_size
        x = int((point[0] + agent.size_area[0] / 2.0) /
                agent.size_area[0] * width)
        y = int((agent.size_area[1] / 2.0 - point[1]) /
                agent.size_area[1] * height)
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    @classmethod
    def _draw_debug_window(cls) -> None:
        if not cls._debug_window_open or not cls._debug_agents:
            return

        panel_width, panel_height = 320, 240
        columns = min(3, max(1, len(cls._debug_agents)))
        rows = int(math.ceil(len(cls._debug_agents) / columns))
        canvas = np.zeros((rows * panel_height, columns * panel_width, 3),
                          dtype=np.uint8)

        for index, agent in enumerate(cls._debug_agents):
            row, column = divmod(index, columns)
            x_offset = column * panel_width
            y_offset = row * panel_height
            panel = np.full((panel_height, panel_width, 3),
                            (55, 55, 55), dtype=np.uint8)
            grid = agent.grid.T
            grid = np.flipud(grid)
            panel_grid = np.full(grid.shape, 120, dtype=np.uint8)
            panel_grid[grid <= 0.1] = 205
            panel_grid[grid >= 0.8] = 25
            panel_grid = cv2.resize(panel_grid, (panel_width, panel_height),
                                    interpolation=cv2.INTER_NEAREST)
            panel[:, :, 0] = panel_grid
            panel[:, :, 1] = panel_grid
            panel[:, :, 2] = panel_grid

            def to_panel(point):
                return agent._debug_world_to_pixel(
                    agent, np.asarray(point, dtype=float),
                    (panel_width, panel_height))

            if len(agent.path_x) > 1:
                path = [to_panel((x, y))
                        for x, y in zip(agent.path_x, agent.path_y)]
                for point_a, point_b in zip(path, path[1:]):
                    cv2.line(panel, point_a, point_b, (0, 220, 220), 2)

            goal = to_panel(agent.goal)
            active_goal = to_panel(agent.active_goal)
            cv2.circle(panel, goal, 6, (255, 255, 0), 2)
            cv2.circle(panel, active_goal, 5, (0, 140, 255), 2)

            measured = agent.measured_gps_position()
            if measured is not None:
                cv2.circle(panel, to_panel(measured), 5, (0, 220, 0), -1)
            cv2.circle(panel, to_panel(agent.true_position()), 3,
                       (0, 0, 220), -1)
            cv2.putText(panel, f"drone {agent.identifier} path={len(agent.path_x)}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            canvas[y_offset:y_offset + panel_height,
                   x_offset:x_offset + panel_width] = panel

        cv2.imshow("Swarm navigation debug", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cls._debug_window_open = False
            cv2.destroyWindow("Swarm navigation debug")

    def _filtered_pose(self) -> tuple[np.ndarray, float]:
        measured_position = self.measured_gps_position()
        if measured_position is None:
            measured_position = (self.filtered_position if
                                 self.filtered_position is not None else
                                 np.zeros(2))
        measured_position = np.asarray(measured_position, dtype=float)

        if self.filtered_position is None:
            self.filtered_position = measured_position.copy()
        else:
            self.filtered_position += self.gps_filter_alpha * (
                measured_position - self.filtered_position)

        measured_angle = self.measured_compass_angle()
        if measured_angle is None:
            measured_angle = (self.filtered_angle
                              if self.filtered_angle is not None else 0.0)
        if self.filtered_angle is None:
            self.filtered_angle = float(measured_angle)
        else:
            angle_error = normalize_angle(
                float(measured_angle) - self.filtered_angle)
            self.filtered_angle = normalize_angle(
                self.filtered_angle + self.compass_filter_alpha * angle_error)

        return self.filtered_position.copy(), self.filtered_angle

    def control(self) -> CommandsDict:
        """Update the map, replan periodically, and follow the next waypoint."""
        position, angle = self._filtered_pose()

        if self.identifier == 0:
            MyDroneNavigation._debug_frame += 1
            if MyDroneNavigation._debug_frame % 5 == 0:
                MyDroneNavigation._draw_debug_window()

        if self.path_x and self.previous_position is not None:
            if np.linalg.norm(position - self.previous_position) < 0.75:
                self.stuck_frames += 1
            else:
                self.stuck_frames = 0
        elif not self.path_x:
            self.stuck_frames = 0
        self.previous_position = position.copy()

        if self.stuck_frames >= 12 and self.recovery_frames == 0:
            self.recovery_frames = 24
            self.recovery_rotation *= -1.0
            self.stuck_frames = 0

        if self.recovery_frames > 0:
            self.recovery_frames -= 1
            return {"forward": 0.0, "lateral": 0.0,
                    "rotation": self.recovery_rotation, "grasper": 0}

        self.iteration += 1
        mapping_angle = self.measured_compass_angle()
        if mapping_angle is None:
            mapping_angle = angle
        self._update_map(position, mapping_angle)

        reached_goal = np.linalg.norm(self.goal - position) < self.resolution
        reached_active_goal = (
            np.linalg.norm(self.active_goal - position) < self.resolution * 1.5
            and self.path_index >= max(0, len(self.path_x) - 4))
        if reached_active_goal and not reached_goal:
            self.path_x, self.path_y = [], []
        needs_plan = ((not self.path_x and not self.avoidance_mode and
                   self.iteration >= self.next_plan_iteration) or
                  (self.iteration % self.replan_period == 0 and
                   not self.avoidance_mode))
        if needs_plan and not reached_goal:
            self._plan(position)

        obstacle_command = self._avoid_obstacle()
        if obstacle_command is not None:
            return obstacle_command

        if reached_goal:
            return {"forward": 0.0, "lateral": 0.0,
                "rotation": 0.0, "grasper": 0}

        if not self.path_x:
            return {"forward": 0.25, "lateral": 0.0,
                    "rotation": 0.0, "grasper": 0}

        remaining_path = np.column_stack((self.path_x[self.path_index:],
                                          self.path_y[self.path_index:]))
        nearest_offset = int(np.argmin(
            np.linalg.norm(remaining_path - position, axis=1)))
        self.path_index += nearest_offset
        while (self.path_index < len(self.path_x) - 1 and
               np.linalg.norm(np.array((self.path_x[self.path_index],
                                        self.path_y[self.path_index])) - position)
               < self.resolution * 2.0):
            self.path_index += 1

        target_index = min(self.path_index + self.lookahead_points,
                           len(self.path_x) - 1)
        target = np.array((self.path_x[target_index],
                           self.path_y[target_index]))

        desired_angle = math.atan2(target[1] - position[1],
                                   target[0] - position[0])
        heading_error = normalize_angle(desired_angle - angle)
        rotation = 0.0 if abs(heading_error) < 0.08 else float(
            np.clip(heading_error * 0.9, -0.8, 0.8))
        forward = float(np.clip(math.cos(heading_error), 0.0, 1.0) * 0.55)
        if abs(heading_error) > math.pi * 0.75:
            forward = 0.0

        return {"forward": forward, "lateral": 0.0,
                "rotation": rotation, "grasper": 0}