import argparse
import asyncio
import aiohttp
import sys
import time
import json
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from enum import Enum
import logging


class Config:
    BASE_URL = "https://challenge.crossmint.io/api"
    REQUEST_DELAY = 0.5  
    MAX_RETRIES = 3
    TIMEOUT = 30


class Colors(Enum):
    BLUE = "blue"
    RED = "red"
    PURPLE = "purple"
    WHITE = "white"


class Directions(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class Logger:
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logger()
    
    def setup_logger(self):
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        if self.verbose:
            self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def success(self, message: str):
        self.logger.info(f"✅ {message}")
    
    def failure(self, message: str):
        self.logger.error(f"❌ {message}")


class Entity(ABC):
   
    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column
        self.endpoint = ""
        self.symbol = ""
    
    @abstractmethod
    def get_payload(self) -> Dict[str, Any]:
        pass
    
    def get_base_payload(self, candidate_id: str) -> Dict[str, Any]:
        return {
            "candidateId": candidate_id,
            "row": self.row,
            "column": self.column
        }
    
    async def create(self, session: aiohttp.ClientSession, candidate_id: str, logger: Logger) -> bool:
        url = f"{Config.BASE_URL}/{self.endpoint}"
        payload = self.get_payload()
        payload.update(self.get_base_payload(candidate_id))
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                async with session.post(url, json=payload, timeout=Config.TIMEOUT) as response:
                    if response.status == 200:
                        logger.debug(f"Created {self} successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Attempt {attempt + 1} failed for {self}: {response.status} - {error_text}")
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self}: {str(e)}")
            
            if attempt < Config.MAX_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))  
        
        logger.failure(f"Failed to create {self} after {Config.MAX_RETRIES} attempts")
        return False
    
    async def delete(self, session: aiohttp.ClientSession, candidate_id: str, logger: Logger) -> bool:
        url = f"{Config.BASE_URL}/{self.endpoint}"
        payload = self.get_base_payload(candidate_id)
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                async with session.delete(url, json=payload, timeout=Config.TIMEOUT) as response:
                    if response.status == 200:
                        logger.debug(f"Deleted {self} successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Delete attempt {attempt + 1} failed for {self}: {response.status} - {error_text}")
                        
            except Exception as e:
                logger.warning(f"Delete attempt {attempt + 1} failed for {self}: {str(e)}")
            
            if attempt < Config.MAX_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))
        
        logger.failure(f"Failed to delete {self} after {Config.MAX_RETRIES} attempts")
        return False
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__} at ({self.row}, {self.column})"


class POLYanet(Entity):
    
    def __init__(self, row: int, column: int):
        super().__init__(row, column)
        self.endpoint = "polyanets"
        self.symbol = "🪐"
    
    def get_payload(self) -> Dict[str, Any]:
        return {}


class SOLoon(Entity):
    
    def __init__(self, row: int, column: int, color: str = "blue"):
        super().__init__(row, column)
        self.endpoint = "soloons"
        self.symbol = "🌙"
        self.color = self._validate_color(color)
    
    def _validate_color(self, color: str) -> str:
        valid_colors = [c.value for c in Colors]
        if color.lower() not in valid_colors:
            return Colors.BLUE.value
        return color.lower()
    
    def get_payload(self) -> Dict[str, Any]:
        return {"color": self.color}
    
    def __str__(self) -> str:
        return f"SOLoon ({self.color}) at ({self.row}, {self.column})"


class ComETH(Entity):
    
    def __init__(self, row: int, column: int, direction: str = "up"):
        super().__init__(row, column)
        self.endpoint = "comeths"
        self.symbol = "☄"
        self.direction = self._validate_direction(direction)
    
    def _validate_direction(self, direction: str) -> str:
        valid_directions = [d.value for d in Directions]
        if direction.lower() not in valid_directions:
            return Directions.UP.value
        return direction.lower()
    
    def get_payload(self) -> Dict[str, Any]:
        return {"direction": self.direction}
    
    def __str__(self) -> str:
        return f"comETH ({self.direction}) at ({self.row}, {self.column})"


class EntityFactory:
    
    @staticmethod
    def create(entity_type: str, row: int, column: int, **kwargs) -> Entity:
        # Create entity based on type
        entity_type = entity_type.lower()
        
        if entity_type in ["polyanet", "🪐"]:
            return POLYanet(row, column)
        elif entity_type in ["soloon", "🌙"]:
            color = kwargs.get("color", "blue")
            return SOLoon(row, column, color)
        elif entity_type in ["cometh", "☄"]:
            direction = kwargs.get("direction", "up")
            return ComETH(row, column, direction)
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")


class GridManager:
    
    def __init__(self, size: int = 30):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.entities = []
    
    def add_entity(self, entity: Entity) -> bool:
        # Add entity to grid
        if self.is_valid_position(entity.row, entity.column):
            self.grid[entity.row][entity.column] = entity
            self.entities.append(entity)
            return True
        return False
    
    def is_valid_position(self, row: int, column: int) -> bool:
        return 0 <= row < self.size and 0 <= column < self.size
    
    def get_entity(self, row: int, column: int) -> Optional[Entity]:
        if self.is_valid_position(row, column):
            return self.grid[row][column]
        return None
    
    def get_adjacent_positions(self, row: int, column: int) -> List[Tuple[int, int]]:
        positions = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, column + dc
            if self.is_valid_position(new_row, new_col):
                positions.append((new_row, new_col))
        
        return positions
    
    def has_adjacent_polyanet(self, row: int, column: int) -> bool:
        adjacent_positions = self.get_adjacent_positions(row, column)
        return any(
            isinstance(self.get_entity(r, c), POLYanet)
            for r, c in adjacent_positions
        )
    
    def display_grid(self, logger: Logger):
        logger.info("Current Megaverse Grid:")
        for row in self.grid:
            row_str = " ".join(entity.symbol if entity else "⬛" for entity in row)
            logger.info(row_str)
    
    def clear(self):
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.entities = []


class PatternGenerator:
    
    @staticmethod
    def create_cross_pattern(grid_manager: GridManager, center_row: int, center_col: int, size: int = 5) -> List[Entity]:
        # Create cross pattern with POLYanets
        entities = []
        
        for i in range(-size, size + 1):
            # Vertical line
            if grid_manager.is_valid_position(center_row + i, center_col):
                entities.append(POLYanet(center_row + i, center_col))
            
            # Horizontal line
            if grid_manager.is_valid_position(center_row, center_col + i):
                entities.append(POLYanet(center_row, center_col + i))
        
        return entities
    
    @staticmethod
    def create_x_pattern(grid_manager: GridManager, center_row: int, center_col: int, size: int = 5) -> List[Entity]:
        # Create X pattern with POLYanets
        entities = []
        
        for i in range(-size, size + 1):
            # Diagonal line (\)
            if grid_manager.is_valid_position(center_row + i, center_col + i):
                entities.append(POLYanet(center_row + i, center_col + i))
            
            # Diagonal line (/)
            if grid_manager.is_valid_position(center_row + i, center_col - i):
                entities.append(POLYanet(center_row + i, center_col - i))
        
        return entities
    
    @staticmethod
    def create_diamond_pattern(grid_manager: GridManager, center_row: int, center_col: int, size: int = 3) -> List[Entity]:
        # Create diamond pattern with POLYanets
        entities = []
        
        for row in range(-size, size + 1):
            for col in range(-size, size + 1):
                if abs(row) + abs(col) == size:
                    new_row, new_col = center_row + row, center_col + col
                    if grid_manager.is_valid_position(new_row, new_col):
                        entities.append(POLYanet(new_row, new_col))
        
        return entities


class ValidationService:
    
    @staticmethod
    def validate_soloon_placement(grid_manager: GridManager, soloon: SOLoon) -> bool:
        return True 
    
    @staticmethod
    def validate_entity_placement(grid_manager: GridManager, entity: Entity) -> bool:
        if isinstance(entity, SOLoon):
            return ValidationService.validate_soloon_placement(grid_manager, entity)
        return True
    
    @staticmethod
    def validate_grid(grid_manager: GridManager) -> List[str]:
        errors = []
        
        for entity in grid_manager.entities:
            if not ValidationService.validate_entity_placement(grid_manager, entity):
                if isinstance(entity, SOLoon):
                    errors.append(f"SOLoon at ({entity.row}, {entity.column}) must be adjacent to a POLYanet")
        
        return errors


class MegaverseBuilder:
    
    def __init__(self, candidate_id: str, logger: Logger):
        self.candidate_id = candidate_id
        self.logger = logger
        self.grid_manager = GridManager()
        self.operation_queue: List[Tuple[str, Entity]] = []
    
    async def fetch_goal_map(self, session: aiohttp.ClientSession) -> Optional[List[List[str]]]:
        url = f"{Config.BASE_URL}/map/{self.candidate_id}/goal"
        
        try:
            async with session.get(url, timeout=Config.TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("goal")
                else:
                    error_text = await response.text()
                    self.logger.failure(f"Failed to fetch goal map: {response.status} - {error_text}")
                    
        except Exception as e:
            self.logger.failure(f"Failed to fetch goal map: {str(e)}")
        
        return None
    
    def parse_goal_map(self, goal_map: List[List[str]]) -> List[Entity]:
        entities: List[Entity] = []
        
        for row_idx, row in enumerate(goal_map):
            for col_idx, cell in enumerate(row):
                if cell == "SPACE":
                    # Empty space, skip
                    continue
                elif cell == "POLYANET":
                    entities.append(EntityFactory.create("polyanet", row_idx, col_idx))
                elif "_SOLOON" in cell:
                    # Extract color from format like "BLUE_SOLOON", "RED_SOLOON"
                    color = cell.split("_")[0].lower()
                    entities.append(EntityFactory.create("soloon", row_idx, col_idx, color=color))
                elif "_COMETH" in cell:
                    # Extract direction from format like "UP_COMETH", "DOWN_COMETH"
                    direction = cell.split("_")[0].lower()
                    entities.append(EntityFactory.create("cometh", row_idx, col_idx, direction=direction))
                else:
                    # Log unknown cell types for debugging
                    self.logger.debug(f"Unknown cell type at ({row_idx}, {col_idx}): {cell}")
        
        return entities
    
    def add_entity(self, entity: Entity) -> bool:
        if ValidationService.validate_entity_placement(self.grid_manager, entity):
            if self.grid_manager.add_entity(entity):
                self.operation_queue.append(("create", entity))
                self.logger.debug(f"Added {entity}")
                return True
            else:
                self.logger.warning(f"Failed to add {entity} - invalid position")
        else:
            self.logger.warning(f"Failed to add {entity} - validation failed")
        return False
    
    async def execute_operations(self, session: aiohttp.ClientSession):
        self.logger.info(f"Executing {len(self.operation_queue)} operations...")
        
        success_count = 0
        failure_count = 0
        
        # Group operations by type for better organization
        polyanets = [op for op in self.operation_queue if isinstance(op[1], POLYanet)]
        soloons = [op for op in self.operation_queue if isinstance(op[1], SOLoon)]
        comeths = [op for op in self.operation_queue if isinstance(op[1], ComETH)]
        
        # Execute POLYanets first (they're foundational)
        self.logger.info(f"Creating {len(polyanets)} POLYanets...")
        for action, entity in polyanets:
            success = await self._execute_single_operation(session, action, entity)
            if success:
                success_count += 1
            else:
                failure_count += 1
            await asyncio.sleep(Config.REQUEST_DELAY)
        
        # Then SOLoons
        self.logger.info(f"Creating {len(soloons)} SOLoons...")
        for action, entity in soloons:
            success = await self._execute_single_operation(session, action, entity)
            if success:
                success_count += 1
            else:
                failure_count += 1
            await asyncio.sleep(Config.REQUEST_DELAY)
        
        # Finally comETHs
        self.logger.info(f"Creating {len(comeths)} comETHs...")
        for action, entity in comeths:
            success = await self._execute_single_operation(session, action, entity)
            if success:
                success_count += 1
            else:
                failure_count += 1
            await asyncio.sleep(Config.REQUEST_DELAY)
        
        self.logger.info(f"Operations completed: {success_count} successful, {failure_count} failed")
        self.operation_queue = []
    
    async def _execute_single_operation(self, session: aiohttp.ClientSession, action: str, entity: Entity) -> bool:
        try:
            if action == "create":
                success = await entity.create(session, self.candidate_id, self.logger)
            elif action == "delete":
                success = await entity.delete(session, self.candidate_id, self.logger)
            else:
                success = False
            
            if success:
                self.logger.success(f"{action.capitalize()}d {entity}")
            else:
                self.logger.failure(f"Failed to {action} {entity}")
            
            return success
                
        except Exception as e:
            self.logger.failure(f"Failed to {action} {entity}: {str(e)}")
            return False
    
    async def build_from_goal(self, session: aiohttp.ClientSession):
        self.logger.info("Fetching goal map...")
        goal_map = await self.fetch_goal_map(session)
        
        if not goal_map:
            self.logger.failure("Could not fetch goal map")
            return
        
        self.logger.info("Parsing goal map...")
        entities = self.parse_goal_map(goal_map)
        self.logger.info(f"Found {len(entities)} entities to create")
        
        # Count entities by type
        polyanet_count = sum(1 for e in entities if isinstance(e, POLYanet))
        soloon_count = sum(1 for e in entities if isinstance(e, SOLoon))
        cometh_count = sum(1 for e in entities if isinstance(e, ComETH))
        
        self.logger.info(f"Entity breakdown: {polyanet_count} POLYanets, {soloon_count} SOLoons, {cometh_count} comETHs")
        
        # Add entities to grid
        added_count = 0
        for entity in entities:
            if self.add_entity(entity):
                added_count += 1
        
        self.logger.info(f"Added {added_count} entities to grid")
        
        # Display grid if verbose
        if self.logger.verbose:
            self.grid_manager.display_grid(self.logger)
        
        # Execute operations
        await self.execute_operations(session)
    
    async def build_custom_pattern(self, session: aiohttp.ClientSession):
        self.logger.info("Building custom X-shaped pattern...")
        
        center_row = self.grid_manager.size // 2
        center_col = self.grid_manager.size // 2
        
        diagonal_size = 10
        polyanets: List[POLYanet] = PatternGenerator.create_x_pattern(
            self.grid_manager, center_row, center_col, size=diagonal_size
        )
        
        for entity in polyanets:
            self.add_entity(entity)
        
        colors = [c.value for c in Colors]
        color_index = 0
        
        for entity in polyanets:
            r, c = entity.row, entity.column
            
            if (r - center_row) == (c - center_col):
                soloon_row, soloon_col = r, c + 1
            
            else:
                soloon_row, soloon_col = r, c - 1
            
            if self.grid_manager.is_valid_position(soloon_row, soloon_col) and not self.grid_manager.get_entity(soloon_row, soloon_col):
                color = colors[color_index % len(colors)]
                soloon = SOLoon(soloon_row, soloon_col, color)
                if self.add_entity(soloon):
                    color_index += 1
        
        tip_positions = [
            (center_row - diagonal_size, center_col - diagonal_size, Directions.UP.value),
            (center_row - diagonal_size, center_col + diagonal_size, Directions.RIGHT.value),
            (center_row + diagonal_size, center_col + diagonal_size, Directions.DOWN.value),
            (center_row + diagonal_size, center_col - diagonal_size, Directions.LEFT.value)
        ]
        
        for (r, c, direction) in tip_positions:
            if self.grid_manager.is_valid_position(r, c) and not self.grid_manager.get_entity(r, c):
                cometh = ComETH(r, c, direction)
                self.add_entity(cometh)
        
        total_entities = len(self.operation_queue)
        count_polys = sum(1 for op in self.operation_queue if isinstance(op[1], POLYanet))
        count_solos = sum(1 for op in self.operation_queue if isinstance(op[1], SOLoon))
        count_coms  = sum(1 for op in self.operation_queue if isinstance(op[1], ComETH))
        
        self.logger.info(f"Custom pattern queued: {count_polys} POLYanets, {count_solos} SOLoons, {count_coms} comETHs  (total {total_entities})")
        
        # If verbose, show the grid in text form
        if self.logger.verbose:
            self.grid_manager.display_grid(self.logger)
        
        await self.execute_operations(session)
    
    
    async def clear_megaverse(self, session: aiohttp.ClientSession):
        self.logger.info("Clearing megaverse...")
        
        url = f"{Config.BASE_URL}/map/{self.candidate_id}"
        try:
            async with session.get(url, timeout=Config.TIMEOUT) as response:
                if response.status == 200:
                    data = await response.json()
                    current_map = data.get("map", [])
                    
                    # Parse current map and create delete operations
                    for row_idx, row in enumerate(current_map):
                        for col_idx, cell in enumerate(row):
                            if cell == "POLYANET":
                                entity = POLYanet(row_idx, col_idx)
                                self.operation_queue.append(("delete", entity))
                            elif "_SOLOON" in cell:
                                color = cell.split("_")[0].lower()
                                entity = SOLoon(row_idx, col_idx, color)
                                self.operation_queue.append(("delete", entity))
                            elif "_COMETH" in cell:
                                direction = cell.split("_")[0].lower()
                                entity = ComETH(row_idx, col_idx, direction)
                                self.operation_queue.append(("delete", entity))
                    
                    self.logger.info(f"Found {len(self.operation_queue)} entities to delete")
                else:
                    self.logger.warning("Could not fetch current map for clearing")
        except Exception as e:
            self.logger.warning(f"Error fetching current map: {str(e)}")
        
        if self.operation_queue:
            await self.execute_operations(session)
        
        self.grid_manager.clear()
        self.logger.success("Megaverse cleared!")


async def main():
    parser = argparse.ArgumentParser(
        description="Megaverse Builder - Create POLYanets, SOLoons, and comETHs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python megaverse_xshape.py --candidate_id=abc123 --verbose
  python megaverse_xshape.py --candidate_id=abc123 --command=goal
  python megaverse_xshape.py --candidate_id=abc123 --command=custom --verbose
  python megaverse_xshape.py --candidate_id=abc123 --command=clear
        """
    )
    
    parser.add_argument(
        "--candidate_id",
        required=True,
        help="Your candidate ID for the Crossmint challenge"
    )
    
    parser.add_argument(
        "--command",
        choices=["goal", "custom", "clear"],
        default="goal",
        help="Command to execute (default: goal)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(verbose=args.verbose)
    
    # Initialize builder
    builder = MegaverseBuilder(args.candidate_id, logger)
    
    logger.info(f"Starting Megaverse Builder")
    logger.info(f"Candidate ID: {args.candidate_id}")
    logger.info(f"Command: {args.command}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("")
    
    async with aiohttp.ClientSession() as session:
        try:
            if args.command == "goal":
                logger.info("=== Building from Goal Map ===")
                await builder.build_from_goal(session)
                
            elif args.command == "custom":
                logger.info("=== Building Custom X-Shape Pattern ===")
                await builder.build_custom_pattern(session)
                
            elif args.command == "clear":
                logger.info("=== Clearing Megaverse ===")
                await builder.clear_megaverse(session)
            
            logger.success("Megaverse building completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
        except Exception as e:
            logger.failure(f"Build process failed: {str(e)}")
            if logger.verbose:
                import traceback
                logger.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
