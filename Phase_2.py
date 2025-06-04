import argparse
import asyncio
import aiohttp
import sys
import json
from typing import List, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
from enum import Enum
import logging


class Config:
    
    BASE_URL = "https://challenge.crossmint.io/api"
    REQUEST_DELAY = 0.5   
    CLEAR_DELAY = 0.02   
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

                    # If get “Too Many Requests,” back off for 1 second and retry
                    if response.status == 429:
                        text = await response.text()
                        logger.warning(f"429 on {self} (attempt {attempt+1}/{Config.MAX_RETRIES}): {text}")
                        await asyncio.sleep(1.0)
                        continue

                    # For any other non-200 status: short exponential backoff
                    text = await response.text()
                    logger.warning(f"Attempt {attempt+1} failed for {self}: {response.status} – {text}")
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} exception for {self}: {e}")

            # Only sleep if going to retry (not on last attempt)
            if attempt < Config.MAX_RETRIES - 1:
                await asyncio.sleep(0.5 * (attempt + 1))

        logger.failure(f"Failed to create {self} after {Config.MAX_RETRIES} attempts")
        return False

    async def delete(self, session: aiohttp.ClientSession, candidate_id: str, logger: Logger) -> bool:

        # Delete this entity on the server (DELETE).
        url = f"{Config.BASE_URL}/{self.endpoint}"
        payload = self.get_base_payload(candidate_id)

        try:
            async with session.delete(url, json=payload, timeout=Config.TIMEOUT) as response:
                # 200 - deleted successfully
                # 404 - nothing to delete (treat as success)
                if response.status in (200, 404):
                    logger.debug(f"Deleted (or not present) {self} with status {response.status}")
                    return True
                else:
                    text = await response.text()
                    logger.warning(f"DELETE returned {response.status} for {self}: {text}")
                    return True
        except Exception as e:
            logger.warning(f"Exception while deleting {self}: {e}. Trying one more time…")
            try:
                async with session.delete(url, json=payload, timeout=Config.TIMEOUT) as response:
                    if response.status in (200, 404):
                        logger.debug(f"Deleted (or not present) {self} on 2nd attempt, status={response.status}")
                        return True
                    else:
                        text = await response.text()
                        logger.warning(f"Second DELETE returned {response.status} for {self}: {text}")
                        return True
            except Exception as e2:
                logger.warning(f"Second attempt exception for {self}: {e2}")
                return True  # give up so we don’t loop forever

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
        valid = [c.value for c in Colors]
        return color.lower() if color.lower() in valid else Colors.BLUE.value

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
        valid = [d.value for d in Directions]
        return direction.lower() if direction.lower() in valid else Directions.UP.value

    def get_payload(self) -> Dict[str, Any]:
        return {"direction": self.direction}

    def __str__(self) -> str:
        return f"comETH ({self.direction}) at ({self.row}, {self.column})"


class EntityFactory:

    @staticmethod
    def create(entity_type: str, row: int, column: int, **kwargs) -> Entity:
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
        self.grid: List[List[Optional[Entity]]] = [[None] * size for _ in range(size)]
        self.entities: List[Entity] = []

    def is_valid_position(self, row: int, column: int) -> bool:
        return 0 <= row < self.size and 0 <= column < self.size

    def get_entity(self, row: int, column: int) -> Optional[Entity]:
        if self.is_valid_position(row, column):
            return self.grid[row][column]
        return None

    def add_entity(self, entity: Entity) -> bool:
        if not self.is_valid_position(entity.row, entity.column):
            return False
        if self.grid[entity.row][entity.column] is not None:
            return False  
        self.grid[entity.row][entity.column] = entity
        self.entities.append(entity)
        return True

    def get_adjacent_positions(self, row: int, column: int) -> List[Tuple[int, int]]:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result: List[Tuple[int, int]] = []
        for dr, dc in offsets:
            r2, c2 = row + dr, column + dc
            if self.is_valid_position(r2, c2):
                result.append((r2, c2))
        return result

    def display_grid(self, logger: Logger):
        logger.info("Current Megaverse Grid:")
        for row in self.grid:
            line = []
            for ent in row:
                if ent is None:
                    line.append("⬛")
                else:
                    line.append(ent.symbol)
            logger.info(" ".join(line))

    def clear(self):
        """Reset our local 30×30 to all‐None."""
        self.grid = [[None] * self.size for _ in range(self.size)]
        self.entities.clear()


class PatternGenerator:

    @staticmethod
    def create_cross_pattern(grid_mgr: GridManager, center_row: int, center_col: int, size: int = 5) -> List[Entity]:
        entities: List[Entity] = []
        for i in range(-size, size + 1):
            # vertical
            if grid_mgr.is_valid_position(center_row + i, center_col):
                entities.append(POLYanet(center_row + i, center_col))
            # horizontal
            if grid_mgr.is_valid_position(center_row, center_col + i):
                entities.append(POLYanet(center_row, center_col + i))
        return entities

    @staticmethod
    def create_x_pattern(grid_mgr: GridManager, center_row: int, center_col: int, size: int = 5) -> List[Entity]:
        entities: List[Entity] = []
        for i in range(-size, size + 1):
            # "\" diagonal
            if grid_mgr.is_valid_position(center_row + i, center_col + i):
                entities.append(POLYanet(center_row + i, center_col + i))
            # "/" diagonal
            if grid_mgr.is_valid_position(center_row + i, center_col - i):
                entities.append(POLYanet(center_row + i, center_col - i))
        return entities

    @staticmethod
    def create_diamond_pattern(grid_mgr: GridManager, center_row: int, center_col: int, size: int = 3) -> List[Entity]:
        entities: List[Entity] = []
        for dr in range(-size, size + 1):
            for dc in range(-size, size + 1):
                if abs(dr) + abs(dc) == size:
                    r2, c2 = center_row + dr, center_col + dc
                    if grid_mgr.is_valid_position(r2, c2):
                        entities.append(POLYanet(r2, c2))
        return entities


class ValidationService:

    @staticmethod
    def validate_soloon_placement(grid_mgr: GridManager, soloon: SOLoon) -> bool:
        return True

    @staticmethod
    def validate_entity_placement(grid_mgr: GridManager, entity: Entity) -> bool:
        if isinstance(entity, SOLoon):
            return ValidationService.validate_soloon_placement(grid_mgr, entity)
        return True

    @staticmethod
    def validate_grid(grid_mgr: GridManager) -> List[str]:
        errors: List[str] = []
        for ent in grid_mgr.entities:
            if isinstance(ent, SOLoon):
                if not ValidationService.has_adjacent_polyanet(grid_mgr, ent.row, ent.column):
                    errors.append(f"SOLoon at ({ent.row},{ent.column}) must be next to a POLYanet")
        return errors

    @staticmethod
    def has_adjacent_polyanet(grid_mgr: GridManager, row: int, column: int) -> bool:
        for (r2, c2) in grid_mgr.get_adjacent_positions(row, column):
            ent = grid_mgr.get_entity(r2, c2)
            if isinstance(ent, POLYanet):
                return True
        return False


class MegaverseBuilder:
    
    def __init__(self, candidate_id: str, logger: Logger):
        self.candidate_id = candidate_id
        self.logger = logger
        self.grid_manager = GridManager()
        self.operation_queue: List[Tuple[str, Entity]] = []

    async def fetch_goal_map(self, session: aiohttp.ClientSession) -> Optional[List[List[str]]]:
        url = f"{Config.BASE_URL}/map/{self.candidate_id}/goal"
        try:
            async with session.get(url, timeout=Config.TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("goal")
                else:
                    txt = await resp.text()
                    self.logger.failure(f"Failed to fetch goal map: {resp.status} – {txt}")
        except Exception as e:
            self.logger.failure(f"Failed to fetch goal map: {e}")
        return None

    def parse_goal_map(self, goal_map: List[List[str]]) -> List[Entity]:
        
        result: List[Entity] = []
        for r_idx, row in enumerate(goal_map):
            for c_idx, cell in enumerate(row):
                if cell == "SPACE":
                    continue
                if cell == "POLYANET":
                    result.append(EntityFactory.create("polyanet", r_idx, c_idx))
                elif cell.endswith("_SOLOON"):
                    color = cell.split("_")[0].lower()
                    result.append(EntityFactory.create("soloon", r_idx, c_idx, color=color))
                elif cell.endswith("_COMETH"):
                    direction = cell.split("_")[0].lower()
                    result.append(EntityFactory.create("cometh", r_idx, c_idx, direction=direction))
                else:
                    self.logger.debug(f"Unknown cell at ({r_idx},{c_idx}): '{cell}'")
        return result

    def add_entity(self, entity: Entity) -> bool:
        if not ValidationService.validate_entity_placement(self.grid_manager, entity):
            self.logger.warning(f"Validation failed for {entity}")
            return False

        if not self.grid_manager.add_entity(entity):
            self.logger.warning(f"Could not place {entity} locally (invalid/occupied).")
            return False

        self.operation_queue.append(("create", entity))
        self.logger.debug(f"Queued create {entity}")
        return True

    async def _execute_single_operation(self, session: aiohttp.ClientSession, action: str, entity: Entity) -> bool:
        try:
            if action == "create":
                ok = await entity.create(session, self.candidate_id, self.logger)
            elif action == "delete":
                ok = await entity.delete(session, self.candidate_id, self.logger)
            else:
                ok = False

            if ok:
                self.logger.success(f"{action.capitalize()}d {entity}")
            else:
                self.logger.failure(f"Failed to {action} {entity}")
            return ok
        except Exception as e:
            self.logger.failure(f"Exception while trying to {action} {entity}: {e}")
            return False

    async def execute_operations(self, session: aiohttp.ClientSession):

        total = len(self.operation_queue)
        self.logger.info(f"Executing {total} operations…")

        # Split by type so that handle POLYanets → SOLoons → comETHs in order:
        polys   = [op for op in self.operation_queue if isinstance(op[1], POLYanet)]
        soloons = [op for op in self.operation_queue if isinstance(op[1], SOLoon)]
        comeths = [op for op in self.operation_queue if isinstance(op[1], ComETH)]

        # POLYanets
        if polys:
            verb = "Deleting" if polys[0][0] == "delete" else "Creating"
            self.logger.info(f"{verb} {len(polys)} POLYanets…")
            for action, ent in polys:
                await self._execute_single_operation(session, action, ent)
                await asyncio.sleep(Config.REQUEST_DELAY)

        # SOLoons
        if soloons:
            verb = "Deleting" if soloons[0][0] == "delete" else "Creating"
            self.logger.info(f"{verb} {len(soloons)} SOLoons…")
            for action, ent in soloons:
                await self._execute_single_operation(session, action, ent)
                await asyncio.sleep(Config.REQUEST_DELAY)

        # comETHs
        if comeths:
            verb = "Deleting" if comeths[0][0] == "delete" else "Creating"
            self.logger.info(f"{verb} {len(comeths)} comETHs…")
            for action, ent in comeths:
                await self._execute_single_operation(session, action, ent)
                await asyncio.sleep(Config.REQUEST_DELAY)

        self.logger.info("Finished executing operations.")
        self.operation_queue.clear()

    async def build_from_goal(self, session: aiohttp.ClientSession):

        self.logger.info("Fetching goal map…")
        goal_map = await self.fetch_goal_map(session)
        if goal_map is None:
            self.logger.failure("Could not fetch goal map; abort.")
            return

        self.logger.info("Parsing goal map…")
        entities = self.parse_goal_map(goal_map)
        self.logger.info(f"Found {len(entities)} entities to create in the goal.")

        # Count types
        poly_count   = sum(1 for e in entities if isinstance(e, POLYanet))
        soloon_count = sum(1 for e in entities if isinstance(e, SOLoon))
        cometh_count = sum(1 for e in entities if isinstance(e, ComETH))
        self.logger.info(f"Entity breakdown: {poly_count} POLYanets, {soloon_count} SOLoons, {cometh_count} comETHs.")

        added = 0
        for e in entities:
            if self.add_entity(e):
                added += 1
        self.logger.info(f"Queued {added} entities for creation.")

        if self.logger.verbose:
            self.grid_manager.display_grid(self.logger)

        await self.execute_operations(session)

    async def build_custom_pattern(self, session: aiohttp.ClientSession):
        
        self.logger.info("Building custom X‐shaped pattern…")

        center = self.grid_manager.size // 2
        polyanets = PatternGenerator.create_x_pattern(self.grid_manager, center, center, size=8)

        for p in polyanets:
            self.add_entity(p)

        colors = [c.value for c in Colors]
        soloon_count = 0
        for poly in polyanets[:20]:
            adj = self.grid_manager.get_adjacent_positions(poly.row, poly.column)
            for (idx, (r2, c2)) in enumerate(adj[:2]):
                if self.grid_manager.get_entity(r2, c2) is None:
                    color = colors[soloon_count % len(colors)]
                    s = SOLoon(r2, c2, color)
                    if self.add_entity(s):
                        soloon_count += 1

        directions = [d.value for d in Directions]
        for i in range(15):
            r2 = center + (i - 7)
            c2 = center + (i - 7) + 3
            if self.grid_manager.is_valid_position(r2, c2) and self.grid_manager.get_entity(r2, c2) is None:
                d = directions[i % len(directions)]
                self.add_entity(ComETH(r2, c2, d))

        total_ops = len(self.operation_queue)
        self.logger.info(f"Queued {total_ops} entities (POLYanets + SOLoons + comETHs) for creation.")

        if self.logger.verbose:
            self.grid_manager.display_grid(self.logger)

        await self.execute_operations(session)

    async def clear_megaverse(self, session: aiohttp.ClientSession):
        
        self.logger.info("Clearing megaverse (brute‐force delete)…")
        size = self.grid_manager.size

        for r in range(size):
            for c in range(size):
                await POLYanet(r, c).delete(session, self.candidate_id, self.logger)
                await asyncio.sleep(Config.CLEAR_DELAY)

                await SOLoon(r, c, "blue").delete(session, self.candidate_id, self.logger)
                await asyncio.sleep(Config.CLEAR_DELAY)

                await ComETH(r, c, "up").delete(session, self.candidate_id, self.logger)
                await asyncio.sleep(Config.CLEAR_DELAY)

        self.grid_manager.clear()
        self.logger.success("Megaverse cleared!")

    async def fetch_goal_map(self, session: aiohttp.ClientSession) -> Optional[List[List[str]]]:
        url = f"{Config.BASE_URL}/map/{self.candidate_id}/goal"
        try:
            async with session.get(url, timeout=Config.TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("goal")
                else:
                    txt = await resp.text()
                    self.logger.failure(f"Failed to fetch goal map: {resp.status} – {txt}")
        except Exception as e:
            self.logger.failure(f"Failed to fetch goal map: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Megaverse Builder – Create POLYanets, SOLoons, and comETHs",
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

    logger = Logger(verbose=args.verbose)
    builder = MegaverseBuilder(args.candidate_id, logger)

    logger.info("Starting Megaverse Builder")
    logger.info(f"Candidate ID: {args.candidate_id}")
    logger.info(f"Command: {args.command}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("")

    async with aiohttp.ClientSession() as session:
        try:
            if args.command == "goal":
                logger.info("=== Clearing any existing megaverse entities ===")
                await builder.clear_megaverse(session)

                logger.info("=== Building from Goal Map ===")
                await builder.build_from_goal(session)

            elif args.command == "custom":
                logger.info("=== Clearing any existing megaverse entities ===")
                await builder.clear_megaverse(session)

                logger.info("=== Building Custom X‐Shape Pattern ===")
                await builder.build_custom_pattern(session)

            elif args.command == "clear":
                logger.info("=== Clearing Megaverse ===")
                await builder.clear_megaverse(session)

            logger.success("Megaverse building completed successfully!")
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
        except Exception as e:
            logger.failure(f"Build process failed: {e}")
            if logger.verbose:
                import traceback
                logger.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())