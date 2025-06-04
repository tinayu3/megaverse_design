import os
import sys
import time
import logging
import argparse
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class AstralPlacementError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class MegaverseClient:
    BASE_URL = "https://challenge.crossmint.io/api"

    def __init__(self, candidate_id: str, timeout: float = 5.0):
        if not candidate_id:
            raise ValueError("MegaverseClient: candidateId cannot be empty.")
        self.candidate_id = candidate_id
        self.timeout = timeout

    def _url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}/{endpoint}"

    def get_goal(self) -> dict:
        
        url = self._url(f"map/{self.candidate_id}/goal")
        logger.info(f"Fetching goal (initializing your phase): GET {url}")
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException as e:
            msg = f"HTTP GET {url} failed: {e}"
            logger.error(msg)
            raise AstralPlacementError(msg)

        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
            except ValueError:
                data = {}
            logger.info(f"[RAW GOAL JSON] {data}")
            return data
        else:
            try:
                body = resp.json()
                detail = body.get("message", body)
            except Exception:
                detail = resp.text or "<no-body>"
            err = f"GET {url} returned {resp.status_code}: {detail}"
            logger.error(err)
            raise AstralPlacementError(err, status_code=resp.status_code)

    def _request(
        self,
        method: str,
        endpoint: str,
        json_payload: Optional[dict] = None,
        max_retries: int = 3
    ) -> dict:
        
        url = self._url(endpoint)
        attempt = 0

        while True:
            attempt += 1
            try:
                resp = requests.request(
                    method=method,
                    url=url,
                    json=json_payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                msg = f"HTTP {method.upper()} {url} failed: {e}"
                logger.error(msg)
                raise AstralPlacementError(msg)

            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    return {}

            # Retry on rate limit (429)
            if resp.status_code == 429 and attempt < max_retries:
                logger.warning(f"Received 429. Retrying in 1s (attempt {attempt}/{max_retries})…")
                time.sleep(1.0)
                continue

            try:
                body = resp.json()
                detail = body.get("message", body)
            except Exception:
                detail = resp.text or "<no-body>"
            err_msg = f"API {method.upper()} {url} returned {resp.status_code}: {detail}"
            logger.error(err_msg)
            raise AstralPlacementError(err_msg, status_code=resp.status_code)

    def create_polyanet(self, row: int, column: int) -> dict:
        
        payload = {
            "candidateId": self.candidate_id,
            "row": row,
            "column": column
        }
        logger.debug(f"[CREATE] Payload: {payload}")
        return self._request(method="post", endpoint="polyanets", json_payload=payload)

    def delete_polyanet(self, row: int, column: int) -> dict:
        
        payload = {
            "candidateId": self.candidate_id,
            "row": row,
            "column": column
        }
        logger.debug(f"[DELETE] Payload: {payload}")
        return self._request(method="delete", endpoint="polyanets", json_payload=payload)

class AstralObject(ABC):
    def __init__(self, client: MegaverseClient, row: int, column: int):
        self.client = client
        self.row = row
        self.column = column
        self._last_response: Optional[dict] = None

    @abstractmethod
    def create(self) -> dict:
        ...

    @abstractmethod
    def delete(self) -> dict:
        ...

    def last_response(self) -> Optional[dict]:
        return self._last_response


class Polyanet(AstralObject):
    def create(self) -> dict:
        logger.info(f"[Polyanet→CREATE] at (row={self.row}, col={self.column})")
        resp = self.client.create_polyanet(self.row, self.column)
        self._last_response = resp
        return resp

    def delete(self) -> dict:
        logger.info(f"[Polyanet→DELETE] at (row={self.row}, col={self.column})")
        resp = self.client.delete_polyanet(self.row, self.column)
        self._last_response = resp
        return resp

class XShapeBuilder:
    
    def __init__(self, client: MegaverseClient, grid_size: int, margin: int, index_offset: int = 0):
        if grid_size < 1:
            raise ValueError("grid_size must be >= 1.")
        if margin < 0 or (margin * 2) >= grid_size:
            raise ValueError("margin must be >= 0 and < grid_size/2.")
        self.client = client

        # N = total number of rows/cols on the server side
        self.N = grid_size
        self.m = margin

        # If the API is 1-based, shift all indices by +1. Otherwise, leave at 0.
        self.INDEX_OFFSET = index_offset

        self.successful: list[Tuple[int, int]] = []
        self.failed: list[Tuple[int, int, str]] = []

    def build(self) -> None:
        
        logger.info("STEP A: Deleting any existing full-size X…")
        total_deletes = 0

        for i0 in range(self.N):
            
            r1 = i0
            c1 = i0
            r2 = i0
            c2 = (self.N - 1) - i0

            server_r1 = r1 + self.INDEX_OFFSET
            server_c1 = c1 + self.INDEX_OFFSET
            server_r2 = r2 + self.INDEX_OFFSET
            server_c2 = c2 + self.INDEX_OFFSET

            total_deletes += 1
            try:
                logger.debug(f"Deleting old Polyanet at ({server_r1}, {server_c1})")
                self.client.delete_polyanet(server_r1, server_c1)
            except AstralPlacementError as e:
                logger.debug(f"   [no POLY at ({server_r1},{server_c1}) — {e}]")

            if (r2, c2) != (r1, c1):
                total_deletes += 1
                try:
                    logger.debug(f"Deleting old Polyanet at ({server_r2}, {server_c2})")
                    self.client.delete_polyanet(server_r2, server_c2)
                except AstralPlacementError as e:
                    logger.debug(f"   [no POLY at ({server_r2},{server_c2}) — {e}]")

        logger.info(f"STEP A done: attempted {total_deletes} DELETE calls on the full-size diagonals.")

        time.sleep(1.0)

        inner_size = self.N - 2 * self.m
        if inner_size <= 0:
            logger.error(f"Inner size = {inner_size} is not positive; aborting.")
            return

        total_creates = 0
        logger.info(
            f"STEP B: Placing an inset X of size {inner_size}×{inner_size} "
            f"inside a {self.N}×{self.N} grid with margin={self.m}…"
        )

        for k in range(inner_size):
            i0 = self.m + k

            r1 = i0
            c1 = i0
            server_r1 = r1 + self.INDEX_OFFSET
            server_c1 = c1 + self.INDEX_OFFSET

            total_creates += 1
            self._attempt_create_with_retries(server_r1, server_c1)

            r2 = i0
            c2 = (self.N - 1) - i0
            if (r2, c2) != (r1, c1):
                server_r2 = r2 + self.INDEX_OFFSET
                server_c2 = c2 + self.INDEX_OFFSET

                total_creates += 1
                self._attempt_create_with_retries(server_r2, server_c2)

            time.sleep(0.8)

        logger.info(f"STEP B done: attempted {total_creates} CREATE calls.")
        logger.info(f"   • Successes: {len(self.successful)}")
        logger.info(f"   • Failures:  {len(self.failed)}")
        if self.failed:
            logger.warning("The following placements failed (row, col, error):")
            for (r, c, err) in self.failed:
                logger.warning(f"   → ({r},{c}): {err}")

    def _attempt_create_with_retries(self, server_row: int, server_col: int) -> None:
        
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                logger.debug(f"Creating POLY at ({server_row}, {server_col}) [Attempt {attempt}/{max_attempts}]")
                self.client.create_polyanet(server_row, server_col)
                self.successful.append((server_row, server_col))
                return
            except AstralPlacementError as e:
                
                if e.status_code == 429:
                    logger.warning(f"Received 429 creating at ({server_row},{server_col}); retrying in 1s… [Attempt {attempt}/{max_attempts}]")
                    time.sleep(1.0)
                    continue
                # Otherwise record as a real failure and stop retrying
                errmsg = str(e)
                logger.warning(f"   [FAILED to create at ({server_row},{server_col}) → {errmsg}]")
                self.failed.append((server_row, server_col, errmsg))
                return

        errmsg = f"Exceeded {max_attempts} attempts due to rate-limiting"
        logger.warning(f"   [FAILED to create at ({server_row},{server_col}) → {errmsg}]")
        self.failed.append((server_row, server_col, errmsg))


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1: Build a smaller inset X of POLYanets inside an N×N grid."
    )
    p.add_argument(
        "--size", "-s",
        type=int,
        default=int(os.getenv("SIZE", "11")),
        help="Grid size N (makes an N×N canvas).",
    )
    p.add_argument(
        "--margin", "-m",
        type=int,
        default=int(os.getenv("MARGIN", "0")),
        help="Margin (rows/cols to leave blank on each edge).",
    )
    p.add_argument(
        "--candidate_id", "-c",
        type=str,
        default=os.getenv("CANDIDATE_ID", ""),
        help="Your candidateId (or set CANDIDATE_ID in the environment).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="If set, print DEBUG logs too.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"** DEBUG: Running with --size={args.size}, --margin={args.margin} **")

    if not args.candidate_id:
        logger.error("Missing --candidate_id (or CANDIDATE_ID env var). Exiting.")
        sys.exit(1)

    try:
        client = MegaverseClient(candidate_id=args.candidate_id)
    except ValueError as e:
        logger.error(f"Cannot initialize MegaverseClient: {e}")
        sys.exit(1)

    try:
        goal_payload = client.get_goal()
    except AstralPlacementError as e:
        logger.error(f"Failed to fetch goal: {e}")
        sys.exit(1)

    INDEX_OFFSET = 0

    if "data" in goal_payload and "gridSize" in goal_payload["data"]:
        real_N = int(goal_payload["data"]["gridSize"])
        logger.info(f"Megaverse DATA says gridSize = {real_N}")
    elif "data" in goal_payload and "width" in goal_payload["data"]:
        real_N = int(goal_payload["data"]["width"])
        logger.info(f"Megaverse DATA says width = {real_N}")
    else:
        real_N = args.size
        logger.warning(
            f"Could not find 'gridSize' or 'width' in get_goal(); falling back to --size={args.size}."
        )

    if "data" in goal_payload and "firstIndex" in goal_payload["data"]:
        fi = int(goal_payload["data"]["firstIndex"])
        logger.info(f"Megaverse DATA says firstIndex = {fi}")
        if fi == 1:
            logger.warning("The server appears to use 1-based indexing! I will add +1 to every row/col.")
            INDEX_OFFSET = 1

    if real_N != 11:
        logger.warning(f"Overriding gridSize={real_N} → 11 so that (2,2) and (2,8) lie inside the board.")
        real_N = 11

    logger.info(f"Using INDEX_OFFSET = {INDEX_OFFSET} to correct for server indexing.")
    logger.info(f"I will treat the grid as {real_N}×{real_N}.")

    try:
        builder = XShapeBuilder(
            client=client,
            grid_size=real_N,
            margin=args.margin,
            index_offset=INDEX_OFFSET
        )
        builder.build()
    except Exception as e:
        logger.exception(f"Error during X-shape build: {e}")
        sys.exit(1)

    logger.info("Finished all steps. Please wait a few seconds, then refresh the Megaverse page.")
    logger.info("If any Saturn icons remain misplaced, check the logs above to see which")
    logger.info("DELETE or CREATE calls failed or if your INDEX_OFFSET was off by one.")
    if builder.failed:
        logger.warning("** NOTE: At least one CREATE failed. See the (row, col, error) list above. **")


if __name__ == "__main__":
    main()