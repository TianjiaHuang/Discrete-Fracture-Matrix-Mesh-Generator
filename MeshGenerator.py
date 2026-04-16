"""Mesh generation utilities for the DFN and TOUGH preprocessing workflow.

The module keeps the original research pipeline intact, but exposes it as a
documented and reusable Python module with a command-line entrypoint.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from typing import Sequence

import meshio
import numpy as np


LOGGER = logging.getLogger(__name__)
DEFAULT_TOLERANCE = 1.0e-4
TOUGH2_REFERENCE_ELEVATION = 2305.0


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounds used to trim external boundary nodes."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    tolerance: float = DEFAULT_TOLERANCE

    @classmethod
    def from_scalar_range(
        cls,
        coord_min: float = 0.0,
        coord_max: float = 0.0,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        z_min: float | None = None,
        z_max: float | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> "BoundingBox":
        return cls(
            x_min=coord_min if x_min is None else x_min,
            x_max=coord_max if x_max is None else x_max,
            y_min=coord_min if y_min is None else y_min,
            y_max=coord_max if y_max is None else y_max,
            z_min=coord_min if z_min is None else z_min,
            z_max=coord_max if z_max is None else z_max,
            tolerance=tolerance,
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        tol = self.tolerance
        return (
            self.x_min - tol <= x <= self.x_max + tol
            and self.y_min - tol <= y <= self.y_max + tol
            and self.z_min - tol <= z <= self.z_max + tol
        )


@dataclass(frozen=True)
class ToughMeshCounts:
    """Element and connection counts needed to assemble TOUGH files."""

    num_ele_mat: int
    num_con_mat: int
    num_ele_dfn: int
    num_con_dfn: int


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end mesh generation pipeline."""

    workdir: Path
    bounds: BoundingBox
    code: str = "TOUGH2"
    aperture: float = 1.0e-4
    dfn_material_id: int = 2


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def run_command(
    command: Sequence[str] | str,
    *,
    cwd: Path,
    shell: bool = False,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a readable error on failure."""

    result = subprocess.run(
        command,
        cwd=str(cwd),
        shell=shell,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}: {command}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def _to_wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    relative_parts = [part.replace("\\", "/") for part in path.parts[1:]]
    return f"/mnt/{drive}/{'/'.join(relative_parts)}"


def _run_wsl_bash(command: str, *, workdir: Path) -> None:
    wsl_workdir = _to_wsl_path(workdir.resolve())
    run_command(["wsl", "bash", "-lc", f"cd {wsl_workdir!r} && {command}"], cwd=workdir)


def _read_token_on_line(path: Path, line_number: int, token_index: int) -> float:
    with path.open("r", encoding="utf-8") as handle:
        for current_line, line in enumerate(handle, start=1):
            if current_line == line_number:
                parts = line.split()
                if token_index >= len(parts):
                    raise ValueError(
                        f"{path} line {line_number} has fewer than {token_index + 1} tokens."
                    )
                return float(parts[token_index])
    raise ValueError(f"{path} has fewer than {line_number} lines.")


def _replace_tokens_on_line(path: Path, line_number: int, updates: Mapping[int, object]) -> None:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    line_index = line_number - 1
    if line_index >= len(lines):
        raise ValueError(f"{path} has fewer than {line_number} lines.")

    parts = lines[line_index].split()
    for token_index, new_value in updates.items():
        if token_index >= len(parts):
            raise ValueError(
                f"{path} line {line_number} has fewer than {token_index + 1} tokens."
            )
        parts[token_index] = str(new_value)
    lines[line_index] = " ".join(parts) + "\n"

    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)


def read_header_counts(path: Path) -> tuple[int, int]:
    """Return the first two numeric values from the first line of a mesh-like file."""

    return int(_read_token_on_line(path, 1, 0)), int(_read_token_on_line(path, 1, 1))


def update_header_counts(path: Path, first_value: int, second_value: int) -> None:
    _replace_tokens_on_line(path, 1, {0: first_value, 1: second_value})


def update_first_token(path: Path, line_number: int, new_value: int) -> None:
    _replace_tokens_on_line(path, line_number, {0: new_value})


def get_tough_mesh_counts(
    workdir: Path,
    *,
    matrix_uge_name: str = "matrix.uge",
    dfn_uge_name: str = "dfn.uge",
) -> ToughMeshCounts:
    """Infer element and connection counts directly from UGE headers."""

    matrix_path = workdir / matrix_uge_name
    dfn_path = workdir / dfn_uge_name
    num_ele_mat = int(_read_token_on_line(matrix_path, 1, 1))
    num_con_mat = int(_read_token_on_line(matrix_path, num_ele_mat + 2, 1))
    num_ele_dfn = int(_read_token_on_line(dfn_path, 1, 1))
    num_con_dfn = int(_read_token_on_line(dfn_path, num_ele_dfn + 2, 1))
    return ToughMeshCounts(num_ele_mat, num_con_mat, num_ele_dfn, num_con_dfn)

def run_dfns_mesh_generator(workdir: Path):
    """Run ``DFNsMeshGenerator3D.bat`` in ``workdir``."""

    bat_file = workdir / "DFNsMeshGenerator3D.bat"
    if not bat_file.exists():
        raise FileNotFoundError(f"DFN batch file not found: {bat_file}")

    LOGGER.info("Running %s", bat_file.name)
    run_command([str(bat_file)], cwd=workdir, shell=True)

def generate_dfn_inp_from_Inputresult(
    workdir: Path,
    input_name: str = "InputResult.txt",
    output_name: str = "dfn_withBoundary.inp",
    num_node: int = 0,
    num_cell: int = 0,
    coord_min: float = 0,
    coord_max: float = 0,
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,
):
    """Generate a trimmed DFN ``.inp`` file from ``InputResult.txt``."""
    input_path = workdir / input_name
    output_path = workdir / output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Generating %s from %s", output_path.name, input_path.name)
    bounds = BoundingBox.from_scalar_range(
        coord_min,
        coord_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        fin.readline()
        fout.write(f"{num_node} {num_cell} 0 0 0\n")

        index_node = 0
        index_new = [0] * (num_node + 1)

        for i in range(1, num_node + 1):
            line = fin.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading node {i} from {input_path}.")

            parts = line.split()
            if len(parts) < 4:
                LOGGER.warning("Skipping malformed node line %d: %s", i, line.rstrip())
                continue

            nums = [float(p) for p in parts]
            x, y, z = nums[1], nums[2], nums[3]
            if not bounds.contains(x, y, z):
                continue

            index_node += 1
            index_new[i] = index_node
            fout.write(f"{index_node} {x:.6f} {y:.6f} {z:.6f}\n")

        fin.readline()

        index_cell = 0
        for i in range(1, num_cell + 1):
            line = fin.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading cell {i} from {input_path}.")

            parts = line.split()
            if len(parts) < 5:
                LOGGER.warning("Skipping malformed cell line %d: %s", i, line.rstrip())
                continue

            nums = [float(p) for p in parts]
            n1 = int(nums[1])
            n2 = int(nums[2])
            n3 = int(nums[3])
            material = int(nums[4])

            if index_new[n1] == 0 or index_new[n2] == 0 or index_new[n3] == 0:
                continue

            index_cell += 1
            fout.write(
                f"{index_cell} {material} tri "
                f"{index_new[n1]} {index_new[n2]} {index_new[n3]}\n"
            )

    LOGGER.info("DFN mesh filtered to %d nodes and %d cells.", index_node, index_cell)

    return index_node, index_cell

def clean_poly_boundary(
    workdir: Path,
    input_name: str = "InputResult.poly",
    output_name: str = "Input.poly",
    num_node: int = 0, 
    num_cell: int = 0,
    coord_min: float = 0,
    coord_max: float = 0,
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,
):
    """Trim nodes and facets from ``InputResult.poly`` and write ``Input.poly``."""
    input_path = workdir / input_name
    output_path = workdir / output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input .poly file not found: {input_path}")

    LOGGER.info("Cleaning %s into %s", input_path.name, output_path.name)
    bounds = BoundingBox.from_scalar_range(
        coord_min,
        coord_max,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
    )

    index_new = [0] * (num_node + 1)
    index_node = 0
    index_cell = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        fin.readline()
        fout.write(f"{num_node}  3  0\n")

        for i in range(1, num_node + 1):
            line = fin.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading node {i} from {input_path}.")

            parts = line.split()
            if len(parts) < 4:
                LOGGER.warning("Skipping malformed node line %d: %s", i, line.rstrip())
                continue

            nums = [float(p) for p in parts]
            x, y, z = nums[1], nums[2], nums[3]
            if not bounds.contains(x, y, z):
                continue

            index_node += 1
            index_new[i] = index_node
            fout.write(f"{index_node}  {x:.6f}  {y:.6f}  {z:.6f}\n")

        fin.readline()
        fout.write(f"{num_cell}  1\n")

        for i in range(1, num_cell + 1):
            line1 = fin.readline()
            if not line1:
                break

            parts1 = line1.split()
            if not parts1:
                continue

            nums1 = [int(float(p)) for p in parts1]
            if len(nums1) < 3:
                LOGGER.warning("Skipping malformed facet header %d: %s", i, line1.rstrip())
                continue

            if nums1[2] > 6:
                line2 = fin.readline()
                if not line2:
                    break

                parts2 = line2.split()
                if len(parts2) < 2:
                    LOGGER.warning("Skipping malformed facet node line %d: %s", i, line2.rstrip())
                    continue

                nums2 = [int(float(p)) for p in parts2]
                cell_id = nums2[0]
                orig_node_ids = nums2[1:]
                if any(index_new[n] == 0 for n in orig_node_ids):
                    continue

                index_cell += 1
                a, b, c = nums1[0], nums1[1], nums1[2]
                fout.write(f"{a}  {b}  {c}\n")

                remapped_nodes = [index_new[n] for n in orig_node_ids]
                n1, n2, n3 = remapped_nodes[0], remapped_nodes[1], remapped_nodes[2]
                fout.write(f"{cell_id}  {n1}  {n2}  {n3}\n")
            else:
                break

    node_after = max(index_new)
    LOGGER.info(".poly mesh filtered to %d nodes and %d facets.", node_after, index_cell)

    return node_after, index_cell


def run_tetgen_in_wsl(project_dir):
    """Run TetGen under WSL in the project directory."""

    workdir = Path(project_dir)
    LOGGER.info("Running TetGen under WSL")
    _run_wsl_bash("./tetgen -pq1.2a20 -Y -V -k Input.poly > tetgen.log 2>&1", workdir=workdir)

def convert_vtk_to_matrix_inp(
    workdir: Path,
    vtk_name: str = "Input.1.vtk",
    output_name: str = "matrix.inp",
):
    """Read TetGen VTK output and write ``matrix.inp`` using ``meshio``."""
    vtk_path = workdir / vtk_name
    output_path = workdir / output_name

    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK file not found: {vtk_path}")

    LOGGER.info("Converting %s to %s", vtk_path.name, output_path.name)
    mesh = meshio.read(str(vtk_path))
    mesh.write(str(output_path))

    return output_path


def convert_matrix_inp_to_new_format(
    workdir: Path,
    input_name: str = "matrix.inp",
    output_name: str = "matrix_new.inp",
    num_node: int = 0,    # modify case-by-case
    num_cell: int = 0,   # modify case-by-case
):
    """Rewrite ``matrix.inp`` into the format expected by LaGriT."""
    input_path = workdir / input_name
    output_path = workdir / output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input matrix.inp not found: {input_path}")

    LOGGER.info("Reformatting %s into %s", input_path.name, output_path.name)
    
    index_node = 0
    index_cell = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for _ in range(4):
            if not fin.readline():
                raise ValueError("Unexpected EOF while skipping header lines")

        fout.write(f"{num_node} {num_cell} 0 0 0\n")

        for line in fin:
            stripped = line.strip()
            if stripped.upper().startswith("*ELEMENT"):
                break
            
            parts = [token.replace(",", "") for token in stripped.split()]
            if len(parts) < 4:
                continue
            
            node_id = int(float(parts[0]))
            x, y, z = map(float, parts[1:4])
            fout.write(f"{node_id} {x:.6f} {y:.6f} {z:.6f}\n")
            index_node += 1

        for line in fin:
            stripped = line.strip()
            if stripped == "":
                break
            
            parts = [token.strip() for token in line.split(',')]
            if len(parts) < 5:
                LOGGER.warning("Skipping malformed element line: %s", line.rstrip())
                continue
            
            cell_id = int(float(parts[0]))
            n1 = int(float(parts[1]))
            n2 = int(float(parts[2]))
            n3 = int(float(parts[3]))
            n4 = int(float(parts[4]))
            fout.write(f"{cell_id} 1 tet {n1} {n2} {n4} {n3}\n")
            index_cell += 1

    LOGGER.info("Matrix mesh filtered to %d nodes and %d cells.", index_node, index_cell)
    
    return index_node, index_cell

def run_lagrit_conversions(project_dir):
    """Run the LaGriT conversion scripts under WSL."""

    workdir = Path(project_dir)
    LOGGER.info("Running LaGriT conversions under WSL")
    _run_wsl_bash("./lagrit < convert_dfn.lgi > lagrit_dfn.log 2>&1", workdir=workdir)
    _run_wsl_bash("./lagrit < convert_matrix.lgi > lagrit_matrix.log 2>&1", workdir=workdir)

def index2mark(index: int):
    """
    Convert index -> (mark, ele_index) where:
      - 99 elements per block
      - mark cycles like AAA, AAB, ..., ZZZ
      - ele_index is 1..99 within each block
    """

    if index <= 0:
        raise ValueError(f"index2mark expects positive index, got {index}")

    idx0 = index - 1
    block = idx0 // 99
    ele_index = idx0 % 99 + 1

    # Convert block (0,1,2,...) into a 3-letter code AAA..ZZZ
    if block >= 26**3:
        raise ValueError(f"index {index} too large for 3-letter alphabetic coding.")

    # Compute letters
    c1 = block // (26*26)
    c2 = (block // 26) % 26
    c3 = block % 26

    mark = chr(ord("A") + c1) + chr(ord("A") + c2) + chr(ord("A") + c3)
    return mark, ele_index


def generate_tough_mesh_and_incon(
    workdir: Path,
    code: str = "TOUGH2",
    aperture: float = 1.0e-4,
    num_ele_mat: int = 0,
    num_con_mat: int = 0,
    num_ele_dfn: int = 0,
    num_con_dfn: int = 0,
    matrix_uge_name: str = "matrix.uge",
    dfn_uge_name: str = "dfn.uge",
    mesh_name: str = "MESH",
    mesh_inactive_name: str = "MESH_inactive",
    incon_name: str = "INCON",
    x_min=None, x_max=None,
    y_min=None, y_max=None,
    z_min=None, z_max=None,
):
    """Build TOUGH ``MESH`` and ``INCON`` files from matrix and DFN UGE files."""

    tol = 1.0e-5
    water_density = 880.0
    workdir = Path(workdir).resolve()
    matrix_path = workdir / matrix_uge_name
    dfn_path = workdir / dfn_uge_name

    if not matrix_path.exists():
        raise FileNotFoundError(f"matrix.uge not found: {matrix_path}")
    if not dfn_path.exists():
        raise FileNotFoundError(f"dfn.uge not found: {dfn_path}")

    mesh_path = workdir / mesh_name
    mesh_inactive_path = workdir / mesh_inactive_name
    incon_path = workdir / incon_name

    LOGGER.info("Generating TOUGH MESH and INCON in %s", workdir)

    # Coordinates of all elements: matrix first, then dfn
    # Use 1-based style indexing (index 0 unused)
    ele_coor = np.zeros((num_ele_mat + num_ele_dfn + 1, 3), dtype=float)

    # Areas for DFN elements (for last CONNE loop)
    area_dfn = np.zeros(num_ele_dfn + 1, dtype=float)  # 1-based

    unknow = 0.0
    space = " "
    type_default = "Matri"

    with matrix_path.open("r", encoding="utf-8") as fid_matrix, dfn_path.open(
        "r", encoding="utf-8"
    ) as fid_dfn, mesh_path.open("w", encoding="utf-8") as fid, mesh_inactive_path.open(
        "w", encoding="utf-8"
    ) as fid_inactive, incon_path.open("w", encoding="utf-8") as fid_incon:

        # INCON header
        fid_incon.write(f"{'INCON':>5s}")

        # --- ELEME section ---
        fid.write(f"{'ELEME':>5s}\n")

        # Skip first line of matrix/dfn files (header)
        _ = fid_matrix.readline()
        _ = fid_dfn.readline()

        # ---- MATRIX ELEMENTS ----
        vol_old = 0.0
        for i in range(1, num_ele_mat + 1):
            line = fid_matrix.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in matrix.uge while reading element {i}")
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed line in matrix.uge at element {i}: {line!r}")

            nums = [float(p) for p in parts]

            index = int(nums[0])
            x, y, z = nums[1], nums[2], nums[3]

            if nums[4] > 0:
                vol = nums[4]
                vol_old = vol
            else:
                vol = vol_old

            ele_coor[i, :] = [x, y, z]
            activity = " "

            mark, ele_index = index2mark(i)
            # Boundary conditions
            if abs(z - z_max) < tol:
                # activity = "I"
                # type_elem = "BOUND"
                vol = 2.0e50
                # type_elem = "EDFMR"
                activity = " "
            if abs(z - z_min) < tol:
                # activity = "I"
                # type_elem = "EDFML"
                vol = 2.0e50
                activity = " "
            if code == "TOUGH2":
                # two lines per block, as in MATLAB
                line1 = "\n%3s%2d%11s%14.8E%15.8E%15.8E%15.8E" % (
                    mark,
                    ele_index,
                    space,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                pressure = water_density * 9.81 * (TOUGH2_REFERENCE_ELEVATION - z)
                line2 = "\n%20.13E%20.13E%20.13E%20.13E" % (pressure, 0.0, 1.0e-8, 200.0)
                fid_incon.write(line1 + line2)
            else:
                line1 = "\n%3s%2d%11s%14.8E%5s%36s%15.8E%15.8E%15.8E%15.8E" % (
                    mark,
                    ele_index,
                    space,
                    0.0,
                    "Aqu",
                    space,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                line2 = "\n%20.13E%20.13E%20.13E" % (2.0e7, 1.0e-5, 50.0)
                fid_incon.write(line1 + line2)

            # ELEME line for matrix element
            # Format: name media volume unknown x y z activity
            # name = 1-char mark + 4-digit index
            line_eleme = "%3s%2d%15s%10.4E%+9.3E%10s%+9.3E%+9.3E%+9.3E%2s\n" % (
                mark,
                ele_index,
                type_default,
                vol,
                unknow,
                space,
                x,
                y,
                z,
                activity,
            )
            fid.write(line_eleme)

        # ---- DFN ELEMENTS ----
        vol_old = 0.0
        type_dfn = "Fract"  # default for DFN
        for i in range(1, num_ele_dfn + 1):
            line = fid_dfn.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in dfn.uge while reading element {i}")
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed line in dfn.uge at element {i}: {line!r}")

            nums = [float(p) for p in parts]

            if nums[4] > 0:
                vol = nums[4] * aperture
                vol_old = vol
            else:
                vol = vol_old

            # Store DFN area (before multiplying by aperture)
            area_dfn[i] = vol / aperture

            x, y, z = nums[1], nums[2], nums[3]
            ii = i + num_ele_mat
            activity = " "
            type_elem = type_dfn

            mark, ele_index = index2mark(ii)

            # Boundary conditions
            if abs(z - z_max) < tol:
                # activity = "I"
                # type_elem = "BOUND"
                vol = 2.0e50
                # type_elem = "EDFMR"
                activity = " "
            if abs(z - z_min) < tol:
                # activity = "I"
                # type_elem = "EDFML"
                vol = 2.0e50
                activity = " "
            if code == "TOUGH2":
                # two lines per block, as in MATLAB
                line1 = "\n%3s%2d%11s%14.8E%15.8E%15.8E%15.8E" % (
                    mark,
                    ele_index,
                    space,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                pressure = water_density * 9.81 * (TOUGH2_REFERENCE_ELEVATION - z)
                line2 = "\n%20.13E%20.13E%20.13E%20.13E" % (pressure, 0.0, 1.0e-8, 200.0)
                fid_incon.write(line1 + line2)
            else:
                line1 = "\n%3s%2d%11s%14.8E%5s%36s%15.8E%15.8E%15.8E%15.8E" % (
                    mark,
                    ele_index,
                    space,
                    0.0,
                    "Aqu",
                    space,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                line2 = "\n%20.13E%20.13E%20.13E" % (2.0e7, 1.0e-5, 50.0)
                fid_incon.write(line1 + line2)

            ele_coor[ii, :] = [x, y, z]

            line_eleme = "%3s%2d%15s%10.4E%+9.3E%10s%+9.3E%+9.3E%+9.3E%2s\n" % (
                mark,
                ele_index,
                type_elem,
                vol,
                unknow,
                space,
                x,
                y,
                z,
                activity,
            )

            if activity == "I":
                fid_inactive.write(line_eleme)
            else:
                fid.write(line_eleme)

        # ---- CONNE section ----
        # read one line from each (as in MATLAB)
        _ = fid_matrix.readline()
        _ = fid_dfn.readline()

        fid.write("\n%5s\n" % "CONNE")
        ConxKi = 1
        beta = 0.0

        # --- MATRIX CONNECTIONS ---
        area_old = 0.0
        for i in range(1, num_con_mat + 1):
            line = fid_matrix.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed matrix CONNE line {i}: {line!r}")

            nums = [float(p) for p in parts]

            conxname1 = int(nums[0])
            conxname2 = int(nums[1])
            face_centroids = np.array(nums[2:5])

            if nums[5] > 0:
                area = nums[5]
                area_old = area
            else:
                area = area_old

            point1 = ele_coor[conxname1, :]
            point2 = ele_coor[conxname2, :]

            conxD1 = np.linalg.norm(point1 - face_centroids)
            conxD2 = np.linalg.norm(point2 - face_centroids)

            mark1, idx1 = index2mark(conxname1)
            mark2, idx2 = index2mark(conxname2)

            point2to1 = point2 - point1
            beta = float(np.dot(point2to1, np.array([0.0, 0.0, -1.0])) / np.linalg.norm(point2to1))
            if abs(beta) < 1.0e-5:
                beta = 0.0

            line_conne = "%3s%2d%3s%2d%20d%10.4e%10.4e%10.4e%+9.3e\n" % (
                mark1,
                idx1,
                mark2,
                idx2,
                ConxKi,
                conxD1,
                conxD2,
                area,
                beta,
            )
            fid.write(line_conne)
            ConxKi += 1

        # --- DFN CONNECTIONS (within DFN) ---
        area_old = 0.0
        for i in range(1, num_con_dfn + 1):
            line = fid_dfn.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Malformed DFN CONNE line {i}: {line!r}")

            nums = [float(p) for p in parts]

            conxname1 = int(nums[0]) + num_ele_mat
            conxname2 = int(nums[1]) + num_ele_mat
            face_centroids = np.array(nums[2:5])

            if nums[5] > 0:
                area = nums[5] * aperture
                area_old = area
            else:
                area = area_old

            point1 = ele_coor[conxname1, :]
            point2 = ele_coor[conxname2, :]

            if np.any(point1 == 0.0) or np.any(point2 == 0.0):
                continue

            mark1, idx1 = index2mark(conxname1)
            mark2, idx2 = index2mark(conxname2)

            conxD1 = np.linalg.norm(point1 - face_centroids)
            conxD2 = np.linalg.norm(point2 - face_centroids)

            point2to1 = point2 - point1
            beta = float(np.dot(point2to1, np.array([0.0, 0.0, -1.0])) / np.linalg.norm(point2to1))
            if abs(beta) < 1.0e-5:
                beta = 0.0

            line_conne = "%3s%2d%3s%2d%20d%10.4e%10.4e%10.4e%+9.3e\n" % (
                mark1,
                idx1,
                mark2,
                idx2,
                ConxKi,
                conxD1,
                conxD2,
                area,
                beta,
            )
            fid.write(line_conne)
            ConxKi += 1

        # --- DFN–MATRIX COUPLING CONNECTIONS ---
        for i in range(1, num_ele_dfn + 1):
            conxname1 = i
            conxname2 = i + num_ele_mat

            area = area_dfn[i]

            mark1, idx1 = index2mark(conxname1)
            mark2, idx2 = index2mark(conxname2)

            conxD1 = aperture / 2.0
            conxD2 = aperture / 2.0

            beta = 0.0

            line_conne = "%3s%2d%3s%2d%20d%10.4e%10.4e%10.4e%+9.3e\n" % (
                mark1,
                idx1,
                mark2,
                idx2,
                ConxKi,
                conxD1,
                conxD2,
                area,
                beta,
            )
            fid.write(line_conne)
            ConxKi += 1

        # Final blank lines as in MATLAB
        fid.write("\n")
        fid_incon.write("\n      \n")
        
        LOGGER.info(
            "Finished writing %s, %s, %s",
            mesh_path.name,
            mesh_inactive_path.name,
            incon_path.name,
        )

def get_first_number_on_line(filename, line_number):
    """Backward-compatible wrapper around the line token helper."""

    return _read_token_on_line(Path(filename), line_number, 0)

def get_second_number_on_line(filename, line_number):
    """Backward-compatible wrapper around the line token helper."""

    return _read_token_on_line(Path(filename), line_number, 1)

def replace_first_two_numbers(filename, new1, new2):
    """Backward-compatible wrapper around the line token update helper."""

    update_header_counts(Path(filename), new1, new2)

def replace_first_number_on_lines(filename, line_updates):
    """Backward-compatible wrapper for updating the first token on many lines."""

    path = Path(filename)
    for line_number, new_value in line_updates.items():
        update_first_token(path, line_number, new_value)

def filter_inp_by_material(
    workdir: Path,
    input_name: str = "dfn_withBoundary.inp",
    output_name: str = "dfn.inp",
    material_id: int = 0,
):
    """Keep only cells with ``material <= material_id`` and drop unused nodes."""
    in_path = workdir / input_name
    out_path = workdir / output_name

    if not in_path.exists():
        raise FileNotFoundError(f"Input inp file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as fin:
        header = fin.readline().split()
        if len(header) < 2:
            raise ValueError(f"Malformed header in {in_path}")
        orig_node_count = int(float(header[0]))
        orig_cell_count = int(float(header[1]))

        nodes = {}
        for _ in range(orig_node_count):
            parts = fin.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Malformed node line in {in_path}")
            node_id = int(float(parts[0]))
            xyz = tuple(float(v) for v in parts[1:4])
            nodes[node_id] = xyz

        cells_kept = []
        used_nodes = set()
        for _ in range(orig_cell_count):
            parts = fin.readline().split()
            if len(parts) < 6:
                continue
            material = int(float(parts[1]))
            if material > material_id:
                continue
            n1, n2, n3 = map(lambda v: int(float(v)), parts[3:6])
            cells_kept.append((material, n1, n2, n3))
            used_nodes.update([n1, n2, n3])

    old_to_new = {}
    new_nodes = []
    for new_id, old_id in enumerate(sorted(used_nodes), start=1):
        old_to_new[old_id] = new_id
        x, y, z = nodes[old_id]
        new_nodes.append((new_id, x, y, z))

    new_cells = []
    for idx, (material, n1, n2, n3) in enumerate(cells_kept, start=1):
        new_cells.append((idx, material, old_to_new[n1], old_to_new[n2], old_to_new[n3]))

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write(f"{len(new_nodes)} {len(new_cells)} 0 0 0\n")
        for nid, x, y, z in new_nodes:
            fout.write(f"{nid} {x:.6f} {y:.6f} {z:.6f}\n")
        for cid, material, n1, n2, n3 in new_cells:
            fout.write(f"{cid} {material} tri {n1} {n2} {n3}\n")

    LOGGER.info(
        "Filtered %s -> %s (nodes: %d -> %d, cells: %d -> %d) for material=%d",
        in_path.name,
        out_path.name,
        orig_node_count,
        len(new_nodes),
        orig_cell_count,
        len(new_cells),
        material_id,
    )

def run_full_pipeline(config: PipelineConfig) -> None:
    """Execute the complete mesh-generation workflow in order."""

    workdir = config.workdir
    input_result_path = workdir / "InputResult.txt"
    input_result_poly_path = workdir / "InputResult.poly"

    run_dfns_mesh_generator(workdir)

    dfn_num_node, _ = read_header_counts(input_result_path)
    dfn_num_cell = int(_read_token_on_line(input_result_path, dfn_num_node + 2, 0))
    node_dfn, cell_dfn = generate_dfn_inp_from_Inputresult(
        workdir,
        num_node=dfn_num_node,
        num_cell=dfn_num_cell,
        x_min=config.bounds.x_min,
        x_max=config.bounds.x_max,
        y_min=config.bounds.y_min,
        y_max=config.bounds.y_max,
        z_min=config.bounds.z_min,
        z_max=config.bounds.z_max,
    )
    update_header_counts(workdir / "dfn_withBoundary.inp", node_dfn, cell_dfn)
    filter_inp_by_material(workdir, material_id=config.dfn_material_id)

    poly_num_node, _ = read_header_counts(input_result_poly_path)
    poly_num_cell = int(_read_token_on_line(input_result_poly_path, poly_num_node + 2, 0))
    new_node, new_cell = clean_poly_boundary(
        workdir,
        num_node=poly_num_node,
        num_cell=poly_num_cell,
        x_min=config.bounds.x_min,
        x_max=config.bounds.x_max,
        y_min=config.bounds.y_min,
        y_max=config.bounds.y_max,
        z_min=config.bounds.z_min,
        z_max=config.bounds.z_max,
    )
    replace_first_number_on_lines(workdir / "Input.poly", {1: new_node, new_node + 2: new_cell})
    run_tetgen_in_wsl(workdir)
    convert_vtk_to_matrix_inp(workdir)
    node_matrix, cell_matrix = convert_matrix_inp_to_new_format(workdir)
    update_header_counts(workdir / "matrix_new.inp", node_matrix, cell_matrix)
    run_lagrit_conversions(workdir)

    counts = get_tough_mesh_counts(workdir)
    generate_tough_mesh_and_incon(
        workdir,
        code=config.code,
        aperture=config.aperture,
        num_ele_mat=counts.num_ele_mat,
        num_con_mat=counts.num_con_mat,
        num_ele_dfn=counts.num_ele_dfn,
        num_con_dfn=counts.num_con_dfn,
        x_min=config.bounds.x_min,
        x_max=config.bounds.x_max,
        y_min=config.bounds.y_min,
        y_max=config.bounds.y_max,
        z_min=config.bounds.z_min,
        z_max=config.bounds.z_max,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for this module."""

    parser = argparse.ArgumentParser(
        description="Generate DFN, matrix, and TOUGH mesh files for the onestep workflow."
    )
    parser.add_argument("--workdir", type=Path, default=Path("."), help="Workflow directory.")
    parser.add_argument("--x-min", type=float, default=5.0)
    parser.add_argument("--x-max", type=float, default=105.0)
    parser.add_argument("--y-min", type=float, default=5.0)
    parser.add_argument("--y-max", type=float, default=105.0)
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--z-max", type=float, default=305.0)
    parser.add_argument("--aperture", type=float, default=1.0e-4)
    parser.add_argument("--code", default="TOUGH2")
    parser.add_argument("--dfn-material-id", type=int, default=2)
    parser.add_argument(
        "--stage",
        choices=("pipeline", "tough"),
        default="pipeline",
        help="Run the full pipeline or only rebuild MESH/INCON from existing UGE files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    config = PipelineConfig(
        workdir=args.workdir.resolve(),
        bounds=BoundingBox(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            z_min=args.z_min,
            z_max=args.z_max,
        ),
        code=args.code,
        aperture=args.aperture,
        dfn_material_id=args.dfn_material_id,
    )

    if args.stage == "pipeline":
        run_full_pipeline(config)
        return

    counts = get_tough_mesh_counts(config.workdir)
    generate_tough_mesh_and_incon(
        config.workdir,
        code=config.code,
        aperture=config.aperture,
        num_ele_mat=counts.num_ele_mat,
        num_con_mat=counts.num_con_mat,
        num_ele_dfn=counts.num_ele_dfn,
        num_con_dfn=counts.num_con_dfn,
        x_min=config.bounds.x_min,
        x_max=config.bounds.x_max,
        y_min=config.bounds.y_min,
        y_max=config.bounds.y_max,
        z_min=config.bounds.z_min,
        z_max=config.bounds.z_max,
    )
    
if __name__ == "__main__":
    main()
