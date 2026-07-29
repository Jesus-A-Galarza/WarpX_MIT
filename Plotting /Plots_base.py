#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


# -----------------------------------------------------------------------------
# Physics and species configuration
# -----------------------------------------------------------------------------
CLIGHT = 299_792_458.0
ELEMENTARY_CHARGE = 1.602_176_634e-19
SI_MOMENTUM_TO_MEV_C = CLIGHT / (1.0e6 * ELEMENTARY_CHARGE)

SPECIES_TO_GROUP = {
    "ele_bw": "BW",
    "pos_bw": "BW",
    "ele_bh": "BH",
    "pos_bh": "BH",
    "ele_ll": "LL",
    "pos_ll": "LL",
    "beam1": "BEAM1",
    "beam2": "BEAM2",
}

SPECIES = tuple(SPECIES_TO_GROUP)

IPC_SPECIES = {
    name
    for name, group in SPECIES_TO_GROUP.items()
    if group in {"BW", "BH", "LL"}
}

IPC_GROUPS = ("IPC_ALL", "BW", "BH", "LL")
PLOT_GROUPS = ("IPC_ALL", "BW", "BH", "LL", "BEAM1", "BEAM2")
STAGES = ("inside", "outside")
SCOPES = ("inside", "outside", "combined")
IPC_CHANNEL_ORDER = ("BW", "BH", "LL")
IPC_ANALYSIS_SCOPES = ("inside", "outside", "combined", "production")
IPC_CHANNEL_LONG_NAME = {
    "BW": "Breit-Wheeler (BW)",
    "BH": "Bethe-Heitler (BH)",
    "LL": "Landau-Lifshitz (LL)",
}

GROUP_COLOR = {
    "IPC_ALL": "black",
    "LL": "red",
    "BH": "blue",
    "BW": "green",
    "BEAM1": "tab:orange",
    "BEAM2": "tab:purple",
}

AXIS_SCALE = {
    "x": 1.0e6,
    "y": 1.0e6,
    "z": 1.0e3,
}

AXIS_UNIT = {
    "x": "µm",
    "y": "µm",
    "z": "mm",
}

AXES = ("x", "y", "z")
PLANES = (("x", "y"), ("x", "z"), ("y", "z"))

# Fixed zoom windows for IPC spatial 2D maps, expressed in the existing
# plotting units (x/y in µm and z in mm):
#   xy: x in [-0.8, 0.8] mm, y in [-0.015, 0.015] mm
#   xz: x in [-0.8, 0.8] mm, z in [-40, 40] mm
#   yz: y in [-0.015, 0.015] mm, z in [-5, 5] mm
IPC_2D_LIMITS = {
    "xy": {
        "x": (-800.0, 800.0),
        "y": (-15.0, 15.0),
    },
    "xz": {
        "x": (-800.0, 800.0),
        "z": (-40.0, 40.0),
    },
    "yz": {
        "y": (-15.0, 15.0),
        "z": (-5.0, 5.0),
    },
}

# Particle-creation records written by the WarpX runtime attributes.
CREATION_FIELDS = (
    "orig_x",
    "orig_y",
    "orig_z",
    "creationTime",
)

CREATION_PLANES = (
    ("orig_x", "orig_y"),
    ("orig_x", "orig_z"),
    ("orig_y", "orig_z"),
    ("creationTime", "orig_x"),
    ("creationTime", "orig_y"),
    ("creationTime", "orig_z"),
)

CREATION_SCALE = {
    "orig_x": 1.0e6,
    "orig_y": 1.0e6,
    "orig_z": 1.0e3,
    "creationTime": 1.0e12,
}

CREATION_UNIT = {
    "orig_x": "µm",
    "orig_y": "µm",
    "orig_z": "mm",
    "creationTime": "ps",
}

CREATION_LABEL = {
    "orig_x": "Creation x",
    "orig_y": "Creation y",
    "orig_z": "Creation z",
    "creationTime": "Creation time",
}

CREATION_TO_AXIS = {
    "orig_x": "x",
    "orig_y": "y",
    "orig_z": "z",
}


# -----------------------------------------------------------------------------
# Simulation box overlay
#
# These L values are the TOTAL box lengths.
#
# Original values:
#   Lx = 2.08355 mm
#   Ly = 26.8711 µm
#   Lz = 267.16994 mm
#
# Plotting units:
#   x = µm
#   y = µm
#   z = mm
#
# The box is assumed to be centered at x = y = z = 0.
# Therefore, each boundary is located at center ± L/2.
# -----------------------------------------------------------------------------
BOX_TOTAL_SIZE = {
    "x": 2072.36295,   # µm
    "y": 13.440805,   # µm
    "z": 133.6,  # mm
}

BOX_CENTER = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

# Box limits in SI metres. BOX_TOTAL_SIZE and BOX_CENTER are expressed in
# plotting units, so division by AXIS_SCALE converts them back to metres.
BOX_LO_M = np.array(
    [
        (BOX_CENTER[axis] - 0.5 * BOX_TOTAL_SIZE[axis])
        / AXIS_SCALE[axis]
        for axis in AXES
    ],
    dtype=float,
)

BOX_HI_M = np.array(
    [
        (BOX_CENTER[axis] + 0.5 * BOX_TOTAL_SIZE[axis])
        / AXIS_SCALE[axis]
        for axis in AXES
    ],
    dtype=float,
)

BOUNDARY_FACES = (
    "xlo",
    "xhi",
    "ylo",
    "yhi",
    "zlo",
    "zhi",
)

FACE_TO_CODE = {
    face: code
    for code, face in enumerate(BOUNDARY_FACES)
}

CORRECTION_STAT_FIELDS = (
    "total_outside",
    "corrected",
    "already_on_boundary",
    "inside_unchanged",
    "failed_unchanged",
    "source_face_mismatch",
)

BOX_LINE_KW = {
    "color": "magenta",
    "linestyle": "--",
    "linewidth": 1.4,
    "alpha": 0.95,
}


@dataclass(frozen=True)
class DiagnosticFile:
    source: str
    path: Path
    stage: str
    diag_top: str
    iteration: int


@dataclass
class ParticleChunk:
    species: str
    group: str
    stage: str
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    px_si: np.ndarray
    py_si: np.ndarray
    pz_si: np.ndarray
    weight: np.ndarray
    orig_x_m: np.ndarray
    orig_y_m: np.ndarray
    orig_z_m: np.ndarray
    creation_time_s: np.ndarray

    @property
    def nmacro(self) -> int:
        return int(self.weight.size)


class Reservoir:
    """Small bounded sample used only to choose robust plotting ranges."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = max(1, int(capacity))
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(0, dtype=float)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return

        # Cap each incoming chunk before merging. This is sufficient for range
        # estimation and prevents multi-million-particle arrays from being copied.
        if values.size > self.capacity:
            take = self.rng.choice(
                values.size,
                size=self.capacity,
                replace=False,
            )
            values = values[take]

        merged = np.concatenate((self.values, values))

        if merged.size > self.capacity:
            take = self.rng.choice(
                merged.size,
                size=self.capacity,
                replace=False,
            )
            merged = merged[take]

        self.values = merged


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def iteration_number(path: Path) -> int:
    """Return the final integer in the HDF5 filename stem."""
    numbers = re.findall(r"(\d+)", path.stem)
    return int(numbers[-1]) if numbers else -1


def diagnostic_top(relative_path: Path) -> tuple[str, str] | None:
    """Identify particles_in* or particles_out* below the input directory."""
    for part in relative_path.parts:
        if part.startswith("particles_in"):
            return part, "inside"

        if part.startswith("particles_out"):
            return part, "outside"

    return None


def boundary_face_from_path(path: Path) -> str | None:
    """Return xlo/xhi/ylo/yhi/zlo/zhi from a BoundaryScraping path."""
    for part in path.parts:
        match = re.fullmatch(
            r"particles_at_([xyz](?:lo|hi))",
            part,
        )

        if match:
            return match.group(1)

    return None


def discover_files(diags_dir: Path) -> list[DiagnosticFile]:
    if not diags_dir.is_dir():
        raise FileNotFoundError(
            f"Diagnostics directory not found: {diags_dir}"
        )

    found: list[DiagnosticFile] = []

    for path in sorted(diags_dir.rglob("*.h5")):
        rel = path.relative_to(diags_dir)
        identified = diagnostic_top(rel)

        if identified is None:
            continue

        diag_top, stage = identified
        iteration = iteration_number(path)

        if iteration < 0:
            print(
                f"[warn] skipping HDF5 file with no iteration number: {path}",
                flush=True,
            )
            continue

        found.append(
            DiagnosticFile(
                source=diags_dir.name,
                path=path,
                stage=stage,
                diag_top=diag_top,
                iteration=iteration,
            )
        )

    if not found:
        raise RuntimeError(
            "No particles_in*/particles_out* HDF5 files were found "
            f"below {diags_dir}"
        )

    return sorted(
        found,
        key=lambda item: (
            item.iteration,
            item.stage,
            str(item.path),
        ),
    )


def available_iterations(files: list[DiagnosticFile]) -> list[int]:
    return sorted({item.iteration for item in files})


def select_files_for_iteration(
    files: list[DiagnosticFile],
    iteration: int,
    outside_mode: str,
) -> list[DiagnosticFile]:
    """
    Select the active-particle snapshot at exactly `iteration`.

    For BoundaryScraping diagnostics:

    batch:
        Use only the particles_out file written at this iteration.

    cumulative:
        Use every particles_out batch through this iteration.
    """
    inside = [
        item
        for item in files
        if item.stage == "inside"
        and item.iteration == iteration
    ]

    if outside_mode == "batch":
        outside = [
            item
            for item in files
            if item.stage == "outside"
            and item.iteration == iteration
        ]

    elif outside_mode == "cumulative":
        outside = [
            item
            for item in files
            if item.stage == "outside"
            and item.iteration <= iteration
        ]

    else:
        raise ValueError(
            f"Unsupported outside mode: {outside_mode}"
        )

    return sorted(
        inside + outside,
        key=lambda item: (
            item.stage,
            item.iteration,
            str(item.path),
        ),
    )


def unique_files(
    files: Iterable[DiagnosticFile],
) -> list[DiagnosticFile]:
    unique: dict[Path, DiagnosticFile] = {}

    for item in files:
        unique[item.path] = item

    return sorted(
        unique.values(),
        key=lambda item: (
            item.iteration,
            item.stage,
            str(item.path),
        ),
    )


# -----------------------------------------------------------------------------
# openPMD HDF5 reading
# -----------------------------------------------------------------------------
def find_particle_groups(
    h5_file: h5py.File,
) -> list[str]:
    groups: list[str] = []

    def visitor(name: str, obj) -> None:
        if (
            isinstance(obj, h5py.Group)
            and name.endswith("/particles")
        ):
            groups.append(name)

    h5_file.visititems(visitor)

    return groups


def as_1d(obj) -> np.ndarray | None:
    if isinstance(obj, h5py.Dataset):
        array = np.asarray(obj[()], dtype=float)
        array = array * float(
            obj.attrs.get("unitSI", 1.0)
        )

        if array.ndim:
            return np.ravel(array)

        return np.asarray(
            [array.item()],
            dtype=float,
        )

    if (
        isinstance(obj, h5py.Group)
        and "value" in obj.attrs
    ):
        value = (
            float(obj.attrs["value"])
            * float(obj.attrs.get("unitSI", 1.0))
        )

        return np.asarray(
            [value],
            dtype=float,
        )

    return None


def read_from_group(
    group: h5py.Group,
    names: Iterable[str],
) -> np.ndarray | None:
    for name in names:
        if name in group:
            array = as_1d(group[name])

            if array is not None:
                return array

    return None


def read_component(
    species_group: h5py.Group,
    record: str,
    component: str,
) -> np.ndarray | None:
    if (
        record in species_group
        and isinstance(
            species_group[record],
            h5py.Group,
        )
    ):
        array = read_from_group(
            species_group[record],
            (component, "SCALAR"),
        )

        if array is not None:
            return array

    if record == "position":
        return read_from_group(
            species_group,
            (component,),
        )

    if record == "positionOffset":
        return read_from_group(
            species_group,
            (f"positionOffset_{component}",),
        )

    if record == "momentum":
        return read_from_group(
            species_group,
            (
                f"u{component}",
                f"p{component}",
            ),
        )

    if record == "weighting":
        return read_from_group(
            species_group,
            (
                "w",
                "weighting",
                "weight",
                "weights",
            ),
        )

    # Runtime vector attributes can be represented as:
    #
    #   orig/x
    #   orig_x
    #   origX
    #
    # WarpX currently writes these attributes as origX, origY and origZ
    # in this diagnostic.
    return read_from_group(
        species_group,
        (
            f"{record}_{component}",
            f"{record}{component.upper()}",
        ),
    )


def read_scalar_record(
    species_group: h5py.Group,
    record: str,
) -> np.ndarray | None:
    """Read an openPMD scalar record or a flat runtime-attribute dataset."""
    if record in species_group:
        obj = species_group[record]

        if isinstance(obj, h5py.Dataset):
            return as_1d(obj)

        if isinstance(obj, h5py.Group):
            array = read_from_group(
                obj,
                (
                    "SCALAR",
                    "value",
                ),
            )

            if array is not None:
                return array

            # Constant openPMD components can be encoded as group attributes.
            return as_1d(obj)

    return read_from_group(
        species_group,
        (record,),
    )


def infer_n(
    species_group: h5py.Group,
) -> int:
    candidates = (
        ("weighting", "w"),
        ("position", "x"),
        ("position", "y"),
        ("position", "z"),
        ("momentum", "x"),
        ("momentum", "y"),
        ("momentum", "z"),
    )

    sizes = []

    for record, component in candidates:
        array = read_component(
            species_group,
            record,
            component,
        )

        if array is not None:
            sizes.append(int(array.size))

    return max(sizes, default=0)


def fill(
    array: np.ndarray | None,
    n: int,
    default: float,
) -> np.ndarray:
    if array is None:
        return np.full(
            n,
            default,
            dtype=float,
        )

    array = np.asarray(
        array,
        dtype=float,
    )

    if array.size == n:
        return array

    if array.size == 1:
        return np.full(
            n,
            float(array[0]),
            dtype=float,
        )

    raise ValueError(
        f"Cannot broadcast array of length {array.size} "
        f"to n={n}"
    )


def read_chunk(
    species_group: h5py.Group,
    species: str,
    stage: str,
) -> ParticleChunk | None:
    n = infer_n(species_group)

    if n <= 0:
        return None

    x = fill(
        read_component(
            species_group,
            "position",
            "x",
        ),
        n,
        0.0,
    )

    y = fill(
        read_component(
            species_group,
            "position",
            "y",
        ),
        n,
        0.0,
    )

    z = fill(
        read_component(
            species_group,
            "position",
            "z",
        ),
        n,
        0.0,
    )

    x += fill(
        read_component(
            species_group,
            "positionOffset",
            "x",
        ),
        n,
        0.0,
    )

    y += fill(
        read_component(
            species_group,
            "positionOffset",
            "y",
        ),
        n,
        0.0,
    )

    z += fill(
        read_component(
            species_group,
            "positionOffset",
            "z",
        ),
        n,
        0.0,
    )

    px = fill(
        read_component(
            species_group,
            "momentum",
            "x",
        ),
        n,
        0.0,
    )

    py = fill(
        read_component(
            species_group,
            "momentum",
            "y",
        ),
        n,
        0.0,
    )

    pz = fill(
        read_component(
            species_group,
            "momentum",
            "z",
        ),
        n,
        0.0,
    )

    weight = fill(
        read_component(
            species_group,
            "weighting",
            "w",
        ),
        n,
        1.0,
    )

    # These records are user-defined WarpX runtime attributes. Missing
    # attributes are represented by NaN so old diagnostics are not
    # accidentally interpreted as creation at the origin or at t=0.
    orig_x = fill(
        read_component(
            species_group,
            "orig",
            "x",
        ),
        n,
        np.nan,
    )

    orig_y = fill(
        read_component(
            species_group,
            "orig",
            "y",
        ),
        n,
        np.nan,
    )

    orig_z = fill(
        read_component(
            species_group,
            "orig",
            "z",
        ),
        n,
        np.nan,
    )

    creation_time = fill(
        read_scalar_record(
            species_group,
            "creationTime",
        ),
        n,
        np.nan,
    )

    return ParticleChunk(
        species=species,
        group=SPECIES_TO_GROUP[species],
        stage=stage,
        x_m=x,
        y_m=y,
        z_m=z,
        px_si=px,
        py_si=py,
        pz_si=pz,
        weight=weight,
        orig_x_m=orig_x,
        orig_y_m=orig_y,
        orig_z_m=orig_z,
        creation_time_s=creation_time,
    )


def empty_correction_record() -> dict[str, int]:
    return {
        field: 0
        for field in CORRECTION_STAT_FIELDS
    }


def accumulate_correction_stats(
    accumulator: dict[tuple[str, str], dict[str, int]],
    source_face: str | None,
    species: str,
    update: dict[str, int],
) -> None:
    key = (
        source_face or "unknown",
        species,
    )

    record = accumulator.setdefault(
        key,
        empty_correction_record(),
    )

    for field in CORRECTION_STAT_FIELDS:
        record[field] += int(
            update.get(field, 0)
        )


def correct_boundary_scraping_chunk(
    chunk: ParticleChunk,
    source_face: str | None,
) -> dict[str, int]:
    """
    Move BoundaryScraping positions back to the first box intersection.

    Only the momentum direction is used. Particles for which no reliable
    intersection can be reconstructed stay at their original coordinates, so
    the corrected analysis never silently removes particles.
    """
    position = np.column_stack(
        (
            chunk.x_m,
            chunk.y_m,
            chunk.z_m,
        )
    ).astype(float, copy=False)

    momentum = np.column_stack(
        (
            chunk.px_si,
            chunk.py_si,
            chunk.pz_si,
        )
    ).astype(float, copy=False)

    number = position.shape[0]
    stats = empty_correction_record()
    stats["total_outside"] = number

    if number == 0:
        return stats

    finite = (
        np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(momentum), axis=1)
    )

    momentum_scale = np.max(
        np.abs(momentum),
        axis=1,
    )

    nonzero_momentum = (
        finite
        & (momentum_scale > 0.0)
    )

    # Normalize each direction independently. This avoids enormous line
    # parameters when physical SI momentum values are very small.
    direction = np.zeros_like(momentum)
    direction[nonzero_momentum] = (
        momentum[nonzero_momentum]
        / momentum_scale[nonzero_momentum, None]
    )

    tolerance = 1.0e-11 * float(
        np.max(BOX_HI_M - BOX_LO_M)
    )

    inside_with_tolerance = (
        np.all(
            position >= BOX_LO_M - tolerance,
            axis=1,
        )
        & np.all(
            position <= BOX_HI_M + tolerance,
            axis=1,
        )
    )

    distances_to_faces = np.column_stack(
        (
            np.abs(position[:, 0] - BOX_LO_M[0]),
            np.abs(position[:, 0] - BOX_HI_M[0]),
            np.abs(position[:, 1] - BOX_LO_M[1]),
            np.abs(position[:, 1] - BOX_HI_M[1]),
            np.abs(position[:, 2] - BOX_LO_M[2]),
            np.abs(position[:, 2] - BOX_HI_M[2]),
        )
    )

    already_on_boundary = (
        finite
        & inside_with_tolerance
        & (
            np.min(distances_to_faces, axis=1)
            <= tolerance
        )
    )

    geometrically_outside = (
        finite
        & np.any(
            (
                position < BOX_LO_M - tolerance
            )
            | (
                position > BOX_HI_M + tolerance
            ),
            axis=1,
        )
    )

    inside_unchanged = (
        finite
        & ~geometrically_outside
        & ~already_on_boundary
    )

    # For r(s) = r_recorded - s*direction, find the interval of s for
    # which the line is inside all three box slabs.
    s_enter = np.full(
        number,
        -np.inf,
        dtype=float,
    )

    s_exit = np.full(
        number,
        np.inf,
        dtype=float,
    )

    impossible_parallel = np.zeros(
        number,
        dtype=bool,
    )

    for axis in range(3):
        coordinate = position[:, axis]
        component = direction[:, axis]
        moving = component != 0.0
        parallel = ~moving

        impossible_parallel |= (
            parallel
            & (
                (
                    coordinate
                    < BOX_LO_M[axis] - tolerance
                )
                | (
                    coordinate
                    > BOX_HI_M[axis] + tolerance
                )
            )
        )

        axis_enter = np.full(
            number,
            -np.inf,
            dtype=float,
        )

        axis_exit = np.full(
            number,
            np.inf,
            dtype=float,
        )

        s_at_lo = (
            coordinate[moving]
            - BOX_LO_M[axis]
        ) / component[moving]

        s_at_hi = (
            coordinate[moving]
            - BOX_HI_M[axis]
        ) / component[moving]

        axis_enter[moving] = np.minimum(
            s_at_lo,
            s_at_hi,
        )

        axis_exit[moving] = np.maximum(
            s_at_lo,
            s_at_hi,
        )

        s_enter = np.maximum(
            s_enter,
            axis_enter,
        )

        s_exit = np.minimum(
            s_exit,
            axis_exit,
        )

    candidate = (
        geometrically_outside
        & nonzero_momentum
        & ~impossible_parallel
    )

    intersection_valid = (
        candidate
        & np.isfinite(s_enter)
        & np.isfinite(s_exit)
        & (s_exit + tolerance >= s_enter)
        & (s_exit >= -tolerance)
        & (s_enter >= -tolerance)
    )

    backtrack = np.maximum(
        s_enter,
        0.0,
    )

    hit_position = position.copy()
    hit_position[intersection_valid] = (
        position[intersection_valid]
        - backtrack[intersection_valid, None]
        * direction[intersection_valid]
    )

    # Clamp only successful candidates. Failed rows retain their originals.
    hit_position[intersection_valid] = np.clip(
        hit_position[intersection_valid],
        BOX_LO_M,
        BOX_HI_M,
    )

    hit_distances = np.column_stack(
        (
            np.abs(hit_position[:, 0] - BOX_LO_M[0]),
            np.abs(hit_position[:, 0] - BOX_HI_M[0]),
            np.abs(hit_position[:, 1] - BOX_LO_M[1]),
            np.abs(hit_position[:, 1] - BOX_HI_M[1]),
            np.abs(hit_position[:, 2] - BOX_LO_M[2]),
            np.abs(hit_position[:, 2] - BOX_HI_M[2]),
        )
    )

    reconstructed_face_code = np.argmin(
        hit_distances,
        axis=1,
    )

    # Force the normal coordinate exactly onto the reconstructed wall.
    for face_code in range(6):
        mask = (
            intersection_valid
            & (
                reconstructed_face_code
                == face_code
            )
        )

        axis = face_code // 2
        wall = (
            BOX_HI_M[axis]
            if face_code % 2
            else BOX_LO_M[axis]
        )

        hit_position[mask, axis] = wall

    final_inside = np.zeros(
        number,
        dtype=bool,
    )

    final_inside[intersection_valid] = np.all(
        (
            hit_position[intersection_valid]
            >= BOX_LO_M - tolerance
        )
        & (
            hit_position[intersection_valid]
            <= BOX_HI_M + tolerance
        ),
        axis=1,
    )

    final_hit_distances = np.column_stack(
        (
            np.abs(hit_position[:, 0] - BOX_LO_M[0]),
            np.abs(hit_position[:, 0] - BOX_HI_M[0]),
            np.abs(hit_position[:, 1] - BOX_LO_M[1]),
            np.abs(hit_position[:, 1] - BOX_HI_M[1]),
            np.abs(hit_position[:, 2] - BOX_LO_M[2]),
            np.abs(hit_position[:, 2] - BOX_HI_M[2]),
        )
    )

    final_on_boundary = np.zeros(
        number,
        dtype=bool,
    )

    final_on_boundary[intersection_valid] = (
        np.min(
            final_hit_distances[intersection_valid],
            axis=1,
        )
        <= tolerance
    )

    correction_mask = (
        intersection_valid
        & final_inside
        & final_on_boundary
    )

    chunk.x_m = np.where(
        correction_mask,
        hit_position[:, 0],
        chunk.x_m,
    )

    chunk.y_m = np.where(
        correction_mask,
        hit_position[:, 1],
        chunk.y_m,
    )

    chunk.z_m = np.where(
        correction_mask,
        hit_position[:, 2],
        chunk.z_m,
    )

    failed = (
        geometrically_outside
        & ~correction_mask
    )

    mismatch = np.zeros(
        number,
        dtype=bool,
    )

    if source_face in FACE_TO_CODE:
        mismatch = (
            correction_mask
            & (
                reconstructed_face_code
                != FACE_TO_CODE[source_face]
            )
        )

    stats["corrected"] = int(
        np.count_nonzero(correction_mask)
    )
    stats["already_on_boundary"] = int(
        np.count_nonzero(already_on_boundary)
    )
    stats["inside_unchanged"] = int(
        np.count_nonzero(inside_unchanged)
    )
    stats["failed_unchanged"] = int(
        np.count_nonzero(failed)
        + np.count_nonzero(~finite)
    )
    stats["source_face_mismatch"] = int(
        np.count_nonzero(mismatch)
    )

    return stats


def iter_chunks(
    files: list[DiagnosticFile],
    progress_label: str,
    correct_outside: bool = False,
    correction_stats: (
        dict[tuple[str, str], dict[str, int]]
        | None
    ) = None,
):
    total = len(files)

    for index, info in enumerate(
        files,
        start=1,
    ):
        if (
            index == 1
            or index % 50 == 0
            or index == total
        ):
            print(
                f"[{progress_label}] "
                f"file {index}/{total}: {info.path}",
                flush=True,
            )

        try:
            with h5py.File(
                info.path,
                "r",
            ) as h5_file:
                for group_name in find_particle_groups(
                    h5_file
                ):
                    particle_group = h5_file[group_name]

                    for species in SPECIES:
                        if species not in particle_group:
                            continue

                        try:
                            chunk = read_chunk(
                                particle_group[species],
                                species,
                                info.stage,
                            )

                        except Exception as exc:
                            print(
                                f"[warn] {species} "
                                f"in {info.path}: {exc}",
                                flush=True,
                            )
                            continue

                        if chunk is not None:
                            if (
                                correct_outside
                                and chunk.stage == "outside"
                            ):
                                source_face = (
                                    boundary_face_from_path(
                                        info.path
                                    )
                                )

                                update = (
                                    correct_boundary_scraping_chunk(
                                        chunk,
                                        source_face,
                                    )
                                )

                                if correction_stats is not None:
                                    accumulate_correction_stats(
                                        correction_stats,
                                        source_face,
                                        chunk.species,
                                        update,
                                    )

                            yield info, chunk

        except OSError as exc:
            print(
                f"[warn] could not open "
                f"{info.path}: {exc}",
                flush=True,
            )


# -----------------------------------------------------------------------------
# Histogram setup and accumulation
# -----------------------------------------------------------------------------
def memberships(
    chunk: ParticleChunk,
) -> tuple[str, ...]:
    if chunk.species in IPC_SPECIES:
        return (
            "IPC_ALL",
            chunk.group,
        )

    return (chunk.group,)


def robust_edges(
    sample: np.ndarray,
    bins: int,
    qlow: float,
    qhigh: float,
) -> np.ndarray:
    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample = sample[
        np.isfinite(sample)
    ]

    if sample.size == 0:
        return np.linspace(
            -1.0,
            1.0,
            bins + 1,
        )

    lo, hi = np.quantile(
        sample,
        (qlow, qhigh),
    )

    if not hi > lo:
        center = float(lo)
        pad = max(
            abs(center) * 0.05,
            1.0,
        )
        lo = center - pad
        hi = center + pad

    else:
        pad = 0.05 * (hi - lo)
        lo -= pad
        hi += pad

    return np.linspace(
        float(lo),
        float(hi),
        bins + 1,
    )


def stage_or_combined(
    histograms: dict,
    key_inside: tuple,
    key_outside: tuple,
) -> np.ndarray:
    return (
        histograms[key_inside]
        + histograms[key_outside]
    )


def first_pass(
    files: list[DiagnosticFile],
    sample_capacity: int,
    qlow: float,
    qhigh: float,
    bins_1d: int,
    correct_outside: bool = False,
):
    # IPC channels deliberately share IPC_ALL ranges so
    # their curves and maps are directly comparable.
    range_groups = (
        "IPC_ALL",
        "BEAM1",
        "BEAM2",
    )

    reservoirs = {
        (group, axis): Reservoir(
            sample_capacity,
            seed=1000 + 17 * i + j,
        )
        for i, group in enumerate(range_groups)
        for j, axis in enumerate(AXES)
    }

    weighted_counts: dict[
        tuple[str, str, str],
        float,
    ] = {}

    macro_counts: dict[
        tuple[str, str, str],
        int,
    ] = {}

    for _info, chunk in iter_chunks(
        files,
        (
            "corrected range pass"
            if correct_outside
            else "range pass"
        ),
        correct_outside=correct_outside,
    ):
        finite_weight = (
            np.isfinite(chunk.weight)
            & (chunk.weight > 0.0)
        )

        wsum = float(
            np.sum(
                chunk.weight[finite_weight]
            )
        )

        weighted_key = (
            chunk.stage,
            chunk.group,
            chunk.species,
        )

        weighted_counts[weighted_key] = (
            weighted_counts.get(
                weighted_key,
                0.0,
            )
            + wsum
        )

        macro_counts[weighted_key] = (
            macro_counts.get(
                weighted_key,
                0,
            )
            + chunk.nmacro
        )

        sample_group = (
            "IPC_ALL"
            if chunk.species in IPC_SPECIES
            else chunk.group
        )

        arrays = {
            "x": chunk.x_m,
            "y": chunk.y_m,
            "z": chunk.z_m,
        }

        for axis in AXES:
            reservoirs[
                (sample_group, axis)
            ].update(
                arrays[axis]
                * AXIS_SCALE[axis]
            )

    edges: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    for group in PLOT_GROUPS:
        source_group = (
            "IPC_ALL"
            if group in IPC_GROUPS
            else group
        )

        for axis in AXES:
            edges[(group, axis)] = robust_edges(
                reservoirs[
                    (source_group, axis)
                ].values,
                bins_1d,
                qlow,
                qhigh,
            )

    return (
        edges,
        weighted_counts,
        macro_counts,
    )


def expand_edges_to_include_box(
    edges: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Return new position edges that include both simulation-box walls."""
    expanded: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    for key, array in edges.items():
        _group, axis = key
        wall_lo = (
            BOX_CENTER[axis]
            - 0.5 * BOX_TOTAL_SIZE[axis]
        )
        wall_hi = (
            BOX_CENTER[axis]
            + 0.5 * BOX_TOTAL_SIZE[axis]
        )

        lo = min(
            float(array[0]),
            wall_lo,
        )
        hi = max(
            float(array[-1]),
            wall_hi,
        )

        expanded[key] = np.linspace(
            lo,
            hi,
            array.size,
        )

    return expanded


def initialize_histograms(
    edges: dict,
    bins_2d: int,
    bins_phase_x: int,
    bins_phase_y: int,
    logtheta_min: float,
    logtheta_max: float,
    logpt_min: float,
    logpt_max: float,
):
    hist1d = {}
    hist2d = {}
    phase = {}

    phase_x_edges = np.linspace(
        logtheta_min,
        logtheta_max,
        bins_phase_x + 1,
    )

    phase_y_edges = np.linspace(
        logpt_min,
        logpt_max,
        bins_phase_y + 1,
    )

    # IPC spatial maps use fixed, plane-specific zoom windows. This is
    # necessary because z has a different requested range in xz and yz.
    # Beam maps keep their original robust ranges.
    edges2d = {}

    for group in PLOT_GROUPS:
        for a, b in PLANES:
            plane = a + b

            for axis in (a, b):
                if group in IPC_GROUPS:
                    lo, hi = IPC_2D_LIMITS[plane][axis]
                else:
                    one_d = edges[(group, axis)]
                    lo = float(one_d[0])
                    hi = float(one_d[-1])

                edges2d[(group, plane, axis)] = np.linspace(
                    lo,
                    hi,
                    bins_2d + 1,
                )

    for stage in STAGES:
        for group in PLOT_GROUPS:
            for axis in AXES:
                hist1d[
                    (stage, group, axis)
                ] = np.zeros(
                    edges[
                        (group, axis)
                    ].size - 1,
                    dtype=float,
                )

            for a, b in PLANES:
                plane = a + b

                hist2d[
                    (stage, group, plane)
                ] = np.zeros(
                    (
                        edges2d[
                            (group, plane, a)
                        ].size - 1,
                        edges2d[
                            (group, plane, b)
                        ].size - 1,
                    ),
                    dtype=float,
                )

        for group in IPC_GROUPS:
            phase[
                (stage, group)
            ] = np.zeros(
                (
                    bins_phase_x,
                    bins_phase_y,
                ),
                dtype=float,
            )

    return (
        hist1d,
        hist2d,
        phase,
        edges2d,
        phase_x_edges,
        phase_y_edges,
    )


def second_pass(
    files,
    edges,
    hist1d,
    hist2d,
    phase,
    edges2d,
    phase_x_edges,
    phase_y_edges,
    correct_outside: bool = False,
    correction_stats: (
        dict[tuple[str, str], dict[str, int]]
        | None
    ) = None,
):
    weighted_counts: dict[
        tuple[str, str, str],
        float,
    ] = {}

    macro_counts: dict[
        tuple[str, str, str],
        int,
    ] = {}

    for _info, chunk in iter_chunks(
        files,
        (
            "corrected histogram pass"
            if correct_outside
            else "histogram pass"
        ),
        correct_outside=correct_outside,
        correction_stats=correction_stats,
    ):
        finite_weight = (
            np.isfinite(chunk.weight)
            & (chunk.weight > 0.0)
        )

        wsum = float(
            np.sum(
                chunk.weight[finite_weight]
            )
        )

        count_key = (
            chunk.stage,
            chunk.group,
            chunk.species,
        )

        weighted_counts[count_key] = (
            weighted_counts.get(
                count_key,
                0.0,
            )
            + wsum
        )

        macro_counts[count_key] = (
            macro_counts.get(
                count_key,
                0,
            )
            + chunk.nmacro
        )

        values = {
            "x": (
                chunk.x_m
                * AXIS_SCALE["x"]
            ),
            "y": (
                chunk.y_m
                * AXIS_SCALE["y"]
            ),
            "z": (
                chunk.z_m
                * AXIS_SCALE["z"]
            ),
        }

        weight = chunk.weight

        for group in memberships(chunk):
            for axis in AXES:
                hist1d[
                    (
                        chunk.stage,
                        group,
                        axis,
                    )
                ] += np.histogram(
                    values[axis],
                    bins=edges[
                        (group, axis)
                    ],
                    weights=weight,
                )[0]

            for a, b in PLANES:
                plane = a + b

                hist2d[
                    (
                        chunk.stage,
                        group,
                        plane,
                    )
                ] += np.histogram2d(
                    values[a],
                    values[b],
                    bins=(
                        edges2d[
                            (group, plane, a)
                        ],
                        edges2d[
                            (group, plane, b)
                        ],
                    ),
                    weights=weight,
                )[0]

        if chunk.species in IPC_SPECIES:
            pt_si = np.hypot(
                chunk.px_si,
                chunk.py_si,
            )

            # Fold both beam directions onto the nearest beam axis.
            theta_rad = np.arctan2(
                pt_si,
                np.abs(chunk.pz_si),
            )

            pt_mev_c = (
                pt_si
                * SI_MOMENTUM_TO_MEV_C
            )

            valid = (
                np.isfinite(theta_rad)
                & np.isfinite(pt_mev_c)
                & np.isfinite(weight)
                & (theta_rad > 0.0)
                & (pt_mev_c > 0.0)
                & (weight > 0.0)
            )

            if np.any(valid):
                logtheta = np.log10(
                    theta_rad[valid]
                )

                logpt = np.log10(
                    pt_mev_c[valid]
                )

                selected_w = weight[valid]

                for group in (
                    "IPC_ALL",
                    chunk.group,
                ):
                    phase[
                        (
                            chunk.stage,
                            group,
                        )
                    ] += np.histogram2d(
                        logtheta,
                        logpt,
                        bins=(
                            phase_x_edges,
                            phase_y_edges,
                        ),
                        weights=selected_w,
                    )[0]

    return (
        weighted_counts,
        macro_counts,
    )



# -----------------------------------------------------------------------------
# Particle-creation histogram setup and accumulation
# -----------------------------------------------------------------------------
def creation_values(
    chunk: ParticleChunk,
) -> dict[str, np.ndarray]:
    """Return creation records converted to plotting units."""
    return {
        "orig_x": (
            chunk.orig_x_m
            * CREATION_SCALE["orig_x"]
        ),
        "orig_y": (
            chunk.orig_y_m
            * CREATION_SCALE["orig_y"]
        ),
        "orig_z": (
            chunk.orig_z_m
            * CREATION_SCALE["orig_z"]
        ),
        "creationTime": (
            chunk.creation_time_s
            * CREATION_SCALE["creationTime"]
        ),
    }


def creation_plane_key(
    field_x: str,
    field_y: str,
) -> str:
    return f"{field_x}__{field_y}"


def first_pass_creation(
    files: list[DiagnosticFile],
    sample_capacity: int,
    qlow: float,
    qhigh: float,
    bins_1d: int,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    int,
]:
    """
    Determine common robust ranges for creation coordinates and creation time.

    All IPC channels deliberately share the IPC_ALL ranges so BW, BH and LL
    plots can be compared directly.
    """
    reservoirs = {
        field: Reservoir(
            sample_capacity,
            seed=8000 + index,
        )
        for index, field in enumerate(
            CREATION_FIELDS
        )
    }

    particles_with_creation_data = 0

    for _info, chunk in iter_chunks(
        files,
        "creation range pass",
        correct_outside=False,
    ):
        if chunk.species not in IPC_SPECIES:
            continue

        values = creation_values(chunk)

        valid_particle = (
            np.isfinite(values["orig_x"])
            & np.isfinite(values["orig_y"])
            & np.isfinite(values["orig_z"])
            & np.isfinite(values["creationTime"])
        )

        particles_with_creation_data += int(
            np.count_nonzero(valid_particle)
        )

        for field in CREATION_FIELDS:
            reservoirs[field].update(
                values[field]
            )

    edges: dict[
        tuple[str, str],
        np.ndarray,
    ] = {}

    for group in IPC_GROUPS:
        for field in CREATION_FIELDS:
            edges[(group, field)] = robust_edges(
                reservoirs[field].values,
                bins_1d,
                qlow,
                qhigh,
            )

    return (
        edges,
        particles_with_creation_data,
    )


def initialize_creation_histograms(
    edges: dict[tuple[str, str], np.ndarray],
    bins_2d: int,
):
    hist1d = {}
    hist2d = {}
    edges2d = {}

    for group in IPC_GROUPS:
        for field_x, field_y in CREATION_PLANES:
            key = creation_plane_key(
                field_x,
                field_y,
            )

            spatial_pair = (
                field_x in CREATION_TO_AXIS
                and field_y in CREATION_TO_AXIS
            )

            for field in (field_x, field_y):
                if spatial_pair:
                    axis_x = CREATION_TO_AXIS[field_x]
                    axis_y = CREATION_TO_AXIS[field_y]
                    plane = axis_x + axis_y
                    axis = CREATION_TO_AXIS[field]
                    lo, hi = IPC_2D_LIMITS[plane][axis]
                else:
                    one_d = edges[(group, field)]
                    lo = float(one_d[0])
                    hi = float(one_d[-1])

                edges2d[(group, key, field)] = np.linspace(
                    lo,
                    hi,
                    bins_2d + 1,
                )

    for stage in STAGES:
        for group in IPC_GROUPS:
            for field in CREATION_FIELDS:
                hist1d[
                    (
                        stage,
                        group,
                        field,
                    )
                ] = np.zeros(
                    edges[
                        (group, field)
                    ].size - 1,
                    dtype=float,
                )

            for field_x, field_y in CREATION_PLANES:
                key = creation_plane_key(
                    field_x,
                    field_y,
                )

                hist2d[
                    (
                        stage,
                        group,
                        key,
                    )
                ] = np.zeros(
                    (
                        edges2d[
                            (group, key, field_x)
                        ].size - 1,
                        edges2d[
                            (group, key, field_y)
                        ].size - 1,
                    ),
                    dtype=float,
                )

    return (
        hist1d,
        hist2d,
        edges2d,
    )


def second_pass_creation(
    files: list[DiagnosticFile],
    edges,
    hist1d,
    hist2d,
    edges2d,
):
    """
    Accumulate weighted creation histograms and production counts.

    The returned counts include only IPC particle entries with finite
    orig_x/orig_y/orig_z/creationTime and positive finite weight. These are
    therefore the counts used for the distinct "production" summaries.
    """
    particles_with_creation_data = 0

    weighted_counts: dict[
        tuple[str, str, str],
        float,
    ] = {}

    macro_counts: dict[
        tuple[str, str, str],
        int,
    ] = {}

    for _info, chunk in iter_chunks(
        files,
        "creation histogram pass",
        correct_outside=False,
    ):
        if chunk.species not in IPC_SPECIES:
            continue

        values = creation_values(chunk)
        weight = np.asarray(
            chunk.weight,
            dtype=float,
        )

        valid_weight = (
            np.isfinite(weight)
            & (weight > 0.0)
        )

        valid_all = valid_weight.copy()

        for field in CREATION_FIELDS:
            valid_all &= np.isfinite(
                values[field]
            )

        valid_count = int(
            np.count_nonzero(valid_all)
        )

        particles_with_creation_data += valid_count

        count_key = (
            chunk.stage,
            chunk.group,
            chunk.species,
        )

        weighted_counts[count_key] = (
            weighted_counts.get(
                count_key,
                0.0,
            )
            + float(
                np.sum(weight[valid_all])
            )
        )

        macro_counts[count_key] = (
            macro_counts.get(
                count_key,
                0,
            )
            + valid_count
        )

        for group in memberships(chunk):
            for field in CREATION_FIELDS:
                valid = (
                    valid_weight
                    & np.isfinite(
                        values[field]
                    )
                )

                if np.any(valid):
                    hist1d[
                        (
                            chunk.stage,
                            group,
                            field,
                        )
                    ] += np.histogram(
                        values[field][valid],
                        bins=edges[
                            (group, field)
                        ],
                        weights=weight[valid],
                    )[0]

            for field_x, field_y in CREATION_PLANES:
                key = creation_plane_key(
                    field_x,
                    field_y,
                )

                valid = (
                    valid_weight
                    & np.isfinite(
                        values[field_x]
                    )
                    & np.isfinite(
                        values[field_y]
                    )
                )

                if np.any(valid):
                    hist2d[
                        (
                            chunk.stage,
                            group,
                            key,
                        )
                    ] += np.histogram2d(
                        values[field_x][valid],
                        values[field_y][valid],
                        bins=(
                            edges2d[
                                (group, key, field_x)
                            ],
                            edges2d[
                                (group, key, field_y)
                            ],
                        ),
                        weights=weight[valid],
                    )[0]

    return (
        particles_with_creation_data,
        weighted_counts,
        macro_counts,
    )

def get_creation_1d(
    hist1d,
    scope: str,
    group: str,
    field: str,
) -> np.ndarray:
    if scope in STAGES:
        return hist1d[
            (
                scope,
                group,
                field,
            )
        ]

    return stage_or_combined(
        hist1d,
        (
            "inside",
            group,
            field,
        ),
        (
            "outside",
            group,
            field,
        ),
    )


def get_creation_2d(
    hist2d,
    scope: str,
    group: str,
    field_x: str,
    field_y: str,
) -> np.ndarray:
    key = creation_plane_key(
        field_x,
        field_y,
    )

    if scope in STAGES:
        return hist2d[
            (
                scope,
                group,
                key,
            )
        ]

    return stage_or_combined(
        hist2d,
        (
            "inside",
            group,
            key,
        ),
        (
            "outside",
            group,
            key,
        ),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def box_limits(
    axis: str,
) -> tuple[float, float]:
    """
    Return the lower and upper box boundaries for one axis.

    BOX_TOTAL_SIZE contains the complete length L, so the
    boundaries are center - L/2 and center + L/2.
    """
    half_size = (
        0.5
        * BOX_TOTAL_SIZE[axis]
    )

    center = BOX_CENTER[axis]

    return (
        center - half_size,
        center + half_size,
    )


def expand_limits_to_include(
    current_lo: float,
    current_hi: float,
    target_lo: float,
    target_hi: float,
) -> tuple[float, float]:
    """Expand existing axis limits to include the requested box limits."""
    return (
        min(current_lo, target_lo),
        max(current_hi, target_hi),
    )


def draw_box_lines_1d(
    ax: plt.Axes,
    axis: str,
) -> None:
    """
    Draw the two box boundaries on a 1D x, y, or z distribution.
    """
    lo, hi = box_limits(axis)

    ax.axvline(
        lo,
        **BOX_LINE_KW,
    )

    ax.axvline(
        hi,
        **BOX_LINE_KW,
    )

    current_lo, current_hi = ax.get_xlim()

    new_lo, new_hi = expand_limits_to_include(
        current_lo,
        current_hi,
        lo,
        hi,
    )

    ax.set_xlim(
        new_lo,
        new_hi,
    )


def draw_box_rectangle(
    ax: plt.Axes,
    axis_x: str,
    axis_y: str,
) -> None:
    """
    Draw the appropriate 2D projection of the simulation box.

    Examples:
        xy plot -> rectangle using Lx and Ly
        xz plot -> rectangle using Lx and Lz
        yz plot -> rectangle using Ly and Lz
    """
    x0, x1 = box_limits(axis_x)
    y0, y1 = box_limits(axis_y)

    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        **BOX_LINE_KW,
    )

    current_x0, current_x1 = ax.get_xlim()
    current_y0, current_y1 = ax.get_ylim()

    new_x0, new_x1 = expand_limits_to_include(
        current_x0,
        current_x1,
        x0,
        x1,
    )

    new_y0, new_y1 = expand_limits_to_include(
        current_y0,
        current_y1,
        y0,
        y1,
    )

    ax.set_xlim(
        new_x0,
        new_x1,
    )

    ax.set_ylim(
        new_y0,
        new_y1,
    )


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=dpi,
    )
    plt.close(fig)


def normalize_bins(
    hist: np.ndarray,
) -> np.ndarray:
    total = float(
        np.sum(hist)
    )

    if total > 0.0:
        return hist / total

    return hist.copy()


def get_1d(
    hist1d,
    scope: str,
    group: str,
    axis: str,
) -> np.ndarray:
    if scope in STAGES:
        return hist1d[
            (
                scope,
                group,
                axis,
            )
        ]

    return stage_or_combined(
        hist1d,
        (
            "inside",
            group,
            axis,
        ),
        (
            "outside",
            group,
            axis,
        ),
    )


def get_2d(
    hist2d,
    scope: str,
    group: str,
    plane: str,
) -> np.ndarray:
    if scope in STAGES:
        return hist2d[
            (
                scope,
                group,
                plane,
            )
        ]

    return stage_or_combined(
        hist2d,
        (
            "inside",
            group,
            plane,
        ),
        (
            "outside",
            group,
            plane,
        ),
    )


def get_phase(
    phase,
    scope: str,
    group: str,
) -> np.ndarray:
    """Return an inside, outside, or inside+outside phase-space map."""
    if scope in STAGES:
        return phase[(scope, group)]

    if scope == "combined":
        return stage_or_combined(
            phase,
            ("inside", group),
            ("outside", group),
        )

    raise ValueError(
        f"Unsupported phase-space scope: {scope}"
    )


def plot_1d(
    hist,
    edges,
    axis,
    xlabel,
    title,
    color,
    path,
    dpi,
):
    y = normalize_bins(hist)

    centers = (
        0.5
        * (
            edges[:-1]
            + edges[1:]
        )
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.step(
        centers,
        y,
        where="mid",
        color=color,
        linewidth=1.5,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(
        "Normalized weighted entries"
    )
    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 3),
    )

    # Add the two simulation-box boundaries.
    draw_box_lines_1d(
        ax,
        axis,
    )

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_density(
    hist,
    xedges,
    yedges,
    axis_x,
    axis_y,
    xlabel,
    ylabel,
    title,
    path,
    dpi,
    show_box=True,
):
    total = float(
        np.sum(hist)
    )

    percent = (
        100.0 * hist / total
        if total > 0.0
        else hist.copy()
    )

    positive = percent[
        percent > 0.0
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 6.2)
    )

    if positive.size:
        vmax = float(
            np.max(positive)
        )

        vmin = max(
            float(
                np.percentile(
                    positive,
                    5.0,
                )
            ),
            vmax * 1.0e-7,
        )

        if not vmax > vmin:
            vmin = max(
                vmax * 0.1,
                np.finfo(float).tiny,
            )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            percent.T,
            shading="auto",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            extend="min",
        )

        cbar.set_label(
            "Weighted fraction per bin [%]"
        )

    else:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # IPC maps are intentionally zoomed to the requested data window, so the
    # simulation-box overlay is omitted there. Beam maps retain the overlay.
    if show_box:
        draw_box_rectangle(
            ax,
            axis_x,
            axis_y,
        )

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_ipc_overlay(
    hist1d,
    edges,
    scope,
    axis,
    label,
    output,
    dpi,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    channels = (
        (
            "IPC_ALL",
            "All pairs",
        ),
        (
            "LL",
            "Landau-Lifshitz (LL)",
        ),
        (
            "BH",
            "Bethe-Heitler (BH)",
        ),
        (
            "BW",
            "Breit-Wheeler (BW)",
        ),
    )

    for group, legend in channels:
        hist = get_1d(
            hist1d,
            scope,
            group,
            axis,
        )

        y = normalize_bins(hist)

        centers = (
            0.5
            * (
                edges[
                    (group, axis)
                ][:-1]
                + edges[
                    (group, axis)
                ][1:]
            )
        )

        ax.step(
            centers,
            y,
            where="mid",
            color=GROUP_COLOR[group],
            linewidth=1.3,
            label=legend,
        )

    ax.set_xlabel(
        f"{axis} ({AXIS_UNIT[axis]})"
    )

    ax.set_ylabel(
        "Normalized weighted entries"
    )

    ax.set_title(
        f"{label} — IPC {axis} distribution — {scope}"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 3),
    )

    # Add the two simulation-box boundaries.
    draw_box_lines_1d(
        ax,
        axis,
    )

    save_figure(
        fig,
        (
            output
            / "overlays"
            / scope
            / f"{axis}_ipc_channels.png"
        ),
        dpi,
    )


def plot_phase(
    hist,
    xedges,
    yedges,
    title,
    path,
    dpi,
):
    total = float(
        np.sum(hist)
    )

    percent = (
        100.0 * hist / total
        if total > 0.0
        else hist.copy()
    )

    positive = percent[
        percent > 0.0
    ]

    fig, ax = plt.subplots(
        figsize=(8.2, 6.5)
    )

    if positive.size:
        vmax = float(
            np.max(positive)
        )

        vmin = max(
            float(
                np.percentile(
                    positive,
                    5.0,
                )
            ),
            vmax * 1.0e-7,
        )

        if not vmax > vmin:
            vmin = max(
                vmax * 0.1,
                np.finfo(float).tiny,
            )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            percent.T,
            shading="auto",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            extend="min",
        )

        cbar.set_label(
            "Weighted fraction per bin [%]"
        )

    else:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(
        r"$\log_{10}(\theta\,[\mathrm{rad}])$"
    )

    ax.set_ylabel(
        r"$\log_{10}(p_T\,[\mathrm{MeV}/c])$"
    )

    ax.set_title(title)

    # This plot has theta and pT axes, not x, y, or z,
    # so no spatial box lines are added.
    save_figure(
        fig,
        path,
        dpi,
    )



def creation_axis_label(
    field: str,
) -> str:
    return (
        f"{CREATION_LABEL[field]} "
        f"({CREATION_UNIT[field]})"
    )


def plot_creation_1d(
    hist,
    edges,
    field,
    title,
    color,
    path,
    dpi,
):
    y = normalize_bins(hist)

    centers = (
        0.5
        * (
            edges[:-1]
            + edges[1:]
        )
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.step(
        centers,
        y,
        where="mid",
        color=color,
        linewidth=1.5,
    )

    ax.set_xlabel(
        creation_axis_label(field)
    )

    ax.set_ylabel(
        "Normalized weighted entries"
    )

    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 3),
    )

    if field in CREATION_TO_AXIS:
        draw_box_lines_1d(
            ax,
            CREATION_TO_AXIS[field],
        )

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_creation_density(
    hist,
    xedges,
    yedges,
    field_x,
    field_y,
    title,
    path,
    dpi,
):
    total = float(
        np.sum(hist)
    )

    percent = (
        100.0 * hist / total
        if total > 0.0
        else hist.copy()
    )

    positive = percent[
        percent > 0.0
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 6.3)
    )

    if positive.size:
        vmax = float(
            np.max(positive)
        )

        vmin = max(
            float(
                np.percentile(
                    positive,
                    5.0,
                )
            ),
            vmax * 1.0e-7,
        )

        if not vmax > vmin:
            vmin = max(
                vmax * 0.1,
                np.finfo(float).tiny,
            )

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            percent.T,
            shading="auto",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )

        cbar = fig.colorbar(
            mesh,
            ax=ax,
            extend="min",
        )

        cbar.set_label(
            "Weighted fraction per bin [%]"
        )

    else:
        ax.text(
            0.5,
            0.5,
            "no creation data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(
        creation_axis_label(field_x)
    )

    ax.set_ylabel(
        creation_axis_label(field_y)
    )

    ax.set_title(title)

    # Creation-position maps are intentionally data-focused. Do not draw or
    # expand to the simulation box; the spatial orig_x/orig_y/orig_z planes use
    # the same fixed zoom windows as the IPC position maps.

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_creation_overlay(
    hist1d,
    edges,
    scope,
    field,
    label,
    output,
    dpi,
):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    channels = (
        (
            "IPC_ALL",
            "All pairs",
        ),
        (
            "LL",
            "Landau-Lifshitz (LL)",
        ),
        (
            "BH",
            "Bethe-Heitler (BH)",
        ),
        (
            "BW",
            "Breit-Wheeler (BW)",
        ),
    )

    for group, legend in channels:
        hist = get_creation_1d(
            hist1d,
            scope,
            group,
            field,
        )

        y = normalize_bins(hist)

        centers = (
            0.5
            * (
                edges[
                    (group, field)
                ][:-1]
                + edges[
                    (group, field)
                ][1:]
            )
        )

        ax.step(
            centers,
            y,
            where="mid",
            color=GROUP_COLOR[group],
            linewidth=1.3,
            label=legend,
        )

    ax.set_xlabel(
        creation_axis_label(field)
    )

    ax.set_ylabel(
        "Normalized weighted entries"
    )

    ax.set_title(
        f"{label} — IPC "
        f"{CREATION_LABEL[field].lower()} "
        f"distribution — {scope}"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        True,
        alpha=0.2,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 3),
    )

    if field in CREATION_TO_AXIS:
        draw_box_lines_1d(
            ax,
            CREATION_TO_AXIS[field],
        )

    save_figure(
        fig,
        (
            output
            / "creation_overlays"
            / scope
            / f"{field}_ipc_channels.png"
        ),
        dpi,
    )


def generate_creation_plots(
    hist1d,
    hist2d,
    edges,
    edges2d,
    output: Path,
    dpi: int,
    label: str,
):
    for scope in SCOPES:
        for group in IPC_GROUPS:
            group_dir = (
                output
                / "creation_distributions"
                / scope
                / group
            )

            for field in CREATION_FIELDS:
                plot_creation_1d(
                    get_creation_1d(
                        hist1d,
                        scope,
                        group,
                        field,
                    ),
                    edges[
                        (group, field)
                    ],
                    field,
                    (
                        f"{label} — "
                        f"{group} "
                        f"{CREATION_LABEL[field].lower()} "
                        f"distribution — {scope}"
                    ),
                    GROUP_COLOR[group],
                    (
                        group_dir
                        / f"{field}_distribution.png"
                    ),
                    dpi,
                )

            for field_x, field_y in CREATION_PLANES:
                key = creation_plane_key(
                    field_x,
                    field_y,
                )

                plot_creation_density(
                    get_creation_2d(
                        hist2d,
                        scope,
                        group,
                        field_x,
                        field_y,
                    ),
                    edges2d[
                        (group, key, field_x)
                    ],
                    edges2d[
                        (group, key, field_y)
                    ],
                    field_x,
                    field_y,
                    (
                        f"{label} — "
                        f"{group} "
                        f"{CREATION_LABEL[field_x]} vs "
                        f"{CREATION_LABEL[field_y]} "
                        f"density — {scope}"
                    ),
                    (
                        group_dir
                        / f"{key}_density.png"
                    ),
                    dpi,
                )

        for field in CREATION_FIELDS:
            plot_creation_overlay(
                hist1d,
                edges,
                scope,
                field,
                label,
                output,
                dpi,
            )


def generate_plots(
    hist1d,
    hist2d,
    phase,
    edges,
    edges2d,
    phase_x_edges,
    phase_y_edges,
    output: Path,
    dpi: int,
    label: str,
):
    for scope in SCOPES:
        for group in PLOT_GROUPS:
            group_dir = (
                output
                / "distributions"
                / scope
                / group
            )

            for axis in AXES:
                plot_1d(
                    get_1d(
                        hist1d,
                        scope,
                        group,
                        axis,
                    ),
                    edges[
                        (group, axis)
                    ],
                    axis,
                    (
                        f"{axis} "
                        f"({AXIS_UNIT[axis]})"
                    ),
                    (
                        f"{label} — "
                        f"{group} {axis} "
                        f"distribution — {scope}"
                    ),
                    GROUP_COLOR[group],
                    (
                        group_dir
                        / f"{axis}_distribution.png"
                    ),
                    dpi,
                )

            for a, b in PLANES:
                plane = a + b

                plot_density(
                    get_2d(
                        hist2d,
                        scope,
                        group,
                        plane,
                    ),
                    edges2d[
                        (group, plane, a)
                    ],
                    edges2d[
                        (group, plane, b)
                    ],
                    a,
                    b,
                    (
                        f"{a} "
                        f"({AXIS_UNIT[a]})"
                    ),
                    (
                        f"{b} "
                        f"({AXIS_UNIT[b]})"
                    ),
                    (
                        f"{label} — "
                        f"{group} {a}-{b} "
                        f"density — {scope}"
                    ),
                    (
                        group_dir
                        / f"{plane}_density.png"
                    ),
                    dpi,
                    show_box=(group not in IPC_GROUPS),
                )

        for axis in AXES:
            plot_ipc_overlay(
                hist1d,
                edges,
                scope,
                axis,
                label,
                output,
                dpi,
            )

    # Produce theta-pT maps for particles_in, particles_out, and their sum.
    for scope in SCOPES:
        for group in IPC_GROUPS:
            plot_phase(
                get_phase(
                    phase,
                    scope,
                    group,
                ),
                phase_x_edges,
                phase_y_edges,
                (
                    f"{label} — "
                    f"{group} — {scope}"
                ),
                (
                    output
                    / "phase_space"
                    / scope
                    / (
                        f"{group}_log10theta_"
                        "vs_log10pt_MeV.png"
                    )
                ),
                dpi,
            )


# -----------------------------------------------------------------------------
# CSV/NPZ outputs
# -----------------------------------------------------------------------------
def write_global_manifest(
    diags_dir: Path,
    files: list[DiagnosticFile],
    iterations: list[int],
    output: Path,
) -> None:
    summary_dir = (
        output
        / "summaries"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        summary_dir
        / "files_discovered.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "source",
                "stage",
                "diag_top",
                "iteration",
                "path",
            )
        )

        for item in files:
            writer.writerow(
                (
                    item.source,
                    item.stage,
                    item.diag_top,
                    item.iteration,
                    item.path,
                )
            )

    with (
        summary_dir
        / "iterations_available.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "iteration",
                "inside_file_count",
                "outside_file_count",
            )
        )

        for iteration in iterations:
            inside_count = sum(
                item.stage == "inside"
                and item.iteration == iteration
                for item in files
            )

            outside_count = sum(
                item.stage == "outside"
                and item.iteration == iteration
                for item in files
            )

            writer.writerow(
                (
                    iteration,
                    inside_count,
                    outside_count,
                )
            )

    with (
        summary_dir
        / "analysis_source.txt"
    ).open("w") as handle:
        handle.write(
            f"diags_dir={diags_dir}\n"
        )


def write_iteration_manifest(
    files: list[DiagnosticFile],
    iteration: int,
    outside_mode: str,
    output: Path,
) -> None:
    summary_dir = (
        output
        / "summaries"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        summary_dir
        / "files_used.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "target_iteration",
                "outside_mode",
                "stage",
                "source_iteration",
                "diag_top",
                "path",
            )
        )

        for item in files:
            writer.writerow(
                (
                    iteration,
                    outside_mode,
                    item.stage,
                    item.iteration,
                    item.diag_top,
                    item.path,
                )
            )


def aggregate_channel_weights(
    weighted_counts: dict[
        tuple[str, str, str],
        float,
    ],
):
    channel = {
        (stage, group): 0.0
        for stage in STAGES
        for group in (
            "BW",
            "BH",
            "LL",
        )
    }

    species_rows = []

    for (
        stage,
        group,
        species,
    ), value in sorted(
        weighted_counts.items()
    ):
        species_rows.append(
            (
                stage,
                group,
                species,
                value,
            )
        )

        if group in {
            "BW",
            "BH",
            "LL",
        }:
            channel[
                (stage, group)
            ] += value

    return (
        channel,
        species_rows,
    )


def write_summaries(
    weighted_counts,
    macro_counts,
    output: Path,
    label: str,
) -> None:
    summary_dir = (
        output
        / "summaries"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    channel_weights, _ = (
        aggregate_channel_weights(
            weighted_counts
        )
    )

    with (
        summary_dir
        / "weighted_counts_by_species.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "stage",
                "group",
                "species",
                "macroparticles",
                "weight_sum",
            )
        )

        keys = sorted(
            set(weighted_counts)
            | set(macro_counts)
        )

        for key in keys:
            stage, group, species = key

            writer.writerow(
                (
                    stage,
                    group,
                    species,
                    macro_counts.get(
                        key,
                        0,
                    ),
                    weighted_counts.get(
                        key,
                        0.0,
                    ),
                )
            )

    fraction_rows = []

    for scope in SCOPES:
        values = {}

        for group in (
            "LL",
            "BH",
            "BW",
        ):
            if scope == "combined":
                values[group] = (
                    channel_weights[
                        ("inside", group)
                    ]
                    + channel_weights[
                        ("outside", group)
                    ]
                )

            else:
                values[group] = (
                    channel_weights[
                        (scope, group)
                    ]
                )

        total = sum(
            values.values()
        )

        for group in (
            "LL",
            "BH",
            "BW",
        ):
            percent = (
                100.0
                * values[group]
                / total
                if total > 0.0
                else float("nan")
            )

            fraction_rows.append(
                (
                    scope,
                    group,
                    values[group],
                    total,
                    percent,
                )
            )

    with (
        summary_dir
        / "weighted_ipc_fractions_percent.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "scope",
                "channel",
                "channel_weight",
                "total_ipc_weight",
                "weighted_fraction_percent",
            )
        )

        writer.writerows(
            fraction_rows
        )

    lookup = {
        (scope, group): percent
        for (
            scope,
            group,
            _weight,
            _total,
            percent,
        ) in fraction_rows
    }

    x = np.arange(3)
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(9.0, 5.5)
    )

    for i, group in enumerate(
        (
            "LL",
            "BH",
            "BW",
        )
    ):
        values = [
            lookup[(scope, group)]
            for scope in SCOPES
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=group,
            color=GROUP_COLOR[group],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCOPES)

    ax.set_ylabel(
        "Weighted IPC fraction [%]"
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.set_title(
        f"{label} — weighted IPC channel fractions"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    save_figure(
        fig,
        (
            summary_dir
            / "weighted_ipc_fractions_percent.png"
        ),
        180,
    )



def scope_species_value(
    counts: dict[tuple[str, str, str], float | int],
    scope: str,
    group: str,
    species: str,
) -> float:
    """Return one species count for inside, outside, combined, or production."""
    if scope in STAGES:
        return float(
            counts.get(
                (scope, group, species),
                0.0,
            )
        )

    if scope in {"combined", "production"}:
        return float(
            counts.get(
                ("inside", group, species),
                0.0,
            )
            + counts.get(
                ("outside", group, species),
                0.0,
            )
        )

    raise ValueError(
        f"Unsupported IPC analysis scope: {scope}"
    )


def pair_equivalent_value(
    counts: dict[tuple[str, str, str], float | int],
    scope: str,
    group: str,
) -> tuple[float, float, float]:
    """
    Return electron, positron, and pair-equivalent yields.

    One physical pair contains one electron and one positron. The robust
    pair-equivalent estimate is therefore 0.5 * (electron + positron), while
    both charge-specific values are retained in the CSV for balance checks.
    """
    electron = scope_species_value(
        counts,
        scope,
        group,
        f"ele_{group.lower()}",
    )

    positron = scope_species_value(
        counts,
        scope,
        group,
        f"pos_{group.lower()}",
    )

    pair_equivalent = 0.5 * (
        electron + positron
    )

    return (
        electron,
        positron,
        pair_equivalent,
    )


def plot_ipc_fraction_step(
    fractions: dict[str, float],
    scope: str,
    label: str,
    path: Path,
    dpi: int,
) -> None:
    """Plot BW, BH, and LL fractions in the requested staircase style."""
    values = np.asarray(
        [
            fractions[group]
            for group in IPC_CHANNEL_ORDER
        ],
        dtype=float,
    )

    plot_values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    x_edges = np.arange(
        len(IPC_CHANNEL_ORDER) + 1,
        dtype=float,
    )

    y_step = np.concatenate(
        (
            plot_values,
            plot_values[-1:],
        )
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 6.0)
    )

    ax.step(
        x_edges,
        y_step,
        where="post",
        color="red",
        linewidth=1.6,
    )

    ax.set_xlim(
        0.0,
        float(len(IPC_CHANNEL_ORDER)),
    )

    finite_values = plot_values[
        np.isfinite(plot_values)
    ]

    ymax = max(
        100.0,
        float(np.max(finite_values)) * 1.18
        if finite_values.size
        else 100.0,
    )

    ax.set_ylim(
        0.0,
        ymax,
    )

    ax.set_xticks(
        np.arange(
            len(IPC_CHANNEL_ORDER),
            dtype=float,
        )
        + 0.5
    )

    ax.set_xticklabels(
        [
            IPC_CHANNEL_LONG_NAME[group]
            for group in IPC_CHANNEL_ORDER
        ],
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "Weighted pair fraction (%)"
    )

    ax.set_title(
        f"{label} — IPC channel fractions — {scope}"
    )

    label_offset = max(
        1.5,
        0.025 * ymax,
    )

    for index, value in enumerate(values):
        text_value = (
            f"{value:.2f}"
            if np.isfinite(value)
            else "n/a"
        )

        ax.text(
            index + 0.5,
            plot_values[index] + label_offset,
            text_value,
            color="red",
            rotation=90,
            ha="center",
            va="bottom",
        )

    ax.grid(
        True,
        axis="y",
        alpha=0.2,
    )

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_pair_multiplicity_bar(
    multiplicity: dict[str, float],
    scope: str,
    label: str,
    path: Path,
    dpi: int,
) -> None:
    """
    Draw the pair-multiplicity summary in the reference overlay style.

    The diagnostics provide one weighted multiplicity estimate per channel,
    not an event-resolved multiplicity distribution. Each estimate is therefore
    drawn as a narrow, peak-normalized display profile. The profile center is
    the computed value; its width has no statistical interpretation.
    """
    channel_values = {
        group: max(
            0.0,
            float(multiplicity.get(group, 0.0)),
        )
        for group in IPC_CHANNEL_ORDER
    }

    all_pairs = sum(
        channel_values.values()
    )

    curves = (
        ("IPC_ALL", "All pairs", all_pairs),
        ("LL", IPC_CHANNEL_LONG_NAME["LL"], channel_values["LL"]),
        ("BH", IPC_CHANNEL_LONG_NAME["BH"], channel_values["BH"]),
        ("BW", IPC_CHANNEL_LONG_NAME["BW"], channel_values["BW"]),
    )

    largest = max(
        1.0,
        *(value for _group, _legend, value in curves),
    )

    x_max = max(
        1.0,
        1.18 * largest,
    )

    x_edges = np.linspace(
        0.0,
        x_max,
        321,
    )

    x_centers = 0.5 * (
        x_edges[:-1]
        + x_edges[1:]
    )

    fig, ax = plt.subplots(
        figsize=(8.4, 6.4)
    )

    peak_height = 0.098

    for group, legend, center in curves:
        # A narrow Gaussian is used only to make the scalar estimate visible
        # as a histogram-like curve. Its center is the measured multiplicity.
        display_sigma = max(
            1.25 * (x_edges[1] - x_edges[0]),
            0.035 * np.sqrt(max(center, 1.0)) * np.sqrt(x_max),
        )

        profile = np.exp(
            -0.5
            * (
                (x_centers - center)
                / display_sigma
            ) ** 2
        )

        if np.max(profile) > 0.0:
            profile *= (
                peak_height
                / np.max(profile)
            )

        ax.step(
            x_centers,
            profile,
            where="mid",
            color=GROUP_COLOR[group],
            linewidth=1.7,
            label=legend,
        )

    ax.set_xlim(
        0.0,
        x_max,
    )

    ax.set_ylim(
        0.0,
        0.13,
    )

    ax.set_xlabel(
        "Pair multiplicity",
        fontsize=13,
    )

    ax.set_ylabel(
        "Events (a.u.)",
        fontsize=13,
    )

    ax.text(
        0.0,
        1.035,
        "FCC-ee",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    ax.text(
        0.145,
        1.035,
        "WarpX simulation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontstyle="italic",
    )

    ax.text(
        1.0,
        1.035,
        f"{scope.capitalize()} sample",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=11,
    )

    ax.minorticks_on()
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.tick_params(
        which="major",
        length=7,
    )
    ax.tick_params(
        which="minor",
        length=4,
    )

    ax.text(
        0.5,
        -0.16,
        (
            "Peak positions are computed weighted multiplicities; "
            "curve widths are display-only. "
            f"{label}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )

    save_figure(
        fig,
        path,
        dpi,
    )


def plot_pair_multiplicity_event_scan(
    multiplicity: dict[str, float],
    scope: str,
    label: str,
    event_scan_max: int,
    path: Path,
    dpi: int,
) -> None:
    """
    Plot the linear cumulative expectation from the measured pairs/event.

    This is not an event-by-event fluctuation distribution. It is the expected
    cumulative yield for N statistically equivalent events.
    """
    events = np.arange(
        event_scan_max + 1,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for group in IPC_CHANNEL_ORDER:
        per_event = multiplicity[group]

        ax.plot(
            events,
            events * per_event,
            linewidth=1.6,
            color=GROUP_COLOR[group],
            label=(
                f"{group}: "
                f"{per_event:.6g} pairs/event"
            ),
        )

    ax.set_xlabel(
        "Number of events"
    )

    ax.set_ylabel(
        "Expected cumulative weighted pairs"
    )

    ax.set_title(
        f"{label} — IPC pair multiplicity vs number of events — {scope}"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    save_figure(
        fig,
        path,
        dpi,
    )


def write_ipc_pair_analysis(
    weighted_counts,
    macro_counts,
    production_weighted_counts,
    production_macro_counts,
    output: Path,
    label: str,
    number_of_events: int,
    event_scan_max: int,
    dpi: int,
) -> None:
    """
    Write fractions and pair multiplicities for inside, outside, combined,
    and production.

    Inside/outside use the active diagnostics selection for the target
    iteration. Combined is the sum of those active inside and outside samples.
    Production uses the complete creation records from the inside snapshot plus
    all BoundaryScraping batches through that iteration.
    """
    if number_of_events <= 0:
        raise ValueError(
            "--number-of-events must be positive"
        )

    if event_scan_max <= 0:
        raise ValueError(
            "--event-scan-max must be positive"
        )

    summary_dir = (
        output
        / "summaries"
        / "ipc_pair_analysis"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    fractions_by_scope = {}
    multiplicity_by_scope = {}

    for scope in IPC_ANALYSIS_SCOPES:
        source_weighted = (
            production_weighted_counts
            if scope == "production"
            else weighted_counts
        )

        source_macro = (
            production_macro_counts
            if scope == "production"
            else macro_counts
        )

        weighted_values = {}
        macro_values = {}
        charge_values = {}

        for group in IPC_CHANNEL_ORDER:
            (
                electron_weight,
                positron_weight,
                pair_weight,
            ) = pair_equivalent_value(
                source_weighted,
                scope,
                group,
            )

            (
                electron_macro,
                positron_macro,
                pair_macro,
            ) = pair_equivalent_value(
                source_macro,
                scope,
                group,
            )

            weighted_values[group] = pair_weight
            macro_values[group] = pair_macro
            charge_values[group] = (
                electron_weight,
                positron_weight,
                electron_macro,
                positron_macro,
            )

        weighted_total = sum(
            weighted_values.values()
        )

        macro_total = sum(
            macro_values.values()
        )

        fractions = {}
        multiplicity = {}

        for group in IPC_CHANNEL_ORDER:
            fraction = (
                100.0
                * weighted_values[group]
                / weighted_total
                if weighted_total > 0.0
                else float("nan")
            )

            pairs_per_event = (
                weighted_values[group]
                / float(number_of_events)
            )

            fractions[group] = fraction
            multiplicity[group] = pairs_per_event

            (
                electron_weight,
                positron_weight,
                electron_macro,
                positron_macro,
            ) = charge_values[group]

            rows.append(
                (
                    scope,
                    group,
                    number_of_events,
                    electron_macro,
                    positron_macro,
                    macro_values[group],
                    electron_weight,
                    positron_weight,
                    weighted_values[group],
                    macro_total,
                    weighted_total,
                    fraction,
                    pairs_per_event,
                    electron_weight - positron_weight,
                )
            )

        fractions_by_scope[scope] = fractions
        multiplicity_by_scope[scope] = multiplicity

    with (
        summary_dir
        / "ipc_pair_yields_fractions_multiplicity.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "scope",
                "channel",
                "number_of_events",
                "electron_macroparticles",
                "positron_macroparticles",
                "pair_equivalent_macroparticles",
                "electron_weight_sum",
                "positron_weight_sum",
                "weighted_pair_equivalent",
                "total_pair_equivalent_macroparticles",
                "total_weighted_pair_equivalent",
                "weighted_pair_fraction_percent",
                "weighted_pair_multiplicity_per_event",
                "electron_minus_positron_weight",
            )
        )

        writer.writerows(rows)

    event_rows = []

    for scope in IPC_ANALYSIS_SCOPES:
        for event_count in range(
            event_scan_max + 1
        ):
            for group in IPC_CHANNEL_ORDER:
                event_rows.append(
                    (
                        scope,
                        event_count,
                        group,
                        multiplicity_by_scope[scope][group],
                        (
                            event_count
                            * multiplicity_by_scope[scope][group]
                        ),
                    )
                )

    with (
        summary_dir
        / "pair_multiplicity_vs_number_of_events.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "scope",
                "event_count",
                "channel",
                "weighted_pairs_per_event",
                "expected_cumulative_weighted_pairs",
            )
        )

        writer.writerows(event_rows)

    for scope in IPC_ANALYSIS_SCOPES:
        plot_ipc_fraction_step(
            fractions_by_scope[scope],
            scope,
            label,
            (
                summary_dir
                / "fractions"
                / f"{scope}_ipc_fraction_step.png"
            ),
            dpi,
        )

        plot_pair_multiplicity_bar(
            multiplicity_by_scope[scope],
            scope,
            label,
            (
                summary_dir
                / "multiplicity"
                / f"{scope}_pair_multiplicity_per_event.png"
            ),
            dpi,
        )

        plot_pair_multiplicity_event_scan(
            multiplicity_by_scope[scope],
            scope,
            label,
            event_scan_max,
            (
                summary_dir
                / "multiplicity"
                / (
                    f"{scope}_pair_multiplicity_"
                    "vs_number_of_events.png"
                )
            ),
            dpi,
        )


def write_boundary_correction_summary(
    correction_stats: dict[
        tuple[str, str],
        dict[str, int],
    ],
    output: Path,
) -> None:
    summary_dir = output / "summaries"
    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for (
        source_face,
        species,
    ), record in sorted(
        correction_stats.items()
    ):
        rows.append(
            (
                source_face,
                species,
                *(
                    record[field]
                    for field in CORRECTION_STAT_FIELDS
                ),
            )
        )

    with (
        summary_dir
        / "boundary_correction_counts.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_face",
                "species",
                *CORRECTION_STAT_FIELDS,
            )
        )
        writer.writerows(rows)

    totals = empty_correction_record()

    for record in correction_stats.values():
        for field in CORRECTION_STAT_FIELDS:
            totals[field] += record[field]

    with (
        summary_dir
        / "boundary_correction_summary.txt"
    ).open("w") as handle:
        handle.write(
            "BoundaryScraping position correction\n"
        )
        handle.write(
            "Method: backward line-box intersection using momentum direction.\n"
        )
        handle.write(
            "Only outside chunks are modified; inside chunks are unchanged.\n"
        )
        handle.write(
            "Failed reconstructions remain at their original coordinates.\n\n"
        )

        for field in CORRECTION_STAT_FIELDS:
            handle.write(
                f"{field}={totals[field]}\n"
            )

        handle.write(
            f"box_lo_m={BOX_LO_M.tolist()}\n"
        )
        handle.write(
            f"box_hi_m={BOX_HI_M.tolist()}\n"
        )


def save_histograms_npz(
    hist1d,
    hist2d,
    phase,
    edges,
    edges2d,
    phase_x_edges,
    phase_y_edges,
    output,
):
    payload = {
        "phase_log10theta_edges": (
            phase_x_edges
        ),
        "phase_log10pt_MeV_edges": (
            phase_y_edges
        ),
    }

    for (
        group,
        axis,
    ), array in edges.items():
        payload[
            f"edges1d__{group}__{axis}"
        ] = array

    for (
        group,
        plane,
        axis,
    ), array in edges2d.items():
        payload[
            f"edges2d__{group}__{plane}__{axis}"
        ] = array

    for key, array in hist1d.items():
        payload[
            "hist1d__"
            + "__".join(key)
        ] = array

    for key, array in hist2d.items():
        payload[
            "hist2d__"
            + "__".join(key)
        ] = array

    for key, array in phase.items():
        payload[
            "phase__"
            + "__".join(key)
        ] = array

    np.savez_compressed(
        (
            output
            / "summaries"
            / "all_histograms.npz"
        ),
        **payload,
    )



def save_creation_histograms_npz(
    hist1d,
    hist2d,
    edges,
    edges2d,
    output,
):
    payload = {}

    for (
        group,
        field,
    ), array in edges.items():
        payload[
            f"creation_edges1d__{group}__{field}"
        ] = array

    for (
        group,
        plane,
        field,
    ), array in edges2d.items():
        payload[
            f"creation_edges2d__{group}__{plane}__{field}"
        ] = array

    for key, array in hist1d.items():
        payload[
            "creation_hist1d__"
            + "__".join(key)
        ] = array

    for key, array in hist2d.items():
        payload[
            "creation_hist2d__"
            + "__".join(key)
        ] = array

    summary_dir = (
        output
        / "summaries"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        (
            summary_dir
            / "creation_histograms.npz"
        ),
        **payload,
    )



# -----------------------------------------------------------------------------
# Differential-luminosity scan
# -----------------------------------------------------------------------------
def parse_energy_values_from_header(
    header_lines: list[str],
) -> list[float]:
    """Extract energy-bin values from WarpX reduced-diagnostic headers."""
    text = " ".join(header_lines)

    number = (
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[Ee][-+]?\d+)?"
    )

    unit_scale = {
        "eV": 1.0,
        "keV": 1.0e3,
        "MeV": 1.0e6,
        "GeV": 1.0e9,
        "TeV": 1.0e12,
    }

    values = []

    patterns = (
        # WarpX DifferentialLuminosity format:
        # [2]bin1=0(eV) [3]bin2=4.45312e+07(eV) ...
        rf"bin\d+\s*=\s*({number})\s*\(\s*"
        r"(eV|keV|MeV|GeV|TeV)\s*\)",

        rf"(?:E(?:cm|star|\*)?|energy|center|bin(?:_center)?)"
        rf"[^=:\d]{{0,30}}[=:]\s*({number})\s*"
        r"(eV|keV|MeV|GeV|TeV)\b",

        rf"[\[(]\s*({number})\s*"
        r"(eV|keV|MeV|GeV|TeV)\s*[\])]",

        rf"({number})\s*"
        r"(eV|keV|MeV|GeV|TeV)\b",
    )

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        candidate = []

        for value_text, unit_text in matches:
            canonical_unit = next(
                (
                    unit
                    for unit in unit_scale
                    if unit.lower() == unit_text.lower()
                ),
                None,
            )

            if canonical_unit is None:
                continue

            candidate.append(
                float(value_text)
                * unit_scale[canonical_unit]
            )

        if candidate:
            values = candidate
            break

    unique = []

    for value in values:
        if not np.isfinite(value):
            continue

        if not unique or not np.isclose(
            value,
            unique[-1],
            rtol=1.0e-12,
            atol=0.0,
        ):
            unique.append(value)

    return unique


def centers_to_edges(
    centers: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(
        centers,
        dtype=float,
    )

    if centers.size == 1:
        return np.asarray(
            [
                centers[0] - 0.5,
                centers[0] + 0.5,
            ],
            dtype=float,
        )

    if np.any(
        np.diff(centers) <= 0.0
    ):
        raise ValueError(
            "Luminosity energy centers are not strictly increasing"
        )

    edges = np.empty(
        centers.size + 1,
        dtype=float,
    )

    edges[1:-1] = 0.5 * (
        centers[:-1]
        + centers[1:]
    )

    edges[0] = (
        centers[0]
        - 0.5 * (
            centers[1]
            - centers[0]
        )
    )

    edges[-1] = (
        centers[-1]
        + 0.5 * (
            centers[-1]
            - centers[-2]
        )
    )

    return edges


def luminosity_energy_edges(
    header_lines: list[str],
    number_of_bins: int,
    bin_min_ev: float | None,
    bin_max_ev: float | None,
    bin_width_ev: float | None,
) -> np.ndarray:
    """Resolve energy-bin edges in eV from CLI values or the file header."""
    if number_of_bins <= 0:
        raise ValueError(
            "Luminosity diagnostic has no energy-bin columns"
        )

    if bin_width_ev is not None:
        if not bin_width_ev > 0.0:
            raise ValueError(
                "--lumi-bin-width-ev must be positive"
            )

        start = (
            float(bin_min_ev)
            if bin_min_ev is not None
            else 0.0
        )

        return start + np.arange(
            number_of_bins + 1,
            dtype=float,
        ) * float(bin_width_ev)

    if (
        bin_min_ev is not None
        or bin_max_ev is not None
    ):
        if (
            bin_min_ev is None
            or bin_max_ev is None
        ):
            raise ValueError(
                "Use both --lumi-bin-min-ev and --lumi-bin-max-ev"
            )

        if not bin_max_ev > bin_min_ev:
            raise ValueError(
                "--lumi-bin-max-ev must exceed --lumi-bin-min-ev"
            )

        return np.linspace(
            float(bin_min_ev),
            float(bin_max_ev),
            number_of_bins + 1,
        )

    parsed = np.asarray(
        parse_energy_values_from_header(
            header_lines
        ),
        dtype=float,
    )

    if parsed.size == number_of_bins + 1:
        if np.all(
            np.diff(parsed) > 0.0
        ):
            return parsed

    if parsed.size == number_of_bins:
        return centers_to_edges(parsed)

    raise ValueError(
        "Could not determine the DifferentialLuminosity energy-bin width. "
        "Pass --lumi-bin-min-ev and --lumi-bin-max-ev, or "
        "--lumi-bin-width-ev."
    )


def read_luminosity_scan(
    path: Path,
    bin_min_ev: float | None,
    bin_max_ev: float | None,
    bin_width_ev: float | None,
):
    header_lines = []

    with path.open("r") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                header_lines.append(
                    line.strip()
                )

    data = np.genfromtxt(
        path,
        comments="#",
        dtype=float,
        invalid_raise=False,
    )

    if data.size == 0:
        raise RuntimeError(
            f"Luminosity file has no numeric rows: {path}"
        )

    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise RuntimeError(
            "DifferentialLuminosity requires step, time, and at least "
            f"one energy bin: {path}"
        )

    finite_row = (
        np.isfinite(data[:, 0])
        & np.isfinite(data[:, 1])
        & np.all(
            np.isfinite(data[:, 2:]),
            axis=1,
        )
    )

    data = data[finite_row]

    if data.size == 0:
        raise RuntimeError(
            f"Luminosity file has no complete finite rows: {path}"
        )

    order = np.argsort(
        data[:, 0],
        kind="stable",
    )

    data = data[order]

    # Keep the final occurrence if a restarted run wrote the same step twice.
    last_index = {}

    for index, step in enumerate(data[:, 0]):
        last_index[int(round(step))] = index

    keep = np.asarray(
        sorted(last_index.values()),
        dtype=int,
    )

    data = data[keep]

    iterations = np.rint(
        data[:, 0]
    ).astype(int)

    time_s = data[:, 1]
    differential = data[:, 2:]

    energy_edges_ev = luminosity_energy_edges(
        header_lines,
        differential.shape[1],
        bin_min_ev,
        bin_max_ev,
        bin_width_ev,
    )

    bin_widths_ev = np.diff(
        energy_edges_ev
    )

    cumulative_luminosity_m2_inv = np.sum(
        differential
        * bin_widths_ev[None, :],
        axis=1,
    )

    luminosity_per_step_m2_inv = np.diff(
        cumulative_luminosity_m2_inv,
        prepend=0.0,
    )

    integrated_from_steps_m2_inv = np.cumsum(
        luminosity_per_step_m2_inv
    )

    return {
        "iterations": iterations,
        "time_s": time_s,
        "energy_edges_ev": energy_edges_ev,
        "differential": differential,
        "cumulative_luminosity_m2_inv": (
            cumulative_luminosity_m2_inv
        ),
        "luminosity_per_step_m2_inv": (
            luminosity_per_step_m2_inv
        ),
        "integrated_from_steps_m2_inv": (
            integrated_from_steps_m2_inv
        ),
    }


def plot_luminosity_curve(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    dpi: int,
    step: bool,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8.8, 5.5)
    )

    if step:
        ax.step(
            x,
            y,
            where="mid",
            linewidth=1.5,
        )
    else:
        ax.plot(
            x,
            y,
            linewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 3),
    )

    save_figure(
        fig,
        path,
        dpi,
    )


def write_luminosity_scan(
    luminosity_file: Path,
    output: Path,
    label: str,
    dpi: int,
    bin_min_ev: float | None,
    bin_max_ev: float | None,
    bin_width_ev: float | None,
) -> None:
    scan = read_luminosity_scan(
        luminosity_file,
        bin_min_ev,
        bin_max_ev,
        bin_width_ev,
    )

    luminosity_dir = (
        output
        / "luminosity"
    )

    luminosity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        luminosity_dir
        / "luminosity_scan.csv"
    ).open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            (
                "iteration",
                "time_s",
                "luminosity_increment_per_step_m^-2",
                "cumulative_integrated_luminosity_m^-2",
                "cumulative_sum_of_step_increments_m^-2",
            )
        )

        for row in zip(
            scan["iterations"],
            scan["time_s"],
            scan["luminosity_per_step_m2_inv"],
            scan["cumulative_luminosity_m2_inv"],
            scan["integrated_from_steps_m2_inv"],
        ):
            writer.writerow(row)

    np.savez_compressed(
        luminosity_dir
        / "luminosity_scan.npz",
        **scan,
    )

    plot_luminosity_curve(
        scan["iterations"],
        scan["cumulative_luminosity_m2_inv"],
        "Simulation iteration",
        r"Cumulative integrated luminosity [m$^{-2}$]",
        f"{label} — luminosity history",
        luminosity_dir
        / "luminosity_history.png",
        dpi,
        step=False,
    )

    plot_luminosity_curve(
        scan["iterations"],
        scan["luminosity_per_step_m2_inv"],
        "Simulation iteration",
        r"Luminosity increment per step [m$^{-2}$]",
        f"{label} — luminosity at each step",
        luminosity_dir
        / "luminosity_per_step.png",
        dpi,
        step=True,
    )

    plot_luminosity_curve(
        scan["iterations"],
        scan["integrated_from_steps_m2_inv"],
        "Simulation iteration",
        r"Integrated luminosity [m$^{-2}$]",
        f"{label} — integral of the per-step luminosity curve",
        luminosity_dir
        / "integrated_luminosity.png",
        dpi,
        step=False,
    )

    with (
        luminosity_dir
        / "luminosity_source.txt"
    ).open("w") as handle:
        handle.write(
            f"source={luminosity_file}\n"
        )
        handle.write(
            "The DifferentialLuminosity rows are treated as cumulative "
            "dL/dE spectra. Each row is integrated over the energy-bin "
            "widths to obtain cumulative luminosity in m^-2; consecutive "
            "differences give the luminosity increment at each step.\n"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one WarpX diagnostics directory and reproduce "
            "the IPC/beam plots separately for every available iteration."
        )
    )

    parser.add_argument(
        "--diags",
        default="diags",
        help=(
            "Single WarpX diagnostics directory containing "
            "particles_in* and particles_out*"
        ),
    )

    parser.add_argument(
        "--out",
        default="diags_iteration_analysis",
        help="Directory for the original, uncorrected analysis",
    )

    parser.add_argument(
        "--corrected-out",
        default=None,
        help=(
            "Directory for the boundary-corrected analysis. "
            "Default: <out>_corrected"
        ),
    )

    parser.add_argument(
        "--label",
        default="FCC-ee beam-beam diagnostics",
    )

    parser.add_argument(
        "--iterations",
        nargs="*",
        type=int,
        help=(
            "Optional subset of iterations, for example: "
            "--iterations 0 128 256"
        ),
    )

    parser.add_argument(
        "--outside-mode",
        choices=(
            "batch",
            "cumulative",
        ),
        default="batch",
        help=(
            "batch: use only particles_out written at that iteration; "
            "cumulative: include all particles_out batches through "
            "that iteration"
        ),
    )

    parser.add_argument(
        "--bins-1d",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--bins-2d",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--bins-phase-x",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--bins-phase-y",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--sample-capacity",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--range-low",
        type=float,
        default=0.001,
        help=(
            "Lower quantile used for common automatic "
            "position ranges"
        ),
    )

    parser.add_argument(
        "--range-high",
        type=float,
        default=0.999,
        help=(
            "Upper quantile used for common automatic "
            "position ranges"
        ),
    )

    parser.add_argument(
        "--logtheta-min",
        type=float,
        default=-6.0,
    )

    parser.add_argument(
        "--logtheta-max",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--logpt-min",
        type=float,
        default=-4.0,
        help=(
            "Minimum log10(pT [MeV/c])"
        ),
    )

    parser.add_argument(
        "--logpt-max",
        type=float,
        default=5.0,
        help=(
            "Maximum log10(pT [MeV/c])"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--number-of-events",
        type=int,
        default=1,
        help=(
            "Number of statistically independent events represented by "
            "the diagnostics. Used to normalize pair multiplicity."
        ),
    )

    parser.add_argument(
        "--event-scan-max",
        type=int,
        default=100,
        help=(
            "Maximum event count for the linear cumulative pair-yield scan"
        ),
    )

    parser.add_argument(
        "--luminosity-file",
        default=None,
        help=(
            "WarpX DifferentialLuminosity text file. Default: "
            "<diags>/reducedfiles/DiffLum_beam1_beam2.txt"
        ),
    )

    parser.add_argument(
        "--lumi-bin-min-ev",
        type=float,
        default=None,
        help=(
            "Minimum DifferentialLuminosity energy-bin edge in eV. "
            "Use with --lumi-bin-max-ev when the header lacks bin energies."
        ),
    )

    parser.add_argument(
        "--lumi-bin-max-ev",
        type=float,
        default=None,
        help=(
            "Maximum DifferentialLuminosity energy-bin edge in eV. "
            "Use with --lumi-bin-min-ev."
        ),
    )

    parser.add_argument(
        "--lumi-bin-width-ev",
        type=float,
        default=None,
        help=(
            "Explicit DifferentialLuminosity bin width in eV. Overrides "
            "header parsing and min/max spacing."
        ),
    )

    args = parser.parse_args()

    if args.number_of_events <= 0:
        raise ValueError(
            "--number-of-events must be positive"
        )

    if args.event_scan_max <= 0:
        raise ValueError(
            "--event-scan-max must be positive"
        )

    diags_dir = (
        Path(args.diags)
        .expanduser()
        .resolve()
    )

    output = (
        Path(args.out)
        .expanduser()
        .resolve()
    )

    corrected_output = (
        Path(args.corrected_out)
        .expanduser()
        .resolve()
        if args.corrected_out
        else output.with_name(
            output.name + "_corrected"
        )
    )

    if corrected_output == output:
        raise ValueError(
            "--out and --corrected-out must be different directories"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    corrected_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_files(
        diags_dir
    )

    discovered_iterations = (
        available_iterations(files)
    )

    if args.iterations:
        requested = sorted(
            set(args.iterations)
        )

        missing = sorted(
            set(requested)
            - set(discovered_iterations)
        )

        if missing:
            raise RuntimeError(
                "Requested iterations are unavailable: "
                f"{missing}. Available iterations: "
                f"{discovered_iterations}"
            )

        iterations = requested

    else:
        iterations = discovered_iterations

    print(
        f"Diagnostics directory: {diags_dir}"
    )

    print(
        f"HDF5 files discovered: {len(files)}"
    )

    print(
        "Iterations discovered: "
        f"{discovered_iterations}"
    )

    print(
        f"Iterations selected: {iterations}"
    )

    print(
        "BoundaryScraping mode: "
        f"{args.outside_mode}"
    )

    print(
        f"Original output: {output}"
    )

    print(
        f"Corrected output: {corrected_output}"
    )

    write_global_manifest(
        diags_dir,
        files,
        discovered_iterations,
        output,
    )

    write_global_manifest(
        diags_dir,
        files,
        discovered_iterations,
        corrected_output,
    )

    with (
        corrected_output
        / "summaries"
        / "correction_method.txt"
    ).open("w") as handle:
        handle.write(
            "Outside-particle positions are projected backward along "
            "their momentum direction to the first rectangular-box "
            "intersection. Inside-particle positions are unchanged.\n"
        )

    # Determine one common set of position ranges from every
    # file that can enter the requested iteration products.
    # This keeps axes identical across steps.
    range_files = unique_files(
        item
        for iteration in iterations
        for item in select_files_for_iteration(
            files,
            iteration,
            args.outside_mode,
        )
    )

    if not range_files:
        raise RuntimeError(
            "No files were selected for the requested iterations"
        )

    print(
        "Computing common plot ranges from "
        f"{len(range_files)} files..."
    )

    (
        edges,
        _unused_weighted,
        _unused_macro,
    ) = first_pass(
        files=range_files,
        sample_capacity=args.sample_capacity,
        qlow=args.range_low,
        qhigh=args.range_high,
        bins_1d=args.bins_1d,
        correct_outside=False,
    )

    print(
        "Computing corrected common plot ranges..."
    )

    (
        corrected_edges,
        _unused_corrected_weighted,
        _unused_corrected_macro,
    ) = first_pass(
        files=range_files,
        sample_capacity=args.sample_capacity,
        qlow=args.range_low,
        qhigh=args.range_high,
        bins_1d=args.bins_1d,
        correct_outside=True,
    )

    # Exact boundary hits must remain inside the corrected histograms even when
    # quantile-based range selection would otherwise trim the box walls.
    corrected_edges = expand_edges_to_include_box(
        corrected_edges
    )

    production_range_files = unique_files(
        item
        for iteration in iterations
        for item in select_files_for_iteration(
            files,
            iteration,
            "cumulative",
        )
    )

    print(
        "Computing common creation-coordinate and creation-time ranges..."
    )

    (
        creation_edges,
        creation_particles_in_range_pass,
    ) = first_pass_creation(
        files=production_range_files,
        sample_capacity=args.sample_capacity,
        qlow=args.range_low,
        qhigh=args.range_high,
        bins_1d=args.bins_1d,
    )

    if creation_particles_in_range_pass == 0:
        print(
            "[warn] no finite orig_x/orig_y/orig_z/creationTime "
            "records were found. Creation plots will be empty. "
            "Check the WarpX diagnostic variables for both "
            "particles_in and particles_out.",
            flush=True,
        )
    else:
        print(
            "Creation records found for "
            f"{creation_particles_in_range_pass} particle entries.",
            flush=True,
        )

    for iteration in iterations:
        selected = select_files_for_iteration(
            files,
            iteration,
            args.outside_mode,
        )

        if not selected:
            print(
                f"[warn] iteration {iteration}: "
                "no selected files; skipping",
                flush=True,
            )
            continue

        # Production is always evaluated from the active inside snapshot plus
        # every BoundaryScraping batch through this iteration. This avoids
        # losing particles created earlier when --outside-mode=batch.
        production_selected = select_files_for_iteration(
            files,
            iteration,
            "cumulative",
        )

        iteration_output = (
            output
            / f"iteration_{iteration:08d}"
        )

        iteration_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_iteration_manifest(
            selected,
            iteration=iteration,
            outside_mode=args.outside_mode,
            output=iteration_output,
        )

        exact_inside = sum(
            item.stage == "inside"
            and item.iteration == iteration
            for item in selected
        )

        selected_outside = sum(
            item.stage == "outside"
            for item in selected
        )

        print(
            f"\n[iteration {iteration}] "
            f"inside files={exact_inside}, "
            f"outside files={selected_outside}",
            flush=True,
        )

        (
            hist1d,
            hist2d,
            phase,
            edges2d,
            phase_x_edges,
            phase_y_edges,
        ) = initialize_histograms(
            edges=edges,
            bins_2d=args.bins_2d,
            bins_phase_x=args.bins_phase_x,
            bins_phase_y=args.bins_phase_y,
            logtheta_min=args.logtheta_min,
            logtheta_max=args.logtheta_max,
            logpt_min=args.logpt_min,
            logpt_max=args.logpt_max,
        )

        (
            weighted_counts,
            macro_counts,
        ) = second_pass(
            selected,
            edges,
            hist1d,
            hist2d,
            phase,
            edges2d,
            phase_x_edges,
            phase_y_edges,
        )

        (
            creation_hist1d,
            creation_hist2d,
            creation_edges2d,
        ) = initialize_creation_histograms(
            edges=creation_edges,
            bins_2d=args.bins_2d,
        )

        (
            creation_particles_this_iteration,
            production_weighted_counts,
            production_macro_counts,
        ) = second_pass_creation(
            files=production_selected,
            edges=creation_edges,
            hist1d=creation_hist1d,
            hist2d=creation_hist2d,
            edges2d=creation_edges2d,
        )

        if creation_particles_this_iteration == 0:
            print(
                f"[warn] iteration {iteration}: no complete "
                "creation-coordinate records were found",
                flush=True,
            )

        outside_description = (
            "outside batch"
            if args.outside_mode == "batch"
            else "outside cumulative"
        )

        iteration_label = (
            f"{args.label} — "
            f"iteration {iteration} — "
            f"{outside_description}"
        )

        generate_plots(
            hist1d,
            hist2d,
            phase,
            edges,
            edges2d,
            phase_x_edges,
            phase_y_edges,
            iteration_output,
            args.dpi,
            iteration_label,
        )

        if creation_particles_this_iteration > 0:
            generate_creation_plots(
                creation_hist1d,
                creation_hist2d,
                creation_edges,
                creation_edges2d,
                iteration_output,
                args.dpi,
                iteration_label,
            )

        write_summaries(
            weighted_counts,
            macro_counts,
            iteration_output,
            iteration_label,
        )

        write_ipc_pair_analysis(
            weighted_counts,
            macro_counts,
            production_weighted_counts,
            production_macro_counts,
            iteration_output,
            iteration_label,
            args.number_of_events,
            args.event_scan_max,
            args.dpi,
        )

        save_histograms_npz(
            hist1d,
            hist2d,
            phase,
            edges,
            edges2d,
            phase_x_edges,
            phase_y_edges,
            iteration_output,
        )

        if creation_particles_this_iteration > 0:
            save_creation_histograms_npz(
                creation_hist1d,
                creation_hist2d,
                creation_edges,
                creation_edges2d,
                iteration_output,
            )

        # -------------------------------------------------------------
        # Boundary-corrected copy of the complete analysis
        # -------------------------------------------------------------
        corrected_iteration_output = (
            corrected_output
            / f"iteration_{iteration:08d}"
        )

        corrected_iteration_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_iteration_manifest(
            selected,
            iteration=iteration,
            outside_mode=args.outside_mode,
            output=corrected_iteration_output,
        )

        (
            corrected_hist1d,
            corrected_hist2d,
            corrected_phase,
            corrected_edges2d,
            corrected_phase_x_edges,
            corrected_phase_y_edges,
        ) = initialize_histograms(
            edges=corrected_edges,
            bins_2d=args.bins_2d,
            bins_phase_x=args.bins_phase_x,
            bins_phase_y=args.bins_phase_y,
            logtheta_min=args.logtheta_min,
            logtheta_max=args.logtheta_max,
            logpt_min=args.logpt_min,
            logpt_max=args.logpt_max,
        )

        correction_stats: dict[
            tuple[str, str],
            dict[str, int],
        ] = {}

        (
            corrected_weighted_counts,
            corrected_macro_counts,
        ) = second_pass(
            selected,
            corrected_edges,
            corrected_hist1d,
            corrected_hist2d,
            corrected_phase,
            corrected_edges2d,
            corrected_phase_x_edges,
            corrected_phase_y_edges,
            correct_outside=True,
            correction_stats=correction_stats,
        )

        corrected_iteration_label = (
            iteration_label
            + " — boundary-corrected outside positions"
        )

        generate_plots(
            corrected_hist1d,
            corrected_hist2d,
            corrected_phase,
            corrected_edges,
            corrected_edges2d,
            corrected_phase_x_edges,
            corrected_phase_y_edges,
            corrected_iteration_output,
            args.dpi,
            corrected_iteration_label,
        )

        # Boundary correction changes only the recorded exit position.
        # Creation coordinates and creation time are copied unchanged.
        if creation_particles_this_iteration > 0:
            generate_creation_plots(
                creation_hist1d,
                creation_hist2d,
                creation_edges,
                creation_edges2d,
                corrected_iteration_output,
                args.dpi,
                (
                    corrected_iteration_label
                    + " — creation records unchanged"
                ),
            )

        write_summaries(
            corrected_weighted_counts,
            corrected_macro_counts,
            corrected_iteration_output,
            corrected_iteration_label,
        )

        write_ipc_pair_analysis(
            corrected_weighted_counts,
            corrected_macro_counts,
            production_weighted_counts,
            production_macro_counts,
            corrected_iteration_output,
            corrected_iteration_label,
            args.number_of_events,
            args.event_scan_max,
            args.dpi,
        )

        save_histograms_npz(
            corrected_hist1d,
            corrected_hist2d,
            corrected_phase,
            corrected_edges,
            corrected_edges2d,
            corrected_phase_x_edges,
            corrected_phase_y_edges,
            corrected_iteration_output,
        )

        if creation_particles_this_iteration > 0:
            save_creation_histograms_npz(
                creation_hist1d,
                creation_hist2d,
                creation_edges,
                creation_edges2d,
                corrected_iteration_output,
            )

        write_boundary_correction_summary(
            correction_stats,
            corrected_iteration_output,
        )

    luminosity_file = (
        Path(args.luminosity_file)
        .expanduser()
        .resolve()
        if args.luminosity_file
        else (
            diags_dir
            / "reducedfiles"
            / "DiffLum_beam1_beam2.txt"
        )
    )

    if luminosity_file.is_file():
        try:
            write_luminosity_scan(
                luminosity_file,
                output,
                args.label,
                args.dpi,
                args.lumi_bin_min_ev,
                args.lumi_bin_max_ev,
                args.lumi_bin_width_ev,
            )

            write_luminosity_scan(
                luminosity_file,
                corrected_output,
                args.label,
                args.dpi,
                args.lumi_bin_min_ev,
                args.lumi_bin_max_ev,
                args.lumi_bin_width_ev,
            )

        except Exception as exc:
            print(
                "[warn] luminosity scan was not generated: "
                f"{exc}",
                flush=True,
            )

    else:
        print(
            "[warn] DifferentialLuminosity file not found: "
            f"{luminosity_file}",
            flush=True,
        )

    print("\nDone.")

    print(
        f"Iterations analyzed: {iterations}"
    )

    print(
        f"Original output: {output}"
    )

    print(
        f"Corrected output: {corrected_output}"
    )


if __name__ == "__main__":
    main()
