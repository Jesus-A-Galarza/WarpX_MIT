#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# =============================================================================
# FIXED CONFIGURATION
# =============================================================================

# The script is intended to be run from the directory that contains `diags`.
SOURCE_DIAGS = Path("diags")

# A complete copy of SOURCE_DIAGS is written here. The source is never opened
# in write mode.
OUTPUT_DIAGS = Path("diags_corrected_boosted")

# Full beam-beam crossing angle and common boost direction.
FULL_CROSSING_ANGLE_MRAD = 30.0
BOOST_SIGN_X = +1

# Only these species receive the momentum boost. BoundaryScraping position
# correction is attempted for every particle species that has usable position
# and momentum records in the selected particles_out files.
IPC_SPECIES = (
    "ele_ll",
    "pos_ll",
    "ele_bh",
    "pos_bh",
    "ele_bw",
    "pos_bw",
)

# Number of particle entries processed at a time.
CHUNK_SIZE = 1_000_000

# Physical constants in SI units.
C_LIGHT = 299_792_458.0
M_ELECTRON = 9.109_383_7139e-31

# Simulation box. These are total lengths, matching the plotting script.
BOX_TOTAL_SIZE_PLOT_UNITS = {
    "x": 2083.55,   # micrometres
    "y": 26.8711,   # micrometres
    "z": 267.170,   # millimetres
}

BOX_CENTER_PLOT_UNITS = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

AXIS_SCALE = {
    "x": 1.0e6,  # m -> micrometres
    "y": 1.0e6,  # m -> micrometres
    "z": 1.0e3,  # m -> millimetres
}

AXES = ("x", "y", "z")
BOUNDARY_FACES = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
FACE_TO_CODE = {face: code for code, face in enumerate(BOUNDARY_FACES)}

BOX_LO_M = np.asarray(
    [
        (
            BOX_CENTER_PLOT_UNITS[axis]
            - 0.5 * BOX_TOTAL_SIZE_PLOT_UNITS[axis]
        )
        / AXIS_SCALE[axis]
        for axis in AXES
    ],
    dtype=np.float64,
)

BOX_HI_M = np.asarray(
    [
        (
            BOX_CENTER_PLOT_UNITS[axis]
            + 0.5 * BOX_TOTAL_SIZE_PLOT_UNITS[axis]
        )
        / AXIS_SCALE[axis]
        for axis in AXES
    ],
    dtype=np.float64,
)

CORRECTION_STAT_FIELDS = (
    "total_entries",
    "geometrically_outside",
    "corrected",
    "already_on_boundary",
    "inside_unchanged",
    "failed_unchanged",
    "source_face_mismatch",
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class ComponentRef:
    """Reference to an openPMD component and its containing HDF5 group."""

    parent: h5py.Group
    name: str
    obj: h5py.Dataset | h5py.Group


@dataclass
class TemporaryWriter:
    """Temporary dataset used to make one component replacement atomic."""

    parent: h5py.Group
    original_name: str
    temporary_name: str
    dataset: h5py.Dataset

    def commit(self) -> None:
        if self.original_name in self.parent:
            del self.parent[self.original_name]
        self.parent.move(self.temporary_name, self.original_name)

    def discard(self) -> None:
        if self.temporary_name in self.parent:
            del self.parent[self.temporary_name]


# =============================================================================
# DIAGNOSTIC DISCOVERY
# =============================================================================

def iteration_number(path: Path) -> int:
    """Extract the final integer from an HDF5 filename."""
    match = re.search(r"(\d+)(?=\.h5$)", path.name)
    return int(match.group(1)) if match else -1


def latest_h5_file(directory: Path) -> Path | None:
    """Return the HDF5 file with the largest filename iteration."""
    files = list(directory.glob("*.h5"))
    if not files:
        return None

    return max(
        files,
        key=lambda path: (
            iteration_number(path),
            path.name,
        ),
    )


def diagnostic_files_to_modify(diags_root: Path) -> list[Path]:
    """
    Select only the latest diagnostic file from each relevant directory.

    This preserves the behavior of the earlier momentum-only script:

      * latest file in each top-level particles_in* directory;
      * latest file in every directory at or below each particles_out* tree.
    """
    selected: list[Path] = []

    for top in sorted(diags_root.iterdir()):
        if not top.is_dir():
            continue

        if top.name.startswith("particles_in"):
            latest = latest_h5_file(top)
            if latest is not None:
                selected.append(latest)

        elif top.name.startswith("particles_out"):
            directories = [top]
            directories.extend(
                sorted(path for path in top.rglob("*") if path.is_dir())
            )

            for directory in directories:
                latest = latest_h5_file(directory)
                if latest is not None:
                    selected.append(latest)

    return list(dict.fromkeys(selected))


def is_outside_file(path: Path, diags_root: Path) -> bool:
    """Return True when the file belongs to a particles_out* diagnostic."""
    relative = path.relative_to(diags_root)
    return any(part.startswith("particles_out") for part in relative.parts)


def boundary_face_from_path(path: Path) -> str | None:
    """Extract xlo/xhi/ylo/yhi/zlo/zhi from a BoundaryScraping path."""
    for part in path.parts:
        match = re.fullmatch(r"particles_at_([xyz](?:lo|hi))", part)
        if match:
            return match.group(1)
    return None


def latest_particles_groups(h5file: h5py.File) -> list[h5py.Group]:
    """Return /data/<largest_iteration>/particles groups in one HDF5 file."""
    found: list[tuple[int, str]] = []

    def visitor(name: str, obj) -> None:
        if not isinstance(obj, h5py.Group):
            return

        match = re.fullmatch(r"data/(\d+)/particles", name)
        if match:
            found.append((int(match.group(1)), name))

    h5file.visititems(visitor)

    if not found:
        return []

    largest_iteration = max(iteration for iteration, _ in found)
    return [
        h5file[name]
        for iteration, name in found
        if iteration == largest_iteration
    ]


# =============================================================================
# OPENPMD COMPONENT ACCESS
# =============================================================================

def get_unit_si(component: h5py.Dataset | h5py.Group) -> float:
    """Read and validate the scalar openPMD unitSI attribute."""
    value = np.asarray(component.attrs.get("unitSI", 1.0))
    if value.size != 1:
        raise ValueError(f"Invalid unitSI attribute on {component.name}")

    unit = float(value.reshape(-1)[0])
    if not np.isfinite(unit) or unit == 0.0:
        raise ValueError(f"Invalid unitSI={unit} on {component.name}")

    return unit


def component_shape(component: h5py.Dataset | h5py.Group) -> tuple[int, ...]:
    """Return the logical shape of a normal or constant openPMD component."""
    if isinstance(component, h5py.Dataset):
        return tuple(int(value) for value in component.shape)

    if isinstance(component, h5py.Group):
        if "shape" not in component.attrs:
            raise ValueError(
                f"Constant component has no shape attribute: {component.name}"
            )
        return tuple(
            int(value)
            for value in np.asarray(component.attrs["shape"]).reshape(-1)
        )

    raise TypeError(f"Unsupported HDF5 object: {component.name}")


def component_length(component: h5py.Dataset | h5py.Group) -> int:
    shape = component_shape(component)
    if len(shape) != 1:
        raise ValueError(
            f"Expected one-dimensional particle component at {component.name}; "
            f"found shape {shape}"
        )
    return shape[0]


def locate_component(
    species_group: h5py.Group,
    record: str,
    component: str,
    *,
    required: bool,
) -> ComponentRef | None:
    """Locate common nested and flat WarpX/openPMD component encodings."""
    if record in species_group and isinstance(species_group[record], h5py.Group):
        record_group = species_group[record]
        if component in record_group:
            return ComponentRef(
                parent=record_group,
                name=component,
                obj=record_group[component],
            )

    flat_names: tuple[str, ...]
    if record == "position":
        flat_names = (component,)
    elif record == "positionOffset":
        flat_names = (f"positionOffset_{component}",)
    elif record == "momentum":
        flat_names = (f"u{component}", f"p{component}")
    else:
        flat_names = (f"{record}_{component}",)

    for name in flat_names:
        if name in species_group:
            return ComponentRef(
                parent=species_group,
                name=name,
                obj=species_group[name],
            )

    if required:
        raise KeyError(
            f"Missing {record}/{component} component in {species_group.name}"
        )
    return None


def locate_production_momentum_component(
    species_group: h5py.Group,
    component: str,
    *,
    required: bool,
) -> ComponentRef | None:
    """Locate origUx/origUy/origUz and common spelling variants.

    These runtime attributes store proper-velocity components gamma*v.
    The diagnostics used here write the flat camel-case names origUx, origUy
    and origUz, but underscore and vector-like encodings are also accepted.
    """
    component = component.lower()

    aliases = (
        f"origU{component}",          # origUx
        f"origU{component.upper()}",  # origUX
        f"orig_u{component}",         # orig_ux
        f"orig_u_{component}",        # orig_u_x
        f"orig_u{component.upper()}", # orig_uX
    )

    for name in aliases:
        if name in species_group:
            return ComponentRef(
                parent=species_group,
                name=name,
                obj=species_group[name],
            )

    for record_name in ("orig_u", "origU"):
        if (
            record_name in species_group
            and isinstance(species_group[record_name], h5py.Group)
        ):
            record_group = species_group[record_name]
            for name in (component, component.upper()):
                if name in record_group:
                    return ComponentRef(
                        parent=record_group,
                        name=name,
                        obj=record_group[name],
                    )

    if required:
        raise KeyError(
            f"Missing production momentum component origU{component} "
            f"in {species_group.name}"
        )

    return None


def read_stored_chunk(
    component: h5py.Dataset | h5py.Group,
    start: int,
    stop: int,
) -> np.ndarray:
    """Read values in stored units, expanding a constant component by chunk."""
    if isinstance(component, h5py.Dataset):
        return np.asarray(component[start:stop], dtype=np.float64)

    if isinstance(component, h5py.Group):
        if "value" not in component.attrs:
            raise ValueError(
                f"Constant component has no value attribute: {component.name}"
            )
        constant_value = float(
            np.asarray(component.attrs["value"]).reshape(-1)[0]
        )
        return np.full(stop - start, constant_value, dtype=np.float64)

    raise TypeError(f"Unsupported component type: {component.name}")


def read_si_chunk(
    component: h5py.Dataset | h5py.Group,
    start: int,
    stop: int,
) -> np.ndarray:
    return read_stored_chunk(component, start, stop) * get_unit_si(component)


def read_optional_si_chunk(
    component_ref: ComponentRef | None,
    start: int,
    stop: int,
) -> np.ndarray:
    if component_ref is None:
        return np.zeros(stop - start, dtype=np.float64)
    return read_si_chunk(component_ref.obj, start, stop)


def dataset_creation_options(
    original: h5py.Dataset | h5py.Group,
    number_of_particles: int,
) -> dict:
    """Choose safe creation options for a temporary replacement dataset."""
    options: dict = {}

    if number_of_particles > 0:
        options["chunks"] = (min(CHUNK_SIZE, number_of_particles),)

    if isinstance(original, h5py.Dataset):
        if original.compression is not None:
            options["compression"] = original.compression
            options["compression_opts"] = original.compression_opts
        if original.shuffle:
            options["shuffle"] = True
        if original.fletcher32:
            options["fletcher32"] = True

    return options


def create_temporary_writer(
    component_ref: ComponentRef,
    number_of_particles: int,
    suffix: str,
    *,
    force_float64: bool = False,
) -> TemporaryWriter:
    """Create a temporary dataset beside an existing component."""
    parent = component_ref.parent
    original = component_ref.obj
    temporary_name = f"__{component_ref.name}_{suffix}_tmp"

    if temporary_name in parent:
        del parent[temporary_name]

    if force_float64 or isinstance(original, h5py.Group):
        dtype = np.dtype(np.float64)
    else:
        dtype = original.dtype
        if not np.issubdtype(dtype, np.floating):
            dtype = np.dtype(np.float64)

    writer = parent.create_dataset(
        temporary_name,
        shape=(number_of_particles,),
        dtype=dtype,
        **dataset_creation_options(original, number_of_particles),
    )

    # Preserve openPMD metadata except the attributes that define a constant
    # component. The logical shape is represented by the new dataset itself.
    for attribute_name, attribute_value in original.attrs.items():
        if attribute_name not in {"value", "shape"}:
            writer.attrs[attribute_name] = attribute_value

    return TemporaryWriter(
        parent=parent,
        original_name=component_ref.name,
        temporary_name=temporary_name,
        dataset=writer,
    )


def validate_equal_lengths(
    named_components: Iterable[tuple[str, ComponentRef | None]],
    expected: int | None = None,
) -> int:
    """Validate component lengths and return the common particle count."""
    lengths: dict[str, int] = {}

    for name, ref in named_components:
        if ref is not None:
            lengths[name] = component_length(ref.obj)

    if expected is not None:
        for name, length in lengths.items():
            if length != expected:
                raise ValueError(
                    f"Component length mismatch for {name}: {length} != {expected}"
                )
        return expected

    unique = set(lengths.values())
    if not unique:
        return 0
    if len(unique) != 1:
        raise ValueError(f"Particle component lengths differ: {lengths}")
    return next(iter(unique))


# =============================================================================
# BOUNDARY POSITION CORRECTION
# =============================================================================

def empty_correction_record() -> dict[str, int]:
    return {field: 0 for field in CORRECTION_STAT_FIELDS}


def reconstruct_boundary_positions(
    position_m: np.ndarray,
    momentum_si: np.ndarray,
    source_face: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Project recorded outside positions backward along the original momentum.

    Returns:
      corrected_position_m, correction_mask, statistics

    Failed candidates retain their original coordinates.
    """
    position = np.asarray(position_m, dtype=np.float64)
    momentum = np.asarray(momentum_si, dtype=np.float64)

    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"Invalid position array shape: {position.shape}")
    if momentum.shape != position.shape:
        raise ValueError(
            f"Momentum shape {momentum.shape} does not match position shape "
            f"{position.shape}"
        )

    number = position.shape[0]
    stats = empty_correction_record()
    stats["total_entries"] = number

    if number == 0:
        return position.copy(), np.zeros(0, dtype=bool), stats

    finite = (
        np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(momentum), axis=1)
    )

    momentum_scale = np.max(np.abs(momentum), axis=1)
    nonzero_momentum = finite & (momentum_scale > 0.0)

    # The direction is dimensionless. Normalizing by each row's largest
    # component avoids unstable line parameters for tiny SI momenta.
    direction = np.zeros_like(momentum)
    direction[nonzero_momentum] = (
        momentum[nonzero_momentum]
        / momentum_scale[nonzero_momentum, None]
    )

    tolerance = 1.0e-11 * float(np.max(BOX_HI_M - BOX_LO_M))

    inside_with_tolerance = (
        np.all(position >= BOX_LO_M - tolerance, axis=1)
        & np.all(position <= BOX_HI_M + tolerance, axis=1)
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
        & (np.min(distances_to_faces, axis=1) <= tolerance)
    )

    geometrically_outside = (
        finite
        & np.any(
            (position < BOX_LO_M - tolerance)
            | (position > BOX_HI_M + tolerance),
            axis=1,
        )
    )

    inside_unchanged = (
        finite
        & ~geometrically_outside
        & ~already_on_boundary
    )

    # For r(s) = r_recorded - s*direction, determine the interval of s for
    # which the line lies inside all three box slabs.
    s_enter = np.full(number, -np.inf, dtype=np.float64)
    s_exit = np.full(number, np.inf, dtype=np.float64)
    impossible_parallel = np.zeros(number, dtype=bool)

    for axis in range(3):
        coordinate = position[:, axis]
        component = direction[:, axis]
        moving = component != 0.0
        parallel = ~moving

        impossible_parallel |= (
            parallel
            & (
                (coordinate < BOX_LO_M[axis] - tolerance)
                | (coordinate > BOX_HI_M[axis] + tolerance)
            )
        )

        axis_enter = np.full(number, -np.inf, dtype=np.float64)
        axis_exit = np.full(number, np.inf, dtype=np.float64)

        s_at_lo = (
            coordinate[moving] - BOX_LO_M[axis]
        ) / component[moving]
        s_at_hi = (
            coordinate[moving] - BOX_HI_M[axis]
        ) / component[moving]

        axis_enter[moving] = np.minimum(s_at_lo, s_at_hi)
        axis_exit[moving] = np.maximum(s_at_lo, s_at_hi)

        s_enter = np.maximum(s_enter, axis_enter)
        s_exit = np.minimum(s_exit, axis_exit)

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

    backtrack = np.maximum(s_enter, 0.0)
    hit_position = position.copy()
    hit_position[intersection_valid] = (
        position[intersection_valid]
        - backtrack[intersection_valid, None]
        * direction[intersection_valid]
    )

    # Clamp successful intersections and force the reconstructed normal
    # coordinate exactly onto the nearest wall.
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

    reconstructed_face_code = np.argmin(hit_distances, axis=1)

    for face_code in range(6):
        mask = intersection_valid & (reconstructed_face_code == face_code)
        axis = face_code // 2
        wall = BOX_HI_M[axis] if face_code % 2 else BOX_LO_M[axis]
        hit_position[mask, axis] = wall

    final_inside = np.zeros(number, dtype=bool)
    final_inside[intersection_valid] = np.all(
        (hit_position[intersection_valid] >= BOX_LO_M - tolerance)
        & (hit_position[intersection_valid] <= BOX_HI_M + tolerance),
        axis=1,
    )

    final_distances = np.column_stack(
        (
            np.abs(hit_position[:, 0] - BOX_LO_M[0]),
            np.abs(hit_position[:, 0] - BOX_HI_M[0]),
            np.abs(hit_position[:, 1] - BOX_LO_M[1]),
            np.abs(hit_position[:, 1] - BOX_HI_M[1]),
            np.abs(hit_position[:, 2] - BOX_LO_M[2]),
            np.abs(hit_position[:, 2] - BOX_HI_M[2]),
        )
    )

    final_on_boundary = np.zeros(number, dtype=bool)
    final_on_boundary[intersection_valid] = (
        np.min(final_distances[intersection_valid], axis=1) <= tolerance
    )

    correction_mask = intersection_valid & final_inside & final_on_boundary
    corrected = position.copy()
    corrected[correction_mask] = hit_position[correction_mask]

    failed = geometrically_outside & ~correction_mask
    mismatch = np.zeros(number, dtype=bool)

    if source_face in FACE_TO_CODE:
        mismatch = (
            correction_mask
            & (reconstructed_face_code != FACE_TO_CODE[source_face])
        )

    stats["geometrically_outside"] = int(
        np.count_nonzero(geometrically_outside)
    )
    stats["corrected"] = int(np.count_nonzero(correction_mask))
    stats["already_on_boundary"] = int(
        np.count_nonzero(already_on_boundary)
    )
    stats["inside_unchanged"] = int(np.count_nonzero(inside_unchanged))
    stats["failed_unchanged"] = int(
        np.count_nonzero(failed) + np.count_nonzero(~finite)
    )
    stats["source_face_mismatch"] = int(np.count_nonzero(mismatch))

    return corrected, correction_mask, stats


def mark_position_correction(
    species_group: h5py.Group,
    source_face: str | None,
    stats: dict[str, int],
) -> None:
    species_group.attrs["boundary_position_correction_applied"] = np.uint8(1)
    species_group.attrs[
        "boundary_position_correction_method"
    ] = "backward_line_box_intersection_original_momentum"
    species_group.attrs[
        "boundary_position_correction_source_face"
    ] = source_face or "unknown"
    species_group.attrs["boundary_position_box_lo_m"] = BOX_LO_M
    species_group.attrs["boundary_position_box_hi_m"] = BOX_HI_M

    for field in CORRECTION_STAT_FIELDS:
        species_group.attrs[f"boundary_position_{field}"] = np.int64(
            stats[field]
        )


def correct_species_positions(
    species_group: h5py.Group,
    source_face: str | None,
) -> dict[str, int]:
    """Correct one species' absolute positions using its original momentum."""
    marker = "boundary_position_correction_applied"
    if int(species_group.attrs.get(marker, 0)) == 1:
        raise RuntimeError(f"{species_group.name} was already position-corrected")

    position_refs = {
        axis: locate_component(
            species_group, "position", axis, required=True
        )
        for axis in AXES
    }
    momentum_refs = {
        axis: locate_component(
            species_group, "momentum", axis, required=True
        )
        for axis in AXES
    }
    offset_refs = {
        axis: locate_component(
            species_group, "positionOffset", axis, required=False
        )
        for axis in AXES
    }

    number_of_particles = validate_equal_lengths(
        [
            *((f"position/{axis}", position_refs[axis]) for axis in AXES),
            *((f"momentum/{axis}", momentum_refs[axis]) for axis in AXES),
        ]
    )

    validate_equal_lengths(
        [(f"positionOffset/{axis}", offset_refs[axis]) for axis in AXES],
        expected=number_of_particles,
    )

    if number_of_particles == 0:
        stats = empty_correction_record()
        mark_position_correction(species_group, source_face, stats)
        return stats

    writers = {
        axis: create_temporary_writer(
            position_refs[axis],
            number_of_particles,
            "boundary_corrected",
        )
        for axis in AXES
    }

    totals = empty_correction_record()

    try:
        for start in range(0, number_of_particles, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, number_of_particles)

            stored_position = {
                axis: read_stored_chunk(position_refs[axis].obj, start, stop)
                for axis in AXES
            }
            position_unit = {
                axis: get_unit_si(position_refs[axis].obj)
                for axis in AXES
            }
            offset_si = {
                axis: read_optional_si_chunk(offset_refs[axis], start, stop)
                for axis in AXES
            }

            absolute_position_m = np.column_stack(
                [
                    stored_position[axis] * position_unit[axis]
                    + offset_si[axis]
                    for axis in AXES
                ]
            )

            original_momentum_si = np.column_stack(
                [
                    read_si_chunk(momentum_refs[axis].obj, start, stop)
                    for axis in AXES
                ]
            )

            corrected_position_m, correction_mask, stats = (
                reconstruct_boundary_positions(
                    absolute_position_m,
                    original_momentum_si,
                    source_face,
                )
            )

            for field in CORRECTION_STAT_FIELDS:
                totals[field] += stats[field]

            for axis_index, axis in enumerate(AXES):
                output_stored = stored_position[axis].copy()
                output_stored[correction_mask] = (
                    corrected_position_m[correction_mask, axis_index]
                    - offset_si[axis][correction_mask]
                ) / position_unit[axis]

                writers[axis].dataset[start:stop] = output_stored.astype(
                    writers[axis].dataset.dtype,
                    copy=False,
                )

        if totals["corrected"] > 0:
            for axis in AXES:
                writers[axis].commit()
        else:
            for axis in AXES:
                writers[axis].discard()

        mark_position_correction(species_group, source_face, totals)
        return totals

    except Exception:
        for writer in writers.values():
            writer.discard()
        raise


# =============================================================================
# IPC PROPAGATED AND PRODUCTION MOMENTUM BOOSTS
# =============================================================================

def mark_species_as_boosted(
    species_group: h5py.Group,
    beta_x: float,
    gamma_boost: float,
) -> None:
    species_group.attrs["ipc_crossing_boost_applied"] = np.uint8(1)
    species_group.attrs["ipc_crossing_transform"] = "lorentz_boost_x"
    species_group.attrs["ipc_crossing_beta_x"] = beta_x
    species_group.attrs["ipc_crossing_gamma"] = gamma_boost
    species_group.attrs[
        "ipc_full_crossing_angle_mrad"
    ] = FULL_CROSSING_ANGLE_MRAD
    species_group.attrs[
        "ipc_boost_applied_after_position_correction"
    ] = np.uint8(1)


def boost_species_momentum(
    species_group: h5py.Group,
    beta_x: float,
    gamma_boost: float,
) -> int:
    """Apply the common x-directed Lorentz boost to one IPC species."""
    marker = "ipc_crossing_boost_applied"
    if int(species_group.attrs.get(marker, 0)) == 1:
        raise RuntimeError(f"{species_group.name} was already momentum-boosted")

    momentum_refs = {
        axis: locate_component(
            species_group, "momentum", axis, required=True
        )
        for axis in AXES
    }

    number_of_particles = validate_equal_lengths(
        [(f"momentum/{axis}", momentum_refs[axis]) for axis in AXES]
    )

    if number_of_particles == 0:
        mark_species_as_boosted(species_group, beta_x, gamma_boost)
        return 0

    px_ref = momentum_refs["x"]
    px_unit = get_unit_si(px_ref.obj)
    px_writer = create_temporary_writer(
        px_ref,
        number_of_particles,
        "ipc_boosted",
        force_float64=False,
    )

    electron_mc = M_ELECTRON * C_LIGHT

    try:
        for start in range(0, number_of_particles, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, number_of_particles)

            px = read_si_chunk(momentum_refs["x"].obj, start, stop)
            py = read_si_chunk(momentum_refs["y"].obj, start, stop)
            pz = read_si_chunk(momentum_refs["z"].obj, start, stop)

            # E/c in SI momentum units.
            energy_over_c = np.sqrt(
                electron_mc**2 + px**2 + py**2 + pz**2
            )

            # px' = gamma * (px + beta_x * E/c); py and pz are unchanged.
            px_boosted = gamma_boost * (
                px + beta_x * energy_over_c
            )

            px_writer.dataset[start:stop] = (
                px_boosted / px_unit
            ).astype(px_writer.dataset.dtype, copy=False)

        px_writer.commit()
        mark_species_as_boosted(species_group, beta_x, gamma_boost)
        return number_of_particles

    except Exception:
        px_writer.discard()
        raise


def mark_production_momentum_as_boosted(
    species_group: h5py.Group,
    beta_x: float,
    gamma_boost: float,
) -> None:
    """Record that origUx was transformed into the crossing-angle frame."""
    species_group.attrs[
        "ipc_production_crossing_boost_applied"
    ] = np.uint8(1)
    species_group.attrs[
        "ipc_production_crossing_transform"
    ] = "lorentz_boost_x"
    species_group.attrs[
        "ipc_production_crossing_beta_x"
    ] = beta_x
    species_group.attrs[
        "ipc_production_crossing_gamma"
    ] = gamma_boost
    species_group.attrs[
        "ipc_production_full_crossing_angle_mrad"
    ] = FULL_CROSSING_ANGLE_MRAD
    species_group.attrs[
        "ipc_production_momentum_storage"
    ] = "proper_velocity_gamma_v_m_per_s"


def boost_species_production_momentum(
    species_group: h5py.Group,
    beta_x: float,
    gamma_boost: float,
) -> int:
    """Lorentz-boost the production-time origUx proper velocity.

    origUx/origUy/origUz store gamma*v in m/s. They are converted to
    physical momentum with p = m_e*gamma*v, boosted in SI momentum units,
    and converted back to gamma*v before origUx is written. The y and z
    production components are unchanged by an x-directed boost.
    """
    marker = "ipc_production_crossing_boost_applied"
    if int(species_group.attrs.get(marker, 0)) == 1:
        raise RuntimeError(
            f"{species_group.name} production momentum was already boosted"
        )

    production_refs = {
        axis: locate_production_momentum_component(
            species_group, axis, required=True
        )
        for axis in AXES
    }

    number_of_particles = validate_equal_lengths(
        [
            (f"origU{axis}", production_refs[axis])
            for axis in AXES
        ]
    )

    if number_of_particles == 0:
        mark_production_momentum_as_boosted(
            species_group, beta_x, gamma_boost
        )
        return 0

    orig_ux_ref = production_refs["x"]
    orig_ux_unit = get_unit_si(orig_ux_ref.obj)
    orig_ux_writer = create_temporary_writer(
        orig_ux_ref,
        number_of_particles,
        "ipc_production_boosted",
        force_float64=False,
    )

    electron_mc = M_ELECTRON * C_LIGHT

    try:
        for start in range(0, number_of_particles, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, number_of_particles)

            # SI proper velocity, gamma*v [m/s].
            ux = read_si_chunk(production_refs["x"].obj, start, stop)
            uy = read_si_chunk(production_refs["y"].obj, start, stop)
            uz = read_si_chunk(production_refs["z"].obj, start, stop)

            # Convert gamma*v to physical momentum p = m_e*gamma*v.
            px = M_ELECTRON * ux
            py = M_ELECTRON * uy
            pz = M_ELECTRON * uz

            energy_over_c = np.sqrt(
                electron_mc**2 + px**2 + py**2 + pz**2
            )

            px_boosted = gamma_boost * (
                px + beta_x * energy_over_c
            )

            # Convert boosted physical momentum back to proper velocity.
            ux_boosted = px_boosted / M_ELECTRON

            orig_ux_writer.dataset[start:stop] = (
                ux_boosted / orig_ux_unit
            ).astype(orig_ux_writer.dataset.dtype, copy=False)

        orig_ux_writer.commit()
        mark_production_momentum_as_boosted(
            species_group, beta_x, gamma_boost
        )
        return number_of_particles

    except Exception:
        orig_ux_writer.discard()
        raise


# =============================================================================
# FILE-LEVEL TRANSFORMATION PASSES
# =============================================================================

def position_correction_pass(
    files: list[Path],
    output_root: Path,
) -> list[dict[str, object]]:
    """Correct outside-particle positions before any momentum is boosted."""
    rows: list[dict[str, object]] = []
    outside_files = [
        path for path in files if is_outside_file(path, output_root)
    ]

    print("\nPASS 1/2: correcting BoundaryScraping positions")

    for path in outside_files:
        source_face = boundary_face_from_path(path)
        print(f"\n[file] {path.relative_to(output_root)}")
        print(f"  source face: {source_face or 'unknown'}")

        with h5py.File(path, "r+") as h5file:
            particle_groups = latest_particles_groups(h5file)

            if not particle_groups:
                print("  [warning] no /data/<iteration>/particles group")
                continue

            for particles_group in particle_groups:
                for species_name in sorted(particles_group.keys()):
                    species_group = particles_group[species_name]
                    if not isinstance(species_group, h5py.Group):
                        continue

                    try:
                        stats = correct_species_positions(
                            species_group,
                            source_face,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"  [skip] {species_group.name}: {exc}")
                        continue

                    print(
                        f"  [position] {species_group.name}: "
                        f"corrected={stats['corrected']:,}, "
                        f"outside={stats['geometrically_outside']:,}, "
                        f"failed={stats['failed_unchanged']:,}"
                    )

                    row: dict[str, object] = {
                        "file": str(path.relative_to(output_root)),
                        "source_face": source_face or "unknown",
                        "species": species_name,
                    }
                    row.update(stats)
                    rows.append(row)

    return rows


def momentum_boost_pass(
    files: list[Path],
    output_root: Path,
    beta_x: float,
    gamma_boost: float,
) -> list[dict[str, object]]:
    """Boost propagated and production IPC momenta after position correction."""
    rows: list[dict[str, object]] = []

    print("\nPASS 2/2: applying propagated and production IPC momentum boosts")

    for path in files:
        print(f"\n[file] {path.relative_to(output_root)}")

        with h5py.File(path, "r+") as h5file:
            particle_groups = latest_particles_groups(h5file)

            if not particle_groups:
                print("  [warning] no /data/<iteration>/particles group")
                continue

            for particles_group in particle_groups:
                for species_name in IPC_SPECIES:
                    if species_name not in particles_group:
                        continue

                    species_group = particles_group[species_name]
                    if not isinstance(species_group, h5py.Group):
                        raise TypeError(
                            f"Species object is not a group: {species_group.name}"
                        )

                    propagated_count = boost_species_momentum(
                        species_group,
                        beta_x,
                        gamma_boost,
                    )

                    try:
                        production_count = boost_species_production_momentum(
                            species_group,
                            beta_x,
                            gamma_boost,
                        )
                    except KeyError as exc:
                        production_count = 0
                        print(
                            f"  [production momentum skip] "
                            f"{species_group.name}: {exc}"
                        )

                    print(
                        f"  [momentum] {species_group.name}: "
                        f"propagated={propagated_count:,}, "
                        f"production={production_count:,}"
                    )

                    rows.append(
                        {
                            "file": str(path.relative_to(output_root)),
                            "species": species_name,
                            "propagated_particles_boosted": propagated_count,
                            "production_particles_boosted": production_count,
                        }
                    )

    return rows


# =============================================================================
# SUMMARIES
# =============================================================================

def write_summaries(
    output_root: Path,
    selected_files: list[Path],
    correction_rows: list[dict[str, object]],
    boost_rows: list[dict[str, object]],
    beta_x: float,
    gamma_boost: float,
) -> None:
    summary_dir = output_root / "transformation_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    correction_csv = summary_dir / "position_correction_counts.csv"
    with correction_csv.open("w", newline="") as handle:
        fieldnames = [
            "file",
            "source_face",
            "species",
            *CORRECTION_STAT_FIELDS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(correction_rows)

    boost_csv = summary_dir / "momentum_boost_counts.csv"
    with boost_csv.open("w", newline="") as handle:
        fieldnames = [
            "file",
            "species",
            "propagated_particles_boosted",
            "production_particles_boosted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boost_rows)

    total_corrected = sum(int(row["corrected"]) for row in correction_rows)
    total_propagated_boosted = sum(
        int(row["propagated_particles_boosted"]) for row in boost_rows
    )
    total_production_boosted = sum(
        int(row["production_particles_boosted"]) for row in boost_rows
    )

    with (summary_dir / "transformation_summary.txt").open("w") as handle:
        handle.write("WarpX diagnostics position correction + IPC boost\n")
        handle.write("==================================================\n\n")
        handle.write(f"source={SOURCE_DIAGS.resolve()}\n")
        handle.write(f"output={output_root}\n")
        handle.write(f"selected_h5_files={len(selected_files)}\n")
        handle.write(f"position_entries_corrected={total_corrected}\n")
        handle.write(
            f"ipc_propagated_particles_boosted={total_propagated_boosted}\n"
        )
        handle.write(
            f"ipc_production_particles_boosted={total_production_boosted}\n\n"
        )
        handle.write(
            "Operation order: all selected BoundaryScraping positions are "
            "corrected first using the original stored momentum; only after "
            "that pass completes are both propagated momentum/x and "
            "production origUx Lorentz-boosted.\n"
        )
        handle.write(
            "Only the latest HDF5 file in particles_in* and in each directory "
            "under particles_out* is transformed. Older files are copied but "
            "left unchanged.\n\n"
        )
        handle.write(f"full_crossing_angle_mrad={FULL_CROSSING_ANGLE_MRAD}\n")
        handle.write(f"boost_sign_x={BOOST_SIGN_X}\n")
        handle.write(f"beta_x={beta_x:.17g}\n")
        handle.write(f"gamma_boost={gamma_boost:.17g}\n")
        handle.write(f"box_lo_m={BOX_LO_M.tolist()}\n")
        handle.write(f"box_hi_m={BOX_HI_M.tolist()}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    source = SOURCE_DIAGS.resolve()
    output = OUTPUT_DIAGS.resolve()

    if not source.is_dir():
        raise FileNotFoundError(
            "The input diagnostics directory does not exist:\n"
            f"{source}"
        )

    if output.exists():
        raise FileExistsError(
            "The output directory already exists:\n"
            f"{output}\n\n"
            "Remove or rename it before running the script again."
        )

    full_angle_rad = FULL_CROSSING_ANGLE_MRAD * 1.0e-3
    half_angle_rad = 0.5 * full_angle_rad
    beta_x = BOOST_SIGN_X * math.sin(half_angle_rad)
    gamma_boost = 1.0 / math.sqrt(1.0 - beta_x**2)

    print(f"Input directory : {source}")
    print(f"Output directory: {output}")
    print(f"Full angle      : {FULL_CROSSING_ANGLE_MRAD} mrad")
    print(f"Half angle      : {half_angle_rad:.12g} rad")
    print(f"beta_x          : {beta_x:.12g}")
    print(f"gamma_boost     : {gamma_boost:.12g}")
    print(f"Box low [m]     : {BOX_LO_M.tolist()}")
    print(f"Box high [m]    : {BOX_HI_M.tolist()}")

    print("\nCopying complete diagnostics tree...")
    shutil.copytree(
        source,
        output,
        copy_function=shutil.copy2,
    )

    selected_files = diagnostic_files_to_modify(output)

    if not selected_files:
        shutil.rmtree(output)
        raise FileNotFoundError(
            "No HDF5 files were found under particles_in* or particles_out*"
        )

    print("\nSelected latest HDF5 files:")
    for path in selected_files:
        print(f"  {path.relative_to(output)}")

    try:
        # This ordering is the central requirement. No boost is written until
        # every selected outside file has completed position correction.
        correction_rows = position_correction_pass(selected_files, output)
        boost_rows = momentum_boost_pass(
            selected_files,
            output,
            beta_x,
            gamma_boost,
        )

        write_summaries(
            output,
            selected_files,
            correction_rows,
            boost_rows,
            beta_x,
            gamma_boost,
        )

    except Exception:
        print(
            "\nAn error occurred while transforming the copied diagnostics.",
            file=sys.stderr,
        )
        print(
            "The original diags directory was not changed.",
            file=sys.stderr,
        )
        print(
            "The partially transformed output directory was retained for "
            "inspection.",
            file=sys.stderr,
        )
        raise

    total_corrected = sum(int(row["corrected"]) for row in correction_rows)
    total_propagated_boosted = sum(
        int(row["propagated_particles_boosted"]) for row in boost_rows
    )
    total_production_boosted = sum(
        int(row["production_particles_boosted"]) for row in boost_rows
    )

    print("\nDone.")
    print(f"Files selected       : {len(selected_files)}")
    print(f"Positions corrected  : {total_corrected:,}")
    print(
        f"IPC propagated boosted: {total_propagated_boosted:,}"
    )
    print(
        f"IPC production boosted: {total_production_boosted:,}"
    )
    print(f"Modified copy        : {output}")
    print(
        "\nThe original diagnostics were preserved. Outside positions were "
        "corrected with original propagated momentum before both propagated "
        "momentum/x and production origUx were boosted."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
