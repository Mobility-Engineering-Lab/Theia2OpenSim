"""Generate an OpenSim ScaleTool setup file and run it via opensim-cmd.

Wraps the same Scaling_Setup.xml structure validated by hand against
TheiaTestData (see sample_data/OpenSim/TestData/Scaling_Setup_TestData.xml):
measurement-based scaling using the DEFAULT_TRC_MARKER_SEGMENT_MAP marker
names, followed by MarkerPlacer using the same static TRC + a self-consistent
coordinate .mot (see workflow_utils.extract_virtual_marker_positions'
reference_pose parameter -- both must share the same reference frame or
MarkerPlacer's IK reports a large, spurious marker error).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

_SCALE_SETUP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40500">
\t<ScaleTool name="{subject_name}">
\t\t<mass>{mass}</mass>
\t\t<height>-1</height>
\t\t<age>-1</age>
\t\t<notes>Unassigned</notes>
\t\t<GenericModelMaker>
\t\t\t<model_file>{model_file}</model_file>
\t\t\t<marker_set_file>{marker_set_file}</marker_set_file>
\t\t</GenericModelMaker>
\t\t<ModelScaler>
\t\t\t<apply>true</apply>
\t\t\t<scaling_order> measurements</scaling_order>
\t\t\t<MeasurementSet>
\t\t\t\t<objects>
\t\t\t\t\t<Measurement name="pelvis">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> LASIS RASIS</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="pelvis">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="femur r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> RASIS RKNEE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="femur_r">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="femur l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> LASIS LKNEE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="femur_l">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="tibia r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> RKNEE RANKLE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="tibia_r">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="tibia l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> LKNEE LANKLE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="tibia_l">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="foot r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> RANKLE RTOE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="talus_r">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t\t<BodyScale name="calcn_r">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t\t<BodyScale name="toes_r">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t\t<BodyScale name="talus_l">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t\t<Measurement name="foot l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<MarkerPairSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<MarkerPair>
\t\t\t\t\t\t\t\t\t<markers> LANKLE LTOE</markers>
\t\t\t\t\t\t\t\t</MarkerPair>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</MarkerPairSet>
\t\t\t\t\t\t<BodyScaleSet>
\t\t\t\t\t\t\t<objects>
\t\t\t\t\t\t\t\t<BodyScale name="calcn_l">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t\t<BodyScale name="toes_l">
\t\t\t\t\t\t\t\t\t<axes> X Y Z</axes>
\t\t\t\t\t\t\t\t</BodyScale>
\t\t\t\t\t\t\t</objects>
\t\t\t\t\t\t\t<groups />
\t\t\t\t\t\t</BodyScaleSet>
\t\t\t\t\t</Measurement>
\t\t\t\t</objects>
\t\t\t\t<groups />
\t\t\t</MeasurementSet>
\t\t\t<ScaleSet>
\t\t\t\t<objects />
\t\t\t\t<groups />
\t\t\t</ScaleSet>
\t\t\t<marker_file>{marker_file}</marker_file>
\t\t\t<time_range> {time_range_start} {time_range_end}</time_range>
\t\t\t<preserve_mass_distribution>false</preserve_mass_distribution>
\t\t\t<output_model_file>{output_model_file}</output_model_file>
\t\t\t<output_scale_file>Unassigned</output_scale_file>
\t\t</ModelScaler>
\t\t<MarkerPlacer>
\t\t\t<apply>true</apply>
\t\t\t<IKTaskSet>
\t\t\t\t<objects>
\t\t\t\t\t<IKMarkerTask name="LASIS">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="RASIS">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="RKNEE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="LKNEE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="RTOE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="LANKLE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="LTOE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="RANKLE">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKMarkerTask name="PELVIS">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>5</weight>
\t\t\t\t\t</IKMarkerTask>
\t\t\t\t\t<IKCoordinateTask name="pelvis_tilt">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="pelvis_list">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="pelvis_rotation">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_flexion_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_adduction_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_rotation_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="knee_angle_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="ankle_angle_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="subtalar_angle_r">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>default_value</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_flexion_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_adduction_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="hip_rotation_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="knee_angle_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="ankle_angle_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>from_file</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t\t<IKCoordinateTask name="subtalar_angle_l">
\t\t\t\t\t\t<apply>true</apply>
\t\t\t\t\t\t<weight>100</weight>
\t\t\t\t\t\t<value_type>default_value</value_type>
\t\t\t\t\t\t<value>0</value>
\t\t\t\t\t</IKCoordinateTask>
\t\t\t\t</objects>
\t\t\t\t<groups />
\t\t\t</IKTaskSet>
\t\t\t<marker_file>{marker_file}</marker_file>
\t\t\t<coordinate_file>{coordinate_file}</coordinate_file>
\t\t\t<time_range> {time_range_start} {time_range_end}</time_range>
\t\t\t<output_motion_file>{output_motion_file}</output_motion_file>
\t\t\t<output_model_file>{output_model_file}</output_model_file>
\t\t\t<output_marker_file>Unassigned</output_marker_file>
\t\t\t<max_marker_movement>-1</max_marker_movement>
\t\t</MarkerPlacer>
\t</ScaleTool>
</OpenSimDocument>
"""


def write_scale_setup_xml(
    output_xml_path: str | Path,
    model_file: str | Path,
    marker_set_file: str | Path,
    marker_file: str | Path,
    coordinate_file: str | Path,
    output_model_file: str | Path,
    time_range: tuple[float, float],
    subject_name: str = "theia2opensim-scaled",
    mass: float = 75.1646,
    output_motion_file: str | Path | None = None,
) -> Path:
    """Write a ScaleTool setup XML from the validated TheiaTestData template.

    All input/output paths are resolved to absolute, forward-slash paths so
    the XML doesn't depend on opensim-cmd's working directory.
    """
    output_xml_path = Path(output_xml_path)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)

    if output_motion_file is None:
        output_motion_file = output_xml_path.with_name(output_xml_path.stem + "_static_ik.mot")

    def _abs(p: str | Path) -> str:
        return Path(p).resolve().as_posix()

    xml_text = _SCALE_SETUP_TEMPLATE.format(
        subject_name=subject_name,
        mass=mass,
        model_file=_abs(model_file),
        marker_set_file=_abs(marker_set_file),
        marker_file=_abs(marker_file),
        coordinate_file=_abs(coordinate_file),
        output_model_file=_abs(output_model_file),
        output_motion_file=_abs(output_motion_file),
        time_range_start=f"{time_range[0]:.6f}",
        time_range_end=f"{time_range[1]:.6f}",
    )

    output_xml_path.write_text(xml_text, encoding="utf-8")
    return output_xml_path


class ScaleToolResult(NamedTuple):
    success: bool
    output_model_file: Path
    marker_rms: float | None
    marker_max: float | None
    marker_max_name: str | None
    stdout: str


def run_scale_tool(
    setup_xml_path: str | Path,
    output_model_file: str | Path,
    opensim_cmd: str | Path | None = None,
) -> ScaleToolResult:
    """Run `opensim-cmd run-tool` on a ScaleTool setup file.

    opensim_cmd: path to opensim-cmd(.exe). Falls back to whatever
    "opensim-cmd" resolves to on PATH if not given.
    """
    exe = str(opensim_cmd) if opensim_cmd is not None else shutil.which("opensim-cmd")
    if not exe or not Path(exe).exists():
        raise FileNotFoundError(
            "opensim-cmd executable not found. Pass --opensim-cmd (or set the "
            "OPENSIM_CMD environment variable) to point at OpenSim's bin/opensim-cmd(.exe)."
        )

    proc = subprocess.run(
        [exe, "run-tool", str(Path(setup_xml_path).resolve())],
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout + proc.stderr

    rms = max_err = None
    max_name = None
    match = re.search(
        r"marker error:\s*RMS\s*=\s*([\d.eE+-]+),\s*max\s*=\s*([\d.eE+-]+)\s*\(([^)]+)\)",
        stdout,
    )
    if match:
        rms = float(match.group(1))
        max_err = float(match.group(2))
        max_name = match.group(3)

    success = proc.returncode == 0 and Path(output_model_file).exists()
    return ScaleToolResult(
        success=success,
        output_model_file=Path(output_model_file),
        marker_rms=rms,
        marker_max=max_err,
        marker_max_name=max_name,
        stdout=stdout,
    )
