from typing import List, Mapping, Optional, Any
from dflow_galaxy.core.pydantic import BaseModel
from dflow_galaxy.core import dispatcher
from ai2_kit.core.artifact import Artifact

from .domain import deepmd, lammps, model_devi, cp2k


class GeneralConfig(BaseModel):
    type_map: List[str]
    mass_map: List[float]


class AppsConfig(BaseModel):
    python: dispatcher.PythonApp
    deepmd: Optional['deepmd.DeepmdApp'] = None
    lammps: Optional['lammps.LammpsApp'] = None
    cp2k: Optional['cp2k.Cp2kApp'] = None


class Orchestration(BaseModel):
    deepmd: Optional[str] = None
    lammps: Optional[str] = None
    model_devi: Optional[str] = None
    cp2k: Optional[str] = None


class TeslaExecutorConfig(dispatcher.ExecutorConfig):
    apps: AppsConfig


class LabelConfig(BaseModel):
    cp2k: Optional['cp2k.Cp2kConfig']


class TrainConfig(BaseModel):
    deepmd: Optional['deepmd.DeepmdConfig']


class ExploreConfig(BaseModel):
    lammps: Optional['lammps.LammpsConfig']


class ScreenConfig(BaseModel):
    model_devi: Optional['model_devi.ModelDeviConfig']


class UpdateConfig(BaseModel):
    until_iter: int
    patch: dict


class WorkflowConfig(BaseModel):
    general: GeneralConfig
    label: LabelConfig
    train: TrainConfig
    explore: ExploreConfig
    screen: ScreenConfig
    update: Optional[UpdateConfig] = None


class TeslaConfig(BaseModel):
    executors: Mapping[str, TeslaExecutorConfig]
    orchestration: Orchestration
    datasets: Mapping[str, Artifact]
    workflow: Any

    def init(self):
        for k, v in self.datasets.items():
            assert 'ancestor' not in v.attrs, f'ancestor already exists: {v.attrs}'
            v.attrs['ancestor'] = k
