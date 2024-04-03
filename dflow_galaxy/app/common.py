from dp.launching.typing import BaseModel, Optional
from dp.launching.typing import BohriumUsername, BohriumTicket, BohriumProjectId
from dp.launching.typing import (
    DflowArgoAPIServer, DflowK8sAPIServer,
    DflowAccessToken, DflowStorageEndpoint,
    DflowStorageRepository, DflowLabels,
)

from dflow.plugins import bohrium
import dflow

from dflow_galaxy.core.log import get_logger
logger = get_logger(__name__)


class DFlowOptions(BaseModel):
    bh_username: Optional[BohriumUsername]
    bh_ticket: Optional[BohriumTicket]
    bh_project_id: Optional[BohriumProjectId]

    dflow_labels: Optional[DflowLabels]
    dflow_argo_api_server: Optional[DflowArgoAPIServer]
    dflow_k8s_api_server: Optional[DflowK8sAPIServer]
    dflow_access_token: Optional[DflowAccessToken]
    dflow_storage_endpoint: Optional[DflowStorageEndpoint]
    dflow_storage_repository: Optional[DflowStorageRepository]


def setup_dflow_context(opts: DFlowOptions):
    """
    setup dflow context based on:
    https://dptechnology.feishu.cn/docx/HYjmdDj36oAksixbviKcbgUinUf
    """

    dflow_config = {
        'host': opts.dflow_argo_api_server,
        "k8s_api_server": opts.dflow_k8s_api_server,
        "token": opts.dflow_access_token,
        "dflow_labels": opts.dflow_labels,
    }
    dflow.config.update(dflow_config)
    logger.info(f"dflow config: {dflow.config}")

    dflow_s3_config = {
        'endpoint': opts.dflow_storage_endpoint,
        'repo_key': opts.dflow_storage_repository,
    }
    dflow.s3_config.update(dflow_s3_config)

    bohrium_config = {
        'username': opts.bh_username,
        'ticket': opts.bh_ticket,
        'project_id': opts.bh_project_id,
    }
    bohrium.config.update(bohrium_config)
    logger.info(f"bohrium config: {bohrium.config}")

    bohrium.config["tiefblue_url"] = "https://tiefblue.dp.tech"
    bohrium.config["bohrium_url"] = "https://bohrium.dp.tech"
    # override config? I have no idea
    dflow.s3_config["repo_key"] = "oss-bohrium"
    # side effect alert!!!
    # the following must be set at the end of all config
    dflow.s3_config['storage_client'] = bohrium.TiefblueClient()
    logger.info(f's3_config: {dflow.s3_config}')
