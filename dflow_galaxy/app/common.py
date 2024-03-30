from dp.launching.typing import BaseModel
from dp.launching.typing import BohriumUsername, BohriumTicket, BohriumProjectId
from dp.launching.typing import (
    DflowArgoAPIServer, DflowK8sAPIServer,
    DflowAccessToken, DflowStorageEndpoint,
    DflowStorageRepository, DflowLabels
)

from dflow.plugins import bohrium
import dflow


class DflowOptions(BaseModel):
    bh_username: BohriumUsername
    bh_ticket: BohriumTicket
    bh_project_id: BohriumProjectId

    dflow_labels: DflowLabels
    dflow_argo_api_server: DflowArgoAPIServer
    dflow_k8s_api_server: DflowK8sAPIServer
    dflow_access_token: DflowAccessToken
    dflow_storage_endpoint: DflowStorageEndpoint
    dflow_storage_repository: DflowStorageRepository


def setup_dflow_context(opts: DflowOptions):
    """
    setup dflow context based on:
    https://dptechnology.feishu.cn/docx/HYjmdDj36oAksixbviKcbgUinUf
    """

    dflow_config = {
        'host': opts.dflow_argo_api_server,
        "k8s_api_server": opts.dflow_k8s_api_server,
        "token": opts.dflow_access_token,
        "dflow_labels": opts.dflow_labels.get_value()
    }
    dflow.config.update(dflow_config)

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

    bohrium.config["tiefblue_url"] = "https://tiefblue.dp.tech"
    bohrium.config["bohrium_url"] = "https://bohrium.dp.tech"
    dflow.s3_config["storage_client"] = bohrium.TiefblueClient()
