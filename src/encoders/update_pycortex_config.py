import configparser
import shutil

import cortex

from encoders.utils import get_logger, load_config

log = get_logger(__name__)


def update_pycortex_config():
    pycortex_config_path = cortex.options.usercfg

    # 1. create back up config
    pycortex_config_backup_path = (
        f"{pycortex_config_path.removesuffix('.cfg')}.backup.cfg"
    )
    shutil.copy(pycortex_config_path, pycortex_config_backup_path)
    log.info(f"Created backup in: {pycortex_config_backup_path}")

    # 2. susbsitute filestore path
    log.info(f"Modifying pycortex config: {pycortex_config_path}")
    pycortex_config = configparser.ConfigParser()
    pycortex_config.read([pycortex_config_path])

    enc_config = load_config()
    pycortex_config["basic"]["filestore"] = enc_config["DATA_DIR"]

    # 3. save pycortex config
    with open(pycortex_config_path, "w") as f_out:
        pycortex_config.write(f_out)

    return 0


if __name__ == "__main__":
    update_pycortex_config()
