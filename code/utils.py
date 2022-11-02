import logging
import os
import torch

logger = logging.getLogger(__name__)

def save_checkpoint(model, config):
    checkpoint = {
            "net": model.state_dict(keep_vars=True),
        }
    ckpt = config["ckpt"]+'/model/'
    os.makedirs(ckpt,exist_ok=True)
    torch.save(checkpoint,
                  '{}/model.pth'.format(ckpt))
    logger.info("Saved model checkpoint to [DIR: %s]", ckpt)