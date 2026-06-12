import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from distill import config
from distill.data import get_dataloader
from distill.lightning_module import LightningModel


def main():
    config.maybe_login()

    model = LightningModel(
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        w_summary=config.W_SUMMARY,
        w_feature=config.W_FEATURE,
        max_steps=config.MAX_STEPS,
        warmup_steps=config.WARMUP_STEPS,
    )
    train_loader = get_dataloader(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, train=True
    )
    val_loader = get_dataloader(
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, train=False
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=config.CKPT_DIR,
        filename="patho-{step:06d}",  # avoid slash-in-metric filename issues
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step")]

    # Optional online benchmark: a cheap linear probe every N steps for a
    # learning-curve signal during training. Needs a small labeled (image, label)
    # probe set. Disabled by default; uncomment and supply loaders to enable.
    #
    # from distill.eval.online_eval import OnlineProbeCallback
    # callbacks.append(OnlineProbeCallback(
    #     train_loader=probe_train_loader, val_loader=probe_val_loader,
    #     every_n_steps=1000, num_classes=4, max_batches=8, name="bach",
    # ))

    trainer = pl.Trainer(
        max_steps=config.MAX_STEPS,
        val_check_interval=config.VAL_CHECK_INTERVAL,
        limit_val_batches=config.VAL_BATCHES,
        num_sanity_val_steps=0,  # streaming val loader: skip the sanity pull
        log_every_n_steps=config.LOG_EVERY,
        accelerator="auto",
        devices=1,
        precision=config.resolve_precision(),
        gradient_clip_val=config.GRAD_CLIP,
        callbacks=callbacks,
        logger=CSVLogger(save_dir="logs", name="patho-distill"),
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
