from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

callbacks = []
wandb_logger = WandbLogger(project='keds')
    
model_path = 'epoch=9-step=594.ckpt'
lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(dirpath='/content/drive/MyDrive/keds/')

model = KEDY()

trainer = pl.Trainer(
                    logger=wandb_logger,                      
                    accelerator='gpu',
                    precision=16,
                    default_root_dir='/content/drive/MyDrive/keds/',
                    callbacks=[lr_monitor, checkpoint_callback],
                    log_every_n_steps=1,
                    min_epochs=1,
                    check_val_every_n_epoch=1,
                     val_check_interval=1,
                     limit_train_batches=150,
                     move_metrics_to_cpu=True
                    )

trainer.fit(model,train_loader, test_loader, ckpt_path='/content/drive/MyDrive/keds/epoch=20-step=594.ckpt'')
