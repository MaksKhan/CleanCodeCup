import torch
import pytorch_lightning as pl
from config import Config
import evaluate
from transformers import ViTImageProcessor, ViTForImageClassification
import sklearn

class KEDY(pl.LightningModule):
    def __init__(self, args=Config):
        super().__init__()
        self.args = args 
        out_ = len(data['id'].unique())
        self.model = efficientnet_v2_s(weights=efficientnet_v2_S_Weights)
        self.model.classifier[1] = torch.nn.Linear(1280,out_//2, bias=True)
        self.model.classifier.append(torch.nn.Linear(out_//2,out_))
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.model(x)
        return x
 
    def create_optimizer(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)
            
    def lr_warmup_config(self):

        def warmup(step):
            """
            This method will be called for ceil(warmup_batches/accum_grad_batches) times,
            warmup_steps has been adjusted accordingly
            """
            if self.args.warmup_steps <= 0:
                factor = 1
            else:
                factor = min(step / self.args.warmup_steps, 1)
            return factor

        opt1 = self.create_optimizer()
        return {
            'frequency': self.args.warmup_batches,
            'optimizer': opt1,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt1, warmup),
                'interval': 'step',
                'frequency': 1,
                'name': 'lr/warmup'
            },
        }


    def lr_decay_config(self):
        opt2 = self.create_optimizer()
        return {
            'frequency': self.args.train_batches - self.args.warmup_batches,
            'optimizer': opt2,
            'lr_scheduler': {

                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                                    opt2, 'min', 
                                    factor=self.args.lrdecay_factor, 
                                    patience=self.args.lrdecay_patience,
                                    threshold=self.args.lrdecay_threshold, 
                                    threshold_mode='rel',  verbose=False,
                                    
                                   ),
                'interval': 'epoch',
                'frequency': 1,
                'monitor': self.args.lrdecay_monitor,
                'strict': False,
                'name': 'lr/reduce_on_plateau',
            }
        }


    def configure_optimizers(self):
        return (
            self.lr_warmup_config(),
            self.lr_decay_config()
        )



    def compute_metrics(self, pred_str, labels_ids):    
        cer = self.get_cer(labels_ids, pred_str)
        string_accuracy = self.get_accuracy(labels_ids, pred_str)
        return cer, string_accuracy

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.log('step', batch_idx, logger=True, on_epoch=True)
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.log("train_loss", loss, logger=True, on_epoch=True)
        # print(loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):        
         
        self.log('step', batch_idx, logger=True, on_epoch=True)
        images, labels = batch
        outputs = self(images)
        preds = outputs.argmax(-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        f1_score = sklearn.metrics.f1_score(labels, preds, average='macro')
        self.log('f1', f1_score, logger=True, on_epoch=True, on_step=True)
        return f1_score

    def predict_step(self, batch, batch_idx):
        # print(batch[1].size())
        path, images = batch
        outputs = self(images)
        preds = outputs.argmax(-1).detach().cpu().numpy()             
        return (path,preds)
