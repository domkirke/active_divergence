import torch, torch.distributions as dist, torchvision as tv, sys, pdb
sys.path.append("../")
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus


def update_dict(memory, new_dict):
    for k, v in new_dict.items():
        if k in memory.keys():
            memory[k].append(v)
        else:
            memory[k] = [v]
    return memory

def accumulate_traces(traces):
    trace_dict = {}
    if "histograms" in traces[0].keys():
        trace_dict['histograms'] = {}
        hist_keys = traces[0]['histograms'].keys()
        for k in hist_keys:
            trace_dict['histograms'][k] = torch.cat([v['histograms'][k] for v in traces])
    if "embeddings" in traces[0].keys():
        trace_dict['embeddings'] = {}
        emb_keys = traces[0]['embeddings'].keys()
        for k in emb_keys:
            trace_dict['embeddings'][k] = torch.cat([v['embeddings'][k] for v in traces])
    return trace_dict



class DissectionMonitor(Callback):

    def __init__(self, n_batches=5, batch_size=1024, monitor_epochs=1, embedding_epochs=10):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.monitor_epochs = monitor_epochs
        self.embedding_epochs = embedding_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        with torch.no_grad():
            if trainer.state.stage == RunningStage.SANITY_CHECKING:
                return
            if trainer.current_epoch % self.monitor_epochs != 0:
                return
            # plot model parameters distribution
            trace = {}
            for k, v  in trainer.model.named_parameters():
                trainer.logger.experiment.add_histogram("params/"+k, v, global_step=trainer.current_epoch)
            n_batch = 0
            traces = []
            for data in trainer.datamodule.train_dataloader(batch_size=self.batch_size):
                trace_out = trainer.model.trace_from_inputs(data)
                n_batch += 1
                traces.append(trace_out)
                if n_batch > self.n_batches:
                    break
                # if not graph:
                #     trainer.logger.experiment.add_graph(trainer.model, data)
                #     graph = True

            trace = accumulate_traces(traces)
            # add outputs
            for k, v in trace['histograms'].items():
                trainer.logger.experiment.add_histogram(k, v, global_step = trainer.current_epoch)

            if trainer.current_epoch % self.embedding_epochs != 0:
                for k, v in trace['embeddings'].items():
                    label_img = None; metadata = None
                    if isinstance(v, dict):
                        label_img = v['label_img']
                        metadata = v['metadata']
                        v = v['data']
                    if len(v.shape) != 2:
                        v = v.reshape(-1, v.size(-1))
                    for dim in range(v.shape[-1]):
                        trainer.logger.experiment.add_histogram(k+"_"+str(dim), v[..., dim], global_step=trainer.current_epoch)
                    trainer.logger.experiment.add_embedding(v, tag="latent", label_img=label_img, metadata=metadata, global_step=trainer.current_epoch)

