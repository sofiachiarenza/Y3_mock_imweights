#!/usr/bin/env python
"""
SYSNet training with snapshot ensemble, MPI-parallelised.
Fixes super() bug in sysnet.SYSNetSnapshotMPI (present in cosmodesi 20250526 and 20260321).
"""
import os
import sys
import torch
from glob import glob
import sysnet


class SYSNetSnapshotMPIFixed(sysnet.SYSNetSnapshotMPI):
    def evaluate(self, dataloader, nn_structure, restore_path):
        snapshot_path = os.path.dirname(restore_path)
        snapshots = sorted(glob(os.path.join(snapshot_path, 'snapshot_*.pth.tar')))

        self.logger.info(f"Restoring parameters from {len(snapshots)} snapshots")
        if len(snapshots) == 0:
            sys.exit('No snapshot files found. Was --snapshot_ensemble passed to training?')

        pred_ensemble, testloss_ensemble = [], []
        for snapshot_i in snapshots:
            test_loss_, hpix_, pred_ = sysnet.SYSNet.evaluate(
                self, dataloader, nn_structure, snapshot_i)
            pred_ensemble.append(pred_)
            testloss_ensemble.append(test_loss_)

        return testloss_ensemble, hpix_, torch.cat(pred_ensemble, 1)


if __name__ == '__main__':
    config = sysnet.parse_cmd_arguments('sysnet_config.yaml')
    pipeline = SYSNetSnapshotMPIFixed(config)
    # T_0=5, T_mult=1: snapshot at every restart, restarts every 5 epochs.
    # With nepochs=100 this gives 20 snapshots per chain.
    pipeline.config.scheduler_kwargs.update(T_0=5, T_mult=1)
    pipeline.run()
