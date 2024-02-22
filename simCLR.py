import torch
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from io_utils import save_config_file, save_checkpoint
from train_utils import accuracy
import os
import numpy as np
import time

class SimCLR(object):

    def __init__(self, args, model, optimizer, scheduler, log_dir, stratified, manifold):
        self.args = args
        self.model = model.to(args.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.stratified = stratified
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.manifold = manifold

    def info_nce_loss(self, features, stratified):
        bs = int(features.shape[0] // 2)
        labels = torch.cat([torch.arange(bs) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        # Normlize the features according to subject
        if stratified == 'stratified':
            features_str = features.clone()
            features_str[:bs, :] = (features[:bs, :] -  features[:bs, :].mean(
                dim=0)) / (features[:bs, :].std(dim=0) + 1e-3)
            features_str[bs:, :] = (features[bs:, :] -  features[bs:, :].mean(
                dim=0)) / (features[bs:, :].std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'bn':
            features_str = features.clone()
            features_str = (features - features.mean(dim=0)) / (features.std(dim=0) + 1e-3)
            features = F.normalize(features_str, dim=1)
        elif stratified == 'no':
            features = F.normalize(features, dim=1)

        features = self.manifold.expmap0(features)
        similarity_matrix = -self.manifold.dist(features[:, None, :], features)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # print('positives:', positives.shape)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        # Put the positive column at the end (when all the entries are the same, the top1 acc will be 0; while if the
        # positive column is at the start, the top1 acc might be exaggerated)
        logits = torch.cat([negatives, positives], dim=1)
        # The label means the last column contain the positive pairs
        labels = torch.ones(logits.shape[0], dtype=torch.long)*(logits.shape[1]-1)
        labels = labels.to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, val_loader, n_times):
        # save config file
        print(self.writer.log_dir)
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs_pretrain} epochs.")

        # train_loss_history = np.zeros(self.args.epochs_pretrain)
        # val_loss_history = np.zeros(self.args.epochs_pretrain)
        bad_count = 0
        best_acc, best_loss = -1, 1000
        model_epochs, optimizer_epochs = {}, {}
        train_top1_history, val_top1_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        train_top5_history, val_top5_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        train_loss_history, val_loss_history = np.zeros(self.args.epochs_pretrain), np.zeros(self.args.epochs_pretrain)
        for epoch_counter in range(self.args.epochs_pretrain):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_acc5 = 0
            # train_loss_mediate, train_acc_mediate = 0, 0
            # loss_batch = torch.tensor(0., requires_grad=True)
            self.model.train()
            for count, (data, truelabels) in enumerate(train_loader):
                data = data.to(self.args.device)
                truelabels = truelabels.to(self.args.device)

                # with autocast(enabled=self.args.fp16_precision):
                features, cls_feat = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss1 = self.criterion(logits, labels)
                loss2 = self.criterion(cls_feat, truelabels.long())
                loss = loss1 + self.args.lamda * loss2
                # loss = self.criterion(logits, labels)
                # loss_batch = loss_batch + loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if np.mod(count+1, n_times)==0:
                #     self.optimizer.zero_grad()
                #     loss_batch.backward()
                #     loss_batch = torch.tensor(0., requires_grad=True)
                #     self.optimizer.step()

                top1, top5 = accuracy(logits, labels, topk=(1,5))

                # if n_iter % self.args.log_every_n_steps == 0:
                #     self.writer.add_scalar('loss', loss, global_step=n_iter)
                #     self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                #     self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

                train_loss = train_loss + loss.data.cpu().numpy()
                train_acc = train_acc + top1[0].cpu().numpy()
                train_acc5 = train_acc5 + top5[0].cpu().numpy()

                # train_loss_mediate += loss.data.cpu().numpy()
                # train_acc_mediate += top1[0].cpu().numpy()
                # if np.mod(count+1, 100) == 0:
                #     print('iter %d, train loss: %.4f, acc: %.3f' % (
                #         count+1, train_loss_mediate/100, train_acc_mediate/100))
                #     train_loss_mediate, train_acc_mediate = 0, 0

            train_loss = train_loss / (count + 1)
            train_acc = train_acc / (count + 1)
            train_acc5 = train_acc5 / (count + 1)

            val_loss = 0
            val_acc = 0
            val_acc5 = 0
            self.model.eval()
            for count, (data, labels) in enumerate(val_loader):
                data = data.to(self.args.device)

                features, _ = self.model(data)
                logits, labels = self.info_nce_loss(features, self.stratified)
                loss = self.criterion(logits, labels)

                top1, top5 = accuracy(logits, labels, topk=(1,5))

                val_loss = val_loss + loss.data.cpu().numpy()
                val_acc = val_acc + top1[0].cpu().numpy()
                val_acc5 = val_acc5 + top5[0].cpu().numpy()

                # if np.mod(count, 5) == 0:
                #     print('val iter %d, top1 acc: %.3f, top5 acc: %.3f' % (count, val_acc / count, val_acc5 / count))
                    # print(logits[0,:])

            val_loss = val_loss / (count + 1)
            val_acc = val_acc / (count + 1)
            val_acc5 = val_acc5 / (count + 1)

            train_top1_history[epoch_counter] = train_acc
            val_top1_history[epoch_counter] = val_acc
            train_top5_history[epoch_counter] = train_acc5
            val_top5_history[epoch_counter] = val_acc5
            train_loss_history[epoch_counter] = train_loss
            val_loss_history[epoch_counter] = val_loss

            model_epochs[epoch_counter] = self.model
            optimizer_epochs[epoch_counter] = self.optimizer

            # warmup for the first 10 epochs
            # if epoch_counter >= 10:
            # No warmup
            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tTrain loss: {train_loss}\tTrain top1 accuracy: {train_acc}")
            logging.debug(f"\tVal loss: {val_loss}\tVal top1 accuracy: {val_acc}")
            print(f"Epoch: {epoch_counter}   Train loss: {train_loss}   Top1 accuracy: {train_acc}   Top5 accuracy: {train_acc5}")
            print(
                f"\tVal loss: {val_loss}   Top1 accuracy: {val_acc}   Top5 accuracy: {val_acc5}")
            # print('learning rate', self.scheduler.get_lr()[0])
            # print('logits:', logits[0])

            if val_acc > best_acc:
                bad_count = 0
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch_counter
            else:
                bad_count += 1

            if bad_count > self.args.max_tol_pretrain:
                break

            end_time = time.time()
            print('time consumed:', end_time - start_time)

        self.best_model = model_epochs[best_epoch]
        self.best_optimizer = optimizer_epochs[best_epoch]

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_epoch)
        save_checkpoint({
            'epoch': best_epoch,
            'state_dict': self.best_model.state_dict(),
            'optimizer': self.best_optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
        save_checkpoint({
            'epoch': epoch_counter,
            'state_dict': model_epochs[epoch_counter].state_dict(),
            'optimizer': optimizer_epochs[epoch_counter].state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

        print('best epoch: %d, train top1 acc:%.3f, top5 acc:%.3f; val top1 acc:%.3f, top5 acc:%.3f, train loss:%.4f, val loss: %.4f' % (
            best_epoch, train_top1_history[best_epoch], train_top5_history[best_epoch],
            val_top1_history[best_epoch], val_top5_history[best_epoch], 
            train_loss_history[best_epoch], val_loss_history[best_epoch]))
        return self.best_model, best_epoch, train_top1_history, val_top1_history, train_top5_history, val_top5_history, train_loss_history, val_loss_history