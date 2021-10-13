import os
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from torch.optim.optimizer import Optimizer

from fer_strong_baseline.utils.logger import TxtLogger
from fer_strong_baseline.utils.meter import AverageMeter
import pdb

from sklearn.metrics import confusion_matrix


class RanLearner():
    def __init__(self, model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler,
                 margin_1,
                 margin_2,
                 beta,
                 relabel_epoch,
                 use_update_w,
                 update_w_start_epoch,
                 logger: TxtLogger,
                 save_dir: str,
                 log_steps=100,
                 device_ids=[0, 1],
                 gradient_accum_steps=1,
                 max_grad_norm=1.0,
                 batch_to_model_inputs_fn=None,
                 early_stop_n=5,
                 ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.logger = logger
        self.device_ids = device_ids
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0
        self.softmax = torch.nn.Softmax()

        self.margin_1 = margin_1
        self.margin_2 = margin_2
        self.beta = beta
        self.relabel_epoch = relabel_epoch
        self.use_update_w = use_update_w
        self.update_w_start_epoch = update_w_start_epoch

    def _accuracy(self,preds, labels):
        return accuracy_score(y_pred= preds, y_true= labels)#(preds == labels).mean()

    def _acc_and_f1(self, preds, labels):
        acc = self._accuracy(preds, labels)
        self.logger.write(classification_report(y_pred= preds, y_true= labels))
        return {
            "acc": acc,
        }

    def _batch_trans(self, batch):
        batch = tuple(t.to(self.device_ids[0]) for t in batch)
        if self.batch_to_model_inputs_fn is None:
            batch_data = {
                'imgs': batch[0],
                'labels': batch[1],
                'indexes': batch[2],
            }
        else:
            batch_data = self.batch_to_model_inputs_fn(batch)
        return  batch_data

    def val(self, val_dataloader : DataLoader):
        eval_loss = 0.0
        eval_celoss = 0.0
        eval_wloss = 0.0
        all_preds = None
        all_labels = None
        self.model.eval()

        for batch in tqdm.tqdm(val_dataloader):
            with torch.no_grad():
                batch_data = self._batch_trans(batch)
                _, pred_loggits = self.model(batch_data['imgs'])
                label = batch_data["labels"].squeeze()
                loss, celoss, wloss = self.loss_fn(pred_loggits, label)

                eval_loss += loss.mean().item()
                eval_celoss += celoss.mean().item()
                if wloss is not None:
                    eval_wloss += wloss.mean().item()
            if all_preds is None:
                all_preds = pred_loggits.detach().cpu().numpy()
                all_labels = batch_data['labels'].detach().cpu().numpy()
            else:
                all_preds = np.append(all_preds, pred_loggits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, batch_data['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / len(val_dataloader)
        eval_celoss = eval_celoss / len(val_dataloader)
        if wloss is not None:
            eval_wloss = eval_wloss / len(val_dataloader)
        all_preds = np.argmax(all_preds, axis=1)
        self.logger.write("steps: {} ". format(self.global_step))
        self.logger.write("all preds shape: ", all_preds.shape)
        result =   self._acc_and_f1(all_preds, all_labels)
        c_matrix = confusion_matrix(all_labels, all_preds)
        self.logger.write("val c_matrix:\n", c_matrix, '\n')
        return result, eval_loss, eval_celoss, eval_wloss

    def train(self, train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              epoches=100, alpha=-1):
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.alpha = alpha

        best_train_score = {'epoch': -1, 'value': 0}
        best_train_total_loss = {'epoch': -1, 'value': np.inf}
        best_train_celoss = {'epoch': -1, 'value': np.inf}
        best_train_wloss = {'epoch': -1, 'value': np.inf}

        best_val_score = {'epoch': -1, 'value': 0}
        best_val_total_loss = {'epoch': -1, 'value': np.inf}
        best_val_celoss = {'epoch': -1, 'value': np.inf}
        best_val_wloss = {'epoch': -1, 'value': np.inf}

        for epoch_i in range(1, epoches + 1):
            running_loss = 0.0
            running_celoss = 0.0
            running_wloss = 0.0
            ce_loss = None
            wloss = None

            correct_sum = 0
            iter_cnt = 0

            epoch_pred = []
            epoch_label = []

            self.model.train()
            for batch_i, (input_first, target_first, input_second, target_second,
                          input_third, target_third, input_forth, target_forth,
                          input_fifth, target_fifth, input_sixth, target_sixth,
                          input_seventh, target_seventh, input_eigth, target_eigth,
                          input_nine, target_nine) \
                    in tqdm.tqdm(enumerate(self.train_dataloader)):

                input = torch.zeros(
                    [input_first.shape[0], input_first.shape[1], input_first.shape[2], input_first.shape[3], 9])

                input[:, :, :, :, 0] = input_first
                input[:, :, :, :, 1] = input_second
                input[:, :, :, :, 2] = input_third
                input[:, :, :, :, 3] = input_forth
                input[:, :, :, :, 4] = input_fifth
                input[:, :, :, :, 5] = input_sixth
                input[:, :, :, :, 6] = input_seventh
                input[:, :, :, :, 7] = input_eigth
                input[:, :, :, :, 8] = input_nine

                input = input.cuda()

                targets = target_first
                if isinstance(targets, list):
                    targets = np.array(targets)
                if isinstance(targets, np.ndarray):
                    targets = torch.from_numpy(target_first)

                targets = targets.squeeze()
                targets = targets.cuda()

                iter_cnt += 1
                self.optimizer.zero_grad()
                outputs, alphas_part_max, alphas_org = self.model(input)

                loss_dict = self.loss_fn.forward_all(outputs, targets)
                loss = loss_dict['loss']
                if 'ce' in loss_dict.keys():
                    ce_loss = loss_dict['ce']
                if 'wloss' in loss_dict.keys():
                    wloss = loss_dict['wloss']

                loss.backward()
                self.optimizer.step()

                running_loss += loss
                if ce_loss:
                    running_celoss += ce_loss
                if wloss:
                    running_wloss += wloss
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num
                epoch_pred.extend(predicts.detach().cpu().numpy().tolist())
                epoch_label.extend(targets.detach().cpu().numpy().tolist())

                # # Relabel samples
                # if epoch_i >= self.relabel_epoch:
                #     sm = torch.softmax(outputs, dim=1)
                #     Pmax, predicted_labels = torch.max(sm, 1)  # predictions
                #     Pgt = torch.gather(sm, 1,
                #                        targets.view(-1, 1)).squeeze()  # retrieve predicted probabilities of targets
                #     true_or_false = Pmax - Pgt > self.margin_2
                #     update_idx = true_or_false.nonzero().squeeze()  # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
                #     label_idx = indexes[update_idx]  # get samples' index in train_loader
                #     relabels = predicted_labels[update_idx]  # predictions where (Pmax - Pgt > margin_2)
                #     self.train_dataloader.dataset.labels[
                #         label_idx.cpu().numpy()] = relabels.cpu().numpy()  # relabel samples in train_loader
            self.scheduler.step()
            acc = correct_sum.float() / float(self.train_dataloader.dataset.__len__())
            running_loss = running_loss / iter_cnt
            running_celoss = running_celoss / iter_cnt
            running_wloss = running_wloss / iter_cnt
            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. CE: %.3f. W: %.3f'  % (epoch_i, acc, running_loss, running_celoss, running_wloss))

            epoch_pred = np.array(epoch_pred)
            epoch_label = np.array(epoch_label)

            c_matirx = confusion_matrix(epoch_label, epoch_pred)  # FIXME
            self.logger.write("train c_matrix:\n", c_matirx, '\n')

            with torch.no_grad():
                running_loss = 0.0
                running_celoss = 0.0
                running_wloss = 0.0
                ce_loss = None
                wloss = None

                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                val_pred = []
                val_label = []
                self.model.eval()

                class_cnt = [1 for _ in range(7)]
                mean_pred = [[0 for _ in range(7)] for _ in range(7)]


                for batch_i, (input_first, target_first, input_second, target_second,
                          input_third, target_third, input_forth, target_forth,
                          input_fifth, target_fifth, input_sixth, target_sixth,
                          input_seventh, target_seventh, input_eigth, target_eigth,
                          input_nine, target_nine) \
                        in enumerate(self.val_dataloader):

                    input = torch.zeros(
                        [input_first.shape[0], input_first.shape[1], input_first.shape[2], input_first.shape[3], 9])

                    input[:, :, :, :, 0] = input_first
                    input[:, :, :, :, 1] = input_second
                    input[:, :, :, :, 2] = input_third
                    input[:, :, :, :, 3] = input_forth
                    input[:, :, :, :, 4] = input_fifth
                    input[:, :, :, :, 5] = input_sixth
                    input[:, :, :, :, 6] = input_seventh
                    input[:, :, :, :, 7] = input_eigth
                    input[:, :, :, :, 8] = input_nine
                    input = input.cuda()

                    targets = target_first
                    if isinstance(targets, list):
                        targets = np.array(targets)
                    if isinstance(targets, np.ndarray):
                        targets = torch.from_numpy(target_first)

                    targets = targets.squeeze()
                    targets = targets.cuda()

                    iter_cnt += 1
                    self.optimizer.zero_grad()
                    outputs, alphas_part_max, alphas_org = self.model(input)
                    loss_dict = self.loss_fn.forward_all(outputs, targets)
                    loss = loss_dict['loss']
                    if 'ce' in loss_dict.keys():
                        ce_loss = loss_dict['ce']
                    if 'wloss' in loss_dict.keys():
                        wloss = loss_dict['wloss']

                    running_loss += loss
                    if ce_loss:
                        running_celoss += ce_loss
                    if wloss:
                        running_wloss += wloss

                    iter_cnt += 1
                    _, predicts = torch.max(outputs, 1)
                    correct_num = torch.eq(predicts, targets)
                    val_pred.extend(predicts.detach().cpu().numpy().tolist())
                    val_label.extend(targets.detach().cpu().numpy().tolist())
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += outputs.size(0)



                    targets_lst = targets.tolist()
                    predicts_lst = self.softmax(outputs).detach().cpu().tolist()
                    # [mean_pred[gt] +    for gt in targets_lst]

                    # for i, pred in enumerate(predicts_lst):
                    #     pseudo_gt = np.argmax(pred)
                    #     for j in range(7):
                    #         mean_pred[pseudo_gt][j] += pred[j]
                    #     class_cnt[pseudo_gt] += 1

                    for pred in predicts_lst:
                        try:
                            mean_pred = [[mean_pred[i][j] + pred[j] * (i == int(np.argmax(pred))) for j in range(7)] for i in range(7)]
                            class_cnt[int(np.argmax(pred))] += 1
                        except Exception as e:
                            pdb.set_trace()

                try:
                    mean_pred = [[mean_pred[i][j]/class_cnt[i] for j in range(7)] for i in range(7)]
                    diag_pred = [mean_pred[i][i] for i in range(7)]
                    mean_pred = [[abs(predefine[i] - diag_pred[i]) for i in range(7)] for predefine in mean_pred]
                except Exception as e:
                    pdb.set_trace()

                # if self.use_update_w:
                #     if epoch_i >= self.update_w_start_epoch and epoch_i < 3:
                #         self.loss_fn.update_w(np.array(mean_pred))

                running_loss = running_loss / iter_cnt
                running_celoss = running_celoss / iter_cnt
                running_wloss = running_wloss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                print('[Epoch %d] Validation accuracy: %.4f. Loss: %.3f. CE: %.3f. W: %.3f' % (
                epoch_i, acc, running_loss, running_celoss, running_wloss))
                val_c_matirx = confusion_matrix(val_label, val_pred)  # FIXME
                self.logger.write("val c_matrix:\n", val_c_matirx, '\n')
                for x in range(val_c_matirx.shape[0]):
                    val_c_matirx[x,x] = 0
                
                # if epoch_i >= self.relabel_epoch * 2:
                #     self.loss_fn.update_w(val_c_matirx)

                if best_val_score['value'] < acc:
                    best_val_score['value'] = acc
                    best_val_score['epoch'] = epoch_i
                    if acc > 0.8665:
                        torch.save({'iter': epoch_i,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(), },
                                   os.path.join(self.save_dir, "epoch" + str(epoch_i) + "_acc" + str(acc) + ".pth"))
                        print('Model saved.')

                self.logger.write("best epoch:\n[epoch{}:{}]\n".format(best_val_score['epoch'], best_val_score['value']))



        return








