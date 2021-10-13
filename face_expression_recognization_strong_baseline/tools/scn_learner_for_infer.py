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


class ScnLearner():
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
            if len(batch) ==4:
                batch_data.update({'img_paths':batch_data[3]})
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


#########################################  init w  #######################################################

        with torch.no_grad():
            val_pred = []
            val_label = []
            bingo_cnt = 0
            sample_cnt = 0

            self.model.eval()

            class_cnt = [1 for _ in range(7)]
            mean_pred = [[0 for _ in range(7)] for _ in range(7)]

            err_img_lst = []

            for batch_i, (imgs, targets, _, img_paths) in enumerate(self.val_dataloader):
                _, outputs = self.model(imgs.cuda())
                targets = targets.cuda()

                _, predicts = torch.max(outputs, 1)
                err_idx = torch.ne(predicts, targets).detach().cpu().numpy()

                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

                val_pred.extend(predicts.detach().cpu().numpy().tolist())
                val_label.extend(targets.detach().cpu().numpy().tolist())

                targets_lst = targets.tolist()
                predicts_lst = self.softmax(outputs).detach().cpu().tolist()
              
                for pred in predicts_lst:
                    try:
                        mean_pred = [[mean_pred[i][j] + pred[j] * (i == int(np.argmax(pred))) for j in range(7)] for i in range(7)]
                        class_cnt[int(np.argmax(pred))] += 1
                    except Exception as e:
                        pdb.set_trace()

                img_paths = np.array(img_paths)
                # pdb.set_trace()
                targets = targets.detach().cpu().numpy()
                predicts = predicts.detach().cpu().numpy()
                tmp = np.stack((img_paths[err_idx], targets[err_idx], predicts[err_idx]), axis=1)
                err_img_lst.append(tmp)

            err_result_path = '/home/yz/workspace/project/face_expression_recognization_strong_baseline_code_branch_3/result/err.txt'
            with open(err_result_path, 'w') as f:
                for batch_err in err_img_lst:
                    for err_one, gt, pred in batch_err:
                        f.write(err_one + "\t"+gt + '\t'+pred +'\n')


            val_c_matirx = confusion_matrix(val_label, val_pred)  # FIXME
            val_c_matirx = val_c_matirx.T
            
            for i in range(7):
                val_c_matirx[i][i]=0

            print('val_c_matirx:', sum(sum(val_c_matirx)))
            # try:
            #     mean_pred = [[mean_pred[i][j]/class_cnt[i] for j in range(7)] for i in range(7)]
            #     diag_pred = [mean_pred[i][i] for i in range(7)]
            #     mean_pred = [[abs(predefine[i] - diag_pred[i]) for i in range(7)] for predefine in mean_pred]
            #     mean_pred = np.array(mean_pred)
            # except Exception as e:
            #     pdb.set_trace()

            self.logger.write("val c_matrix:\n", val_c_matirx, '\n')
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            # print('[Epoch %d] Validation accuracy: %.4f. Loss: %.3f. CE: %.3f. W: %.3f' % (
            # epoch_i, acc, running_loss, running_celoss, running_wloss))

            print('Validation accuracy: %.4f' % (acc))
            
            if self.use_update_w:
                self.loss_fn.init_w_line_norm(val_c_matirx)

        return 








