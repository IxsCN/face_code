import  os
import  torch
import  tqdm
import  numpy as np
from torch.utils.data import  Dataset, DataLoader
from sklearn.metrics import classification_report,accuracy_score
# from torch.optim.optimizer import Optimizer

from fer_strong_baseline.utils.logger import TxtLogger
from fer_strong_baseline.utils.meter import AverageMeter
import pdb

from sklearn.metrics import confusion_matrix

import traceback
import logging
import nni
from nni.utils import merge_parameter

class SimpleLearner():
    def __init__(self, 
                 model : torch.nn.Module,
                 loss_fn : torch.nn.Module,
                 optimizer,
                 scheduler,
                 logger,
                 nnilogger,
                 device,

                 save_dir : str,
                 log_steps = 100,
                 gradient_accum_steps = 1,
                 max_grad_norm = 1.0,
                 batch_to_model_inputs_fn = None,
                 early_stop_n = 999,
                 **kwargs
                 ):
                 
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.nnilogger = nnilogger
        self.device =device


        self.save_dir = save_dir
        self.log_steps = log_steps
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        
        self.global_step = 0
        self.step_n = 0
        self.stop = False

        self.need_result_list = None

    def test_and_save(self, epo=-1):
        # m, loss, celoss, wloss = self.val(self.val_dataloader)
        if self.need_result_list:
            m, loss, celoss, wloss, all_preds, all_labels, all_img_fns = self.multi_crop_val(self.val_dataloader)
        else:
            m, loss, celoss, wloss = self.multi_crop_val(self.val_dataloader)
        acc = m['acc']

        if self.best_val_score['value'] < acc:
            self.best_val_score['epoch'] = epo + 1
            self.best_val_score['value'] = acc

            self.early_n = 0

            ######################
            #  close model save  #
            ######################
            model_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(self.model.state_dict(), model_path)

        if self.best_val_total_loss['value'] > loss:
            self.best_val_total_loss['epoch'] = epo + 1
            self.best_val_total_loss['value'] = loss

        if wloss is not None:
            if self.best_val_celoss['value'] > celoss:
                self.best_val_celoss['epoch'] = epo + 1
                self.best_val_celoss['value'] = celoss

            if self.best_val_wloss['value'] > wloss:
                self.best_val_wloss['epoch'] = epo + 1
                self.best_val_wloss['value'] = wloss
                
                # if float(self.alpha) >0.000001:
                #     early_n = 0
                # else:
                #     print('ce loop, not update early_n!\n')
                # model_path = os.path.join(self.save_dir, 'best_wloss.pth')
                # torch.save(self.model.state_dict(), model_path)

        else:
            self.best_val_celoss = self.best_val_total_loss

        if wloss is not None:
            self.logger.write('[{0}] ' \
                        ' lr= {1:.6f}' \
                        ' avg_val_loss= {2:.4f}' \
                        ' avg_val_celoss= {3:.4f} ' \
                        ' avg_val_wloss={4:.4f}' \
                        ' avg_val_acc={5:.4f} '.format(
                        # ' best_val_loss=\{epoch:{6:d},value:{7:.4f}\} ' \
                        # ' best_val_celoss=\{epoch:{8:d},value:{9:.4f}\}' \
                        # ' best_val_wloss=\{epoch:{10:d},value:{11:.4f}\}' \
                        # ' best_val_acc=\{epoch:{12:d},value:{13:.4f}\}'. \
                epo + 1,
                self.scheduler.get_lr()[0],
                loss,
                celoss,
                wloss,
                acc,
                # best_val_total_loss.get('epoch'), best_val_total_loss.get('value'),
                # best_val_celoss.get('epoch'), best_val_celoss.get('value'),
                # best_val_wloss.get('epoch'), best_val_wloss.get('value'),
                # best_val_score.get('epoch'), best_val_score.get('value')
            ))
            self.logger.write('\nbest_val_total_loss:', self.best_val_total_loss.__str__())
            self.logger.write('\nbest_val_celoss:', self.best_val_celoss.__str__())
            self.logger.write('\nbest_val_wloss:', self.best_val_wloss.__str__())
            self.logger.write('\nbest_val_score:', self.best_val_score.__str__())
        else:
            status = '[{0}] ' \
                        ' lr= {1:.6f}' \
                        ' avg_val_loss= {2:.4f}' \
                        ' avg_val_celoss= {3:.4f} ' \
                        ' avg_val_acc={4:.4f} '.format(
                        # ' best_val_loss=\{epoch:{5:d},value:{6:.4f}\} ' \
                        # ' best_val_celoss=\{epoch:{7:d},value:{8:.4f}\}' \
                        # ' best_val_acc=\{epoch:{9:d},value:{10:.4f}\}'. \

                epo + 1,
                self.scheduler.get_lr()[0],
                loss,
                loss,
                acc,
                # best_val_total_loss.get('epoch'), best_val_total_loss.get('value'),
                # best_val_celoss.get('epoch'), best_val_celoss.get('value'),
                # best_val_score.get('epoch'), best_val_score.get('value')
            )
            self.logger.write('\nbest_val_total_loss:', self.best_val_total_loss.__str__())
            self.logger.write('\nbest_val_celoss:', self.best_val_celoss.__str__())
            self.logger.write('\nbest_val_score:', self.best_val_score.__str__())

        # self.logger.write(status)

        self.early_n += 1
        self.logger.write("=="*50)

        if self.early_n > self.early_stop_n:
            self.logger.write('early stopped!')
            self.stop = True

        if self.need_result_list:
            return acc, all_preds, all_labels, all_img_fns
        return acc


    def train_epoch(self, epo):
        epoch_pred = []
        epoch_label = []
        
        train_avg_loss = AverageMeter()
        train_avg_celoss = AverageMeter()
        train_avg_wloss = AverageMeter()
        train_avg_acc = AverageMeter()
        data_iter = tqdm.tqdm(self.train_dataloader)
        train_wloss = None

        for batch in data_iter:
            self.model.train()
            self.optimizer.zero_grad()
            batch_data = self._batch_trans(batch)
            train_loss, train_celoss, train_wloss, acc, pred, label = self.step(self.step_n, batch_data)
            epoch_pred.append(pred)
            epoch_label.append(label)
            train_avg_loss.update(train_loss.item(),1)
            train_avg_celoss.update(train_celoss.item(),1)
            train_avg_acc.update(acc,1)

            # import pdb; pdb.set_trace()
            if train_wloss:
                train_avg_wloss.update(train_wloss.item(), 1)
                status = '[{0}] lr= {1:.6f} loss= {2:.4f} avg_loss= {3:.4f} avg_ce_loss={4:.4f} avg_wloss {5:.4f} avg_acc={6:.4f} '.format(
                epo + 1, self.scheduler.get_lr()[0],
                train_loss.item(), train_avg_loss.avg, train_avg_celoss.avg, train_avg_wloss.avg, train_avg_acc.avg )
            else:
                status = '[{0}] lr= {1:.6f} loss= {2:.4f} avg_loss= {3:.4f} avg_ce_loss={4:.4f} avg_acc={5:.4f} '.format(
                    epo + 1, self.scheduler.get_lr()[0],
                    train_loss.item(), train_avg_loss.avg, train_avg_celoss.avg, train_avg_acc.avg)

            data_iter.set_description(status)
            self.step_n +=1


        self.scheduler.step()  # Update learning rate schedule
        self.model.zero_grad()

        if self.best_train_score['value'] < train_avg_acc.avg:
            self.best_train_score['epoch'] = epo + 1
            self.best_train_score['value'] = train_avg_acc.avg

        if self.best_train_total_loss['value'] > train_avg_loss.avg:
            self.best_train_total_loss['epoch'] = epo + 1
            self.best_train_total_loss['value'] = train_avg_loss.avg

        if self.best_train_celoss['value'] > train_avg_celoss.avg:
            self.best_train_celoss['epoch'] = epo + 1
            self.best_train_celoss['value'] = train_avg_celoss.avg

        if self.best_train_wloss['value'] >  train_avg_wloss.avg:
            self.best_train_wloss['epoch'] = epo + 1
            self.best_train_wloss['value'] = train_avg_wloss.avg
        if train_wloss:
            self.logger.write('[{0}] ' \
                        ' lr= {1:.6f}' \
                        ' avg_train_loss= {2:.4f}' \
                        ' avg_train_celoss= {3:.4f} ' \
                        ' avg_train_wloss={4:.4f}' \
                        ' avg_train_acc={5:.4f} '.format(
                    epo + 1,
                    self.scheduler.get_lr()[0],
                    train_avg_loss.avg,
                    train_avg_celoss.avg,
                    train_avg_wloss.avg,
                    train_avg_acc.avg
            ))
            self.logger.write('\nbest_train_total_loss:', self.best_train_total_loss.__str__())
            self.logger.write('\nbest_train_celoss:', self.best_train_celoss.__str__())
            self.logger.write('\nbest_train_wloss:', self.best_train_wloss.__str__())
            self.logger.write('\nbest_train_score:', self.best_train_score.__str__())
            
        else:
            self.logger.write('[{0}] ' \
                        ' lr= {1:.6f}' \
                        ' avg_train_loss= {2:.4f}' \
                        ' avg_train_celoss= {3:.4f} ' \
                        ' avg_train_acc={4:.4f} '.format(
                epo + 1,
                self.scheduler.get_lr()[0],
                train_avg_loss.avg,
                train_avg_celoss.avg,
                train_avg_acc.avg,
            ))

            self.logger.write('\nbest_train_total_loss:', self.best_train_total_loss.__str__())
            self.logger.write('\nbest_train_celoss:', self.best_train_celoss.__str__())
            self.logger.write('\nbest_train_score:', self.best_train_score.__str__())
           
        # self.logger.write(status)
        # calc train confusion matrix
        assert len(epoch_pred) == len(epoch_label), print("lens of pred and label not equal!")
        epoch_pred_1 = epoch_pred[0]
        epoch_label_1 = epoch_label[0]
        for i in range(1, len(epoch_pred)):
            try:
                epoch_pred_1 = np.hstack([epoch_pred_1, epoch_pred[i]])
                epoch_label_1 = np.hstack([epoch_label_1, epoch_label[i]])
            except:
                import pdb; pdb.set_trace()
        # epoch_pred = np.array(epoch_pred)
        # epoch_label = np.array(epoch_label)
        # epoch_pred = epoch_pred.reshape(-1)
        # epoch_label = epoch_label.reshape(-1)

        epoch_pred = epoch_pred_1
        epoch_label = epoch_label_1

        c_matirx = confusion_matrix(epoch_label, epoch_pred)  # FIXME
        
        self.logger.write("train c_matrix:\n", c_matirx, '\n')

    def validation(self, val_dataloader : DataLoader,
        need_result_list : bool):
        self.val_dataloader = val_dataloader
        self.need_result_list = need_result_list

        if self.need_result_list:
            self.val_dataset = self.val_dataloader.dataset


        # val set log
        self.best_val_score = {'epoch':-1, 'value':0}
        self.best_val_total_loss =  {'epoch':-1, 'value':np.inf}
        self.best_val_celoss =  {'epoch':-1, 'value':np.inf}
        self.best_val_wloss = {'epoch':-1, 'value':np.inf}
        if self.need_result_list:
            test_acc, all_preds, all_labels, all_img_fns = self.test_and_save(-1)
            return test_acc, all_preds, all_labels, all_img_fns

        test_acc = self.test_and_save(-1)
        return test_acc

    def training(self, train_dataloader : DataLoader,
              val_dataloader : DataLoader,
              epoches = 100):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # train set log
        self.best_train_score = {'epoch':-1, 'value':0}
        self.best_train_total_loss = {'epoch':-1, 'value':np.inf}
        self.best_train_celoss =  {'epoch':-1, 'value':np.inf}
        self.best_train_wloss =  {'epoch':-1, 'value':np.inf}

        # val set log
        self.best_val_score = {'epoch':-1, 'value':0}
        self.best_val_total_loss =  {'epoch':-1, 'value':np.inf}
        self.best_val_celoss =  {'epoch':-1, 'value':np.inf}
        self.best_val_wloss = {'epoch':-1, 'value':np.inf}

        early_n = 0
        for epo in range(epoches):
            if self.stop:
                break

            self.train_epoch(epo)
            test_acc = self.test_and_save(epo)
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            self.logger.info('test accuracy %g', test_acc)
            self.logger.info('Pipe send intermediate result done.')

            self.nnilogger.info('test accuracy %g', test_acc)
            self.nnilogger.info('Pipe send intermediate result done.')


        # report final result
        nni.report_final_result(self.best_val_score['value'])
        self.logger.info('Final result is %g', self.best_val_score['value'])
        self.logger.info('Send final result done.')

        self.nnilogger.info('Final result is %g', self.best_val_score['value'])
        self.nnilogger.info('Send final result done.')

        return


    def step(self, step_n, batch_data: dict):
        # _, loggits = self.model(batch_data['imgs'])
        _, loggits = self.model(batch_data['imgs'])
        label = batch_data['labels']
        # pdb.set_trace()
        label = label.squeeze()
        # loss, ce_loss, wloss = self.loss_fn(loggits, label)
        loss_dict = self.loss_fn.forward_all(loggits, label)
        # {"loss": loss, "ce": ce_loss, "wloss": wloss}

        loss = loss_dict["loss"] if "loss" in loss_dict else None
        ce_loss = loss_dict["ce"] if "ce" in loss_dict else None
        wloss = loss_dict["wloss"] if "wloss" in loss_dict else None
        
        # if self.gradient_accum_steps > 1:
        #     loss = loss / self.gradient_accum_steps

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # if (step_n + 1) % self.gradient_accum_steps == 0:
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        loggits = loggits.detach().cpu().numpy()
        pred = np.argmax(loggits, axis=1)
        label = label.detach().cpu().numpy()
        acc = self._accuracy(pred, label)

        return loss, ce_loss, wloss, acc, pred, label

    def _accuracy(self, preds, labels):
        return accuracy_score(y_pred= preds, y_true= labels)#(preds == labels).mean()

    def _acc_and_f1(self, preds, labels):
        acc = self._accuracy(preds, labels)
        self.logger.write(classification_report(y_pred= preds, y_true= labels))
        return {
            "acc": acc,
        }

    def _batch_trans(self, batch):
        # batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
        imgs = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        if self.batch_to_model_inputs_fn is None:
            batch_data = {
                'imgs': imgs,
                'labels': labels,
                'idxs':batch[2] if len(batch) >2 else None,
                'img_fns':batch[3] if len(batch) >3 else None,
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
                _, pred_loggits = self.model( batch_data['imgs'])
                label = batch_data["labels"].squeeze()
                # loss, celoss, wloss = self.loss_fn(pred_loggits,label )
                loss_dict = self.loss_fn.forward_all(pred_loggits, label)

                loss = loss_dict["loss"] if "loss" in loss_dict else None
                celoss = loss_dict["ce"] if "ce" in loss_dict else None
                wloss = loss_dict["wloss"] if "wloss" in loss_dict else None

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

    

    def multi_crop_val(self, val_dataloader : DataLoader):
        eval_loss = 0.0
        eval_celoss = 0.0
        eval_wloss = 0.0
        all_preds = None
        all_labels = None
        all_img_fns = None
        self.model.eval()


        for b_id, batch in tqdm.tqdm(enumerate(val_dataloader)):
            with torch.no_grad():
                batch_data = self._batch_trans(batch)

                batch_imgs = batch_data['imgs']
                bs, ncrops, c,h,w = batch_imgs.size()

                batch_imgs = batch_imgs.view(-1,c,h,w)
                _, pred_loggits =  self.model(batch_imgs)
                pred_loggits = pred_loggits.view(bs,ncrops,-1).mean(1)

                label = batch_data["labels"].squeeze()
                
                # loss, celoss, wloss = self.loss_fn(pred_loggits,label )
                loss_dict = self.loss_fn.forward_all(pred_loggits, label)

                loss = loss_dict["loss"] if "loss" in loss_dict else None
                celoss = loss_dict["ce"] if "ce" in loss_dict else None
                wloss = loss_dict["wloss"] if "wloss" in loss_dict else None

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
        
            if self.need_result_list:
                img_fns = batch_data['img_fns']
                # batchsize = label.shape[0]
                # img_fn_and_labels = self.val_dataset.imgs[b_id * batchsize:(b_id+1) * batchsize]
                # img_fns = np.array([x[0] for x in img_fn_and_labels])

                # label_from_dataset = np.array([x[1] for x in img_fn_and_labels])
                # label_from_batch = label.detach().cpu().numpy()

                # try:
                #     assert (label_from_dataset==label_from_batch).all()
                # except:
                #     pdb.set_trace()

                if all_img_fns is None:
                    all_img_fns = img_fns

                else:
                    try:
                        all_img_fns = np.append(all_img_fns, img_fns, axis=0)
                    except:
                        pdb.set_trace()

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
        if self.need_result_list:
            return result, eval_loss, eval_celoss, eval_wloss, all_preds, all_labels, all_img_fns
        return result, eval_loss, eval_celoss, eval_wloss

    
