import signal
from transModel import *
from processData import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import time
import matplotlib.pyplot as plt

class ModelManagement():
    def __init__(self, model, train_dataloader, test_dataloader=None, device=torch.device('cpu')):
        # === static ===
        self.EPOCH_PROGRESS_COUNT = 10
        self.EPOCH_CHECKPOINT_COUNT = 100
        self.EPOCH_MIN_CHECKPOINT = 5
        self.PATIENCE_EPOCH = 3
        self.MEAN_EPOCH = 3
        self.HARD_LOSS_GAP_THRESHOLD = 0.6
        self.SOFT_LOSS_GAP_THRESHOLD = 0.3
        self.HARD_LOSS_GL_THRESHOLD = 0.5
        self.SOFT_LOSS_GL_THRESHOLD = 0.2
        self.EXACT_GAP_LIMIT = 10
        self.EXACT_NO_GAP = 1.2
        self.EXACT_GAP_JUMP = 1.5
        self.TOP_K = 3
        # === init ===
        self.model = model
        self.train_dl = train_dataloader
        self.test_dl = test_dataloader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=5e-6)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-6)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        # === dynamic related with model===
        self.test_loss = float('inf')
        self.train_loss = float('inf')
        self.best_test_loss = float('inf')
        self.best_train_loss = float('inf')
        # === dynamic related with mgmt===
        self.epoch_count = 0
        self.train_loss_list = []
        self.test_loss_list = []
        self.best_checkpoints = dict()
        # === tmp & plt & flags ===
        self.epoch = 0
        self.epochs = 0
        self.lr = 1e-4
        self.monitor_flag = []
        self.roll_back_flag = False
        self.patience = self.PATIENCE_EPOCH
        self.last_exact_gap = 1
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.train_line1 = None
        self.test_line1 = None
        self.train_line2 = None
        self.test_line2 = None
        # for manual exit
        self._register_signal_handler()

    def _register_signal_handler(self):
        #  SIGINT(Ctrl+C)   SIGTERM
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)

    def _handle_termination(self, signum, frame):
        print(f"\n!!! CATCH SIGNAL: {signum}, SAVING BEFORE TERMINATING!!!\n...")
        try:
            self.save_checkpoint()
            print("SAVED SUCCESS, EXIT NOW")
        except Exception as e:
            print(f"SAVED FAILED: {e}, EXIT")
        finally:
            signal.signal(signal.SIGINT, self.original_sigint)
            signal.signal(signal.SIGTERM, self.original_sigterm)
            exit(0)

    def init_train(self, lr=1e-4, weight_decay=5e-6, factor=0.5, patience=10, min_lr=1e-6):
        self.model.train()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.lr = lr

    def init_eval(self):
        self.model.eval()
        self.model.to(self.device)

    def progress_info(self, force=False):
        if self.monitor_flag:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}|{self.test_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}|{self.best_test_loss:.4f}, lr: {self.lr:.6f}; "
                f"\n ---> Monitor: {','.join(self.monitor_flag)}")
            self.monitor_flag = []
        elif (self.epoch + 1) % self.EPOCH_PROGRESS_COUNT == 0:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}|{self.test_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}|{self.best_test_loss:.4f}, lr: {self.lr:.6f}")
        elif force:
            logging.info(
                f"[{self.epoch + 1}/{self.epochs}]|Epoch_{self.epoch_count}] -> Loss: {self.train_loss:.4f}|{self.test_loss:.4f}, "
                f"Best loss: {self.best_train_loss:.4f}|{self.best_test_loss:.4f}, lr: {self.lr:.6f}")

        if self.epoch_count % self.EPOCH_CHECKPOINT_COUNT == 0:
            self.save_checkpoint()

    def save_checkpoint(self, ckp_name=''):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'best_test_loss': self.best_test_loss,
            'best_train_loss': self.best_train_loss
        }
        if ckp_name:
            weight_path = './saves/' + ckp_name
        else:
            weight_path = f'./saves/CheckPoint_Ep{self.epoch_count}_{self.train_loss:.4f}_{self.test_loss:.4f}.pth'
        torch.save(checkpoint, weight_path)
        logging.info(f"checkpoint: {weight_path} Saved")

    def load_checkpoint(self, ckp_name='', only_weights=False):
        if not ckp_name:
            print('No checkpoint provided.')
            return
        weight_path = './saves/' + ckp_name
        try:
            ckpt = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            if not only_weights:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scheduler.load_state_dict(ckpt["scheduler"])
                self.train_loss = ckpt["train_loss"]
                self.test_loss = ckpt["test_loss"]
                self.best_test_loss = ckpt["best_test_loss"]
                self.best_train_loss = ckpt["best_train_loss"]
            logging.info(f"checkpoint: {weight_path} Loaded")
        except Exception as e:
            logging.error(f"load_statues Error: {e}", exc_info=True)

    def save_state(self, state_name=''):
        manager_state = {
            'epoch_count': self.epoch_count,
            'train_loss_list': self.train_loss_list,
            'test_loss_list': self.test_loss_list,
            'best_checkpoints': self.best_checkpoints
        }
        if state_name:
            state_path = './saves/' + state_name
        else:
            state_path = f'./saves/State_Ep{self.epoch_count}_{self.best_test_loss:.4f}.pkl'
        with open(state_path, "wb") as f:
            pickle.dump(manager_state, f)
        logging.info(f"State saved at {state_path}")

    def load_state(self, state_name=''):
        if not state_name:
            print('No state provided.')
            return
        state_path = './saves/' + state_name
        try:
            with open(state_path, 'rb') as f:
                manager_state = pickle.load(f)
                self.epoch_count = manager_state['epoch_count']
                self.train_loss_list = manager_state['train_loss_list']
                self.test_loss_list = manager_state['test_loss_list']
                self.best_checkpoints = manager_state['best_checkpoints']
                logging.info(f"State: {state_path} Loaded")
        except Exception as e:
            logging.error(f"load_statues Error: {e}", exc_info=True)

    def clear_state(self):
        pass

    def save_best(self):
        self.save_checkpoint('best_test.pth')
        self.best_checkpoints[self.epoch_count] = (self.train_loss, self.test_loss)
        self.patience = self.PATIENCE_EPOCH

    def roll_back(self):
        self.load_checkpoint('best_test.pth', False)
        self.patience = self.PATIENCE_EPOCH

    def trans_data2dev(self, *args):
        transferred_args = []
        for arg in args:
            try:
                transferred_arg = arg.to(self.device, non_blocking=True)
                transferred_args.append(transferred_arg)
            except Exception as e:
                logging.error(f"trans_data2dev Error: {e}", exc_info=True)
                transferred_args.append(arg)

        if len(transferred_args) == 1:
            return transferred_args[0]
        return tuple(transferred_args)

    def init_dashboard(self):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # ion
        plt.switch_backend('TkAgg')
        plt.ion()
        # draw
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 6), num="Loss Dashboard")
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Linear Scale')
        self.ax1.grid(alpha=0.3)
        self.ax2.set_yscale('log')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Loss')
        self.ax2.set_title('Log Scale')
        self.ax2.grid(alpha=0.3)
        # init
        self.train_line1, = self.ax1.plot(range(1, self.epoch_count+1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=4)
        self.train_line2, = self.ax2.plot(range(1, self.epoch_count + 1),
                                        self.train_loss_list,
                                        label='Train Loss',
                                        marker='o',
                                        markersize=4)
        self.test_line1, = self.ax1.plot(range(1, self.epoch_count+1),
                                       self.test_loss_list,
                                       label='Test loss',
                                       marker='s',
                                       markersize=4)
        self.test_line2, = self.ax2.plot(range(1, self.epoch_count + 1),
                                       self.test_loss_list,
                                       label='Test loss',
                                       marker='s',
                                       markersize=4)
        # update
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_dashboard(self):
        self.train_line1.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line1.set_ydata(self.train_loss_list)
        self.train_line2.set_xdata(range(1, len(self.train_loss_list) + 1))
        self.train_line2.set_ydata(self.train_loss_list)
        self.test_line1.set_xdata(range(1, len(self.test_loss_list) + 1))
        self.test_line1.set_ydata(self.test_loss_list)
        self.test_line2.set_xdata(range(1, len(self.test_loss_list) + 1))
        self.test_line2.set_ydata(self.test_loss_list)
        # auto-set
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        # update
        self.ax1.legend()
        self.ax2.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # optional
        #time.sleep(0.01)

    def show_dashboard(self):
        plt.ioff()
        plt.show()

    def loss_algorithm(self):
        # 1. 初始免检测轮次
        # 在全局训练次数<EPOCH_MIN_CHECKPOINT时，仅保存loss，不做任何处理
        if self.epoch_count < self.EPOCH_MIN_CHECKPOINT:
            self.save_best()
            return

        # 2. 绝对比GAP判断 （硬回滚指标）
        # 绝对比GAP， >1说明测试loss大于训练loss，限制一定的绝对比例
        exact_gap = self.test_loss / self.train_loss
        # 绝对比GAP数量级差异，直接回滚
        if exact_gap > self.EXACT_GAP_LIMIT:
            self.monitor_flag.append(f'Exact Gap too High! ({exact_gap:.2f}>{self.EXACT_GAP_LIMIT}), roll back!')
            self.roll_back_flag = True
            return
        # 本次绝对比GAP对比之前的对比GAP，容许一定的扩大，但不允许跳变，否则直接回滚
        if (exact_gap > self.EXACT_NO_GAP) and (exact_gap > self.last_exact_gap * self.EXACT_GAP_JUMP):
            self.monitor_flag.append(f'Exact Gap Jump! ({exact_gap:.2f}->{self.last_exact_gap:.2f}), roll back!')
            self.roll_back_flag = True
            return
        self.last_exact_gap = exact_gap
        # 测试集loss下降
        tobe_save_best = False
        if self.test_loss < self.best_test_loss:
            # GAP也没有过拟合趋势，保存最佳loss
            if exact_gap < self.EXACT_NO_GAP:
                self.save_best()
                self.monitor_flag.append(f'Save Best Loss! ({self.best_test_loss:.4f}->{self.test_loss:.4f})')
                self.best_test_loss = self.test_loss
                return
            else:
                # 虽然loss下降了，但GAP存疑，先待定，等软检测完成
                tobe_save_best = True

        # 3. 滑动窗口归一化后的绝对差（硬回滚+软检测）
        # 计算最近的MEAN_EPOCH个epoch的loss训练集平均值作为基数，不含本epoch
        loss_base = sum(self.train_loss_list[-(self.MEAN_EPOCH+1):-1])/self.MEAN_EPOCH
        # 把loss按base归一化到同一数量级
        norm_train_loss = self.train_loss / loss_base
        norm_test_loss = self.test_loss / loss_base
        # 偏差： 相差的值及比例
        loss_gap = norm_test_loss - norm_train_loss
        # 偏差过大
        # 如：base=2, train=1,test=2，归一后为0.5,1，GAP是0.5
        # 过大直接硬回滚，中等则消耗耐心值
        if loss_gap > self.HARD_LOSS_GAP_THRESHOLD:
            self.monitor_flag.append(f'MEAN Exact Gap too High! ({loss_gap:.2f}->{loss_base:.2f}), roll back!')
            self.roll_back_flag = True
            return
        elif loss_gap > self.SOFT_LOSS_GAP_THRESHOLD:
            self.monitor_flag.append(f'MEAN Exact Gap become High! ({loss_gap:.2f}->{loss_base:.2f}), Patience: {self.patience}->{self.patience-1}')
            self.patience -= 1
            tobe_save_best = False

        # 4. 背离（硬回滚+软检测）
        # 背离：泛化损失（Generalization Loss，GL），训练损失还在下降，但测试损失已经停止下降甚至上升
        # 通常 GL>10%~30% 判定为轻度过拟合，>50% 为重度过拟合；
        g_loss = (self.test_loss - self.best_test_loss)/self.best_test_loss
        if self.train_loss < self.best_train_loss:
            self.best_train_loss = self.train_loss
            if g_loss > self.HARD_LOSS_GL_THRESHOLD:
                self.monitor_flag.append(f'GL Gap too High! ({g_loss:.2f}), roll back!')
                self.roll_back_flag = True
                return
            elif g_loss > self.SOFT_LOSS_GL_THRESHOLD:
                self.monitor_flag.append(f'GL Gap become High! ({g_loss:.2f}), Patience: {self.patience}->{self.patience-1}')
                self.patience -= 1
                tobe_save_best = False

        # 5. 如果耐心值耗完，则回滚
        if self.patience <= 0:
            self.monitor_flag.append(f'GL Gap too High! ({g_loss:.2f}), roll back!')
            self.roll_back_flag = True
            return

        # 6. 待定的最佳测试loss，GAP可接受，则保存
        if tobe_save_best:
            self.save_best()
            self.monitor_flag.append(f'Save Best Loss! (Soft)({self.best_test_loss:.4f}->{self.test_loss:.4f})')
            self.best_test_loss = self.test_loss


    def get_batch_loss(self, one_pack_data):
        # unpack: Source -> processData
        src, tgt_input, src_mask, tgt_mask, tgt_tgt = one_pack_data
        src, tgt_input, src_mask, tgt_mask, tgt_tgt = self.trans_data2dev(src, tgt_input, src_mask, tgt_mask, tgt_tgt)
        # forward
        output, _, _, _, _, _, _ = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        # loss
        output_flat = output.reshape(-1, VOCAB_SIZE)
        tgt_flat = tgt_tgt.reshape(-1)
        loss = self.criterion(output_flat, tgt_flat)
        return loss

    def get_batch_output(self, one_pack_data):
        # unpack: Source -> processData
        src, tgt_input, src_mask, tgt_mask, tgt_tgt = one_pack_data
        src, tgt_input, src_mask, tgt_mask, tgt_tgt = self.trans_data2dev(src, tgt_input, src_mask, tgt_mask, tgt_tgt)
        # forward
        output, _, _, _, _, _, _ = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
        return output.cpu()

    def train_epoch(self):
        epoch_loss = torch.tensor(0.0, device=self.device)
        for one_pack_data in self.train_dl:
            self.optimizer.zero_grad()
            loss = self.get_batch_loss(one_pack_data)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        return epoch_loss.item()/len(self.train_dl)

    def eval_test_epoch(self):
        epoch_loss = torch.tensor(0.0, device=self.device)
        for one_pack_data in self.test_dl:
            loss = self.get_batch_loss(one_pack_data)
            epoch_loss += loss
        return epoch_loss.item()/len(self.test_dl)

    def train_epochs(self, eps):
        self.epochs = eps
        for ep in range(self.epochs):
            self.epoch = ep
            self.model.train()
            self.train_loss = self.train_epoch()
            self.train_loss_list.append(self.train_loss)

            self.model.eval()
            with torch.no_grad():
                self.test_loss = self.eval_test_epoch()
                self.test_loss_list.append(self.test_loss)

            self.loss_algorithm()
            self.scheduler.step(self.test_loss)
            self.lr = self.optimizer.param_groups[0]['lr']
            self.epoch_count += 1

            self.progress_info()
            self.update_dashboard()

            if self.roll_back_flag:
                self.roll_back()
                self.roll_back_flag = False

        self.save_checkpoint()

    def predict_manual(self, txt, is_cn=False):
        # pre-process
        src_ids = sce2id_fillpad(txt, is_cn=is_cn)
        src_tensor = torch.tensor([src_ids], dtype=torch.long)
        src_mask = generate_src_mask(src_tensor)

        # init output
        tgt_ids = [BOS_ID]
        pred_text = []

        with torch.no_grad():
            for i in range(MAX_LEN - 1):
                # pad & mask
                tgt_padded = tgt_ids + [PAD_ID] * (MAX_LEN - len(tgt_ids))
                tgt_tensor = torch.tensor([tgt_padded], dtype=torch.long)
                tgt_mask = generate_tgt_mask(tgt_tensor)
                # infer
                src_tensor, tgt_tensor, src_mask, tgt_mask = self.trans_data2dev(src_tensor, tgt_tensor, src_mask, tgt_mask)
                output, _, _, _, _, _, _ = self.model(src_tensor, tgt_tensor, src_mask, tgt_mask)
                last_token_output = output[0, -1, :]
                logging.debug(f"last_token_output: {last_token_output.shape}")
                # top 3
                probs = torch.softmax(last_token_output, dim=-1)
                top3_probs, top3_ids = torch.topk(probs, 3)

                next_token = "Next token: "
                for j in range(3):
                    token = idx2token[top3_ids[j].item()]
                    prob = top3_probs[j].item()
                    next_token += f"[{token}|{prob:.3f}]"
                print(next_token)

                # next id
                next_id = top3_ids[0].item()
                tgt_ids.append(next_id)
                pred_text.append(idx2token[next_id])

                if next_id == EOS_ID:
                    break

    def predict_auto(self):
        logging.info("Start auto testing...")
        test_length = len(self.test_dl)
        i=1
        for one_pack_data in self.test_dl:
            src, _, _, _, tgt_tgt = one_pack_data
            output = self.get_batch_output(one_pack_data)
            for j in range(DEFAULT_BATCH_SIZE):
                ids_output = output[j]
                probs = torch.softmax(ids_output, dim=-1)
                top_probs, top_ids = torch.topk(probs, 1)
                src_tokens = [idx2token[id] for id in src[j].tolist()]
                tgt_tokens = [idx2token[id] for id in tgt_tgt[j].tolist()]
                top_ids_flat = [i[0] for i in top_ids.tolist()]
                pred_tokens = [idx2token[id] for id in top_ids_flat]
                pred_probs = [f'{p[0]:.4f}' for p in top_probs.tolist()]
                print(f'Input_[{i}|{test_length}][{j}|{DEFAULT_BATCH_SIZE}] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                print(f'Src: {' '.join(src_tokens)}')
                print(f'Tgt: {''.join(tgt_tokens)}')
                print(f'Pred: {''.join(pred_tokens)}')
                print(f'Probs: {','.join(pred_probs)}')
                print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            i+=1

if __name__ == '__main__':
    print('init model...')
    model = MyTransf()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('init ModelManagement...')
    m_mgmt = ModelManagement(model, None, None, dev)
    print('Empty, Do nothing, Exit...')