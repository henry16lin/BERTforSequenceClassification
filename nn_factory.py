import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class nn_factory():
    def __init__(self, model, device, tokenizer):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer
    
    def fit(self, epoch, optimizer, train_loader, val_loader, model_save_path):
        val_loss, val_acc = np.Inf, 0.
        train_loss_hist, train_acc_hist = [],[]
        val_loss_hist, val_acc_hist = [],[]

        for ep in range(1, epoch + 1):
            epoch_begin = time.time()
            cur_train_loss, cur_train_acc = self.train(train_loader, optimizer, ep)
            cur_val_loss, cur_val_acc = self.val(val_loader)

            print('elapse: %.2fs \n' % (time.time() - epoch_begin))

            if cur_val_loss <= val_loss:
                print('improve validataion loss, saving model...\n')
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_save_path, 'best_model.pt'
                           % (ep, cur_val_loss, cur_val_acc)))

                val_loss = cur_val_loss
                val_acc = cur_val_acc

            train_loss_hist.append(cur_train_loss)
            train_acc_hist.append(cur_train_acc)
            val_loss_hist.append(cur_val_loss)
            val_acc_hist.append(cur_val_acc)

        # save final model
        state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(state, os.path.join(self.model_save_path, 'last_model.pt'))

        ### graph train hist ###
        fig = plt.figure()
        plt.plot(train_loss_hist)
        plt.plot(val_loss_hist)
        plt.legend(['train loss','val loss'], loc='best')
        plt.savefig(os.path.join(self.model_save_path, 'loss.jpg'))
        plt.close(fig)
        fig = plt.figure()
        plt.plot(train_acc_hist)
        plt.plot(val_acc_hist)
        plt.legend(['train acc', 'val acc'], loc='best')
        plt.savefig(os.path.join(self.model_save_path, 'acc.jpg'))
        plt.close(fig)
    
    def train(self, train_loader, optimizer, epoch):
        print('[epoch %d]train on %d data......'%(epoch, len(train_loader.dataset)))
        train_loss, correct = np.Inf, 0

        self.model.train()
        for data, label in tqdm(train_loader):
            device_data = {}
            for k, v in data.items():
                device_data[k] = v.to(self.device)
            device_label = label.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(device_data)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, device_label)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += pred.eq(device_label).sum().item()

        train_loss /= len(train_loader.dataset)
        acc = correct/len(train_loader.dataset)

        print('training set: average loss: %.4f, acc: %d/%d(%.3f%%)' %(train_loss,
              correct, len(train_loader.dataset), 100 * acc))

        return train_loss, acc


    def val(val_loader):
        print('validation on %d data......'%len(val_loader.dataset))
        self.model.eval()
        val_loss, correct = np.Inf, 0.
        with torch.no_grad():
            for data, label in val_loader:
                device_data = {}
                for k, v in data.items():
                    device_data[k] = v.to(self.device)
                device_label = label.to(self.device)
                
                output = self.model(device_data)

                criterion = nn.CrossEntropyLoss()
                val_loss += criterion(output, device_label).item() #sum up batch loss

                pred = output.argmax(dim=1)
                correct += pred.eq(device_label).sum().item()
            val_loss /= len(val_loader.dataset)  # avg of sum of batch loss
            acc = correct/len(val_loader.dataset)

        print('Val set:Average loss:%.4f, acc:%d/%d(%.3f%%)' %(val_loss,
              correct, len(val_loader.dataset), 100. * acc))

        return val_loss, acc
    
    
    def predict_proba(self, sentence):
        wrapped_input = self.tokenizer(sentence, max_length=200, add_special_tokens=True, 
                                       truncation=True, padding='max_length', return_tensors="pt")

        with torch.no_grad():
            log_prob = F.log_softmax(self.model(wrapped_input))
            pred_prob = torch.exp(log_prob).data.cpu().numpy()

        return pred_prob


    def predict(self, sentence):
        pred_prob = self.predict_proba(sentence)
        pred_ind = np.argmax(pred_prob, axis=1)

        return pred_ind