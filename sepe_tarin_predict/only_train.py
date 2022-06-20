import gzip
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from layer3_deep_convnet import DeepConvNet
from common.only_trainer import Trainer
import pickle

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

#데이터 읽기(추가되는 데이터)
with open('testdata3D_N.pkl', 'rb') as f:
	datasetL = pickle.load(f)
xt_train, tt_train = datasetL

x_train = np.concatenate([x_train, xt_train])
t_train = np.concatenate([t_train, tt_train])

# 시간이 오래 걸릴 경우 데이터를 줄인다.
#x_train, t_train = x_train[:10000], t_train[:10000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = DeepConvNet()

trainer = Trainer(network, x_train, t_train,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보존
network.save_params("final_params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()