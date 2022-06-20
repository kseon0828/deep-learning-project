import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from layer3_deep_convnet import DeepConvNet
import pickle

pkl_file = "final_params.pkl"
network = DeepConvNet()

for i in range(1,4):
    with open('DataSet%d_3D.pkl'%i, 'rb') as f:
        datasetL = pickle.load(f)

    x_test, t_test = datasetL

    network.load_params(pkl_file)
    print("%d's accauracy"%i)
    print(network.accuracy(x_test, t_test), end='\n\n')