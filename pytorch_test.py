# coding:utf-8
# @TIME         : 2022/8/21 10:32 
# @Author       : acer
# @Project      : CPCS-v2
# @File Name    : pytorch_test.py
#
#                          _ooOoo_
#                         o8888888o
#                         88" . "88                              
#                         (| ^_^ |)
#                         O\  =  /O
#                      ____/`---'\____
#                    .'  \\|     |//  `.
#                   /  \\|||  :  |||//  \
#                  /  _||||| -:- |||||-  \
#                  |   | \\\  -  /// |   |
#                  | \_|  ''\---/''  |   |
#                  \  .-\__  `-`  ___/-. /
#                ___`. .'  /--.--\  `. . ___
#              ."" '<  `.___\_<|>_/___.'  >'"".
#            | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#            \  \ `-.   \_ __\ /__ _/   .-` /  /
#      ========`-.____`-.___\_____/___.-`____.-'========
#                           `=---='
#      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                 佛祖保佑             永无BUG

import torch
seed = 20
torch.manual_seed(seed)

batch = torch.randint(10, [20, 1])
print(batch)

a = torch.rand([10, 10])

dig = torch.eye(10)
print(dig)

print(a)

b = torch.argmax(a, dim=1, keepdim=True)

print(b)

# label = torch.tensor([])
for i, num in enumerate(batch):
    idx = b[num]
    if i == 0:
        label = dig[idx]
    else:
        label = torch.cat((label, dig[idx]), dim=0)

print(label.squeeze())

A = [0,12,3,65,6,9,6,0,2,1]
dict = {}
for i in range(0, len(A)):
    dict[i] = A[i]
a = sorted(dict.items(), key=lambda x: x[1])
print(a)
print(sorted(A, reverse=True))

test = torch.rand([20, 10])
print(test)
test = test.reshape(10, 2, 10)
print(test)

