import torch

# 2.2.1 创建行向量
rvec = torch.empty(12)
print(rvec)
# tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]) 

#获取张量属性
print(rvec.shape)
# torch.Size([12])
print(rvec.size())
# torch.Size([12])

#改变形状
rvec = rvec.view(3, 4)
print(rvec.shape)
# torch.Size([3, 4]) (新张量和旧张量共享同一块内存)

nvec = rvec.clone().view(3, 4)
#使用clone()之后不与rvec共享同一块内存


#创建各元素为0，形状为(2, 3, 4)的张量
zerovec = torch.zeros(2, 3, 4)
print(zerovec)

'''
tensor([[[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]],

        [[0., 0., 0., 0.],
         [0., 0., 0., 0.],
         [0., 0., 0., 0.]]])
'''

#创建各元素为1的张量，指定类型为long型的张量
onevec = torch.ones(2, 3, 4, dtype = torch.long)
print(onevec)
'''
tensor([[[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]],

        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]])
'''

#创建指定元素的张量
assigned_vec = torch.tensor([1, 2, 3])
print(assigned_vec)
# tensor([1, 2, 3])

#创建形状为(3, 4)的随机数张量
ranvec = torch.rand(3, 4)
print(ranvec)
'''
tensor([[0.1238, 0.5242, 0.1967, 0.6718],
        [0.0164, 0.0121, 0.4536, 0.3567],
        [0.1717, 0.0430, 0.7121, 0.5674]])
'''

#2.2.2 运算
x = torch.ones(1, 2, dtype = torch.long)
y = torch.tensor([2, 3])
sum = x + y
print(sum)
# tensor([[3, 4]])

mul = x * y
print(mul)
# tensor([[2, 3]]) 此处的乘法是每一个对应数字互相乘起来

#2.2.3 广播机制
ten1 = torch.arange(2, 4).view(1, 2)
print(ten1)
# tensor([[2, 3]])
ten2 = torch.arange(2, 5).view(3, 1)
print(ten2)
'''
tensor([[2],
        [3],
        [4]])
'''

sum_ten = ten1 + ten2
print(sum_ten)
'''
tensor([[4, 5],
        [5, 6],
        [6, 7]])
'''

# 对一个张量所有元素进行求和并转换为python自带的类型输出
tensor = torch.ones(2, 3, dtype = torch.long)
print(tensor.sum().item())
# result: 6

# 2.2.4 索引
x = torch.arange(2, 14).view(3, 4)
print(x)
'''
tensor([[ 2,  3,  4,  5],
        [ 6,  7,  8,  9],
        [10, 11, 12, 13]])
'''
y = x[1:3]
print(y)
'''
tensor([[ 6,  7,  8,  9],
        [10, 11, 12, 13]])
'''

y += 1
print(x[1:3])
'''
tensor([[ 7,  8,  9, 10],
        [11, 12, 13, 14]])
'''

# 改变y会连带着x一起改变

# 2.2.5 运算的内存开销

# 如果两个实例的ID一致，则代表着它们对应的内存空间地址相同，反之则不同

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)

# false

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before)
# true

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before)
# true


# 2.2.6 Tensor 和 NumPy之间的转换
# Tensor 转 NumPy
a = torch.arange(2, 6).view(4)
b = a.numpy()
print(a, b)
# tensor([2, 3, 4, 5]) [2 3 4 5]，二者使用同样的内存
b += 1
print(a)
# tensor([3, 4, 5, 6])

# NumPy转Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
# tensor([1., 1., 1., 1., 1.], dtype=torch.float64)



