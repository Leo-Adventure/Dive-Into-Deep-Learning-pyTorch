from numpy import require
import torch

x = torch.ones(2,2,requires_grad = True)# 必须要显式指定，否则默认为False
print(x)
print(x.grad_fn)
# None

y = x + 2
print(y)
print(y.grad_fn)
# 注意x是直接创建的，所以它没有grad_fn, 
# 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。

# .grad_fn属性是创建该Tensor的Function

z = y * y * 3
out = z.mean()
print(z, out)
'''
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) 
tensor(27., grad_fn=<MeanBackward0>)
'''

out.backward() # 会追踪在out上面的所有梯度计算
print(x.grad)

# 再次反向传播
out2 = x.sum()
out2.backward()
print(x.grad)
'''
tensor([[5.5000, 5.5000],
        [5.5000, 5.5000]])
'''

# 如果不清空x的梯度，则grad会随着调用函数的增加而进行累加
out3 = x.sum()
out3.backward() # 再次进行追踪
print(x.grad)
'''
tensor([[6.5000, 6.5000],
        [6.5000, 6.5000]])
'''
x.grad.data.zero_()
print(x.grad)
'''
tensor([[0., 0.],
        [0., 0.]])
'''
out3.backward()
print(x.grad)
'''
tensor([[1., 1.],
        [1., 1.]])
'''

# 使用with torch.no_grad()中断梯度追踪
x = torch.tensor(1.0, requires_grad = True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

print(x, x.requires_grad)
print(y1, y1.requires_grad)
print(y2, y2.requires_grad)
print(y3, y3.requires_grad)

# important！！
# 求y3的梯度
y3.backward()
print(x.grad)
# 由于y3 = y1 + y2但是Y2被with torch.no_grad()包裹，所以y2不会回传梯度
# tensor(2.)

# 如果只是想要修改tensor的数值，但是又不希望被autograd记录（即不希望影响反向传播），那么就对tensor.data进行操作
x = torch.ones(1, requires_grad = True)

print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100 # 只是改变x的数值，不会影响最终求梯度的结果
# x *= 100 # 叶子节点不能进行原地操作（in_place operation）

y.backward()
print(x)
print(x.grad)
