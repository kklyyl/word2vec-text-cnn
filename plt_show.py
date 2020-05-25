import matplotlib.pyplot as plt

#图形输入值
input_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
#图形输出值

train_acc = [0.062, 0.188,0.125, 0.344,  0.406,0.281,0.328, 0.312, 0.328,0.484,0.469,0.578,0.594, 0.578,0.688, 0.750,0.719,  0.672, 0.688, 0.734, 0.812,  0.734, 0.812, 0.734, 0.875, 0.891, 0.875,0.844, 0.922,  0.938, 0.938,0.891,  0.844
     ]
val_acc=[0.156,0.203,0.234,0.247,0.289,0.312,0.344,0.360,0.376,0.409,0.419,0.435,0.448,0.462,0.476,0.476,0.492,0.501, 0.502,0.510, 0.508,0.520,0.526,0.530, 0.538, 0.546,0.539,0.551,0.554, 0.549,0.554,0.558,0.559,]

#plot根据列表绘制出有意义的图形，linewidth是图形线宽，可省略
plt.plot(input_values,train_acc,linewidth=5)
plt.plot(input_values,val_acc,linewidth=5)

#设置图标标题
plt.title(" fitting curve",fontsize = 24)
#设置坐标轴标签
plt.xlabel("steps",fontsize = 14)
plt.ylabel("acc value",fontsize = 14)
#设置刻度标记的大小
plt.tick_params(axis='both',labelsize = 14)
#打开matplotlib查看器，并显示绘制图形
plt.show()