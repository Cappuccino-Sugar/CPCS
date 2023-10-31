"""
    绘图 填充交叉区域
"""
import numpy as np
import matplotlib.pyplot as mp

# 1.准备数据
x = np.linspace(0, 8*np.pi, 1000)
sin_x = np.sin(x)
cos_x = np.cos(x/2) / 2


# 2.创建窗口
mp.figure("Fill", facecolor="lightgray", figsize=(10, 6))
mp.title("Fill", fontsize=20)

# 3.X刻度
x_ticks = np.linspace(0, 8*np.pi, 9)
x_ticks_label = [0, r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$",
                 r"$5\pi$", r"$6\pi$", r"$7\pi$", r"$8\pi$"]
mp.xticks(x_ticks, x_ticks_label)

aix = mp.gca()
aix.xaxis.set_minor_locator(mp.MultipleLocator(np.pi/2))
mp.grid(linestyle=":")

# 4.绘制曲线
mp.plot(x, sin_x, c="b", label=r"$y=sin(x)$")
mp.plot(x, cos_x, c="r", label=r"$y=\frac{cos(\frac{x}{2})}{2}$")

# 5.绘制填充
mp.fill_between(x, sin_x, cos_x, sin_x > cos_x, color="dodgerblue", alpha=0.3)
mp.fill_between(x, sin_x, cos_x, sin_x < cos_x, color="orangered", alpha=0.3)

mp.legend()
mp.show()