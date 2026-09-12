import io

# ---- daily log ----
p = '.workbuddy/memory/2026-09-08.md'
s = io.open(p, encoding='utf-8').read()
anchor = '- 任务记录 jobs/JOB-2026-0906-043_nondiag_freq_batch.md。'
add = anchor + '''

## C1 频率单位、内部模式投影与043记录补正批（JOB-2026-0906-044，新增量化 0）

- **043 频率全部撤回**（频率缩小 42.695x：质量转电子质量却用 amu 系数 5140.487）：撤回"无负模""-1.2 为平转噪声""极低频平坦势盆""可接受表征进入下一阶段"；补正 042（撤回"梯度接近 DFT 非谐性极限"，无证据）。
- **后处理来源如实登记**：磁盘 c1_nondiag_freq.py 崩溃（eigh 解包 3 vs 2）于写 harmonic_analysis 前；最终 JSON 频率由另一次临时内联命令补入——不以崩溃脚本冒充成功版本。
- **成本修正**：SCF **2**（端点复现 + mf_clean.kernel() 额外 1 次）；D2 构造=源码推算 **42 次 d2_grad FD**（无运行时计数器）；步长敏感性**未完成**；原始未对称化矩阵**缺失**。
- **单位修复**：SI 推导 amu 5140.4871 / 电子质量 219474.6314（比 42.695）；两路线一致 **8.3e-9 cm^-1**。
- **15 维内部子空间**（MW 空间 sqrt(m) 平动转动、SVD 秩=6、正交补 21x15、不删最低六个）：**15 模含 1 个真实负模 -72.186 cm^-1**（外部重叠 6.1e-17，NH3 libration/torsion，H 位移 0.55 Bohr）；单体模与 026 参考吻合（787/1034/1335/1342/1676/1680/3533/3660）。
- **PySCF 交叉核对**：imaginary_freq=False + mass= -> **n_imaginary=1**，最大差 **3.6e-6 cm^-1**（另发现 PySCF 复数虚频被 float() 丢弃 -> 假 0.0）。
- **梯度投影修正**：内部占 99.9998%、外部 1.46e-7、重构残差 1e-20；梯度同时在低频分子间模与高频分子内模，**不在负曲率方向**（负模仅 5.5e-7）；修正 043"主要在高频模"。
- **无 SCF 验证全过**：026 NH3 6 模 4.4e-8 / O3 3 模 1.6e-8；合成矩阵 15 预置值全恢复 4.7e-11、负模保留、外部曲率 1e-17、刚性运动不变 4.8e-11。
- 任务记录 jobs/JOB-2026-0906-044_freq_fix_batch.md。'''
if anchor in s:
    s = s.replace(anchor, add, 1)
    io.open(p, 'w', encoding='utf-8').write(s)
    print('daily log appended')
else:
    io.open(p, 'a', encoding='utf-8').write(add)
    print('daily log appended at end')

# ---- MEMORY.md ----
p2 = '.workbuddy/memory/MEMORY.md'
s2 = io.open(p2, encoding='utf-8').read()
a = ('- **C1 表征完成**：ΔE=-2.258 kcal/mol、N-O≈3.13 Å、无负曲率、两个优化器确认；'
     '待总指挥决定：接受表征进入下一阶段（文献三维取向核对或 C2）vs 继续收敛；'
     '用户确认水体类型/pH/波长 → 水相扩展。')
b = ('- **044 批（2026-09-08，新增量化 0）**：**043 频率结论全部撤回**——单位错误'
     '（质量转电子质量却用 amu 系数 → 频率缩小 **42.695x**）＋投影错误；最终 JSON '
     '频率由崩溃后的临时内联命令补入（不以崩溃脚本冒充）。修正后 **15 维内部子空间'
     '分析（MW sqrt(m) 平动转动、SVD 秩=6、正交补 21x15）→ 15 模含 1 个真实内部负模'
     ' **-72.186 cm^-1**（NH3 libration，外部重叠 6.1e-17）**；PySCF n_imaginary=1 '
     '交叉核对 3.6e-6 cm^-1；梯度内部占 99.9998%、不在负曲率方向；无 SCF 验证全过'
     '（026 NH3/O3 恢复、合成矩阵）。\n'
     '- **C1 现状**：ΔE=-2.258 kcal/mol、N-O≈3.13 Å、**非驻点且存在内部负曲率方向**'
     '——不可据此接受表征；待总指挥决定后续路线；用户确认水体类型/pH/波长 → 水相扩展。')
if a in s2:
    s2 = s2.replace(a, b, 1)
else:
    s2 = s2 + '\n' + b + '\n'
io.open(p2, 'w', encoding='utf-8').write(s2)
print('MEMORY.md updated')
