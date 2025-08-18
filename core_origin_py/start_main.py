import pandas as pd
import streamlit as st
import time 
from datetime import datetime
from io import BytesIO
from utils import ImputerPipeline  # 从 utils.py 导入类
def start():
    # GUI_模块
    st.header("😆欢迎来到Westbeacon的小站")
    st.markdown("""
    这是一个基于streamlit构建的应用集合, 构建了许多经济管理类学术小工具,帮助更好快速的完成论文。
    - 📈数据插补模块
    - 📊数据合并模块
    - 📩综合指数集结模块
    - 🔍数据探索模块
    - 🗄可解释性机器学习模块
    - 🎨数据可视化模块
    - 🌍空间计量模块
    #### 👈在侧边栏选择一个模块来使用吧
    """)
    
    st.markdown("""    ### 📝主要功能
    #### 🔍 数据插补
        - 查看数据描述性统计信息
        - 支持多种缺失值处理方法（如均值、KNN、多重插补等）
        - 可视化对比插补前后的缺失情况
    #### 📁 数据合并
        - 支持 CSV 和 Excel 文件导入
        - 提供多种表连接方式（左连接、右连接、内连接等）
    #### 📊 数据集结
        - 支持多种客观赋权法的综合指数方法
        - 📈 熵权法（Entropy Weight Method）
        - 🎯 Topsis法（逼近理想解排序法）
        - 📊 变异系数法（Coefficient of Variation）
        - 🧠 主成分分析法（PCA）
        - 🌐 灰色关联法（Grey Relational Analysis）
    #### 📈 数据探索
        - 中位数分组：适用于实证研究中的异质性分析
        - 指标比重计算：用于论文图表支持与结果展示
        - 宽面板转长面板：方便时序数据分析与建模
    #### 🗄机器学习可解释性
        - SHAP值解释：理解模型预测背后的驱动因素
        - ALE累计局部效应：理解模型预测的局部效应
    #### 🎨绘图
        - 相关系数图
        - 联合分布图
        - 异质性分析：森林图
    #### 🌍空间计量
        - 空间滞后项生成
        - 空间溢出效应边界--权重矩阵法
        - 空间溢出效应边界--虚拟变量法
    """
)
