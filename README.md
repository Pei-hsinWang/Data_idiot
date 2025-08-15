# 🔬 Data idiots Analytics Suite

数据处理和分析脚本工具

<!-- PROJECT SHIELDS -->


[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->



<br />
<div align="center">


  <h3 align="center">🔬 DataPro Analytics Suite</h3>

  <p align="center">
    一站式数据处理桌面应用
    <br />
    <a href="https://github.com/Pei-hsinWang/Data_idiot"><strong>探索项目文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Pei-hsinWang/Data_idiot">查看 Demo</a>
    ·
    <a href="https://github.com/Pei-hsinWang/Data_idiot/issues">报告 Bug</a>
    ·
    <a href="https://github.com/Pei-hsinWang/Data_idiot/issues">提出新功能建议</a>
  </p>
</div>

## ✨ 产品特点

<div align="center">
  <table>
    <tr>
      <td align="center"><b>📊 直观可视化</b></td>
      <td align="center"><b>⚡ 高效处理</b></td>
      <td align="center"><b>🔒 本地运行</b></td>
    </tr>
    <tr>
      <td>交互式图表与数据探索</td>
      <td>傻瓜式点击交互处理数据集</td>
      <td>数据安全，无需上传至云端</td>
    </tr>
  </table>
</div>


## 🔒 系统要求
- Windows 10/11 (64位)
- 至少4GB RAM
- 800MB可用磁盘空间

## 🚀 部署说明
本应用为 Windows 平台打包的桌面程序，部署方式如下：
1. 访问 [GitHub项目页面](https://github.com/Pei-hsinWang/Data_idiot)
2. 找到右侧的Release(发布)版本，点击下载最新版本的`exe`自解压文件
3. 运行下载好的文件进行自解压。
4. 解压完成后，双击目录下的 `启动文件.exe` 即可运行应用。

无需联网或安装额外依赖，所有依赖均已打包在安装包中。

#### 命令行启动：
1. 打开命令行窗口，进入项目目录。
2. 运行以下命令启动应用：
```cmd
streamlit run main.py
```

## 🛠️ 技术栈与框架

本项目基于以下技术构建：

- **前端界面**: [Streamlit](https://streamlit.io) - 快速构建数据应用的 Python 框架
- **打包工具**: [PyStand](https://github.com/skywind3000/PyStand/) - 嵌入式打包 Python 解释器与依赖库打包
- **数据可视化**: [Matplotlib](https://matplotlib.org/)
- **数据处理**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **缺失值插补**: [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/),[xgboost](https://xgboost.readthedocs.io/en/latest/)
- **ML与可解释性**: [Scikit-learn](https://scikit-learn.org/), [SHAP](https://shap.readthedocs.io/en/latest/),[xgboost](https://xgboost.readthedocs.io/en/latest/),[lightgbm](https://lightgbm.readthedocs.io/en/latest/),[PyALE](https://github.com/DanaJomar/PyALE)


## 📱 界面预览

以下是 Data_idiot的主要功能页面截图，直观展示了各模块的交互方式与可视化效果：

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;">
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/main.png" alt="主界面" width="100%">
    <p>主界面：功能选择中心</p>
  </div>

  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_Imputation_preview.png" alt="数据插补工具页面" width="100%">
    <p>数据插补：多种缺失值处理方法可视化展示</p>
  </div>

  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_Merge_preview.png" alt="数据集合并工具页面" width="100%">
    <p>数据合并：支持多格式导入与灵活合并策略</p>
  </div>

  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_exploration_preview.png" alt="数据探索页面" width="100%">
    <p>数据探索：中位数分组、指标比重分析等实用功能</p>
  </div>

  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Interpretable_ML_preview.png" alt="机器学习可解释性页面" width="100%">
    <p>机器学习可解释性：SHAP值分析与图形导出</p>
  </div>

  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Shap_preview_2.png" alt="Shap分析2" width="100%">
    <p>SHAP值散点图：特征与输出关系可视化</p>
  </div>  
    <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/ale_preview.png" alt="Ale累计局部效应图" width="100%">
    <p>Ale累计局部效应图</p>
  </div>
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/cor_fig_preview.png" alt="相关系数热力图" width="100%">
    <p>相关系数热力图：展示变量之间的相关关系</p>
  </div>
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/joint_distribution_preview.png" alt="变量联合分布图" width="100%">
    <p>变量联合分布图：初步探索变量之间的关系</p>
  </div>
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/hete_analysis_preview.png" alt="异质性分析森林图" width="100%">
    <p>异质性分析森林图：可视化展现异质性分析结果</p>
  </div>
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/specon_preview.png" alt="空间计量工具预览" width="100%">
    <p>空间计量工具</p>  
  </div>
  <div style="text-align: center; max-width: 45%;">
    <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/spillover_preview.png" alt="空间溢出效应边界图" width="100%">
    <p>空间溢出效应边界图</p>

  </div>  
</div>

## 📋 主要功能

### 🔍 数据插补
- 查看数据描述性统计信息  
- 支持多种缺失值处理方法（如均值、KNN、多重插补等）  
- 可视化对比插补前后的缺失情况  

### 📁 数据合并
- 支持 `CSV` 和 `xlsx` 文件导入  
- 提供多种表连接方式（左连接、右连接、内连接等）  

### 📊 综合指数计算
#### **支持多种客观赋权法的综合指数方法**
1. 📈 **熵权法**（Entropy Weight Method）
2. 🎯 **Topsis法**（逼近理想解排序法）
3. 📊 **变异系数法**（Coefficient of Variation）
4. 🧠 **主成分分析法**（PCA）
5. 🌐 **灰色关联法**（Grey Relational Analysis）

### 📈 数据探索
- **中位数分组**：适用于实证研究中的异质性分析  
- **指标比重计算****：用于论文图表支持与结果展示  
- **宽面板转长面板**：方便时序数据分析与建模  

### 🗄机器学习可解释性
- **SHAP值解释**：理解模型预测背后的驱动因素  
- **ALE图解释**：ALE图展示了特征对模型预测的平均影响。

### 🎨 绘图工具
- **相关系数热力图**：展示变量之间的相关关系
- **变量联合分布图**：初步探索变量之间的关系
- **异质性分析森林图**：可视化展现异质性分析结果
- **空间溢出效应边界图**：显示空间溢出效应边界

### 🌍 空间计量工具
- **空间滞后项生成**：用于计算Stata没有的**SLX**和**SDEM**模型
- **空间门槛模型——权重矩阵法**：用于空间溢出效应的边界计算
- **空间门槛模型——虚拟变量法**：用于空间溢出效应的边界计算

## ❓ 常见问题

**Q: 应用运行缓慢怎么办?**
A: 关闭其他占用内存较大的程序，或尝试降低数据集大小。

**Q: 支持Mac或Linux系统吗?**
A: 目前仅支持Windows系统。

## 📄 许可证

本项目采用 [MIT许可证](LICENSE) 开源

<!-- links -->
[contributors-shield]: https://img.shields.io/github/contributors/Pei-hsinWang/Data_idiot?style=flat-square
[contributors-url]: https://github.com/Pei-hsinWang/Data_idiot/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/Pei-hsinWang/Data_idiot?style=flat-square
[forks-url]: https://github.com/Pei-hsinWang/Data_idiot/network/members

[stars-shield]: https://img.shields.io/github/stars/Pei-hsinWang/Data_idiot?style=flat-square
[stars-url]: https://github.com/Pei-hsinWang/Data_idiot/stargazers

[issues-shield]: https://img.shields.io/github/issues/Pei-hsinWang/Data_idiot?style=flat-square
[issues-url]: https://github.com/Pei-hsinWang/Data_idiot/issues

[license-shield]: https://img.shields.io/github/license/Pei-hsinWang/Data_idiot?style=flat-square
[license-url]: https://github.com/Pei-hsinWang/Data_idiot/blob/main/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-blue
[linkedin-url]: https://www.linkedin.com/in/你的-linkedin-用户名/
