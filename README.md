
# 🚀 Data_idiot 
> **一站式经管实证研究助手 | 零代码 · 本地化 · 可视化**

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Issues][issues-shield]][issues-url]



## 💡 项目简介

**Data_idiot** 是一款专为经管类实证研究设计的桌面端数据处理工具。

它集成了从**数据清洗**、**指标构建**到**模型解释**的全流程工具包。无需编写复杂的 Python/R/Stata 代码，只需通过**鼠标点击交互**，即可完成缺失值插补、综合指数计算、空间计量建模及高质量绘图任务。

---

## 🛠️ 核心功能模块

### 1. 🧹 数据清洗与预处理
- **智能插补**：支持均值、KNN、多重插补等多种算法，并提供插补前后缺失情况可视化对比。
- **灵活合并**：支持 `CSV`/`xlsx` 导入，提供左/右/内/外连接等多种表连接策略。
- **数据重塑**：一键实现宽面板转长面板，适配时序分析需求。

### 2. 📐 综合指数构建
内置多种主流客观赋权法，自动计算权重并生成指数：
- 📈 **熵权法** (Entropy Weight Method)
- 🎯 **TOPSIS** (逼近理想解排序法)
- 📊 **变异系数法** (Coefficient of Variation)
- 🧠 **主成分分析** (PCA)
- 🌐 **灰色关联分析** (Grey Relational Analysis)

### 3. 📈 深度数据探索
- **异质性分析**：自动进行中位数分组，快速识别组间差异。
- **指标比重**：计算并展示关键指标的贡献度，辅助论文图表制作。
- **联合分布**：初步探索变量间的非线性关系。

### 4. 🤖 机器学习与可解释性
不仅提供预测，更关注“为什么”：
- **SHAP 值分析**：精准量化特征对模型预测的驱动因素（支持全局与局部解释）。
- **ALE 图分析**：展示特征对模型预测的平均边际效应，克服 SHAP 在高维下的局限。
- **支持模型**：XGBoost, LightGBM, Scikit-Learn 全系列。

### 5. 🌍 空间计量专用工具
解决传统软件（如 Stata）难以实现的复杂空间模型：
- **空间滞后项生成**：支持 **SLX** 和 **SDEM** 模型变量构建。
- **空间门槛模型**：提供“权重矩阵法”与“虚拟变量法”两种边界计算策略。
- **溢出效应可视化**：制作空间溢出效应边界图。

### 6. 🎨 学术绘图工具箱
- 🔥 相关系数热力图
- 🌲 异质性分析森林图
- 🗺️ 空间溢出效应边界图
- 📉 变量联合分布图

---

## 🚀 快速开始

本应用为 **Windows** 平台打包的独立桌面程序，**无需安装 Python 环境**，开箱即用。

### 方式一：桌面版（推荐）
1. 前往 [Releases 页面](https://github.com/Pei-hsinWang/Data_idiot/releases) 下载最新版本的 `.exe` 自解压包。
2. 运行下载的文件进行自解压。
3. 双击目录中的 `启动文件.exe` 即可启动应用。

### 方式二：源码运行（开发者）
如果你希望自定义功能或跨平台运行：
```bash
# 1. 克隆项目
git clone https://github.com/Pei-hsinWang/Data_idiot.git
cd Data_idiot

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
streamlit run main.py
```

---

## 🖥️ 界面预览

<div align="center">
  <table style="width:100%; border-collapse: collapse;">
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Web_page.png" alt="主界面" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">主界面：功能选择中心</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_Imputation_preview.png" alt="数据插补工具页面" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">数据插补：多种缺失值处理方法可视化展示</p>
      </td>
    </tr>
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_Merge_preview.png" alt="数据集合并工具页面" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">数据合并：支持多格式导入与灵活合并策略</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Data_exploration_preview.png" alt="数据探索页面" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">数据探索：中位数分组、指标比重分析等实用功能</p>
      </td>
    </tr>
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Interpretable_ML_preview.png" alt="机器学习可解释性页面" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">机器学习可解释性：SHAP值分析与图形导出</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/Shap_preview_2.png" alt="Shap分析2" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">SHAP值散点图：特征与输出关系可视化</p>
      </td>
    </tr>
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/ale_preview.png" alt="Ale累计局部效应图" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">Ale累计局部效应图</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/cor_fig_preview.png" alt="相关系数热力图" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">相关系数热力图：展示变量之间的相关关系</p>
      </td>
    </tr>
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/joint_distribution_preview.png" alt="变量联合分布图" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">变量联合分布图：初步探索变量之间的关系</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/hete_analysis_preview.png" alt="异质性分析森林图" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">异质性分析森林图：可视化展现异质性分析结果</p>
      </td>
    </tr>
    <tr>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/specon_preview.png" alt="空间计量工具预览" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">空间计量工具</p>
      </td>
      <td style="text-align: center; padding: 10px;">
        <img src="https://raw.githubusercontent.com/Pei-hsinWang/Data_idiot/master/preview_fig/spillover_preview.png" alt="空间溢出效应边界图" width="100%" style="max-width: 400px; border-radius: 6px; border: 1px solid #eee;">
        <p style="margin-top: 10px; font-weight: 500;">空间溢出效应边界图</p>
      </td>
    </tr>
  </table>
</div>


---

## 📋 系统要求

- **操作系统**: Windows 10 / 11 (64位)
- **内存**: 至少 4GB RAM (建议 8GB 以上以处理大型数据集)
- **磁盘空间**: 至少 800MB 可用空间
- **网络**: 首次运行无需联网，后续使用完全离线

> **注意**：目前暂不支持 macOS 或 Linux 系统的桌面版安装包（源码运行模式除外）。

---

## 🛠️ 技术栈

本项目基于强大的 Python 生态构建：

- **应用框架**: [Streamlit](https://streamlit.io) (交互式前端)
- **打包方案**: [PyStand](https://github.com/skywind3000/PyStand/) (嵌入式 Python 运行时)
- **数据处理**: `Pandas`, `NumPy`
- **机器学习**: `Scikit-learn`, `XGBoost`, `LightGBM`
- **模型解释**: `SHAP`, `PyALE`
- **可视化**: `Matplotlib`, `Seaborn`

---

## ❓ 常见问题 (FAQ)

**Q: 运行速度缓慢怎么办？**
A: 请尝试关闭其他占用内存的程序。如果数据集过大（超过 10 万行），建议在数据探索前先行抽样或使用服务器版源码运行。

**Q: 数据会上传到服务器吗？**
A: **绝对不会**。本软件设计初衷即为本地化运行，所有数据处理均在您的本地计算机内存中完成，关闭软件后数据即释放。

**Q: 如何反馈 Bug 或请求新功能？**
A: 欢迎在 [Issues 页面](https://github.com/Pei-hsinWang/Data_idiot/issues) 提交您的问题或建议。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源，允许免费用于学术研究和商业用途。



[forks-shield]: https://img.shields.io/github/forks/Pei-hsinWang/Data_idiot.svg?style=flat-square
[forks-url]: https://github.com/Pei-hsinWang/Data_idiot/network/members
[stars-shield]: https://img.shields.io/github/stars/Pei-hsinWang/Data_idiot.svg?style=flat-square
[stars-url]: https://github.com/Pei-hsinWang/Data_idiot/stargazers
[issues-shield]: https://img.shields.io/github/issues/Pei-hsinWang/Data_idiot.svg?style=flat-square
[issues-url]: https://github.com/Pei-hsinWang/Data_idiot/issues
[license-shield]: https://img.shields.io/github/license/Pei-hsinWang/Data_idiot.svg?style=flat-square
[license-url]: https://github.com/Pei-hsinWang/Data_idiot/blob/master/LICENSE


