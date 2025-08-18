import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np
# 导入必要的模型库
from sklearn.model_selection import  train_test_split
# 导入自由模块
from utils import DataUtils  
from utils import DataExporter
from ml_utils import SHAPAnalyzer
from ml_utils import Ale
# set the page title and icon
st.set_page_config(page_title="Data_Imputation", page_icon="📈")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常
                              
                    """)
# set GUI title and description
st.title("📈 机器学习可解释性")
st.markdown("""
            ### 使用说明👋
            - 1. 上传一个 `Excel` 文件，支持`xlsx` 和 `csv` 格式，大小不超过200Mb
            - 2. 选择用于预测的**特征列**
            - 3. 选择被预测的**目标列**
            - 4. 选择进行拟合的机器学习模型
            - 5. 选择**任务类型**
            - 6. 选择绘图参数
            - 7. 完成训练并打印结果
            """)


# 主体功能区
tab1, tab2 = st.tabs(["1️⃣ 功能一: Shap值法",
                           "2️⃣ 功能二: ALE累计局部效应"])
# ======================= 功能一：Shap值法 =======================
with tab1:
    st.subheader("1️⃣ Shap值法")

    uploaded_file = st.file_uploader("上传`xlsx`或`csv`文件", type=["csv", "xlsx"],key="shap")
    
    if uploaded_file is not None:
        # 读取上传的文件
        df_shap = DataUtils.read_file(uploaded_file)
        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df_shap.head())

        # 选择用于预测的特征列
        choising_cols = df_shap.select_dtypes(include=[np.number]).columns.tolist()
        if not choising_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            feature_cols = st.multiselect("选择用于预测的**特征列**", choising_cols, default=choising_cols[1:])
        # 选择被预测的目标列
        target_col = st.selectbox("选择被预测的**目标列**", df_shap.columns)

        # 模型选择
        model_name = st.selectbox("选择使用的模型", ["xgboost", "lightgbm", "random_forest"])

        # 任务类型
        task_type = st.selectbox("选择任务类型", ["regression", "classification"])

        # Top 特征数量
        num_top_features = st.slider("选择要显示的Top特征数量", min_value=1, max_value=20, value=6)

        # 散点图设置
        scatter_density = st.checkbox("启用点密度采样", value=True)
        max_points = st.slider("每个散点图最多显示的点数", min_value=50, max_value=500, value=100)
        scatter_alpha = st.slider("散点透明度", min_value=0.1, max_value=1.0, value=0.7)
        scatter_rows = st.slider("每列显示的散点图数量", min_value=1, max_value=5, value=2)

        # 图像分辨率
        fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1800, value=600)

        # 调参设置
        param_search_method = st.selectbox("选择参数搜索方法", ["optuna","grid_search"])
        n_trials = st.slider("Optuna 试验次数", min_value=10, max_value=100, value=30)
        cv = st.slider("交叉验证折数", min_value=2, max_value=10, value=5)

        # 使用示例
        if st.button("开始分析"):
            analyzer = SHAPAnalyzer(
                df=df_shap,           # 数据集
                feature_cols=df_shap[feature_cols],  # 特征列
                target_col=df_shap[target_col],  # 目标列
                num_top_features=num_top_features,    # 显示最重要的特征数量
                scatter_rows=scatter_rows,        # 每列显示的散点图数量
                fig_dpi=fig_dpi,           # 图像分辨率
                scatter_density=scatter_density,  # 启用点密度采样
                max_points=max_points,        # 每个散点图最多显示的点数
                scatter_alpha=scatter_alpha,     # 散点透明度
                scatter_size=45,       # 散点大小
                model_name=model_name,  # 使用选定模型
                param_search_method=param_search_method,  # 参数搜索方法
                n_trials=n_trials,           # Optuna 试验次数
                cv=cv,                  # 交叉验证折数
                task_type=task_type  # 任务类型
            )
            # 分析并可视化
            analyzer.analyze_and_visualize(show_plot=False)
            # 绘制
            image_buffers = analyzer.export_fig(analyzer.fig, dpi=fig_dpi)

            # 展示预览图像

            st.image(image_buffers['png'], caption="特征重要性图与SHAP依赖图")
            # 提供多种格式下载按钮
            col1, col2, col3 = st.columns(3)
            col4, col5, col6  = st.columns(3)

            with col1:
                st.download_button(
                    label="📥 下载 PNG 图像",
                    data=image_buffers['png'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            with col2:
                st.download_button(
                    label="📄 下载 PDF 图像",
                    data=image_buffers['pdf'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

            with col3:
                st.download_button(
                    label="📐 下载 SVG 图像",
                    data=image_buffers['svg'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml"
                )
            with col4:
                st.download_button(
                    label="📜 下载 EPS 图像",
                    data=image_buffers['eps'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                    mime="image/eps"
                )            

            with col5:
                st.download_button(
                    label="🖼️ 下载 TIFF 图像",
                    data=image_buffers['tiff'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                    mime="image/tiff"
                )

            with col6:
                st.download_button(
                    label="📷 下载 JPG 图像",
                    data=image_buffers['jpg'],
                    file_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")
# ======================= 功能二：ALE累计局部效应 =======================
with tab2:
    st.subheader("2️⃣ ALE累计局部效应")
    uploaded_file = st.file_uploader("上传`xlsx`或`csv`文件", type=["csv", "xlsx"],key="ALE")
    
    df = None
    if uploaded_file is not None:
        # 读取上传的文件
        df = DataUtils.read_file(uploaded_file)
        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df.head())

    if df is not None:
        # 选择用于预测的特征列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            feature_cols = st.multiselect("选择用于预测的**特征列**", numeric_cols, default=numeric_cols[1:],key="ALE_features")
        # 选择被预测的目标列
        target_col = st.selectbox("选择被预测的**目标列**", df.columns,key="ALE_target")

        # 模型选择
        model_name = st.selectbox("选择使用的模型", ["xgboost", "lightgbm","gradient_boosting", "random_forest"],key="ALE_model")

        # 任务类型
        task_type = st.selectbox("选择任务类型", ["auto_detect", "regression", "classification"], key="ALE_task")
        
        # 模型寻优算法选择
        optimization_method = st.selectbox("选择模型寻优算法", ["GridSearchCV", "optuna"], key="ALE_optimization")
        
        # 初始化session_state存储模型
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'model_params' not in st.session_state:
            st.session_state.model_params = {}
        if 'X_test' not in st.session_state:
            st.session_state.X_test = None
        # 语言选择
        zn = bool(st.checkbox("是否选择中文绘图, 默认使用English",key="joint_zn"))        
        # 训练按钮
        train_button = st.button("开始训练模型", key="train_ale")                

        # 检查是否需要重新训练模型
        current_params = {
            'model_name': model_name,
            'feature_cols': feature_cols,
            'target_col': target_col,
            'task_type': task_type,
            'optimization_method': optimization_method
        }
        
        need_retrain = train_button and feature_cols and target_col in df.columns
         
        if need_retrain:
            with st.spinner("模型训练中..."):
                # 准备数据
                X = df[feature_cols]
                y = df[target_col]
                
                # 自动检测任务类型
                ale_analyzer = Ale(df,random_state=42,zn=zn)
                if task_type == "auto_detect":
                    task_type = ale_analyzer._determine_task_type(y)
                    st.info(f"自动检测到任务类型: {task_type}")
                
                # 分割训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 训练模型
                best_model, best_params = ale_analyzer.train_model(
                    X_train, y_train, X_test,y_test, model_name, task_type, optimization_method
                )
                
                # 保存模型和参数到session_state
                st.session_state.ale_analyzer = ale_analyzer
                st.session_state.trained_model = best_model
                st.session_state.model_params = current_params
                st.session_state.X_test = X_test
                st.success(f"模型训练完成！最佳参数: {best_params}")
        elif not feature_cols or target_col not in df.columns:
            st.warning("请先选择有效的特征列和目标列")
        
        # 只有当模型训练完成后才显示特征选择和绘图功能
        if st.session_state.trained_model is not None:

            # 图像分辨率
            fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1200, value=600,key="cor_fig_dpi")
            # 间隔宽度
            interval_width = st.slider("地毯图间隔宽度", min_value=20, max_value=100, value=50,key="grid_size") 

            st.subheader("ALE累计局部效应图")

            feature_to_plot = st.selectbox("选择要可视化的特征", feature_cols, key="ale_feature")
            st.markdown("🔑ALE图绘制——单特征")
            if st.button("生成ALE图", key="ale_button"):
                # 生成图像但不立即显示
                fig = st.session_state.ale_analyzer.plot_ale(st.session_state.trained_model,
                                            st.session_state.X_test,
                                            feature_to_plot,
                                            grid_num=interval_width,
                                            figsize=(9, 6),
                                            bootstrap_uncertainty=True,
                                            bootstrap_reps=100,
                                            show_mean_curve=True,
                                            show_ci_band=True,
                                            show_plot=False)
                # 导出图像为多种格式
                image_buffers = DataExporter.export_fig(fig, dpi=fig_dpi)
                #st.pyplot(fig)
                # 预览图像
                st.image(image_buffers['png'], caption="ALE 累计局部效应图")
                                # 提供多种格式下载按钮
                col1, col2, col3 = st.columns(3)
                col4, col5, col6  = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 下载 PNG 图像",
                        data=image_buffers['png'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )

                with col2:
                    st.download_button(
                        label="📄 下载 PDF 图像",
                        data=image_buffers['pdf'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

                with col3:
                    st.download_button(
                        label="📐 下载 SVG 图像",
                        data=image_buffers['svg'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                        mime="image/svg+xml"
                    )
                with col4:
                    st.download_button(
                        label="📜 下载 EPS 图像",
                        data=image_buffers['eps'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                        mime="image/eps"
                    )            

                with col5:
                    st.download_button(
                        label="🖼️ 下载 TIFF 图像",
                        data=image_buffers['tiff'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                        mime="image/tiff"
                    )

                with col6:
                    st.download_button(
                        label="📷 下载 JPG 图像",
                        data=image_buffers['jpg'],
                        file_name=f"aleplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg"
                    )              
            else:
                st.warning("请选择可视化特征后点击开始")
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")