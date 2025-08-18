import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import DataUtils  # 从 utils.py 导入类
from utils import DataExporter
from utils import IndicatorsAggregation
# set the page title and icon
st.set_page_config(page_title="Indicator_Aggregation", page_icon="")

st.title("📩 数据集结工具")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常
                              
                    """)

# 主体功能区
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✅ 熵权法", 
                                        "✅ 熵权TOPSIS法",
                                        "✅ 变异系数法",  
                                        "✅ 主成分分析法",
                                        "✅ 灰色关联法"])
# ======================= 1️⃣ 熵权法 =======================
with tab1:
    st.subheader("1️⃣熵权法")
     
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择指标方向 (正向或负向)
            - 下载处理后的结果
            """)    
    with st.expander("🔍 熵权法简介"):
         st.markdown("""
             熵权法是一种基于 **信息熵** 的指标权重计算方法。它通过计算各指标的信息熵来确定指标的权重，信息熵越大，说明该指标的信息量越小，权重越低；反之，信息熵越小，说明该指标的信息量越大，权重越高。
                    """)
         st.markdown("""
             熵权法的步骤如下：
             1. 计算每个指标的 **信息熵**
             2. 计算每个指标的 **信息增益**
             3. 计算每个指标的 **权重**
             4. 根据指标方向（正向或负向）调整权重
             5. 输出结果
             """)
         st.markdown("""### 🧮 公式推导：""")
         st.latex(r"""
            \text{正向指标归一化: } x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}
        """)
         st.latex(r"""
            \text{负向指标归一化: } x'_{ij} = \frac{\max(x_j) - x_{ij}}{\max(x_j) - \min(x_j)}
        """)
         st.latex(r"""
            \text{概率计算: } p_{ij} = \frac{x'_{ij}}{\sum_{i=1}^{n} x'_{ij}}
        """)
         st.latex(r"""
            \text{信息熵: } e_j = -\frac{1}{\ln n} \sum_{i=1}^{n} p_{ij} \ln p_{ij}
        """)
         st.latex(r"""
            \text{差异系数: } d_j = 1 - e_j
        """)
         st.latex(r"""
            \text{权重计算: } w_j = \frac{d_j}{\sum_{j=1}^{m} d_j}
        """)
         st.latex(r"""
            \text{最终得分: } S_i = \sum_{j=1}^{m} w_j \cdot x'_{ij}
        """)
         st.markdown("""
            最终得分由加权归一化值求和得到，用于综合评价多个对象的优劣排序。
        """)
         
    uploaded_file_Entropy = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="median")
    
    if uploaded_file_Entropy is not None:
        # 使用封装方法读取文件
        df_entropy = DataUtils.read_file(uploaded_file_Entropy)

        st.write("原始数据预览：")
        st.dataframe(df_entropy.head())
       # 统计原始数据缺失值
        stats_df = DataUtils.get_missing_stats(df_entropy)
        st.markdown("原始数据缺失值统计")
        st.dataframe(stats_df)

        # 选择用于计算熵权的数值列
        numeric_cols = df_entropy.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于熵权计算的列", numeric_cols, default=numeric_cols[:])

            # 新增：为每个选中的列单独选择指标方向
            directions = {}
            st.markdown("### 设置各列的指标方向")
            for col in selected_cols:
                dir_key = f"dir_{col}"  # 使用唯一 key 避免冲突
                direction = st.selectbox(f"{col} 的指标方向", ["正向指标", "负向指标"], key=dir_key)
                directions[col] = direction

            if st.button("开始计算"):
                with st.spinner('🔄 正比处理并计算权重，请稍等...'):
                    # 调用 utils 中的方法进行熵权法计算
                    result_df,score_df = IndicatorsAggregation.entropy_weight_method(df_entropy[selected_cols], cols=selected_cols, directions=directions)

                st.success(f"✅ 熵权法计算完成，已根据各列方向计算权重与得分")
                
                st.dataframe(result_df[['信息熵', '差异系数', '权重']])  
                st.dataframe(score_df[['得分', '排名']])

                # 获取导出参数
                export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="export_ewm")

                # 获取 MIME 类型和扩展名
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                # 是否同时导出两个表
                export_both = st.checkbox("同时导出权重与得分结果", value=True, key="export_both")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((result_df, score_df), export_format, sheet_names=("权重结果", "得分结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(result_df, export_format)
                        score_str = DataExporter.convert_df_to_format(score_df, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出权重结果
                    export_data = DataExporter.convert_df_to_format(result_df, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"entropy_weight_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")


# ======================= 2️⃣ 熵权Topsis法 =======================
with tab2:
    st.subheader("2️⃣ 熵权TOPSIS法简介")
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持 xlsx 和 csv 格式
            - 选择指标方向 (正向或负向)
            - 下载处理后的结果
            """)    
    with st.expander("🔍 方法说明"):
        st.markdown("""
            熵权TOPSIS法是一种结合 **熵权法** 与 **TOPSIS 排序评价法** 的综合评价方法。
            
            - **熵权法**：用于确定每个指标的客观权重；
            - **TOPSIS**：利用权重对数据进行归一化后，计算每个对象与最优解和最劣解的距离，并据此排序。
        """)
        st.markdown("""
            熵权TOPSIS法的主要步骤：
            1. 对数据进行归一化处理；
            2. 计算各指标的信息熵和权重；
            3. 构建加权决策矩阵；
            4. 找出正理想解（最大值）和负理想解（最小值）；
            5. 计算每个对象到理想解的距离；
            6. 计算相对接近度并排序。
        """)
        st.markdown("### 🧮 公式推导")
        st.latex(r"\text{正向指标归一化: } x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{负向指标归一化: } x'_{ij} = \frac{\max(x_j) - x_{ij}}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{概率计算: } p_{ij} = \frac{x'_{ij}}{\sum_{i=1}^{n} x'_{ij}}")
        st.latex(r"\text{信息熵: } e_j = -\frac{1}{\ln n} \sum_{i=1}^{n} p_{ij} \ln p_{ij}")
        st.latex(r"\text{差异系数: } d_j = 1 - e_j")
        st.latex(r"\text{权重计算: } w_j = \frac{d_j}{\sum_{j=1}^{m} d_j}")
        st.latex(r"\text{加权归一化矩阵: } v_{ij} = w_j \cdot x'_{ij}")
        st.latex(r"\text{正理想解: } v^+_j = \max(v_{ij})")
        st.latex(r"\text{负理想解: } v^-_j = \min(v_{ij})")
        st.latex(r"\text{距离计算: } D^+_i = \sqrt{\sum_{j=1}^{m}(v_{ij} - v^+_j)^2}")
        st.latex(r"\text{距离计算: } D^-_i = \sqrt{\sum_{j=1}^{m}(v_{ij} - v^-_j)^2}")
        st.latex(r"\text{相对接近度: } C_i = \frac{D^-_i}{D^+_i + D^-_i}")

    uploaded_file_topsis = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="topsis")

    if uploaded_file_topsis is not None:
        df_topsis = DataUtils.read_file(uploaded_file_topsis)
        st.write("原始数据预览：")
        st.dataframe(df_topsis.head())

        # 统计缺失值
        stats_df = DataUtils.get_missing_stats(df_topsis)
        st.markdown("原始数据缺失值统计")
        st.dataframe(stats_df)

        numeric_cols = df_topsis.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于熵权TOPSIS计算的列", numeric_cols, default=numeric_cols[:])
            directions = {}
            for col in selected_cols:
                dir_key = f"direction_{col}"
                direction = st.selectbox(f"{col} 的指标方向", ["正向指标", "负向指标"], index=0, key=dir_key)
                directions[col] = direction

            if st.button("开始计算"):
                with st.spinner('🔄 正在计算熵权TOPSIS得分，请稍等...'):
                    weight_df,score_df = IndicatorsAggregation.entropy_weight_topsis_method(df_topsis[selected_cols], directions=directions)

                st.success("✅ 熵权TOPSIS计算完成！")
                st.dataframe(weight_df)
                st.dataframe(score_df)

                export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="export_topsis")
                
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)
                
                export_both = st.checkbox("同时导出权重与得分结果", value=True, key="export_both")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((weight_df, score_df), export_format, sheet_names=("权重结果", "得分结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(weight_df, export_format)
                        score_str = DataExporter.convert_df_to_format(score_df, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出权重结果
                    export_data = DataExporter.convert_df_to_format(weight_df, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"entropy_topsis_weight_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")  

# ======================= 3️⃣ 变异系数法 =======================
with tab3:
    st.subheader("3️⃣ 变异系数法")

    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择多个数值列进行变异系数法计算
            - 选择每列的指标方向（正向/负向）
            - 下载处理后的结果
            """)
    
    with st.expander("🔍 方法说明"):
        st.markdown("""
            变异系数法是一种基于 **标准差与均值比值** 的客观赋权方法。

            它通过计算各指标的变异系数来确定其权重，变异系数越大，说明该指标波动性越强，权重越高；
            反之，变异系数越小，说明该指标越稳定，权重越低。
                   """)
        st.markdown("""
            变异系数法的主要步骤：
            1. 对数据进行归一化处理（区分正向/负向指标）
            2. 计算每列的均值与标准差
            3. 计算每列的变异系数（CV = 标准差 / 均值）
            4. 根据变异系数计算权重
            5. 加权合成得分并排序
            """)
        st.markdown("### 🧮 公式推导")
        st.latex(r"\text{正向指标归一化: } x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{负向指标归一化: } x'_{ij} = \frac{\max(x_j) - x_{ij}}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{均值: } \mu_j = \frac{1}{n} \sum_{i=1}^{n} x'_{ij}")
        st.latex(r"\text{标准差: } \sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x'_{ij} - \mu_j)^2}")
        st.latex(r"\text{变异系数: } CV_j = \frac{\sigma_j}{\mu_j}")
        st.latex(r"\text{权重计算: } w_j = \frac{CV_j}{\sum_{j=1}^{m} CV_j}")
        st.latex(r"\text{最终得分: } S_i = \sum_{j=1}^{m} w_j \cdot x'_{ij}")

    uploaded_file_cv = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="cv")

    if uploaded_file_cv is not None:
        df_cv = DataUtils.read_file(uploaded_file_cv)
        st.write("原始数据预览：")
        st.dataframe(df_cv.head())

        # 统计缺失值
        stats_df = DataUtils.get_missing_stats(df_cv)
        st.markdown("原始数据缺失值统计")
        st.dataframe(stats_df)

        numeric_cols = df_cv.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于变异系数法计算的列", numeric_cols, default=numeric_cols[:])

            directions = {}
            st.markdown("### 设置各列的指标方向")
            for col in selected_cols:
                dir_key = f"dir_cv_{col}"  # 使用唯一 key 避免冲突
                direction = st.selectbox(f"{col} 的指标方向", ["正向指标", "负向指标"], key=dir_key)
                directions[col] = direction

            if st.button("开始计算"):
                with st.spinner('🔄 正在计算变异系数法权重与得分，请稍等...'):
                    weight_df,score_df = IndicatorsAggregation.coefficient_of_variation_method(df_cv[selected_cols], cols=selected_cols, directions=directions)

                st.success("✅ 变异系数法计算完成！")
                st.dataframe(weight_df[['变异系数', '权重']])
                st.dataframe(score_df[['得分']])

                export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="export_cv")
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)
                export_both = st.checkbox("同时导出权重与得分结果", value=True, key="export_both")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((weight_df, score_df), export_format, sheet_names=("权重结果", "得分结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(weight_df, export_format)
                        score_str = DataExporter.convert_df_to_format(score_df, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出权重结果
                    export_data = DataExporter.convert_df_to_format(weight_df, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"entropy_topsis_weight_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。") 
# ======================= 4️⃣ 主成分分析法 =====================
with tab4:
    st.subheader("4️⃣ 主成分分析法")

    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择多个数值列进行主成分分析
            - 选择每列的指标方向（正向/负向）
            - 设置累计方差贡献率阈值来决定保留的主成分数量
            - 下载处理后的结果
            """)
    
    with st.expander("🔍 方法说明"):
        st.markdown("""
            主成分分析是一种基于数据协方差矩阵的降维方法，通过提取主要信息减少冗余并保留数据的主要特征。

            它通过计算各主成分的方差贡献率，并根据用户设定的累计贡献率来决定保留的主成分数量。
            """)
        st.markdown("""
            主成分分析的主要步骤：
            1. 对数据进行标准化处理（区分正向/负向指标）
            2. 计算协方差矩阵和特征值
            3. 提取主成分（按累计贡献率判断保留个数）
            4. 计算每个样本在前几个主成分上的得分
            5. 加权合成最终综合得分
            """)
        st.markdown("""
        主成分分析前会对数据进行 **Z-Score 标准化（Z标准化）** 处理，公式如下：
        
        $$
        z = \\frac{x - \\mu}{\\sigma}
        $$
        
        - $ x $: 原始值
        - $ \\mu $: 该列均值
        - $ \\sigma $: 该列标准差
        
        这样可以消除量纲差异的影响，使各指标具有可比性，负向指标会提前做正向化处理。
        """)        
        st.markdown("### 🧮 公式推导")
        st.latex(r"\text{协方差矩阵: } \Sigma = \frac{1}{n-1} X^T X")
        st.latex(r"\text{特征值分解: } \Sigma v_i = \lambda_i v_i")
        st.latex(r"\text{方差贡献率: } \eta_j = \frac{\lambda_j}{\sum_{i=1}^{m} \lambda_i}")
        st.latex(r"\text{累计方差贡献率: } \eta_{total} = \sum_{j=1}^{k} \eta_j ")
        st.latex(r"\text{综合得分: } S_i = \sum_{j=1}^{k} w_j \cdot PC_j(i)")
        
    uploaded_file_pca = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="pca")

    if uploaded_file_pca is not None:
        df_pca = DataUtils.read_file(uploaded_file_pca)
        st.write("原始数据预览：")
        st.dataframe(df_pca.head())

        # 统计缺失值
        stats_df = DataUtils.get_missing_stats(df_pca)
        st.markdown("原始数据缺失值统计")
        st.dataframe(stats_df)

        numeric_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于主成分分析的列", numeric_cols, default=numeric_cols[:])

            directions = {}
            st.markdown("### 设置各列的指标方向")
            for col in selected_cols:
                dir_key = f"dir_pca_{col}"  # 使用唯一 key 避免冲突
                direction = st.selectbox(f"{col} 的指标方向", ["正向指标", "负向指标"], key=dir_key)
                directions[col] = direction

            variance_ratio = st.slider("选择累计方差贡献率阈值", min_value=0.5, max_value=1.0, value=0.85, step=0.01, key="variance_ratio")

            if st.button("开始计算", key="start_pca"):
                with st.spinner('🔄 正在进行主成分分析，请稍等...'):
                    weight_df, score_df, fig = IndicatorsAggregation.pca_method(df_pca[selected_cols],
                                                                                cols=selected_cols,
                                                                                directions=directions,
                                                                                threshold=variance_ratio
                    )
                st.success("✅ 主成分分析完成！")
                st.write(weight_df[['主成分', '方差贡献率', '累计贡献率']])
                st.dataframe(score_df[['综合得分']].head())

                st.markdown("### 主成分方差贡献图")
                st.pyplot(fig)  # ✅ 这里展示绘图对象

                export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="export_pca")
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)
                export_both = st.checkbox("同时导出权重与得分结果", value=True, key="export_both_pca")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((weight_df, score_df), export_format, sheet_names=("权重结果", "得分结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(weight_df, export_format)
                        score_str = DataExporter.convert_df_to_format(score_df, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出权重结果
                    export_data = DataExporter.convert_df_to_format(weight_df, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)


                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"pca_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")
# ======================= 5️⃣ 灰色关联法 ====================
with tab5:
    st.subheader("5️⃣ 灰色关联法")

    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持 xlsx 和 csv 格式
            - 选择多个数值列进行灰色关联分析
            - 设置每列的指标方向（正向/负向）
            - 下载处理后的结果
            """)

    with st.expander("🔍 方法说明"):
        st.markdown("""
            灰色关联法是一种基于 **灰色系统理论** 的多指标综合评价方法。它通过计算各指标之间的关联度来确定指标的重要性和优先级。
            
            该方法适用于数据量较少、信息不完全明确的场景，具有较强的鲁棒性。
                   """)
        st.markdown("""
            灰色关联法的主要步骤：
            1. 对数据进行归一化处理；
            2. 构建参考序列（通常为最优序列）；
            3. 计算每个样本与参考序列的关联系数；
            4. 计算平均关联度作为权重；
            5. 加权合成综合得分并排序；
            """)
        st.markdown("### 🧮 公式推导")
        st.latex(r"\text{正向指标归一化: } x'_{ij} = \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{负向指标归一化: } x'_{ij} = \frac{\max(x_j) - x_{ij}}{\max(x_j) - \min(x_j)}")
        st.latex(r"\text{关联系数: } \gamma_{ij} = \frac{\min_k\min_i|\Delta_{ij}| + \rho \max_k\max_i|\Delta_{ij}|}{|\Delta_{ij}| + \rho \max_k\max_i|\Delta_{ij}|}")
        st.latex(r"\text{综合得分: } S_i = \sum_{j=1}^{m} w_j \cdot \gamma_{ij}")

    uploaded_file_gra = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="gra")

    if uploaded_file_gra is not None:
        df_gra = DataUtils.read_file(uploaded_file_gra)
        st.write("原始数据预览：")
        st.dataframe(df_gra.head())

        # 统计缺失值
        stats_df = DataUtils.get_missing_stats(df_gra)
        st.markdown("原始数据缺失值统计")
        st.dataframe(stats_df)

        numeric_cols = df_gra.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于灰色关联分析的列", numeric_cols, default=numeric_cols[:])

            directions = {}
            st.markdown("### 设置各列的指标方向")
            for col in selected_cols:
                dir_key = f"dir_gra_{col}"
                direction = st.selectbox(f"{col} 的指标方向", ["正向指标", "负向指标"], key=dir_key)
                directions[col] = direction

            if st.button("开始计算", key="start_gra"):
                with st.spinner('🔄 正在进行灰色关联分析，请稍等...'):
                    weight_df, score_df = IndicatorsAggregation.grey_relational_analysis(df_gra[selected_cols], cols=selected_cols, directions=directions)

                st.success("✅ 灰色关联分析已完成！")
                st.dataframe(weight_df[['灰色关联度']])
                st.dataframe(score_df[['综合得分', '排名']])

                # 可视化部分
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(weight_df.index, weight_df['灰色关联度'], color='skyblue')
                ax.set_xlabel('灰色关联度')
                ax.set_ylabel('指标')
                ax.set_title('各指标灰色关联度分布')

                for index, value in enumerate(weight_df['灰色关联度']):
                    ax.text(value, index, f'{value:.4f}', va='center', ha='left')

                st.pyplot(fig)

                export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="export_gra")
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)
                export_both = st.checkbox("同时导出权重与得分结果", value=True, key="export_both_gra")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((weight_df, score_df), export_format, sheet_names=("权重结果", "得分结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(weight_df, export_format)
                        score_str = DataExporter.convert_df_to_format(score_df, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出权重结果
                    export_data = DataExporter.convert_df_to_format(weight_df, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"gra_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")