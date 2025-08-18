import streamlit as st
import pandas as pd
import numpy as np
# 从 utils.py 导入类
from utils import DataUtils, DataExporter,Spatial_Eco
st.title("🌍 空间计量工具")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常                              
                    """)
# 主体功能区
tab1, tab2, tab3 = st.tabs(["1️⃣ 空间滞后项生成",
                            "2️⃣ 空间溢出效应边界——权重矩阵法",
                            "3️⃣ 空间溢出效应边界——虚拟变量法"])


# ======================= 功能一：空间滞后项生成 =======================
with tab1:
    st.subheader("1️⃣ 空间滞后项生成")
    st.markdown(f"""
    #### 🤷‍♂️为什么要计算空间滞后项？
    **Stata**常用的空间计量包中缺少了**空间自变量滞后模型(SLX)**和**空间杜宾误差模型(SDEM)**
    - 将制作好的空间滞后项用**ols**进行估计既可以实现**SLX**模型
    - 将制作好的空间滞后项用**sem(空间误差模型)**进行估计可以实现**SDEM**模型
    """)
    st.markdown(""" 
    #### 🤷‍♀️为什么要专门写个计算空间滞后项的程序？
    - **Trash**和**逆天**的Stata语法！
    - Stata**羸弱**的矩阵计算能力
    - 大多数经管研究使用的是面板数据, 使得矩阵乘法计算需要额外处理逻辑

    ### 使用说明👋
    - 上传整理好的 权重矩阵的`Excel` 文件以及 变量数据的`Excel`文件, 支持`xlsx` 和 `csv` 格式     
    - 选择需要计算滞后项的变量列
    - 选择导出格式`csv`、`xlsx`格式
    - 下载处理后的结果
    - 🖋 注：权重矩阵的格式为: 矩阵的**行列数必须相等**(方阵), 矩阵的行列与数据的id(索引)对应。 推荐如果是城市数据按城市行政区划号排序, 企业数据按股票代码排序。
    """)    
    weight_matrix = st.file_uploader("上传权重矩阵的CSV或Excel文件", type=["csv", "xlsx"], key="weight_matrix")
    variable      = st.file_uploader("上传变量的CSV或Excel文件", type=["csv", "xlsx"], key="variable")
    with st.expander("🔍 数据样式"):
        st.markdown("""
        **反距离矩阵示例(节选)：**
        |北京	|天津	|石家庄	|唐山	|秦皇岛|
        |---|---|---|---|---|
        |0	|1/127	|1/285	|1/172	|1/237|
        |1/127	|0	|1/283	|1/98	|1/182|
        |1/285	|1/283	|0	|1/381	|1/464|
        |1/172	|1/98	|1/381	|0	|1/83|
        |1/237	|1/182	|1/464	|1/83	|0| 
        """) 
        st.markdown("""
        **数据示例(节选)：**
        |地区	|行政区划代码	|id	|year	|DID
        |-----|---|---|---|---|
        |北京市	|110000	|1	|2012	|1
        |北京市	|110000	|1	|2013	|1
        |北京市	|110000	|1	|2014	|1
        |天津市	|120000	|2	|2012	|0
        |天津市	|120000	|2	|2013	|1
        |天津市	|120000	|2	|2014	|1
        |石家庄市	|130100	|3	|2012	|1
        |石家庄市	|130100	|3	|2013	|1
        |石家庄市	|130100	|3	|2014	|1
        |唐山市	|130200	|4	|2012	|0
        |唐山市	|130200	|4	|2013	|0
        |唐山市	|130200	|4	|2014	|0
        |秦皇岛市	|130300	|5	|2012	|0
        |秦皇岛市	|130300	|5	|2013	|0
        |秦皇岛市	|130300	|5	|2014	|0
        """)
   
    if weight_matrix and variable:
        weight = DataUtils.read_file(weight_matrix,)
        var    = DataUtils.read_file(variable)
        
        st.markdown("第一个文件预览")
        st.dataframe(weight.head())
        st.markdown(f"权重矩阵形状{weight.shape}")
        
        st.markdown("第二个文件预览")
        st.dataframe(var.head())
        st.markdown("第二个文件描述性统计")
        st.dataframe(var.describe().drop(['25%', '50%', '75%']))

        # 选择用于计算滞后项的数值列,默认为所有数值列
        numeric_cols = var.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_cols = st.multiselect("选择用于计算滞后项的列", numeric_cols, default=numeric_cols[:],key= "selected_lag_cols") 
        # 选择id标识符列,默认为id
        id_col = st.selectbox("选择id标识符列", numeric_cols,key='id_cols')
        # 选择年份标识列,默认为year
        year_col = st.selectbox("选择年份标识列", numeric_cols,key='year_cols')

        if st.button("开始计算",key="spatial_lage_button"):

            with st.spinner('🔄 正比处理并计算，请稍等...'):
                W_sp = Spatial_Eco()
                result= W_sp.compute_weighted_panel_multi_variables (var,weight,id_col=id_col,year_col=year_col, value_cols = selected_cols ,normalize_weights=True)
            
            export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="ratio_export")
            # 获取 MIME 类型和扩展名
            mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

            # 生成字节流
            export_data = DataExporter.convert_df_to_format(result, export_format)

            st.download_button(
                label=f"📥 点击下载 {export_format.upper()} 文件",
                data=export_data,
                file_name=f"spatial_lag_result.{file_extension}",
                mime=mime_type
            )    
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")

# ======================= 功能二：空间溢出效应边界——权重矩阵法 =======================
with tab2:
    st.subheader("2️⃣ 空间溢出效应边界——权重矩阵法")
    st.markdown("""
            ### 1.权重矩阵法workfolow:
            - 首先，空间距离矩阵的特征，对矩阵进行**裁剪**，从而得到一系列阈值矩阵。
            - 然后，将**阈值矩阵作为权重矩阵**，进行空间计量分析
            - 最后，绘制**空间溢出效应衰减边界图**
            """)
    st.markdown("""
            ### 2. 权重矩阵裁剪逻辑
            - `if` 该距离矩阵元素<=阈值:
                - 保留该元素值
            - `else`:
                - 令元素值置为0
            """)
    st.markdown(  
            """          
            ### 3. 使用说明👋
            - 上传整理好的 空间距离矩阵的`Excel` 文件，例如，地理距离、交通距离等
            - ✍注：不是取导数之后的反地理距离矩阵
            - 选择裁剪参数：距离阈值、初始值、终值
            - 选择导出格式`csv`、`xlsx`格式
            - 下载处理后的结果
            """)
    with st.expander("🔍 数据样式"):
        st.markdown("""
        **距离矩阵示例(节选)：**
        |北京	|天津	|石家庄	|唐山	|秦皇岛|
        |---|---|---|---|---|
        |0	|127	|285	|172	|237|
        |127	|0	|283	|98	|182|
        |285	|283	|0	|381	|464|
        |172	|98	|381	|0	|83|
        |237	|182	|464	|83	|0| 
        """)

        st.markdown("""
        **数据示例(节选)：**
        |地区	|行政区划代码	|id	|year	|DID
        |-----|---|---|---|---|
        |北京市	|110000	|1	|2012	|1
        |北京市	|110000	|1	|2013	|1
        |北京市	|110000	|1	|2014	|1
        |天津市	|120000	|2	|2012	|0
        |天津市	|120000	|2	|2013	|1
        |天津市	|120000	|2	|2014	|1
        |石家庄市	|130100	|3	|2012	|1
        |石家庄市	|130100	|3	|2013	|1
        |石家庄市	|130100	|3	|2014	|1
        |唐山市	|130200	|4	|2012	|0
        |唐山市	|130200	|4	|2013	|0
        |唐山市	|130200	|4	|2014	|0
        |秦皇岛市	|130300	|5	|2012	|0
        |秦皇岛市	|130300	|5	|2013	|0
        |秦皇岛市	|130300	|5	|2014	|0
        """)

    with st.expander("🔍 方法"):
        st.markdown("**空间门槛权重矩阵**")
        st.latex(r"W_{ij} = \begin{cases} \frac{1}{d_{ij}} & d_{ij} \geq d_{threshold} \\ 0 & d_{ij} < d_{threshold} \end{cases}")
        st.markdown("""
        - $d_{ij}$: 两个城市之间的距离
        - $d_{threshold}$: 距离阈值
        """)     
    uploaded_file_matrix = st.file_uploader("上传空间溢出矩阵的CSV或Excel文件", type=["csv", "xlsx"], key="spillover_matrix")
    if uploaded_file_matrix is not None:
        try:
            spillover_matrix = DataUtils.read_file(uploaded_file_matrix)
            st.markdown("第一个文件预览")
            st.dataframe(spillover_matrix.head())
        except Exception as e:
            st.error(f"无法读取文件：{e}")

        # 选择距离阈值
        distance_threshold = st.slider("选择距离阈值", min_value=0, max_value=200, value=50, step=10,key="matrix_distance_threshold")
        # 选择距离初始值，默认值为50
        initial_value = st.slider("选择初始值", min_value=0, max_value=200, value=50, step=10,key="matrix_initial_value")
        # 选择距离终值，默认值为400
        final_value = st.slider("选择终值", min_value=0, max_value=1000, value=400, step=50,key="matrix_final_value")

        # 初始化 Spatial_Eco 实例
        spatial_eco = Spatial_Eco()

        if st.button("开始计算",key="weight_botton"):
            with st.spinner('🔄 正比处理并计算，请稍等...'):
            # 调用函数生成多个阈值矩阵
                matrices = spatial_eco.spatial_spillover_matrix(spillover_matrix, step=distance_threshold, begin_distance=initial_value, end_distance=final_value)     
                st.session_state.matrices = matrices
                st.success('✅ 计算完成！')            
            
        # 显示结果
        if 'matrices' in st.session_state:
            matrices = st.session_state.matrices

            export_format = "xlsx" 
            st.info('☝️ 导出结果格式为xlsx，每个工作簿对应一个距离阈值区间')
            if st.button("导出结果"):    
                with st.spinner("正在导出结果..."):
                    # 根据导出格式处理字典数据
                    # 对于Excel格式，将字典中的每个DataFrame保存为不同的工作表
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df in matrices.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    export_data = output.getvalue()
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_extension = "xlsx"
                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"spillover_matrices.{file_extension}",
                    mime=mime_type
                )
    else:
        st.warning("请上传距离矩阵文件以便继续操作。")


# ======================= 功能三：空间溢出效应边界——虚拟变量法 =======================
with tab3:
    st.subheader("3️⃣ 空间溢出效应边界——虚拟变量法")
    st.markdown("""
            ### 1. 虚拟变量法workfolow:
            - 首先，根据距离阈值生成虚拟变量
            - 然后，将**虚拟变量作为控制变量**，进行空间计量分析
            - 最后，获得虚拟变量系数值，绘制**空间溢出效应衰减边界图**
            """)
    st.markdown("""
            ### 2. 虚拟变量生成逻辑
            - `if` 该年没有处理组，则所有虚拟变量设为0:
            - `else`:
                - 1. 计算每个样本到最近处理组的距离，取最小值
                - 2. 比较阈值与最小值，若小于阈值，则设为1，否则设为0
                - 3. 遍历阈值范围得到多个虚拟变量
            """)
    with st.expander("🔍 数据样式说明(重要)"):
        st.markdown("""
        **距离矩阵示例(节选)：**
        |110000	|120000	|130100	|130200	|130300|
        |---|---|---|---|---|
        |0	|127	|285	|172	|237|
        |127	|0	|283	|98	|182|
        |285	|283	|0	|381	|464|
        |172	|98	|381	|0	|83|
        |237	|182	|464	|83	|0| 
        """)
        st.markdown("""
        **数据示例(节选)：**
        |地区	|行政区划代码	|id	|year	|DID
        |-----|---|---|---|---|
        |北京市	|110000	|1	|2012	|1
        |北京市	|110000	|1	|2013	|1
        |北京市	|110000	|1	|2014	|1
        |天津市	|120000	|2	|2012	|0
        |天津市	|120000	|2	|2013	|1
        |天津市	|120000	|2	|2014	|1
        |石家庄市	|130100	|3	|2012	|1
        |石家庄市	|130100	|3	|2013	|1
        |石家庄市	|130100	|3	|2014	|1
        |唐山市	|130200	|4	|2012	|0
        |唐山市	|130200	|4	|2013	|0
        |唐山市	|130200	|4	|2014	|0
        |秦皇岛市	|130300	|5	|2012	|0
        |秦皇岛市	|130300	|5	|2013	|0
        |秦皇岛市	|130300	|5	|2014	|0

        """)
    with st.expander("🔍 方法与参考文献"):
        st.markdown("**估计方程**")
        st.latex(r"{Y_{it}} = {\beta _0} + {\beta _1}{D_{it}} + \sum\nolimits_{s = star\_d}^{end\_d} {{\delta _s}dummy_{it}^s + \gamma {X_{it}} + {\mu _i} + {\lambda _t}}  + {\varepsilon _{it}}")
        st.markdown("""
        在经典的双向固定效应TWFE法中，我们引入了一组地理虚拟变量(0,1)。
        - **具体而言**：如果在t年距离样本i(s-50,s)的范围内存在处理组，那么$dummy_{it}^s=1$,否则$dummy_{it}^s=0$
        - **系数解释**：进行ols估计后，得到虚拟变量的估计系数，即表示为空间效应的大小
        - **参考文献**：[1]曹清峰.国家级新区对区域经济增长的带动效应——基于70大中城市的经验证据[J].中国工业经济,2020,(07):43-60.DOI:10.19581/j.cnki.ciejournal.2020.07.014.

        """)    
    st.markdown(  
            """          
            ### 3. 使用说明👋
            - 上传整理好的 空间溢出矩阵的`Excel` 文件
            - 选择距离阈值范围，点击开始计算
            - 选择导出格式`csv`、`xlsx`格式
            - 下载处理后的结果
            """)
    uploaded_file_dummy_matrix = st.file_uploader("上传空间溢出矩阵的CSV或Excel文件", type=["csv", "xlsx"], key="dummy_matrix")
    uploaded_file_data = st.file_uploader("上传数据集的CSV或Excel文件", type=["csv", "xlsx"], key="dummy_data")

    if uploaded_file_dummy_matrix and uploaded_file_data is not None:
        try:
            spillover_matrix = DataUtils.read_file(uploaded_file_dummy_matrix,header=None)
            data = DataUtils.read_file(uploaded_file_data)
            st.markdown("第一个文件预览")
            st.dataframe(spillover_matrix.head())
            st.markdown(f"距离矩阵形状{spillover_matrix.shape}")
            st.markdown("第二个文件预览")
            st.dataframe(data.head())
            st.markdown(f"数据集形状{data.shape}")

        except Exception as e:
            st.error(f"无法读取文件：{e}")
        # 选择列
        choising_cols = data.columns.tolist()
        # id列名
        id_col = st.selectbox("选择ID列，要与距离矩阵行一一对应，推荐用行政区划代码", choising_cols,key="dummy_id_col") 
        # year列
        year_col = st.selectbox("选择年份列", choising_cols,key="dummy_year_col")
        # treat列
        treat_col = st.selectbox("选择 treatment 列，即did列", choising_cols,key="dummy_treat_col")

        # 开始年份
        start_year = st.number_input("选择开始年份", min_value=2000, max_value=2025, value=2006,key="dummy_start_year")
        # 结束年份
        end_year = st.number_input("选择结束年份", min_value=2000, max_value=2025, value=2021,key="dummy_end_year")
        
        # 选择距离阈值
        distance_threshold = st.slider("选择距离阈值", min_value=0, max_value=200, value=50, step=10,key="dummy_distance_threshold")
        # 选择距离初始值，默认值为50
        initial_value = st.slider("选择初始值", min_value=0, max_value=200, value=50, step=10,key="dummy_initial_value")
        # 选择距离终值，默认值为400
        final_value = st.slider("选择终值", min_value=0, max_value=1000, value=400, step=50,key="dummy_final_value")
        # 初始化 Spatial_Eco 实例
        spatial_eco = Spatial_Eco()

        if st.button("开始计算",key="dummy_button"):

            with st.spinner('🔄 正比处理并计算，请稍等...'):
            # 调用函数生成多个阈值矩阵
                config = {
                    'dist_df': spillover_matrix,      # 你提供的距离矩阵文件
                    'policy_data': data,           # 政策数据文件
                    'id_col': id_col,  # 注意：这里要匹配政策表中的列名，你原始代码中是 '地区'，但数据可能是行政区划代码
                    'year_col': year_col,
                    'treat_col': treat_col,
                    'start_year': start_year,
                    'end_year': end_year,
                    'thresholds': list(range(initial_value , final_value, distance_threshold))
                }
                dummy = spatial_eco.distance_dummies(**config)     
                st.session_state.dummy = dummy
                st.success('✅ 计算完成！')      
                
        # 显示结果
        if 'dummy' in st.session_state:
            dummy = st.session_state.dummy

            export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="dummy_export")
            # 获取 MIME 类型和扩展名
            mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

            # 生成字节流
            export_data = DataExporter.convert_df_to_format(dummy, export_format)

            st.download_button(
                label=f"📥 点击下载 {export_format.upper()} 文件",
                data=export_data,
                file_name=f"distance_dummy_result.{file_extension}",
                mime=mime_type
            )    
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")




