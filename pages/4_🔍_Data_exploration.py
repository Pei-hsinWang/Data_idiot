import streamlit as st
import pandas as pd
import numpy as np
from utils import DataUtils  # 从 utils.py 导入类
from utils import DataExporter

st.title("📊 数据探索")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常
                              
                    """)
# 主体功能区
tab1, tab2, tab3 = st.tabs(["✅ 中位数分组",
                            "✅ 指标比重计算",
                            "✅ 宽面板转长面板"])

# ======================= 功能一：中位数分组 =======================
with tab1:
    st.subheader("1️⃣ 中位数分组")
     
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择用于分组的数值列
            - 选择导出格式csv、xlsx格式
            - 下载处理后的结果
            """)    
    uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="median")
    
    # 添加示例说明：
    with st.expander("💡 示例说明"):
        st.markdown("""
        ### 示例
                    
        | Year | Region | Emissions |
        |------|--------|-----------|
        | 2020 | Beijing   | 100    |
        | 2021 | Beijing   | 80     |
        | 2013 | Shenzhen  | 60     |
        - 请选择“Emissions”作为数值列
        - 返回结果如下：
                    
        | Year | Region | Emissions |Emissions_group|
        |------|--------|-----------|---------------|
        | 2020 | Beijing   | 100    |1              |
        | 2021 | Beijing   | 80     |1              |
        | 2013 | Shenzhen  | 60     |0              |      

        """)      

    
    if uploaded_file is not None:
        # 使用封装方法读取文件
        df_median = DataUtils.read_file(uploaded_file)

        st.write("原始数据预览：")
        st.dataframe(df_median.head())

        # 选择数字列
        numeric_cols = df_median.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            selected_col = st.selectbox("选择用于分组的数值列", numeric_cols)
            export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0)

            if st.button("☝️执行分组"):
                df_grouped, median_val = DataUtils.median_grouping(df_median, selected_col)
                st.success(f"已基于列 '{selected_col}' 的中位数 {median_val:.2f} 分组")
                st.dataframe(df_grouped[[selected_col, f"{selected_col}_group"]].head())

                # 获取 MIME 类型和扩展名
                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                # 生成字节流
                with st.spinner('🔄 正在生成下载文件，请稍等...'):
                    export_data = DataExporter.convert_df_to_format(df_grouped, export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"grouped_data.{file_extension}",
                    mime=mime_type
                )
        st.info('☝️ 在结果中生成分组虚拟变量列，大于中位数的值标记为1，否则标记为0。')   
    else:
        st.warning("请上传Excel文件。")

# ======================= 功能二：指标比重计算 =======================
with tab2:
    st.subheader("2️⃣ 指标比重计算👋")
    
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择年份列、年份值
            - 选择条件列及对应的值（如地区）
            - 选择目标列（如碳排放量）
            - 点击【开始计算】获取占比结果并下载
            """)    
    # 添加示例说明：
    with st.expander("💡 示例说明"):
        st.markdown("""
        ### 示例1: 求2013年深圳市的碳排放占比
        | Year | Region | Emissions |
        |------|--------|-----------|
        | 2020 | Beijing   | 100       |
        | 2021 | Beijing   | 80        |
        | 2013 | Shenzhen  | 60        |
        - 请选择“Year”作为筛选列
        - 请选择"2013"作为筛选值
        - 请选择"Region"作为条件列
        - 请选择"Shenzhen"作为条件值
        - 请选择"Emissions"作为值列
        - 点击【开始计算】获取占比结果
        ### 示例2: 求2020年低碳建设城市(LCC=1) 的碳排放占比
        | Year | LCC | Emissions |
        |------|-----|-----------|
        | 2020 | 1   | 100       |
        | 2021 | 1   | 80        |
        | 2020 | 0   | 60        |
        - 请选择“Year”作为筛选列
        - 请选择"2020"作为筛选值
        - 请选择"LCC"作为条件列
        - 请选择"1"作为条件值
        - 请选择"Emissions"作为值列
        - 点击【开始计算】获取占比结果
        """)        
    uploaded_file_ratio = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="ratio")

    if uploaded_file_ratio is not None:
        # 使用封装方法读取文件
        df_ratio = DataUtils.read_file(uploaded_file_ratio)

        st.write("原始数据预览：")
        st.dataframe(df_ratio.head())

        # 用户选择年份列
        cols = df_ratio.columns.tolist()
        if not cols:
            st.warning("⚠️ 数据中无可用列，请上传有效数据。")
        else:

             # 获取筛选列和筛选值
            filter_col = st.selectbox("选择筛选列", cols)

            filter_col_value = df_ratio[filter_col].dropna().unique().tolist()
            
            filter_value = st.selectbox("选择筛选值", filter_col_value)

            # 获取条件列和条件值
            condition_col = st.selectbox("选择条件列", df_ratio.columns.tolist())
            
            condition_col_value = df_ratio[condition_col].dropna().unique().tolist()
            condition_value = st.selectbox(f"选择条件值", condition_col_value)

            # 用户选择目标列（如碳排放量、销售额）
            target_col = st.selectbox("作为值列", df_ratio.columns.tolist())

            export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="ratio_export")

            if st.button("☝️开始计算"):
                with st.spinner('🔄 正在计算占比，请稍等...'):
                    # 计算占比
                    ratio = DataUtils.calculate_ratio(
                        df_ratio,
                        filter_col     = filter_col,         # ✅ 列名，如 "Year"
                        filter_value   = filter_value,       # ✅ 值，如 2020
                        condition_col  = condition_col,      # ✅ 条件列名，如 "Region"
                        condition_value= condition_value,    # ✅ 条件值，如 "Asia"
                        target_col     = target_col          # ✅ 目标列名，如 "Emissions"
                    )
                    st.success(f"在{filter_value}，{condition_value}的'{target_col}'指标占比为: {ratio:.4%}")

                    # 构造结果 DataFrame
                    result_df = pd.DataFrame({
                        '筛选值': [filter_value],
                        '条件列': [condition_col_value],
                        '条件值': [condition_value],
                        '目标列': [target_col],
                        '占比': [f"{ratio:.4%}"]
                    })

                    # 获取 MIME 类型和扩展名
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                    # 生成字节流
                    export_data = DataExporter.convert_df_to_format(result_df, export_format)

                    st.download_button(
                        label=f"📥 点击下载 {export_format.upper()} 文件",
                        data=export_data,
                        file_name=f"ratio_result.{file_extension}",
                        mime=mime_type
                    )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")

# ======================= 功能三：宽面板转长面板 =======================
with tab3:
    st.subheader("3️⃣ 宽面板转长面板")
    
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持xlsx 和 csv 格式
            - 选择需要转换的列
            - 点击【开始转换】获取长面板结果并下载
            """)    
    with st.expander("🔍 宽面板转长面板示例"):
        st.markdown("""
            ### 宽面板数据
            | 年份 | 北京  | 上海 |石家庄 | 合肥 |
            |------|------|------|------|------|
            | 2020 |15432 |23456 |18765 | 20123|
            | 2021 |17234 |25678 |19876 | 21345|
            | 2022 |18123 |27890 |20432 | 22456|
            | 2023 |16543 |24321 |19234 | 23567|
            | 2024 |17890 |26789 |21098 | 24654|
            - 请选择"北京", "上海"等列作为需要转换的列
            - 请选择"年份"作为转换参考列
            - ☝️点击【开始转换】获取占比结果
            """)

    uploaded_file_wide = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="wide_to_long")

    if uploaded_file_wide is not None:
        # 使用封装方法读取文件
        df_wide = DataUtils.read_file(uploaded_file_wide)

        st.write("原始数据预览：")
        st.dataframe(df_wide.head())

        # 用户选择需要转换的列
        cols = df_wide.columns.tolist()
        if not cols:
            st.warning("⚠️ 数据中无可用列，请上传有效数据。")
        else:
            selected_cols = st.multiselect("选择需要转换的列", cols,default=cols[1:], help="请选择需要转换为长面板的列，通常是指标列。")
            choice_col    = st.selectbox("选择转换参考列", options=cols, index=0, help="请选择转换参考列，通常为时间列。")
            export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0, key="wide_export")

            mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)            
            
            if st.button("☝️开始转换"):
                with st.spinner('🔄 正在转换为长面板，请稍等...'):                   
                    df_long = DataUtils.wide_to_long(df_wide, id_vars= choice_col, value_vars=selected_cols)
                    st.success(f"已将宽面板转换为长面板，包含 {len(df_long)} 行数据")
                    st.dataframe(df_long.head(10))

                mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)
                export_both = st.checkbox("同时导出宽面板与长面板", value=True, key="export_both_gra")

                if export_both:
                    # 合并两个 DataFrame
                    if export_format == "xlsx":
                        # 导出为 Excel，使用两个 sheet
                        export_data = DataExporter.convert_df_to_format((df_long, df_wide), export_format, sheet_names=("长面板结果", "宽面板结果"))
                    else:
                        # 导出为 CSV，拼接成一个字符串
                        result_str = DataExporter.convert_df_to_format(df_long, export_format)
                        score_str = DataExporter.convert_df_to_format(df_wide, export_format)
                        export_data = (result_str + "\n\n" + score_str).encode('utf-8')
                        mime_type = "text/csv"
                else:
                    # 只导出长面板结果
                    export_data = DataExporter.convert_df_to_format(df_long, export_format)
                    mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

                st.download_button(
                    label=f"📥 点击下载 {export_format.upper()} 文件",
                    data=export_data,
                    file_name=f"gra_result.{file_extension}",
                    mime=mime_type
                )

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")