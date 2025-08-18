import streamlit as st
import pandas as pd
from utils import DataUtils  # 从 utils.py 导入类
from utils import DataExporter
# set the page title and icon
st.set_page_config(page_title="Data_Merge", page_icon="📊")

st.title("📊 数据合并工具")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常
                              
                    """)
st.markdown("""
            ### 使用说明👋
            - 上传两个 Excel 文件, 支持xlsx 和 csv 格式
            - 选择合并的匹配列（公共列）
            - 选择匹配方法（交际，并集，左连接，右连接)
            - 下载合并后的结果
            """)

# GUI_模块
st.markdown('### 第一步: 数据导入')

# 上传文件
uploaded_file1 = st.file_uploader("上传第一个 excel 文件", type=["xlsx", "csv"], key="file1")
uploaded_file2 = st.file_uploader("上传第二个 excel 文件", type=["xlsx", "csv"], key="file2")

# 当用户上传文件后执行以下代码块
if uploaded_file1 and uploaded_file2:
    df1 = DataUtils.read_file(uploaded_file1)
    df2 = DataUtils.read_file(uploaded_file2)

    st.markdown("第一个文件预览")
    st.dataframe(df1.head())
    stats_df1 = DataUtils.get_missing_stats(df1)
    st.markdown("第一个文件的缺失值统计")
    st.dataframe(stats_df1)

    st.markdown("第二个文件预览")
    st.dataframe(df2.head())
    stats_df2 = DataUtils.get_missing_stats(df2)
    st.markdown("第二个文件的缺失值统计")
    st.dataframe(stats_df2)

    st.markdown("### 第二步：按列合并")

    common_cols = list(set(df1.columns) & set(df2.columns))

    if not common_cols:
        st.error("❌ 两个文件没有公共列，无法进行列合并。请检查上传的数据。")
    else:
        on_columns = st.multiselect(
            "选择用于合并或匹配的列",
            options=common_cols,
            default=[common_cols[0]] if len(common_cols) >= 1 else []
        )

        merge_how = st.selectbox(
            "选择合并方式",
            options=["inner", "outer" ,"left", "right"],
            index=0
        )
        # 合并方式说明
        merge_explanations = {
            "inner": "默认1: 内连接 (保留两表交集)：只保留两个表中都能匹配上的行。",
            "left":  "左连接：保留左表所有行，右表无匹配则填充 NaN。",
            "right": "右连接：保留右表所有行，左表无匹配则填充 NaN。",
            "outer": "默认2:外连接 (保留两表并集)：保留两个表所有行，无匹配则填充 NaN。"
        }

        selected_merge = merge_how
        st.info(f"📘 当前选择的合并方式说明：\n\n{merge_explanations[selected_merge]}")
        if not on_columns:
            st.warning("⚠️ 请至少选择一个用于合并的列。")
        else:
            if st.button("✅ 执行合并"):
                with st.spinner("🔄 正在合并数据，请稍等..."):
                    merged_df = pd.merge(df1, df2, on=on_columns, how=merge_how)
                    st.session_state.merged_df = merged_df
                    st.success(f"✅ 合并完成（{merge_how} join）")

    # 显示合并结果
    if 'merged_df' in st.session_state:
        merged_df = st.session_state.merged_df

        st.subheader("合并后的数据预览")
        st.dataframe(merged_df.head())

        # 显示缺失值统计
        stats_merged_df = DataUtils.get_missing_stats(merged_df)
        st.markdown("合并后的缺失值统计")
        st.dataframe(stats_merged_df)

        export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0)

        if st.button("📥 生成下载文件"):
            with st.spinner('🔄 正在生成文件，请稍等...'):
                export_data = DataExporter.convert_df_to_format(merged_df, export_format)

            mime_type, ext = DataExporter.get_mime_and_extension(export_format)

            st.download_button(
                label=f"📥 点击下载 {export_format.upper()} 文件",
                data=export_data,
                file_name=f"merged_data{ext}",
                mime=mime_type
            )
else:
    st.warning("请上传两个 xlsx 文件以便继续操作。")