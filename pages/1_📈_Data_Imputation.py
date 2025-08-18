import pandas as pd
import streamlit as st
import time 
from datetime import datetime
from io import BytesIO
# 自有模块
from utils import ImputerPipeline  
from utils import DataUtils  
from utils import ImputationConfig
from utils import DataExporter
# set the page title and icon
st.set_page_config(page_title="Data_Imputation", page_icon="📈")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常
                              
                    """)
# set GUI title and description
st.title("📈 数据插补工具")
st.markdown("""
            ### 使用说明👋
            - 上传一个 excel 文件，支持xlsx 和 csv 格式，大小不超过200Mb
            - 选择需要插补的列和插补方法，查看缺失情况
            - 选择插补方法，最后点击开始插补，支持一种或多种方法，查看插补数据情况
            - 下载插补后的结果。
            """)

# 定义插补方法及参数
params = ImputationConfig.get_params()

# GUI_模块

st.markdown('### 第一步: 数据导入')

# 创建文件上传器，用户可选择文件
uploaded_file = st.file_uploader("选择文件,支持拖拽和浏览,支持xlsx和csv格式: ",type=['xlsx','csv'])
# 当用户上传文件后执行以下代码块

df = pd.DataFrame()
if uploaded_file is not None:

    if uploaded_file.name.endswith('.xlsx'):
        # 如果文件后缀为xlsx，则读取为Excel文件
        start_time = time.time()
        df = pd.read_excel(uploaded_file)
        st.write(f"读取Excel文件耗时: {time.time() - start_time:.3f} 秒")
    else:
        start_time = time.time()
        df = pd.read_csv(uploaded_file)
        st.write(f"读取Excel文件耗时: {time.time() - start_time:.3f} 秒")
        
    st.write(df.head(5))
    # ✅ 所有需要 df 的后续逻辑都放在这里
    st.markdown('### 第二步: 插补列选择')
    # 选择需要插补的列
    columns = st.multiselect('选择需要插补的列: 剔除无关数据,例如id和时间等', df.columns)

    # 展示用户选择的结果
    st.write('你的选择', columns)  # 输出用户的选择

    # 提取用户选择的数据
    df_impute = df[columns].copy()

    # 数据预览
    st.write('数据预览:', df_impute.head())
    
    # 插补前的缺失值统计
    st.markdown("插补前的缺失值统计")
    before_stats = DataUtils.get_missing_stats(df_impute)
    st.dataframe(before_stats)

    # 数据插补
    st.markdown('### 第三步: 插补方法选择')

    method = st.multiselect(
        '选择一个或多个插补方法',  # 问题描述
        ['线性插值', '三次样条插值', '均值插补', '中位数插补', 'MICE', 'KNN', 'XGBoost'],  # 可选方法
        ['均值插补', 'KNN'])  # 默认已选择 Yellow 和 Red
    st.info('☝️ 线性插值不能填补两端的缺失值')

        # 定义中文选项到英文方法名的映射
    method_mapping = {
        '线性插值': 'linear_interpolation',
        '三次样条插值': 'cubic_spline',
        '均值插补': 'mean',
        '中位数插补': 'median',
        'MICE': 'mice',
        'KNN': 'knn',
        'XGBoost': 'xgboost'
        }
        # 将用户选择的中文选项转换为对应的英文方法名
    mapped_methods = [method_mapping[m] for m in method]    

    # 选择插补方法
    if st.button('开始插补'):

        start_time = time.time()
        # 调用ImputerPipeline类, 生成实例对象
        imputer = ImputerPipeline(methods= mapped_methods, params=params)
        # 数据插补,调用实例方法
        result = imputer.fit_transform(df_impute)

        # 将 result 保存到 session_state，供后续使用
        st.session_state.result = result

        st.success(f"插补完成，耗时 {time.time()- start_time:.3f} 秒")
        # 显示插补结果
        st.write("插补后的数据预览：", result.head(5))

        # 显示插补后的缺失值统计
        st.markdown(" 插补后的缺失值统计")
        st.dataframe(DataUtils.get_missing_stats(result))

    # ✅ 提供下载按钮
    if 'result' in st.session_state:
        st.markdown('### 第四步: 下载结果')
        # 生成带时间戳的文件名
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"插补结果_{current_time}.xlsx"

        export_format = st.selectbox("选择导出格式", options=["xlsx", "csv"], index=0)
         # 获取 MIME 类型和扩展名
        mime_type, file_extension = DataExporter.get_mime_and_extension(export_format)

        # 生成字节流
        with st.spinner('🔄 正在生成下载文件，请稍等...'):
            result = st.session_state.result
            export_data = DataExporter.convert_df_to_format(result, export_format)

            st.download_button(
                label=f"📥 点击下载 {export_format.upper()} 文件",
                data=export_data,
                file_name=f"filename.{file_extension}",
                mime=mime_type
            )        
        # # 将 DataFrame 转为 Excel 字节流
        # output = BytesIO()
        # with pd.ExcelWriter(output, engine='openpyxl') as writer:
        #     st.session_state.result.to_excel(writer, index=False)
        # excel_data = output.getvalue()
        
        # # 提供下载按钮
        # st.download_button(
        #     label="📥 下载结果",
        #     data=excel_data,
        #     file_name=filename,
        #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        # )
    else:
        st.info('☝️ 请点击开始插补')

else:
     # 如果用户未上传文件，则显示提示信息
     st.info('☝️ 请上传数据文件')
