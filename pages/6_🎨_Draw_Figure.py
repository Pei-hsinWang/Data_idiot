import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
# 自有模块
from utils import DataUtils,Draw_Figure

# set the page title and icon
st.set_page_config(page_title="Draw_Figure", page_icon="")


st.title("📩 绘图工具")
st.sidebar.markdown("""
                    ## 关注作者
                    - ✉️ GitHub: [Pei-hsin Wang](https://github.com/Pei-hsinWang)
                    - ✉️ 公众号: 拒绝H0的日常   
                    """)

# 主体功能区
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 相关系数图", 
                            "2️⃣ 联合分布图",
                            "3️⃣ 异质性分析: 森林图",
                            "4️⃣ 空间溢出效应边界图"])
# ======================= 1️⃣ 相关系数图 =======================
with tab1:
    st.subheader("1️⃣相关系数图")
     
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持`xlsx` 和 `csv` 格式
            - 选择需要计算的特征列
            - 下载处理后的结果
            """)   

    uploaded_file_cor = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="cor")
    if uploaded_file_cor is not None:

        # 读取上传的文件
        df_cor = DataUtils.read_file(uploaded_file_cor)

        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df_cor.head(5))

        # 选择用于计算相关系数的特征列
        choising_cols = df_cor.select_dtypes(include=[np.number]).columns.tolist()       
        if not choising_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            feature_cols = st.multiselect("选择用于计算相关系数的**特征列**", choising_cols, default=choising_cols[0:])
        
        # 图像分辨率
        fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1200, value=600,key="cor_fig_dpi")        
        
        
        # 配色方案选择
        color_maps = ['Reds', 'Blues', 'YlOrRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'plasma', 'inferno']
        selected_cmap = st.selectbox("选择配色方案", options=color_maps, index=0)        

        # 语言选择
        zn = bool(st.checkbox("是否选择中文绘图, 默认使用English"))
        st.markdown(":violet-badge[:material/star: 变量(特征)标签过长会导致图像显示问题,建议使用短标签 ]")
        # 使用示例
        if st.button("开始分析",key="cor_button"):

            if len(feature_cols) < 2:
                st.warning("请至少选择两个数值列进行相关性分析")
            else:
                
                # 创建绘图实例
                drawer = Draw_Figure(zn)
                
                # 生成图像但不立即显示
                fig = drawer.correlation_matrix(df_cor, feature_cols, show_plot=False, cmap=selected_cmap)

                
                # 导出图像为多种格式
                image_buffers = drawer.export_fig(fig, dpi=fig_dpi)

                # 预览图像
                st.image(image_buffers['png'], caption="相关系数矩阵热力图")
                
            # 提供多种格式下载按钮
            col1, col2, col3 = st.columns(3)
            col4, col5, col6  = st.columns(3)

            with col1:
                st.download_button(
                    label="📥 下载 PNG 图像",
                    data=image_buffers['png'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

            with col2:
                st.download_button(
                    label="📄 下载 PDF 图像",
                    data=image_buffers['pdf'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

            with col3:
                st.download_button(
                    label="📐 下载 SVG 图像",
                    data=image_buffers['svg'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml"
                )
            with col4:
                st.download_button(
                    label="📜 下载 EPS 图像",
                    data=image_buffers['eps'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                    mime="image/eps"
                )            

            with col5:
                st.download_button(
                    label="🖼️ 下载 TIFF 图像",
                    data=image_buffers['tiff'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                    mime="image/tiff"
                )

            with col6:
                st.download_button(
                    label="📷 下载 JPG 图像",
                    data=image_buffers['jpg'],
                    file_name=f"corplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")
# ======================= 2️⃣ 联合分布图 =======================
with tab2:
    st.subheader("2️⃣ 联合分布图")
     
    st.markdown("""
            ### 使用说明👋
            - 上传 Excel 文件, 支持`xlsx` 和 `csv` 格式
            - 选择需要计算的`x`列和`y`列
            - 下载处理后的结果
            """)   

    uploaded_file_joint = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="joint")
    if uploaded_file_joint is not None:
        # 读取上传的文件
        df_joint = DataUtils.read_file(uploaded_file_joint)
        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df_joint.head(5))

        # 选择用于计算联合分布的特征列
        choising_cols = df_joint.select_dtypes(include=[np.number]).columns.tolist()       
        if not choising_cols:
            st.warning("⚠️ 数据中无数值列，请上传包含数值列的数据。")
        else:
            if len(choising_cols) < 2:
                st.warning("⚠️ 数据中至少需要两个数值列来进行相关性分析。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    feature_x = st.selectbox("选择自变量x", options=choising_cols, index=0, key="feature_x")
                with col2:
                    feature_y = st.selectbox("选择因变量y", options=[c for c in choising_cols if c != feature_x], index=0, key="feature_y")
                

            feature_cols = [feature_x, feature_y]

            data = df_joint[feature_cols]

            # 图像分辨率
            fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1200, value=600,key="joint_fig_dpi")

           
            # 颜色中文名 → 颜色值
            color_name_map = {
                "科研蓝": "#2E86AB",
                "墨绿":   "#3E7D59",
                "砖红":   "#A23B72",
                "深灰":   "#50514F",
                "橘棕":   "#F18F01",
                "淡紫灰": "#C7B8EA"
            }
            # 下拉框：用户看到的是中文，返回的是 Hex
            choising_cols = st.selectbox(
                "选择配色方案",
                options=list(color_name_map.keys()),
                index=0,
                format_func=lambda x: x  # 保持中文显示
            )
            # 真正需要的颜色值
            selected_color = color_name_map[choising_cols] # 拿中文名作为key，获取对应的Hex值            

            # 语言选择
            zn = bool(st.checkbox("是否选择中文绘图, 默认使用English",key="joint_zn"))           
            # 使用示例
            if st.button("开始分析",key="joint_button"):
                if len(feature_cols) < 2:
                    st.warning("请至少选择两个数值列进行联合分布分析")
                else:
                    
                    # 创建绘图实例
                    drawer = Draw_Figure(zn)
                    
                    # 生成图像但不立即显示
                    fig = drawer.joint_distribution_plot(data,feature_x,feature_y,show_plot=False,color=selected_color)

                    
                    # 导出图像为多种格式
                    image_buffers = drawer.export_fig(fig, dpi=fig_dpi)

                    # 预览图像
                    st.image(image_buffers['png'], caption="联合分布分图")               
            
                # 提供多种格式下载按钮
                col1, col2, col3 = st.columns(3)
                col4, col5, col6  = st.columns(3)

                with col1:
                    st.download_button(
                        label="📥 下载 PNG 图像",
                        data=image_buffers['png'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )

                with col2:
                    st.download_button(
                        label="📄 下载 PDF 图像",
                        data=image_buffers['pdf'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

                with col3:
                    st.download_button(
                        label="📐 下载 SVG 图像",
                        data=image_buffers['svg'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                        mime="image/svg+xml"
                    )
                with col4:
                    st.download_button(
                        label="📜 下载 EPS 图像",
                        data=image_buffers['eps'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                        mime="image/eps"
                    )            

                with col5:
                    st.download_button(
                        label="🖼️ 下载 TIFF 图像",
                        data=image_buffers['tiff'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                        mime="image/tiff"
                    )

                with col6:
                    st.download_button(
                        label="📷 下载 JPG 图像",
                        data=image_buffers['jpg'],
                        file_name=f"jointplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        mime="image/jpeg"
                    )                
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")
# ======================= 3️⃣ 异质性分析: 森林图 ===============
with tab3:
    st.subheader("3️⃣ 异质性分析: 森林图")
     
    st.markdown("""
            ### 使用说明👋
            - 上传 `Excel` 文件, 支持`xlsx` 和 `csv` 格式
            #### 包括以下三列内容：
            -  `subgroup` 列: 子组标签(名称)
            -  `coef` 列: 回归系数值
            -  `se` 列: 标准误 

            """)
    st.markdown(
    ":violet-badge[:material/star: coef和se从stata结果中复制过来]:orange-badge[:material/star: subgroup手工填写]")
    st.markdown("""
            ### 示例数据
        | subgroup | coef    | se       |
        |----------|---------|----------|
        | lowpress | 0.42    | 0.08     |
        | highpress| 0.35    | 0.09     |
        | resource | 0.28    | 0.10     |
        | non_resource| 0.55 | 0.07     |      
    """)
    uploaded_file_forest = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="forest")
    
    if uploaded_file_forest is not None:
        # 读取上传的文件
        df_forest = DataUtils.read_file(uploaded_file_forest)
        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df_forest.head(5))

        # 图像分辨率
        fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1200, value=600,key="forest_fig_dpi")
        # 语言选择
        zn = bool(st.checkbox("是否选择中文绘图, 默认使用English",key="forest_zn"))        
        # 检查列是否存在
        required_cols = ["subgroup", "coef", "se"]

        if not all(col in df_forest.columns for col in required_cols):
            st.warning("⚠️ 数据中缺少必要列，请上传包含 'subgroup', 'coef', 'se' 列的数据。")
        else:
            # 获取所有子组名称
            unique_subgroups = df_forest["subgroup"].unique().tolist()

            # 用户颜色配置
            st.markdown("### 🎨 自定义子组颜色")
            color_map = {}
            default_color = "#000000"

            # 遍历所有子组，提供颜色选择器
            for subgroup in unique_subgroups:
                selected_color = st.color_picker(
                    f"选择 '{subgroup}' 的颜色",
                    value=default_color,
                    key=f"color_picker_{subgroup}"
                )
                color_map[subgroup] = selected_color
        # 创建绘图实例
        drawer = Draw_Figure(zn)

        # 生成图像但不立即显示
        fig = drawer.forest_plot(df_forest, 
                                subgroup_col="subgroup", 
                                coef_col="coef", 
                                se_col="se", 
                                color_map=color_map, 
                                show_plot=False)        
        # 导出图像为多种格式
        image_buffers = drawer.export_fig(fig, dpi=fig_dpi)

        # 预览图像
        st.image(image_buffers['png'], caption="森林图（异质性分析）", use_container_width=True)                
        # 提供多种格式下载按钮
        col1, col2, col3 = st.columns(3)
        col4, col5, col6  = st.columns(3)

        with col1:
            st.download_button(
                label="📥 下载 PNG 图像",
                data=image_buffers['png'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

        with col2:
            st.download_button(
                label="📄 下载 PDF 图像",
                data=image_buffers['pdf'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

        with col3:
            st.download_button(
                label="📐 下载 SVG 图像",
                data=image_buffers['svg'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                mime="image/svg+xml"
            )
        with col4:
            st.download_button(
                label="📜 下载 EPS 图像",
                data=image_buffers['eps'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                mime="image/eps"
            )            

        with col5:
            st.download_button(
                label="🖼️ 下载 TIFF 图像",
                data=image_buffers['tiff'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                mime="image/tiff"
            )

        with col6:
            st.download_button(
                label="📷 下载 JPG 图像",
                data=image_buffers['jpg'],
                file_name=f"forestplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg"
            )                        
    else:
        st.warning("请上传Excel或CSV文件以继续操作。")
                
# ======================= 4️⃣ 空间溢出效应边界图 ===================
with tab4:
    st.markdown("""
            ### 使用说明👋
            - 1. 上传 `Excel` 文件, 支持`xlsx` 和 `csv` 格式
            - 2. 选择距离阈值(步长)，终值
            - 3. 选择绘图的距离范围
            - 4. 选择置信区间值, 99%,95%,90%
            - 5. 考虑是否加入参考线
            - 6. 选择系数列、t值列或标准误列
            - 7. 考虑输入x和y轴标题
            - 8. 选择绘图颜色
            - 9. 导出图像
            #### 至少包括以下三列内容：
            -  `distance` 列: 距离值
            -  `coef` 列: 回归系数值
            -  `se` 列: 标准误 

            """)
    st.markdown(
    ":violet-badge[:material/star: coef和se从stata或matlab结果中复制过来]")
    st.markdown("""
            ### 示例数据
        | distance| coef    | se       |
        |---- |---------|----------|
        | 50 | -0.1074  | 0.07     |
        | 100| -0.1762  | 0.08     |
        | 150| -0.1818  | 0.10     |
        | 200| -0.1709  | 0.11     |      
    """)    
    uploaded_file_spatialstat = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"], key="spatialstat")
    if uploaded_file_spatialstat is not None:
        df_spatialstat = DataUtils.read_file(uploaded_file_spatialstat)
        # 显示原始数据
        st.write("原始数据预览：")
        st.dataframe(df_spatialstat.head(5))     
        # 选择距离阈值(步长：默认50)
        distance_threshold = st.slider("选择距离阈值", min_value=0, max_value=200, value=50, step=50,key="spatialstat_distance_threshold")
        # 选择距离初始值，默认值为0
        initial_value = 0
        # 选择距离终值，默认值为1000
        final_value = st.slider("选择终值", min_value=0, max_value=1000, value=1000, step=50,key="spatialstat_final_value")
        st.markdown(f"👋数据的距离区间: {initial_value}km 到 {final_value}km")
        # 绘图的距离区间
        filter_min_threshold = st.slider("选择绘图的起点距离", min_value=0, max_value=200, value=50, step=50,key="spatialstat_min_threshold")
        filter_max_threshold = st.slider("选择绘图的终点距离", min_value=0, max_value=1000, value=600, step=50,key="spatialstat_max_threshold")
        st.markdown(f"👋绘图的距离区间: {filter_min_threshold}km 到 {filter_max_threshold}km")
        # 绘图的置信区间
        confidence_interval = st.selectbox("请输入置信区间", options=[0.99,0.95, 0.90], key="spatialstat_confidence_interval")
        
        # x轴参考线：用于划分区间
        x_line_value = st.text_input("请输入x轴参考线值: 用于划分区间 (默认为空置，即不加参考线) ", value=None, key="spatialstat_x_line_value")
        if x_line_value is not None:
            x_line_value = int(x_line_value)
        else:
            x_line_value = None
        # 选择列
        choising_cols = df_spatialstat.columns.tolist()
        # 系数列名
        coef_col = st.selectbox("请选择系数列", choising_cols, key="spatialstat_coef_col")
        # t值列名 
        t_value_col = st.text_input("请输入t值列名(如果有标准误列,不需要填, 默认为空值)",  value=None, key="spatialstat_t_value_col")
        # 标准误差列名
        se_col = st.text_input("请输入标准误差列名(如果有t值列,不需要填, 默认为空值)",  value=None, key="spatialstat_se_col")

        # 处理空值选择
        if t_value_col == "无":
            t_value_col = None
        if se_col == "无":
            se_col = None

        # 验证输入
        if t_value_col is None and se_col is None:
            st.warning("请至少选择标准误差列或t值列之一")
            st.stop()
        elif t_value_col is not None and se_col is not None:
            st.warning("请只选择标准误差列或t值列之一，不要同时选择两个")
            st.stop()

        # x轴标题
        x_title = st.text_input("请输入x轴标题", value="Geographic Distance (km)", key="spatialstat_x_title")
        # y轴标题
        y_title = st.text_input("请输入y轴标题", value="Indirect Effect", key="spatialstat_y_title")
        # 颜色选择
        color = st.color_picker("颜色选择", "#1f77b4",key="spatialstat_color")
        # 图像分辨率
        fig_dpi = st.slider("图像分辨率 (DPI)", min_value=300, max_value=1200, value=600,key="spatialstat_fig_dpi")
        # 语言选择
        zn = bool(st.checkbox("是否选择中文绘图, 默认使用English",key="spatialstat_zn"))   
        # 实例化对象
        drawer = Draw_Figure(zn)
        # 绘制图像
        fig = drawer.spatial_decay_plot(df_spatialstat, 
                                coef_col=coef_col, 
                                t_value_col=t_value_col,
                                se_col=se_col, 
                                Distance_threshold=list(range(initial_value + distance_threshold, final_value + distance_threshold, distance_threshold)),
                                filter_min_threshold=filter_min_threshold,
                                filter_max_threshold=filter_max_threshold,
                                confidence_interval=confidence_interval,

                                figsize=(10, 6),
                                x_line_value=x_line_value,
                                x_title=x_title,
                                y_title=y_title,
                                show_plot=False,
                                color=color)

        # 导出图像为多种格式
        image_buffers = drawer.export_fig(fig, dpi=fig_dpi)

        # 预览图像
        st.image(image_buffers['png'], caption="空间衰减边界图", use_container_width=True)                
        # 提供多种格式下载按钮
        col1, col2, col3 = st.columns(3)
        col4, col5, col6  = st.columns(3)

        with col1:
            st.download_button(
                label="📥 下载 PNG 图像",
                data=image_buffers['png'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )

        with col2:
            st.download_button(
                label="📄 下载 PDF 图像",
                data=image_buffers['pdf'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

        with col3:
            st.download_button(
                label="📐 下载 SVG 图像",
                data=image_buffers['svg'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                mime="image/svg+xml"
            )
        with col4:
            st.download_button(
                label="📜 下载 EPS 图像",
                data=image_buffers['eps'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eps",
                mime="image/eps"
            )            

        with col5:
            st.download_button(
                label="🖼️ 下载 TIFF 图像",
                data=image_buffers['tiff'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tiff",
                mime="image/tiff"
            )

        with col6:
            st.download_button(
                label="📷 下载 JPG 图像",
                data=image_buffers['jpg'],
                file_name=f"spatialdistance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg"
            )                                

    else:
        st.warning("请上传Excel或CSV文件以继续操作。")