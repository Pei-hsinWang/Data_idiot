import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # 启用实验性功能
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from scipy.stats import linregress 
import xgboost as xgb
from io import BytesIO
import streamlit as st

# 数据填补模块
class ImputerPipeline:

    """
    用于按顺序执行多种缺失值插补方法的类。
    支持自定义每种插补方法的参数，并能输出参数详情。
    
    参数:
        methods (list): 插补方法列表。
        params (dict): 每个方法对应的参数配置字典。
    """
    def __init__(self, methods, params=None):
        self.methods = methods
        self.params = params or {}
        self._validate_methods()
    
    def _validate_methods(self):
        """验证所有方法是否有效。"""
        valid_methods = [
            'linear_interpolation', 
            'quadratic_spline', 
            'cubic_spline', 
            'mean',
            'median', 
            'mice', 
            'knn', 
            'xgboost'
        ]
        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method: {method}")  

    def fit_transform(self, df):
        """
        对输入 DataFrame 应用插补方法。
        
        参数:
            df (pd.DataFrame): 原始数据框。
        
        返回:
            pd.DataFrame: 处理后的数据框。
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_df = df[numeric_cols]

        for method in self.methods:
            if method == 'linear_interpolation':
                numeric_df = self._interpolate(numeric_df, 'linear')
            elif method == 'quadratic_spline':
                numeric_df = self._interpolate(numeric_df, 'spline', order=2)
            elif method == 'cubic_spline':
                numeric_df = self._interpolate(numeric_df, 'spline', order=3)
            elif method == 'mean':
                numeric_df = self._fill_mean(numeric_df)
            elif method == 'median':
                numeric_df = self._fill_median(numeric_df)
            elif method == 'mice':
                numeric_df = self._impute_mice(numeric_df)
            elif method == 'knn':
                numeric_df = self._impute_knn(numeric_df)
            elif method == 'xgboost':
                numeric_df = self._impute_xgboost(numeric_df)

        df[numeric_cols] = numeric_df
        return df

    def _interpolate(self, df, method, order=None):
        method_params = self.params.get(method, {})
        limit_direction = method_params.get('limit_direction', 'both')
        limit = method_params.get('limit', None)
        return df.interpolate(
            method=method,
            order=order,
            limit_direction=limit_direction,
            limit=limit,
            axis=0,
            inplace=False
        )

    def _fill_mean(self, df):
        for col in df.columns:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
        return df

    def _fill_median(self, df):
        for col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        return df

    def _impute_mice(self, df):
        method_params = self.params.get('mice', {})
        estimator = method_params.get('estimator', BayesianRidge())
        max_iter = method_params.get('max_iter', 10)
        random_state = method_params.get('random_state', 0)

        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            random_state=random_state
        )
        numeric_array = imputer.fit_transform(df)
        return pd.DataFrame(numeric_array, columns=df.columns, index=df.index)

    def _impute_knn(self, df):
        method_params = self.params.get('knn', {})
        n_neighbors = method_params.get('n_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        numeric_array = imputer.fit_transform(df)
        return pd.DataFrame(numeric_array, columns=df.columns, index=df.index)

    def _impute_xgboost(self, df):
        method_params = self.params.get('xgboost', {})
        n_estimators = method_params.get('n_estimators', 100)
        random_state = method_params.get('random_state', 0)

        for col in df.columns:
            if not df[col].isnull().any():
                continue
            X = df.drop(columns=[col]).copy()
            y = df[col].copy()
            missing_idx = y[y.isnull()].index
            X_train = X.loc[~missing_idx]
            y_train = y.loc[~missing_idx]
            X_test = X.loc[missing_idx]

            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_state
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            df.loc[missing_idx, col] = preds
        return df

# 插补参数配置模块  
class ImputationConfig:
    # 获取插补参数
    @staticmethod
    def get_params():
        return {
            'linear_interpolation': {
                'limit_direction': 'both',
                'limit': 2
            },
            'mice': {
                'estimator': BayesianRidge(),
                'max_iter': 5,
                'random_state': 42
            },
            'knn': {
                'n_neighbors': 3
            },
            'xgboost': {
                'n_estimators': 50,
                'random_state': 42
            }
        }

# 工具模块
class DataUtils:
    # 获取 DataFrame 中每个变量的缺失值数量和比例
    @staticmethod
    def get_missing_stats(df):
        """
        返回 DataFrame 中每个变量的缺失值数量和比例。
    
        参数:
            df (pd.DataFrame): 输入数据框。
        
        返回:
            pd.DataFrame: 包含缺失值数量和比例的统计表。
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("输入必须是 pandas DataFrame")
        df = df.replace('.', np.nan)
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        return pd.DataFrame({
            '缺失值数量': missing_count,
            '缺失值比例(%)': missing_percentage.round(2)
        })
    
    # 读取文件方法
    @staticmethod
    def read_file(file,header=0):
        """
        根据文件类型自动读取 .xlsx 或 .csv 文件。

        参数:
            file (str): 文件路径。
            header (int): 表头行数，默认为0。

        返回:
            pandas.DataFrame: 读取后的数据。
        """
        if file.name.endswith('.xlsx'):
            return pd.read_excel(file, header=header)
        elif file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            raise ValueError("❌ 不支持的文件格式，请上传 .xlsx 或 .csv 文件。")
    
    @staticmethod
    # 工具函数 - 中位数分组
    def median_grouping(df, column):
        # 计算指定列的中位数
        median = df[column].median()
        # 根据中位数创建新列：大于中位数的值标记为1，否则标记为0
        df[f"{column}_group"] = np.where(df[column] > median, 1, 0)
        # 返回更新后的 DataFrame 和计算得到的中位数
        return df, median
    
    # 工具函数 - 指标比重计算
    @staticmethod
    def calculate_ratio(df, filter_col, filter_value, condition_col, condition_value, target_col):
        """
        计算某条件下目标列的占比。
        """
        filtered_df = df[df[filter_col] == filter_value]
        group_sum = filtered_df.groupby(condition_col)[target_col].sum()
        total = group_sum.sum()
        value = group_sum.get(condition_value, 0)
        return value / total if total != 0 else 0.0
    # 工具函数 - 宽面板转长面板
    @staticmethod
    def wide_to_long(df, id_vars=None, value_vars=None, var_name='variable', value_name='value'):
        """
        将宽面板数据转换为长面板格式
        
        参数:
            df (pd.DataFrame): 输入的宽面板 DataFrame
            id_vars (list): 不需要转换的标识变量列（如 Year、ID）
            value_vars (list): 需要转换的宽格式列（如 Region_A_Sales, Region_B_Sales）
            var_name (str): 新生成的变量名列名，默认 'variable'
            value_name (str): 新生成的值列名，默认 'value'
            
        返回:
            pd.DataFrame: 转换后的长面板 DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("输入必须是一个 pandas DataFrame")

        if value_vars is None:
            value_vars = df.columns.tolist()
            if id_vars is not None:
                value_vars = [col for col in value_vars if col not in id_vars]

        if not value_vars:
            raise ValueError("必须指定至少一列用于转换的列（value_vars）")

        # 使用 pd.melt 进行长面板转换
        long_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )

        return long_df

# 综合指数模块
class IndicatorsAggregation:
    """
    综合评价类：支持多种指标权重与得分计算方法。
    
    当前已实现：
    - entropy_weight_method: 熵权法
    - entropy_weight_topsis_method: 熵权TOPSIS法
    - coefficient_of_variation_method: 变异系数法
    - pca_method: 主成分分析法
    - grey_relation_analysis_method: 灰色关联分析法
    """

    @staticmethod
    def entropy_weight_method(df, cols, directions=None):
        """
        对指定的多列数据执行熵权法，返回每列的权重和得分
        
        参数:
            df (pd.DataFrame): 输入数据框
            cols (list): 要参与计算的数值列名列表
            directions (dict): 每列的方向 {"col1": "正向指标", "col2": "负向指标"}，默认全为正向
        
        返回:
            pd.DataFrame: 包含权重、信息熵等中间结果的数据框
        """
        if directions is None:
            directions = {col: "正向指标" for col in cols}

        # 归一化处理
        normalized_df = pd.DataFrame()
        for col in cols:
            x_min = df[col].min()
            x_max = df[col].max()

            if x_max == x_min:
                normalized_df[col + "_normalized"] = 0.5
            else:
                if directions.get(col, "正向指标") == "正向指标":
                    normalized_df[col + "_normalized"] = (df[col] - x_min) / (x_max - x_min)
                else:
                    normalized_df[col + "_normalized"] = (x_max - df[col]) / (x_max - x_min)

        # 计算每列的概率 p_ij
        p = normalized_df / normalized_df.sum()
        p = p.fillna(0) + 1e-10  # 防止 log(0)

        # 计算信息熵 e_j
        entropy = -(p * np.log(p)).sum() / np.log(len(df))

        # 差异系数 d_j 与 权重 w_j
        d = 1 - entropy
        weight = d / d.sum() if d.sum() != 0 else 0.0
        
        # 数据集结：加权合成得分
        weighted_score = (normalized_df * weight).sum(axis=1)

        # 构造结果 DataFrame
        result_df = pd.DataFrame({
            '信息熵': entropy,
            '差异系数': d,
            '权重': weight,
        })
        score_df = pd.DataFrame({
            '得分': weighted_score,
            '排名': weighted_score.rank(method='dense', ascending=False).astype(int)
        })

        return result_df, score_df
    

    @staticmethod
    def entropy_weight_topsis_method(df, cols=None, directions=None):
        """
        实现熵权+TOPSIS综合评价方法
        
        参数:
            df (pd.DataFrame): 输入数据框
            cols (list): 要参与计算的数值列名列表
            directions (dict): 每列的方向 {"col1": "正向指标", "col2": "负向指标"}，默认全为正向
            
        返回:
            pd.DataFrame: 包含权重、得分、排名等字段的数据框
        """
        if cols is None:
            cols = df.columns.tolist()

        if directions is None:
            directions = {col: "正向指标" for col in cols}

        # 归一化处理
        normalized_df = pd.DataFrame()
        for col in cols:
            x_min = df[col].min()
            x_max = df[col].max()

            if x_max == x_min:
                normalized_df[col + "_normalized"] = 0.5
            else:
                if directions.get(col, "正向指标") == "正向指标":
                    normalized_df[col + "_normalized"] = (df[col] - x_min) / (x_max - x_min)
                else:
                    normalized_df[col + "_normalized"] = (x_max - df[col]) / (x_max - x_min)

        # 计算概率和信息熵
        p = normalized_df / normalized_df.sum()
        p = p.fillna(0) + 1e-10
        entropy = -(p * np.log(p)).sum() / np.log(len(df))
        d = 1 - entropy
        weight = d / d.sum()

        # 加权归一化矩阵
        weighted_normalized = normalized_df * weight.values

        # 正理想解 & 负理想解
        positive_ideal = weighted_normalized.max()
        negative_ideal = weighted_normalized.min()

        # 距离计算
        distance_positive = np.sqrt(((weighted_normalized - positive_ideal) ** 2).sum(axis=1))
        distance_negative = np.sqrt(((weighted_normalized - negative_ideal) ** 2).sum(axis=1))

        # 相对接近度
        closeness = distance_negative / (distance_positive + distance_negative)

        # 构造结果 DataFrame
        weigth_df = pd.DataFrame({
            '权重': weight,
        })
        score_df = pd.DataFrame({
            '到正理想解距离': distance_positive,
            '到负理想解距离': distance_negative,
            '相对接近度(得分)': closeness,
        })

        return weigth_df, score_df
    
    @staticmethod
    def coefficient_of_variation_method(df, cols=None, directions=None):
        """
        实现变异系数法综合评价方法
        
        参数:
            df (pd.DataFrame): 输入数据框
            cols (list): 要参与计算的数值列名列表
            directions (dict): 每列的方向 {"col1": "正向指标", "col2": "负向指标"}，默认全为正向
            
        返回:
            pd.DataFrame: 包含变异系数、权重、得分等字段的数据框
        """
        if cols is None:
            cols = df.columns.tolist()

        if directions is None:
            directions = {col: "正向指标" for col in cols}

        # 归一化处理
        normalized_df = pd.DataFrame()
        for col in cols:
            x_min = df[col].min()
            x_max = df[col].max()

            if x_max == x_min:
                normalized_df[col + "_normalized"] = 0.5
            else:
                if directions.get(col, "正向指标") == "正向指标":
                    normalized_df[col + "_normalized"] = (df[col] - x_min) / (x_max - x_min)
                else:
                    normalized_df[col + "_normalized"] = (x_max - df[col]) / (x_max - x_min)

        # 计算变异系数
        mean = normalized_df.mean()
        std = normalized_df.std()
        cv = std / mean

        # 计算权重
        weight = cv / cv.sum()

        # 计算得分
        score = (normalized_df * weight).sum(axis=1)

        # 构造结果 DataFrame
        weight_df = pd.DataFrame({
            '变异系数': cv,
            '权重': weight,
        })
        score_df  = pd.DataFrame({
            '得分': score
        })
        return weight_df,score_df
    
    @staticmethod
    def pca_method(df, cols=None, directions=None, threshold=0.85):
        """
        实现主成分分析法
        
        参数:
            df (pd.DataFrame): 输入数据框
            cols (list): 要参与计算的数值列名列表
            directions (dict): 每列的方向 {"col1": "正向指标", "col2": "负向指标"}，默认全为正向
            threshold (float): 累计方差贡献率阈值，默认 0.85
            
        返回:
            pd.DataFrame: 包含权重、综合得分等字段的数据框（score_df）
            dict: 包含碎石图对象的字典（weight_df）
            matplotlib.figure.Figure: 可视化图表对象（用于前端展示）
        """

        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if cols is None:
            cols = df.columns.tolist()

        if directions is None:
            directions = {col: "正向指标" for col in cols}

        # 归一化与方向调整
        X = df[cols].values
        X_scaled = StandardScaler().fit_transform(X)

        for i, col in enumerate(cols):
            if directions[col] == "负向指标":
                X_scaled[:, i] = -X_scaled[:, i]

        # 主成分分析
        pca = PCA()
        pca.fit(X_scaled)
        
        # 累计贡献率
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_selected_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

        # 计算综合得分
        scores = pca.transform(X_scaled)
        weights = pca.explained_variance_ratio_[:n_selected_components]
        total_scores = scores[:, :n_selected_components].dot(weights)
        
        # 构造结果 DataFrame
        score_df = pd.DataFrame({
            '综合得分': total_scores,
            '排名': pd.Series(total_scores).rank(method='dense').astype(int)
        }, index=df.index)

        weight_df = pd.DataFrame({
            '主成分': np.arange(1, len(pca.explained_variance_ratio_) + 1),
            '方差贡献率': pca.explained_variance_ratio_,
            '累计贡献率': cumulative_variance_ratio
        })

        # 绘图部分
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, label='主成分方差贡献率')
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('方差贡献率')

        ax2 = ax1.twinx()
        ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, color='red', marker='o', linestyle='--', label='累计方差贡献率')
        ax2.set_ylabel('累计方差贡献率')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')

        ax1.grid(True)

        return weight_df, score_df, fig
    
    @staticmethod
    def grey_relational_analysis(df, cols=None, directions=None):
        """
        实现灰色关联分析法
        
        参数:
            df (pd.DataFrame): 输入数据框
            cols (list): 要参与计算的数值列名列表
            directions (dict): 每列的方向 {"col1": "正向指标", "col2": "负向指标"}，默认全为正向
            
        返回:
            pd.DataFrame: 包含灰色关联度的数据框（weight_df）
            pd.DataFrame: 包含综合得分和排名的数据框（score_df）
        """
        import numpy as np
        import pandas as pd

        if cols is None:
            cols = df.columns.tolist()

        if directions is None:
            directions = {col: "正向指标" for col in cols}

        # 归一化与方向调整
        X = df[cols].values.copy()

        for i, col in enumerate(cols):
            col_min = X[:, i].min()
            col_max = X[:, i].max()
            if col_min == col_max:
                X[:, i] = 0.5
            else:
                if directions[col] == "正向指标":
                    X[:, i] = (X[:, i] - col_min) / (col_max - col_min)
                else:
                    X[:, i] = (col_max - X[:, i]) / (col_max - col_min)

        # 构造参考序列（最优序列）
        ref_seq = np.max(X, axis=0)

        # 计算差分矩阵
        delta = np.abs(X - ref_seq)

        # 计算最小差和最大差
        min_diff = np.min(np.min(delta))
        max_diff = np.max(np.max(delta))

        # 关联系数计算
        rho = 0.5
        gamma = (min_diff + rho * max_diff) / (delta + rho * max_diff)

        # 计算平均关联度作为权重
        weights = np.mean(gamma, axis=0)
        weight_df = pd.DataFrame({
            '灰色关联度': weights
        }, index=cols)

        # 计算综合得分
        scores = gamma.dot(weights)
        score_df = pd.DataFrame({
            '综合得分': scores,
            '排名': pd.Series(scores).rank(method='dense', ascending=False).astype(int)
        }, index=df.index)

        return weight_df, score_df

# 缓存与导出模块
class DataExporter:
    @staticmethod
    @st.cache_data
    def convert_df_to_format(data, format_type, sheet_names=("Sheet1", "Sheet2")):
        """
        将 DataFrame 转换为指定格式的字节流数据。

        参数:
            df (pd.DataFrame): 要导出的数据框。
            format_type (str): 导出格式，支持 "xlsx" 或 "csv"。

        返回:
            bytes: 文件字节数据。
        """
        if format_type == "xlsx":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    result_df, score_df = data
                    result_df.to_excel(writer, sheet_name=sheet_names[0], index=False)
                    score_df.to_excel(writer, sheet_name=sheet_names[1], index=False)
                else:
                    data.to_excel(writer, sheet_name=sheet_names[0], index=False)
            return output.getvalue()
        elif format_type == "csv":
            if isinstance(data, (list, tuple)) and len(data) == 2:
                result_df, score_df = data
                return result_df.to_csv(index=False) + "\n\n" + score_df.to_csv(index=False)
            else:
                return data.to_csv(index=False)
        else:
            raise ValueError(f"不支持的导出格式: {format_type}")

    @staticmethod
    def get_mime_and_extension(format_type):
        """
        获取对应格式的 MIME 类型和文件扩展名。

        参数:
            format_type (str): 导出格式，支持 "xlsx" 或 "csv"。

        返回:
            tuple: (mime_type, file_extension)
        """
        mime_map = {
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "csv": "text/csv"
        }
        return mime_map[format_type], f".{format_type}"

    # 生成多种格式的图像字节流
    @staticmethod
    def export_fig(fig, dpi=600):
        formats = {
            'png': BytesIO(),
            'pdf': BytesIO(),
            'svg': BytesIO(),
            'eps': BytesIO(),
            'tiff': BytesIO(),
            'jpg': BytesIO()
        }

        # 保存为不同格式
        fig.savefig(formats['png'], format='png', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['pdf'], format='pdf', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['svg'], format='svg', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['eps'], format='eps', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['tiff'], format='tiff', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['jpg'], format='jpg', dpi=dpi, bbox_inches='tight')

        # 重置缓冲区指针
        for buf in formats.values():
            buf.seek(0)

        return formats

# 绘图模块
class Draw_Figure():

    def __init__(self,zn=False):
        self.apply_style(zn)
    # 设置中英文样式
    @staticmethod
    def apply_style(zn=False):
        # 强制转换布尔值
        zn = bool(zn)
        if zn:
            config = {
            "font.family": "serif",
            "font.serif": ["SimSun"],
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 0.8,
            "errorbar.capsize": 2,
            "axes.unicode_minus": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "mathtext.fontset": "stix",
            "text.usetex": False,
        }
        else:
            config = {
                "font.family": "serif",
                "font.serif": ["Times New Roman"],  # 级联
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.linewidth": 0.8,
                "lines.linewidth": 0.8,
                "errorbar.capsize": 2,
                "axes.unicode_minus": False,
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "text.usetex": False
            }
        plt.rcParams.update(config)
    

    # 计算相关系数矩阵并绘制热力图
    def correlation_matrix(self,df,feature_cols ,show_plot=True, cmap='viridis'):

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 计算相关系数矩阵
        corr = df[feature_cols].corr()
        
        # 使用 seaborn 默认颜色映射绘制热力图
        heatmap = sns.heatmap(
            corr, 
            annot=True, 
            fmt=".2f", 
            cmap=cmap,  # 支持 'viridis', 'coolwarm', 'cividis', 'magma' 等
            square=True, 
            cbar_kws={"shrink": .8},
            ax=ax
        )     
        
        # 自动调整布局
        fig.tight_layout()
        
        # 如果不需要显示，则关闭图像
        if not show_plot:
            plt.close(fig)
            return fig
        
        return fig
    def joint_distribution_plot(self, df, x, y, figsize=(10, 6), show_plot=True,color='#1f77b4'):
        """
        绘制特征变量和目标变量的联合分布图，并添加线性回归线及方程

        参数：
        - df: 数据集
        - x: X轴特征名称
        - y: Y轴特征名称
        - figsize: 图像大小
        - show_plot: 是否显示图像
        """
        # 均值填补缺失值
        df_clean = df[[x, y]].copy()
        df_clean[x] = df_clean[x].fillna(df_clean[x].mean())
        df_clean[y] = df_clean[y].fillna(df_clean[y].mean())

        # 使用 seaborn 绘制联合分布图，调整散点样式以增强回归线可视性
        g = sns.jointplot(
            data=df_clean,
            x=x,
            y=y,
            color=color,
            kind="reg",
            height=figsize[1],
            scatter_kws={'alpha': 0.8, 's': 20}  # 正确方式：通过 scatter_kws 控制散点样式
        )

        # 获取当前图表的Axes对象
        ax = g.ax_joint

        # 计算线性回归参数
        slope, intercept, r_value, p_value, std_err = linregress(df_clean[x], df_clean[y])
        equation = f'$y = {slope:.2f}x + {intercept:.2f}$\n$R^2 = {r_value**2:.2f}$'
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # 自动调整布局
        g.fig.tight_layout()

        # 如果不需要显示，则关闭图像
        if not show_plot:
            plt.close(g.fig)
            return g.fig

        return g.fig    
    # 绘制森林图
    def forest_plot(self, df, subgroup_col, coef_col, se_col, color_map=None, default_color="#999999", figsize=(6, 4), show_plot=True):
        """
        绘制森林图(Forest Plot)用于异质性分析

        参数：
        - df: 数据集
        - subgroup_col: 子组列名 (如 "subgroup")
        - coef_col: 回归系数列名 (如 "coef")
        - se_col: 标准误列名 (如 "se")
        - color_map: 子组到颜色的映射字典
        - default_color: 默认颜色（用于未在 color_map 中定义的子组）
        - figsize: 图像大小
        - show_plot: 是否显示图像
        """
        df = df.copy()
        df["lower"] = df[coef_col] - 1.96 * df[se_col]
        df["upper"] = df[coef_col] + 1.96 * df[se_col]

        # 设置颜色
        if color_map is None:
            color_map = {}
        df["color"] = df[subgroup_col].map(color_map).fillna(default_color)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(df))

        # 逐行绘制 errorbar
        for idx, (_, row) in enumerate(df.iterrows()):
            ax.errorbar(
                x=row[coef_col],
                y=y_pos[idx],
                xerr=[[row[coef_col] - row["lower"]], [row["upper"] - row[coef_col]]],
                fmt="o",
                color=row["color"],
                capsize=4,
                capthick=1.5,
                markersize=6
            )

        # 添加零线
        ax.axvline(0, color="gray", linestyle="--")

        # 设置坐标轴和标题
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df[subgroup_col])
        ax.set_xlabel("Coefficient (95% CI)")
        # 反转 Y轴方向，使 Y轴从上到下递减
        ax.invert_yaxis()
        # 自动调整布局
        fig.tight_layout()

        # 如果不需要显示，则关闭图像
        if not show_plot:
            plt.close(fig)
            return fig

        return fig

    # 生成多种格式的图像字节流
    def export_fig(self, fig, dpi=600):
        formats = {
            'png': BytesIO(),
            'pdf': BytesIO(),
            'svg': BytesIO(),
            'eps': BytesIO(),
            'tiff': BytesIO(),
            'jpg': BytesIO()
        }

        # 保存为不同格式
        fig.savefig(formats['png'], format='png', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['pdf'], format='pdf', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['svg'], format='svg', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['eps'], format='eps', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['tiff'], format='tiff', dpi=dpi, bbox_inches='tight')
        fig.savefig(formats['jpg'], format='jpg', dpi=dpi, bbox_inches='tight')

        # 重置缓冲区指针
        for buf in formats.values():
            buf.seek(0)

        return formats

    # 空间衰减边界图
    def spatial_decay_plot(self, 
                            df, 
                            coef_col: str = None, 
                            t_value_col: str = None,
                            se_col: str = None, 
                            Distance_threshold: np.array = None,
                            filter_min_threshold:int = 50,
                            filter_max_threshold:int  = 600,
                            confidence_interval: float = 0.95,
                            figsize=(10, 6),
                            x_line_value:int=None,
                            x_title:str="Geographic Distance (km)",
                            y_title:str="Indirect Effect",
                            show_plot=True,
                            color='#1f77b4'):
        """
        绘制空间衰减边界图

        参数：
        - df: DataFrame, 包含距离和效应值的数据集
        - coef_col: str, 效应值列名
        - t_value_col: str, t值列名
        - se_col: str, 标准误列名
        - Distance_threshold: list, 距离阈值列表
        - filter_min_threshold: int, 最小距离阈值
        - filter_max_threshold: int, 最大距离阈值
        - confidence_interval: float, 置信区间
        - figsize: tuple, 图像大小
        - x_line_value: int, 参考线值
        - x_title: str, X轴标题
        - y_title: str, Y轴标题
        - show_plot: bool, 是否显示图像
        - color: str, 图像颜色

        """
        # 输入验证
        if coef_col is None:
            raise ValueError("必须提供效应值列名 (coef_col)")
        
        if se_col is None and t_value_col is None:
            raise ValueError("必须提供标准误列名 (se_col) 或 t值列名 (t_value_col)")

        # 如果没有提供距离阈值，则假设数据行与距离一一对应（第1行对应50km，第2行对应100km，以此类推）
        if Distance_threshold is None:
            Distance_threshold = np.arange(50, 50 * (len(df) + 1), 50)
        else:
            # 确保 Distance_threshold 是 NumPy 数组
            Distance_threshold = np.array(Distance_threshold)
        
        # 验证数据长度一致性
        if len(df) != len(Distance_threshold):
            raise ValueError(f"数据长度不匹配：df行数为{len(df)}，Distance_threshold长度为{len(Distance_threshold)}")
        
        # 直接使用数据的行索引与距离阈值对应
        distances = Distance_threshold
        coeff = np.array(df[coef_col], dtype=float)
        
        if se_col is not None:
            se = np.array(df[se_col], dtype=float)
            # 确保标准误为正数
            se = np.abs(se)
        else:
            if t_value_col is not None:
                t_values = np.array(df[t_value_col], dtype=float)
                # 计算标准误并确保为正数
                se = np.abs(coeff / t_values)
            else:
                raise ValueError("请输入标准误列名或t值列名")

        # 筛选数据，只保留指定距离范围内的数据
        mask = (distances >= filter_min_threshold) & (distances <= filter_max_threshold)
        filtered_distances = distances[mask]
        coeff = coeff[mask]
        se = se[mask]

        # 检查是否有无效值
        valid_mask = ~(np.isnan(coeff) | np.isnan(se) | np.isinf(coeff) | np.isinf(se))
        coeff = coeff[valid_mask]
        se = se[valid_mask]
        filtered_distances = filtered_distances[valid_mask]

        # 创建画布
        fig, ax = plt.subplots(figsize=figsize)

        # 设置置信区间系数
        if confidence_interval == 0.90:
            z_score = 1.645
        elif confidence_interval == 0.95:
            z_score = 1.96
        elif confidence_interval == 0.99:
            z_score = 2.576
        else:
            raise ValueError(f"不支持的置信区间: {confidence_interval}")

        # 统一绘图逻辑
        ax.errorbar(
            filtered_distances, coeff,
            yerr=z_score * se,
            fmt='o', markersize=8, capsize=5, capthick=1.5, elinewidth=1.5,
            color=color, label=f'Indirect Effect ({int(confidence_interval*100)}% CI)'
        )

        # 添加 y=0 参考线，更换颜色
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
        # 添加参考线，以划分区间
        if x_line_value is not None:
            ax.axvline(x_line_value, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        # 设置标题和标签
        ax.set_xlabel(x_title, fontsize=14, labelpad=10)
        ax.set_ylabel(y_title, fontsize=14, labelpad=10)

        # 设置横轴刻度和范围
        ax.set_xticks(np.arange(filter_min_threshold, filter_max_threshold + 1, 50)) 
        ax.set_xlim(filter_min_threshold - 25, filter_max_threshold + 25)  # 设置横轴范围留出一些边距
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # 自动调整布局
        fig.tight_layout()

        # 如果不需要显示，则关闭图像
        if not show_plot:
            plt.close(fig)
            return fig

        return fig


# 空间计量模块
class Spatial_Eco:
    @staticmethod
    def compute_weighted_panel_multi_variables(df, weight_matrix, id_col='id', year_col='年份', value_cols=None, normalize_weights=True):
        """
        对面板数据中的多个变量按年份分组，并与权重矩阵进行矩阵乘法运算，最后还原为长面板格式
        
        Parameters:
        df: 长面板数据DataFrame
        weight_matrix: 权重矩阵
        id_col: ID列名
        year_col: 年份列名
        value_cols: 需要处理的变量名列（列表格式）
        normalize_weights: 是否标准化权重矩阵
        
        Returns:
        result_df: 加权后的长面板格式结果
        """
        
        # 确保权重矩阵为numpy数组
        weight_array = np.asarray(weight_matrix)
        
        # 如果需要标准化权重矩阵
        if normalize_weights:
            row_sums = weight_array.sum(axis=1, keepdims=True)
            # 避免除以零的情况
            row_sums = np.where(row_sums == 0, 1, row_sums)
            weight_array = weight_array / row_sums

        # 如果未指定变量列，则默认处理所有数值型列（除了年份列和ID列）
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if year_col in value_cols:
                value_cols.remove(year_col)
            if id_col in value_cols:
                value_cols.remove(id_col)
        
        # 存储结果
        result_list = []
        
        # 按年份分组计算
        for year, group in df.groupby(year_col):
            # 确保数据按ID列排序，以与权重矩阵对齐
            group_sorted = group.sort_values(by=id_col).reset_index(drop=True)
            
            # 为每个变量执行矩阵乘法
            weighted_data = {}
            # 保留标识符列
            weighted_data[id_col] = group_sorted[id_col].values
            weighted_data[year_col] = [year] * len(group_sorted)
            
            # 复制非数值列
            non_numeric_cols = [col for col in group_sorted.columns if col not in value_cols + [id_col, year_col]]
            for col in non_numeric_cols:
                weighted_data[col] = group_sorted[col].values
            
            # 对指定的数值列进行加权计算
            for col in value_cols:
                values = group_sorted[col].values
                
                # 检查维度是否匹配
                if weight_array.shape[1] != len(values):
                    raise ValueError(f"权重矩阵列数({weight_array.shape[1]})与{year}年{col}变量数据长度({len(values)})不匹配")
                
                # 执行矩阵乘法
                weighted_data[col + '_weighted'] = weight_array.dot(values)
            
            # 添加原始值列
            for col in value_cols:
                weighted_data[col] = group_sorted[col].values
                
            result_list.append(pd.DataFrame(weighted_data))
        
        # 合并所有年份的结果
        result_df = pd.concat(result_list, ignore_index=True)
        
        # 按id和年份排序
        result_df.sort_values(by=[id_col, year_col], inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        return result_df

    @staticmethod
    def spatial_spillover_matrix(weight_matrix,step=50,begin_distance=50,end_distance=700):
        """
        根据不同的距离阈值生成空间溢出矩阵
        
        Parameters:
        weight_matrix: 权重矩阵
        step: 距离阈值的步长
        begin_distance: 起始距离阈值
        end_distance: 结束距离阈值
        normalize_weights: 是否标准化权重
        
        Returns:
        dict: 包含多个距离阈值对应的空间溢出矩阵的字典
        """
        # 创建存储矩阵的字典
        matrix_dict = {}
        
        # 计算距离阈值范围
        current_distance = begin_distance
        while current_distance <= end_distance:
            # 根据距离阈值生成空间溢出矩阵
            matrix_dict[f'D_threshold{current_distance}'] = weight_matrix.applymap(
                lambda x: 0 if x < current_distance else (1/x if x != 0 else 0))
            current_distance += step
        return matrix_dict

    @staticmethod
    def distance_dummies(
        dist_df: pd.DataFrame,
        policy_data: pd.DataFrame,
        id_col: str = '行政区划代码',
        year_col: str = '年份',
        treat_col: str = None,
        start_year: int = 2005,
        end_year: int = 2020,
        thresholds: list = None
    ):
        """
        基于已有地理距离矩阵和政策数据，生成距离阈值虚拟变量。
        """
        
        # 默认距离阈值设置：50km到900km，间隔50km
        if thresholds is None:
            thresholds = list(range(50, 950, 50))  # 50km 到 900km

        print("加载距离矩阵...")

        # 提取城市代码列表（第一行数据）
        cities = dist_df.iloc[0].tolist()
        # 解决关键问题：清洗城市代码，去除.0后缀并转换为字符串
        # 例如：110000.0 -> '110000'
        cities = [str(int(city)) if isinstance(city, (int, float)) else str(city).split('.')[0] 
                for city in cities]
        
        # 提取距离数据矩阵（除第一行外的所有数据）
        dist_matrix = dist_df.iloc[1:].copy()
        #dist_matrix = dist_df.copy()
        # 设置列名为清洗后的城市代码
        dist_matrix.columns = cities
        # 设置行索引为清洗后的城市代码
        dist_matrix.index = cities
        # 确保距离数据为浮点数类型
        dist_matrix = dist_matrix.astype(float)

        print(f"距离矩阵加载完成，形状: {dist_matrix.shape}")

        print("加载政策数据...")

        # 清洗政策数据中的行政区划代码：确保为字符串并去除空格
        policy_data[id_col] = policy_data[id_col].astype(str).str.strip()
        # 进一步清洗：去除可能存在的.0后缀
        policy_data[id_col] = policy_data[id_col].str.split('.').str[0]

        results = []

        # 调试信息：显示清洗后的数据格式
        print("清洗后距离矩阵中的城市代码示例:", list(dist_matrix.columns)[:5])
        print("清洗后政策数据中的城市代码示例:", list(policy_data[id_col])[:5])

        for year in range(start_year, end_year + 1):
            print(f"处理年份: {year}")
            # 筛选特定年份的政策数据
            yearly = policy_data[policy_data[year_col] == year]
            if yearly.empty:
                print(f"⚠️  年份 {year} 无政策数据，跳过")
                continue

            # 构建 {city_id: LCC_status} 字典，确保城市ID为字符串类型
            treat_dict = yearly.set_index(id_col)[treat_col].astype(int).to_dict()

            # 获取所有试点城市（LCC=1），并确保ID格式统一
            treated_cities = [str(cid).split('.')[0] for cid, status in treat_dict.items() if status == 1]
            # 获取所有城市列表，并统一格式
            all_cities = [str(cid).split('.')[0] for cid in treat_dict.keys()]

            if not treated_cities:
                # 该年无试点城市，所有虚拟变量设为0
                for city in all_cities:
                    row = {'Year': year, id_col: city}
                    for th in thresholds:
                        row[f'dummy_{th}'] = 0
                    results.append(row)
            else:
                # 对每个城市计算到最近试点城市的距离并生成虚拟变量
                for city in all_cities:
                    # 关键检查：确保城市代码在距离矩阵中存在
                    if city not in dist_matrix.columns:
                        print(f"⚠️  城市 {city} 不在距离矩阵中，跳过")
                        continue

                    # 计算该城市到所有试点城市的距离，取最小值
                    distances = dist_matrix.loc[city, treated_cities]
                    min_dist = distances.min()

                    # 根据不同阈值生成虚拟变量
                    row = {'Year': year, id_col: city}
                    for th in thresholds:
                        row[f'dummy_{th}'] = 1 if min_dist <= th else 0
                    results.append(row)

        # 构建结果 DataFrame
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(by=[id_col, 'Year']).reset_index(drop=True)

        return result_df