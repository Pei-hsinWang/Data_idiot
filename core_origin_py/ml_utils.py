import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import gridspec
import matplotlib.transforms as mtrans
import scienceplots
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import optuna
import joblib
from io import BytesIO
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from PyALE import ale
class SHAPAnalyzer:
    def __init__(self, df,feature_cols, target_col,
                 num_top_features=6, scatter_rows=2, 
                 model_params=None, test_size=0.2, 
                 random_state=42, fig_dpi=300,
                 scatter_density=True, max_points=500,
                 scatter_alpha=0.6, scatter_size=40,
                 param_search_method=None,model_name='random_forest',
                 n_trials=50, cv=5, task_type='auto',
                 custom_labels=None):
        """
        增加的参数：
        - model_name: 模型名称，可选 'random_forest', 'lasso', 'gradient_boosting', 'xgboost', 'lightgbm'
        - param_search_method: 调参方法，可选 None, 'grid_search', 'optuna'
        - n_trials: Optuna 的试验次数
        - cv: 交叉验证折数
        """
        """
        增加的参数：
        - custom_labels: 自定义子图标签列表，如 ['特征A', '特征B']，若为空则用 '(a)', '(b)' 补充
        """
        # 初始化配置参数
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.num_top_features = num_top_features
        self.scatter_rows = scatter_rows
        self.model_params = model_params or {'n_estimators': 100, 'random_state': random_state}
        self.test_size = test_size
        self.random_state = random_state
        self.fig_dpi = fig_dpi

        # 新增样本采样，来解决大样本散点图点过多的不美观问题
        self.scatter_density = scatter_density  # 是否使用点密度采样
        self.max_points = max_points            # 最大显示点数
        self.scatter_alpha = scatter_alpha      # 散点透明度
        self.scatter_size = scatter_size        # 散点大小      

        # 新增机器学习方法与自动化调参
        self.model_name = model_name
        self.param_search_method = param_search_method
        self.n_trials = n_trials
        self.cv = cv

        # 新增参数:task_type: 任务类型，可选 'auto', 'regression', 'classification'
        self.task_type = task_type  
        # 新增参数:custom_labels: 自定义标签
        self.custom_labels = custom_labels 
        # 定义模型及其默认参数
        self.model_params = {
            'random_forest': {'n_estimators': 100, 'random_state': random_state},
            'lasso': {'alpha': 1.0, 'random_state': random_state},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': random_state},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': random_state},
            'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': random_state}
        }
        # 分类任务专用参数空间
        self.classification_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'objective': ['binary:logistic', 'multi:softmax'],
                'eval_metric': ['logloss']
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50],
                'objective': ['binary', 'multiclass']
            }
        }

        # 回归任务参数空间保持不变
        self.regression_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50]
            }
        }

        # 定义超参数搜索空间
        self.param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50]
            }
        }       
        # 配置图形默认设置
        # plt.rcParams['font.family'] = 'Times New Roman'
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.rcParams['font.weight'] = 'bold'
        # plt.rcParams['axes.labelweight'] = 'bold'
        # plt.rcParams['axes.titleweight'] = 'bold'
        plt.style.use('science') # 使用SciencePlots风格
        plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲
    def load_data(self):
        """加载数据集并分离特征和目标变量"""
        self.y = self.target_col
        self.X = self.feature_cols
        self.feature_names = self.X.columns[0:].tolist()
        self.num_features = self.X.shape[1]
        self.num_samples = self.X.shape[0]

        print(f"样本数: {self.num_samples}, 特征数: {self.num_features}")
        print(f"目标变量: {self.y}")
        print(f"特征列表: {self.feature_names[:5]}...")


    def _determine_task_type(self):
        """自动判断任务类型"""
        unique_values = np.unique(self.y)
        if len(unique_values) <= 10 and np.all(np.round(unique_values) == unique_values):
            return 'classification'
        else:
            return 'regression'    
    

    def train_model(self):
        """根据选择的模型和调参方法训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        # 自动判断任务类型
        if self.task_type == 'auto':
            self.task_type = self._determine_task_type()

        # 构建基础模型
        if self.model_name == 'random_forest':
            if self.task_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(random_state=self.random_state)
            else:
                base_model = RandomForestRegressor(random_state=self.random_state)
        elif self.model_name == 'gradient_boosting':
            if self.task_type == 'classification':
                from sklearn.ensemble import GradientBoostingClassifier
                base_model = GradientBoostingClassifier(random_state=self.random_state)
            else:
                base_model = GradientBoostingRegressor(random_state=self.random_state)
        elif self.model_name == 'xgboost':
            if self.task_type == 'classification':
                from xgboost import XGBClassifier
                base_model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            else:
                base_model = XGBRegressor(random_state=self.random_state, eval_metric='logloss')
        elif self.model_name == 'lightgbm':
            if self.task_type == 'classification':
                from lightgbm import LGBMClassifier
                base_model = LGBMClassifier(random_state=self.random_state)
            else:
                base_model = LGBMRegressor(random_state=self.random_state)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_name}")

        # 如果没有启用自动调参，直接使用默认参数训练
        if self.param_search_method is None:
            self.model = base_model.set_params(**self.model_params[self.model_name])
            self.model.fit(X_train, y_train)
            print(f"使用默认参数训练完成 (R²={self.model.score(X_test, y_test):.3f})")
            return

        # 否则使用指定的调参方法
        if self.task_type == 'classification':
            param_grid = self.classification_param_grids[self.model_name]
        else:
            param_grid = self.regression_param_grids[self.model_name]

        if self.param_search_method == 'grid_search':
            grid_search = GridSearchCV(base_model, param_grid, scoring='accuracy' if self.task_type == 'classification' else 'r2', cv=self.cv, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_

        elif self.param_search_method == 'optuna':

            def objective(trial):
                params = {}
                for param_name, values in param_grid.items():
                    if isinstance(values, list):
                        if all(isinstance(v, int) for v in values):
                            params[param_name] = trial.suggest_categorical(param_name, values)
                        elif all(isinstance(v, float) for v in values):
                            min_val = min(values)
                            max_val = max(values)
                            params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                        else:
                            params[param_name] = trial.suggest_categorical(param_name, values)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, [values])

                model = base_model.set_params(**params)
                model.fit(X_train, y_train)
                if self.task_type == 'classification':
                    score = model.score(X_test, y_test)
                else:
                    score = model.score(X_test, y_test)
                return -score  # 因为Optuna默认最小化目标函数

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.n_trials)
            self.model = base_model.set_params(**study.best_params)
            self.model.fit(X_train, y_train)
            best_score = -study.best_value
            best_params = study.best_params

        else:
            raise ValueError(f"不支持的调参方法: {self.param_search_method}")

        print(f"模型训练完成 ({'Accuracy' if self.task_type == 'classification' else 'R²'}={best_score:.3f})")
        print(f"最佳参数: {best_params}")


    def compute_shap_values(self):
        """计算SHAP值并排序特征重要性"""
        if self.task_type == 'classification':
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.X)  # 多分类返回列表
        else:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(self.X)

        # 计算平均绝对SHAP值
        if self.task_type == 'classification':
            # 对多类情况取平均
            mean_abs_shap_per_class = [np.mean(np.abs(sv), axis=0) for sv in self.shap_values]
            self.mean_abs_shap = np.mean(mean_abs_shap_per_class, axis=0)
        else:
            self.mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        self.sorted_indices = np.argsort(self.mean_abs_shap)[::-1]
        self.sorted_features = [self.feature_names[i] for i in self.sorted_indices]
        self.sorted_shap_values = self.mean_abs_shap[self.sorted_indices]

        # 创建特征重要性DataFrame
        self.feature_importance = pd.DataFrame({
            '特征名称': self.sorted_features,
            '平均绝对SHAP值': self.sorted_shap_values
        })
        print("\n特征重要性排名:")
        print(self.feature_importance.head(10))

        # 计算预测值和残差
        self.y_pred = self.model.predict(self.X)
        if self.task_type == 'regression':
            self.residuals = self.y - self.y_pred

    def generate_colors(self):
        """为特征生成颜色映射"""
        n = len(self.feature_names)
        if n <= 20:
            cmap = plt.get_cmap('tab20')
            self.colors = list(cmap.colors[:n])
        elif n <= 40:
            cmap1 = plt.get_cmap('tab20b')
            cmap2 = plt.get_cmap('tab20c')
            self.colors = list(cmap1.colors) + list(cmap2.colors)[:n]
        else:
            hsv = plt.get_cmap('hsv')
            self.colors = [hsv(i / n) for i in range(n)]

    def setup_figure_layout(self):
        """设置图形布局和尺寸"""
        # 确定实际要绘制的散点图数量
        num_scatter = min(self.num_top_features, self.num_features)
        
        # 计算布局参数
        scatter_cols = (num_scatter + self.scatter_rows - 1) // self.scatter_rows
        total_cols = 1 + scatter_cols  # 1列用于条形图
        
        # 计算图形尺寸
        base_width = 4
        bar_width_ratio = 1.5
        fig_width = bar_width_ratio * base_width + scatter_cols * base_width
        fig_height = self.scatter_rows * base_width * 0.9
        
        # 创建图形和网格
        self.fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(self.scatter_rows, total_cols, figure=self.fig,
                              width_ratios=[bar_width_ratio] + [1]*scatter_cols,
                              wspace=0.4, hspace=0.55)
        
        # 创建子图
        self.ax_bar = self.fig.add_subplot(gs[:, 0])
        self.ax_scatters = []
        
        for i in range(num_scatter):
            row = i // scatter_cols
            col = 1 + (i % scatter_cols)
            self.ax_scatters.append(self.fig.add_subplot(gs[row, col]))
    #    设置子图标签
    def _generate_subplot_labels(self):
        """生成子图标签，优先使用自定义标签，否则使用 (a), (b), ..."""
        if self.custom_labels and isinstance(self.custom_labels, list):
            labels = self.custom_labels[:self.num_top_features]
        else:
            labels = []

        # 补足剩余部分为 (a), (b), ...
        from string import ascii_lowercase
        for i in range(len(labels), self.num_top_features):
            labels.append(f"({ascii_lowercase[i]})")
        
        return labels    
    def plot_feature_importance(self):
        """绘制特征重要性条形图"""
        legend_handles = []
        marker_size = 10
        
        for i, feat in enumerate(self.sorted_features):
            orig_idx = self.feature_names.index(feat)
            color = self.colors[orig_idx] if orig_idx < len(self.colors) else 'gray'
            
            self.ax_bar.barh(feat, self.sorted_shap_values[i], color=color)
            
            # 为前6个特征创建图例
            if i < 6:
                handle = plt.Line2D([0], [0], marker='s', linestyle='None',
                                  markersize=marker_size, markerfacecolor=color,
                                  markeredgecolor=color, label=feat)
                legend_handles.append(handle)
        
        self.ax_bar.invert_yaxis()
        self.ax_bar.set_xlabel('Mean(|SHAP| value)', fontsize=10)
        self.ax_bar.tick_params(axis='both', labelsize=10)
        self.ax_bar.legend(handles=legend_handles, fontsize=10, frameon=False)
        self.ax_bar.grid(False)
        
        # 加粗边框
        for spine in self.ax_bar.spines.values():
            spine.set_linewidth(1.5)

    def polynomial_fit(self, x, a, b, c):
        """二次多项式拟合函数"""
        return a * x**2 + b * x + c
    
    def density_based_sampling(self, x, y, n_samples=500):
        """
        基于点密度进行采样，在高密度区域保留更多点
        """
        if len(x) <= n_samples:
            return x, y, np.ones(len(x))
        
        # 计算点密度
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        density = kde(xy)
        
        # 归一化密度值
        normalized_density = density / density.sum()
        
        # 按密度概率采样
        sample_indices = np.random.choice(
            len(x), size=n_samples, 
            replace=False, 
            p=normalized_density
        )
        
        # 计算采样权重用于透明度调整
        weights = density[sample_indices] / density.max()
        
        return x[sample_indices], y[sample_indices], weights
        
    def plot_shap_dependencies(self):
        """绘制SHAP依赖图，使用点密度采样提高可读性"""
        subplot_labels = self._generate_subplot_labels()  # 获取标签列表
        for i, ax in enumerate(self.ax_scatters):
            orig_idx = self.sorted_indices[i]
            feat = self.feature_names[orig_idx]
            color = self.colors[orig_idx]
            
            # 获取数据
            x_data = self.X.iloc[:, orig_idx].values
            #x_data = self.X[:, orig_idx]
            if self.task_type == 'classification':
                y_data = self.shap_values[0][:, orig_idx]  # 取第一个类别的SHAP值
            else:
                y_data = self.shap_values[:, orig_idx]
            
            # 点密度采样
            if self.scatter_density and len(x_data) > self.max_points:
                x_plot, y_plot, weights = self.density_based_sampling(
                    x_data, y_data, n_samples=self.max_points
                )
                # 使用权重调整透明度
                alpha_values = self.scatter_alpha * weights
            else:
                x_plot, y_plot = x_data, y_data
                alpha_values = self.scatter_alpha
            
            # 绘制散点图和零线
            ax.scatter(
                x_plot, y_plot, 
                color=color, 
                alpha=alpha_values, 
                s=self.scatter_size,
                edgecolor='none'
            )
            ax.axhline(0, color='gray', linestyle='--', linewidth=1.2)
            
            # 尝试非线性拟合（使用所有数据点）
            try:
                sorted_idx = np.argsort(x_data)
                x_sorted = x_data[sorted_idx]
                y_sorted = y_data[sorted_idx]
                
                popt, pcov = curve_fit(self.polynomial_fit, x_sorted, y_sorted, maxfev=10000)
                x_fit = np.linspace(min(x_data), max(x_data), 100)
                y_fit = self.polynomial_fit(x_fit, *popt)
                
                ax.plot(x_fit, y_fit, color='black', linewidth=2.5, label='非线性拟合')
                ax.plot(x_fit, y_fit, color=color, linewidth=1.5)
                
                # 绘制置信区间
                perr = np.sqrt(np.diag(pcov))
                try:
                    y_upper = self.polynomial_fit(x_fit, *(popt + 1.96 * perr))
                    y_lower = self.polynomial_fit(x_fit, *(popt - 1.96 * perr))
                    ax.fill_between(x_fit, y_lower, y_upper, color=color, alpha=0.15)
                except Exception:
                    pass
            except RuntimeError:
                ax.text(0.5, 0.5, '拟合失败', fontsize=16, color='red', 
                       ha='center', va='center', transform=ax.transAxes)
            
            # 设置轴标签和格式
            ax.set_xlabel(feat, fontsize=18)
            ax.set_ylabel('SHAP value', fontsize=18)
            ax.tick_params(labelsize=16)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(False)
            ax.set_box_aspect(1)
            
            # 加粗边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                
            # 添加特征名称标题
            # ax.set_title(feat, fontsize=16, pad=10)

            # 在左上角添加子图标签
            ax.text(
                -0.45,1.0, subplot_labels[i],
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
            )
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
    
    def analyze_and_visualize(self, show_plot=True):
        """执行完整分析流程"""
        # 流程步骤
        self.load_data()
        self.train_model()
        self.compute_shap_values()
        self.generate_colors()
        self.setup_figure_layout()
        self.plot_feature_importance()
        self.plot_shap_dependencies()

        # 控制是否显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()  # 防止图像自动显示        plt.show()

class Ale:
    def __init__(self, df, random_state=42,zn=False):
        self.df = df
        self.random_state = random_state
        self.apply_style(zn)
        self.model_params = {
            'random_forest': {'n_estimators': 100, 'random_state': self.random_state},
            'lasso': {'alpha': 1.0, 'random_state': self.random_state},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': self.random_state},
            'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': self.random_state},
            'lightgbm': {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'random_state': self.random_state}
        }
        # 分类任务专用参数空间
        self.classification_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'objective': ['binary:logistic', 'multi:softmax'],
                'eval_metric': ['logloss']
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50],
                'objective': ['binary', 'multiclass']
            }
        }

        # 回归任务参数空间
        self.regression_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50]
            }
        }
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
        #mpl.rc("figure", figsize=(9, 6))
    def _determine_task_type(self, target_series):
        """自动判断任务类型"""
        unique_values = np.unique(target_series)
        # 检查是否为分类任务（类别数量少且为整数）
        if len(unique_values) <= 10 and np.all(np.round(unique_values) == unique_values):
            return 'classification'
        else:
            return 'regression'
    
    def _get_model_and_params(self, model_name, task_type):
        """根据模型名称和任务类型获取模型和参数网格"""
        # 选择模型
        if model_name == 'random_forest':
            model = RandomForestRegressor(**self.model_params['random_forest']) if task_type == 'regression' else RandomForestClassifier(**self.model_params['random_forest'])
            param_grid = self.regression_param_grids['random_forest'] if task_type == 'regression' else self.classification_param_grids['random_forest']
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.model_params['gradient_boosting']) if task_type == 'regression' else GradientBoostingClassifier(**self.model_params['gradient_boosting'])
            param_grid = self.regression_param_grids['gradient_boosting'] if task_type == 'regression' else self.classification_param_grids['gradient_boosting']
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(**self.model_params['xgboost']) if task_type == 'regression' else xgb.XGBClassifier(**self.model_params['xgboost'])
            param_grid = self.regression_param_grids['xgboost'] if task_type == 'regression' else self.classification_param_grids['xgboost']
        elif model_name == 'lightgbm':
            model = lgb.LGBMRegressor(**self.model_params['lightgbm']) if task_type == 'regression' else lgb.LGBMClassifier(**self.model_params['lightgbm'])
            param_grid = self.regression_param_grids['lightgbm'] if task_type == 'regression' else self.classification_param_grids['lightgbm']
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return model, param_grid
    
    def _get_optuna_params(self, trial, model_name, task_type):
        """为Optuna优化生成参数建议"""
        params = {}
        if model_name == 'random_forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
            params['max_depth'] = trial.suggest_categorical('max_depth', [None, 5, 10])
            if task_type == 'classification':
                params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        elif model_name == 'gradient_boosting':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 7)
        elif model_name == 'xgboost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 7)
            if task_type == 'classification':
                params['objective'] = trial.suggest_categorical('objective', ['binary:logistic', 'multi:softmax'])
        elif model_name == 'lightgbm':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200, step=50)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 50)
            if task_type == 'classification':
                params['objective'] = trial.suggest_categorical('objective', ['binary', 'multiclass'])
        return params
    
    def train_model(self, X_train, y_train, X_test,y_test, model_name, task_type, optimization_method='GridSearchCV'):
        """训练模型并返回最佳模型"""
        model, param_grid = self._get_model_and_params(model_name, task_type)
        
        if optimization_method == "GridSearchCV":
            scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring)
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            import optuna
            def objective(trial):
                params = self._get_optuna_params(trial, model_name, task_type)
                model.set_params(**params)
                model.fit(X_train, y_train)
                return model.score(X_test, y_test)
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            best_model = model.set_params(**study.best_params)
            best_model.fit(X_train, y_train)
            return best_model, study.best_params
    def plot_ale(self, model, X, feature_name,
                grid_num=50, figsize=(6, 4),
                bootstrap_uncertainty=False, bootstrap_reps=100,
                show_bootstrap_curves=True, show_mean_curve=True, show_ci_band=False,
                show_plot=True):
        """绘制 ALE 图并使用 Bootstrap 分析不确定性（符合 Nature 风格）"""

        # 初始化存储 ALE 曲线
        ale_curves = []

        if bootstrap_uncertainty:
            for _ in range(bootstrap_reps):
                # 有放回抽样
                indices = np.random.choice(X.index, size=int(len(X) * 0.8), replace=True)
                X_sampled = X.loc[indices]

                # 计算 ALE
                res_df_sampled = ale(
                    X_sampled,
                    model,
                    [feature_name],
                    grid_size=grid_num,
                    impute_empty_cells=False,
                    contour=True,
                    include_CI=False,
                    plot=False
                )

                # 外推插值到主网格
                x_sampled = res_df_sampled.index.values
                y_sampled = res_df_sampled['eff'].values

                if len(x_sampled) < 2:
                    continue  # 跳过无效样本

                f = interp1d(x_sampled, y_sampled, kind='linear', fill_value='extrapolate')
                filled_curve = f(X[feature_name].sort_values().unique())

                # 插值结果重新索引为统一网格
                ale_curve = pd.Series(filled_curve, index=X[feature_name].sort_values().unique())
                ale_curve = ale_curve.reindex(X[feature_name].sort_values().unique()).interpolate().values

                ale_curves.append(ale_curve)

        # 主网格用于绘图
        feature_grid = X[feature_name].sort_values().unique()

        # 对每个数据点添加一个小的随机偏移（“抖动”）
        sorted_values = X[feature_name].sort_values()
        values_diff = abs(sorted_values.shift() - sorted_values)
        rug = X.apply(
            lambda row: row[feature_name]
            + np.random.uniform(
                -values_diff[values_diff > 0].min() / 2,
                values_diff[values_diff > 0].min() / 2,
            ),
            axis=1,
        )

        fig, ax = plt.subplots(figsize=figsize)

        # # 绘制所有扰动曲线（浅绿色，半透明）
        # if show_bootstrap_curves and bootstrap_uncertainty:
        #     for curve in ale_curves:
        #         ax.plot(feature_grid, curve, color='lightgreen', alpha=0.2)

        # 绘制平均曲线（黑色实线）
        if show_mean_curve and bootstrap_uncertainty and len(ale_curves) > 0:
            mean_curve = np.mean(ale_curves, axis=0)
            ax.plot(feature_grid, mean_curve, color='#1f1f1f', lw=2, label='Mean ALE')

        # 绘制置信区间带（5%-95%）
        if show_ci_band and bootstrap_uncertainty and len(ale_curves) > 0:
            lower = np.percentile(ale_curves, 0, axis=0)
            upper = np.percentile(ale_curves, 100, axis=0)
            ax.fill_between(feature_grid, lower, upper, color='#8bb3d1', alpha=0.25, label='5–95% Interval')

        # 设置 Y 轴下限，避免地毯图被裁剪
        if show_mean_curve and bootstrap_uncertainty and len(ale_curves) > 0:
            mean_min = np.min(mean_curve)
            mean_max = np.max(mean_curve)
            ax.set_ylim(bottom=mean_min - 0.5 * abs(mean_min), top=mean_max + 0.5 * abs(mean_max))
        else:
            ax.set_ylim(bottom=np.min(X[feature_name]), top=np.max(X[feature_name]))

        # 绘制地毯图（贴着图像底部）
        rug_y_value = ax.get_ylim()[0] + 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        tr = mtrans.offset_copy(ax.transData, fig=fig, x=0.0, y=-5, units="points")
        ax.plot(
            rug,
            [rug_y_value] * len(rug),
            "|",
            color='black',
            alpha=0.6,
            transform=tr,
        )

        # 设置标签和标题
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Accumulated Local Effect")
        #ax.legend()
        fig.tight_layout()

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