import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalysisTool:
    """
    Advanced statistical analysis tool for financial data including trend analysis,
    forecasting, correlation analysis, and budget variance calculations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Statistical Analysis Tool.
        
        Args:
            logger: Logger instance for debugging and monitoring
        """
        self.logger = logger or logging.getLogger(__name__)
        self.analysis_cache = {}
        
        # Statistical thresholds
        self.significance_level = 0.05
        self.confidence_level = 0.95
        
        self.logger.info("Statistical Analysis Tool initialized")
    
    async def perform_analysis(
        self, 
        analysis_type: str, 
        data_source: str, 
        data: Optional[Union[List[Dict], pd.DataFrame]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for statistical analysis.
        
        Args:
            analysis_type: Type of analysis to perform
            data_source: Source of the data (for context)
            data: Raw data to analyze
            **kwargs: Additional parameters specific to analysis type
            
        Returns:
            Analysis results with statistical metrics and insights
        """
        try:
            self.logger.info(f"Performing {analysis_type} analysis on {data_source}")
            
            # Convert data to DataFrame if needed
            if data is not None:
                df = self._prepare_data(data)
            else:
                df = pd.DataFrame()  # Empty DataFrame for demo/testing
            
            # Route to appropriate analysis method
            analysis_methods = {
                "trend_analysis": self._perform_trend_analysis,
                "correlation": self._perform_correlation_analysis,
                "forecast": self._perform_forecasting,
                "distribution": self._perform_distribution_analysis,
                "outlier_detection": self._perform_outlier_detection,
                "seasonality": self._perform_seasonality_analysis,
                "budget_variance": self._perform_budget_variance_analysis,
                "performance_metrics": self._calculate_performance_metrics
            }
            
            analysis_method = analysis_methods.get(analysis_type)
            if not analysis_method:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Perform the analysis
            results = await analysis_method(df, **kwargs)
            
            # Add metadata
            results.update({
                "analysis_type": analysis_type,
                "data_source": data_source,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_points": len(df) if not df.empty else 0,
                "parameters": kwargs
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type,
                "data_source": data_source
            }
    
    def _prepare_data(self, data: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """Convert input data to standardized DataFrame format."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a list of dictionaries or pandas DataFrame")
        
        # Standardize common column names
        column_mapping = {
            'transaction_date': 'date',
            'transaction_amount': 'amount',
            'spending_amount': 'amount'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Parse dates if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df
    
    async def _perform_trend_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis including linear/polynomial trends,
        trend strength, and change points.
        """
        try:
            metrics = kwargs.get('metrics', ['amount'])
            time_period = kwargs.get('time_period', 'daily')
            polynomial_degree = kwargs.get('polynomial_degree', 2)
            
            if df.empty or 'date' not in df.columns:
                return {"error": "No date data available for trend analysis"}
            
            results = {}
            
            for metric in metrics:
                if metric not in df.columns:
                    continue
                
                # Aggregate by time period
                aggregated_data = self._aggregate_by_period(df, metric, time_period)
                
                if len(aggregated_data) < 3:
                    results[metric] = {"error": "Insufficient data points"}
                    continue
                
                # Extract time series data
                dates = aggregated_data.index
                values = aggregated_data.values
                
                # Convert dates to numeric for regression
                x_numeric = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
                y = values
                
                # Linear trend analysis
                linear_model = LinearRegression()
                linear_model.fit(x_numeric, y)
                linear_trend = linear_model.predict(x_numeric)
                linear_slope = linear_model.coef_[0]
                linear_r2 = r2_score(y, linear_trend)
                
                # Polynomial trend analysis
                poly_features = PolynomialFeatures(degree=polynomial_degree)
                x_poly = poly_features.fit_transform(x_numeric)
                poly_model = Ridge(alpha=1.0)  # Ridge regression to prevent overfitting
                poly_model.fit(x_poly, y)
                poly_trend = poly_model.predict(x_poly)
                poly_r2 = r2_score(y, poly_trend)
                
                # Statistical significance of trend
                correlation, p_value = stats.pearsonr(x_numeric.flatten(), y)
                is_significant = p_value < self.significance_level
                
                # Trend strength classification
                trend_strength = self._classify_trend_strength(abs(correlation))
                trend_direction = "increasing" if linear_slope > 0 else "decreasing" if linear_slope < 0 else "stable"
                
                # Change point detection (simplified)
                change_points = self._detect_change_points(y, dates)
                
                # Volatility analysis
                volatility = np.std(y)
                coefficient_of_variation = volatility / np.mean(y) if np.mean(y) != 0 else 0
                
                # Confidence intervals for trend
                confidence_intervals = self._calculate_trend_confidence_intervals(
                    x_numeric, y, linear_model
                )
                
                results[metric] = {
                    "linear_trend": {
                        "slope": float(linear_slope),
                        "r_squared": float(linear_r2),
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "is_significant": is_significant,
                        "p_value": float(p_value),
                        "correlation": float(correlation)
                    },
                    "polynomial_trend": {
                        "degree": polynomial_degree,
                        "r_squared": float(poly_r2),
                        "better_than_linear": poly_r2 > linear_r2
                    },
                    "volatility": {
                        "standard_deviation": float(volatility),
                        "coefficient_of_variation": float(coefficient_of_variation),
                        "volatility_level": self._classify_volatility(coefficient_of_variation)
                    },
                    "change_points": change_points,
                    "confidence_intervals": confidence_intervals,
                    "summary": self._generate_trend_summary(
                        trend_direction, trend_strength, is_significant, linear_slope, time_period
                    )
                }
            
            return {
                "success": True,
                "results": results,
                "time_period": time_period,
                "analysis_period": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_correlation_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform correlation analysis between different spending categories or metrics.
        """
        try:
            categories = kwargs.get('categories', [])
            correlation_method = kwargs.get('method', 'pearson')  # pearson, spearman, kendall
            
            if df.empty:
                return {"error": "No data available for correlation analysis"}
            
            # If categories specified, filter data
            if categories and 'category' in df.columns:
                df = df[df['category'].isin(categories)]
            
            # Create correlation matrix based on available data
            if 'category' in df.columns and 'amount' in df.columns:
                # Pivot data to get categories as columns
                pivot_data = df.pivot_table(
                    index='date' if 'date' in df.columns else df.index,
                    columns='category',
                    values='amount',
                    aggfunc='sum',
                    fill_value=0
                )
            else:
                # Use numeric columns for correlation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                pivot_data = df[numeric_cols]
            
            if pivot_data.empty or len(pivot_data.columns) < 2:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Calculate correlation matrix
            if correlation_method == 'pearson':
                corr_matrix = pivot_data.corr(method='pearson')
            elif correlation_method == 'spearman':
                corr_matrix = pivot_data.corr(method='spearman')
            elif correlation_method == 'kendall':
                corr_matrix = pivot_data.corr(method='kendall')
            else:
                corr_matrix = pivot_data.corr(method='pearson')
            
            # Calculate p-values for significance testing
            p_values = self._calculate_correlation_p_values(pivot_data, correlation_method)
            
            # Find significant correlations
            significant_correlations = []
            strong_correlations = []
            
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr_value = corr_matrix.loc[col1, col2]
                        p_value = p_values.loc[col1, col2] if col1 in p_values.index and col2 in p_values.columns else 1.0
                        
                        if not np.isnan(corr_value):
                            correlation_info = {
                                "category_1": col1,
                                "category_2": col2,
                                "correlation": float(corr_value),
                                "p_value": float(p_value),
                                "is_significant": p_value < self.significance_level,
                                "strength": self._classify_correlation_strength(abs(corr_value)),
                                "direction": "positive" if corr_value > 0 else "negative"
                            }
                            
                            if p_value < self.significance_level:
                                significant_correlations.append(correlation_info)
                            
                            if abs(corr_value) > 0.5:
                                strong_correlations.append(correlation_info)
            
            # Sort by absolute correlation strength
            significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                "success": True,
                "correlation_matrix": corr_matrix.to_dict(),
                "p_values": p_values.to_dict(),
                "significant_correlations": significant_correlations,
                "strong_correlations": strong_correlations,
                "method": correlation_method,
                "significance_level": self.significance_level,
                "summary": self._generate_correlation_summary(significant_correlations, strong_correlations)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_forecasting(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Perform time series forecasting using multiple methods.
        """
        try:
            forecast_periods = kwargs.get('forecast_periods', 30)
            metric = kwargs.get('metric', 'amount')
            methods = kwargs.get('methods', ['linear', 'exponential_smoothing', 'arima'])
            
            if df.empty or 'date' not in df.columns or metric not in df.columns:
                return {"error": "Insufficient data for forecasting"}
            
            # Prepare time series data
            ts_data = self._prepare_time_series(df, metric)
            
            if len(ts_data) < 10:
                return {"error": "Insufficient historical data for reliable forecasting"}
            
            forecasts = {}
            
            # Linear trend forecasting
            if 'linear' in methods:
                forecasts['linear'] = self._linear_forecast(ts_data, forecast_periods)
            
            # Exponential smoothing
            if 'exponential_smoothing' in methods:
                forecasts['exponential_smoothing'] = self._exponential_smoothing_forecast(
                    ts_data, forecast_periods
                )
            
            # ARIMA forecasting
            if 'arima' in methods:
                forecasts['arima'] = self._arima_forecast(ts_data, forecast_periods)
            
            # Ensemble forecast (average of all methods)
            if len(forecasts) > 1:
                forecasts['ensemble'] = self._create_ensemble_forecast(forecasts, forecast_periods)
            
            # Calculate forecast accuracy metrics on historical data
            accuracy_metrics = {}
            for method, forecast_data in forecasts.items():
                if method != 'ensemble':
                    accuracy_metrics[method] = self._calculate_forecast_accuracy(
                        ts_data, forecast_data, forecast_periods
                    )
            
            # Generate forecast insights
            insights = self._generate_forecast_insights(forecasts, ts_data, forecast_periods)
            
            return {
                "success": True,
                "forecasts": forecasts,
                "accuracy_metrics": accuracy_metrics,
                "forecast_periods": forecast_periods,
                "historical_data_points": len(ts_data),
                "insights": insights,
                "recommended_method": self._recommend_forecast_method(accuracy_metrics)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_distribution_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze the statistical distribution of financial data.
        """
        try:
            metric = kwargs.get('metric', 'amount')
            
            if df.empty or metric not in df.columns:
                return {"error": f"No {metric} data available for distribution analysis"}
            
            data = df[metric].dropna()
            
            if len(data) < 5:
                return {"error": "Insufficient data for distribution analysis"}
            
            # Basic statistics
            basic_stats = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "mode": float(data.mode().iloc[0]) if not data.mode().empty else None,
                "std": float(data.std()),
                "variance": float(data.var()),
                "min": float(data.min()),
                "max": float(data.max()),
                "range": float(data.max() - data.min()),
                "q1": float(data.quantile(0.25)),
                "q3": float(data.quantile(0.75)),
                "iqr": float(data.quantile(0.75) - data.quantile(0.25))
            }
            
            # Shape statistics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            
            shape_stats = {
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "skewness_interpretation": self._interpret_skewness(skewness),
                "kurtosis_interpretation": self._interpret_kurtosis(kurtosis)
            }
            
            # Normality tests
            normality_tests = {}
            
            if len(data) >= 8:  # Minimum for Shapiro-Wilk
                shapiro_stat, shapiro_p = stats.shapiro(data)
                normality_tests['shapiro_wilk'] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > self.significance_level
                }
            
            if len(data) >= 20:  # Better for larger samples
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                normality_tests['kolmogorov_smirnov'] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_p > self.significance_level
                }
            
            # Distribution fitting
            distribution_fits = self._fit_distributions(data)
            
            # Percentiles
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                percentiles[f"p{p}"] = float(data.quantile(p/100))
            
            return {
                "success": True,
                "basic_statistics": basic_stats,
                "shape_statistics": shape_stats,
                "normality_tests": normality_tests,
                "distribution_fits": distribution_fits,
                "percentiles": percentiles,
                "summary": self._generate_distribution_summary(basic_stats, shape_stats, normality_tests)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_outlier_detection(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Detect outliers using multiple statistical methods.
        """
        try:
            metric = kwargs.get('metric', 'amount')
            methods = kwargs.get('methods', ['iqr', 'zscore', 'modified_zscore'])
            
            if df.empty or metric not in df.columns:
                return {"error": f"No {metric} data available for outlier detection"}
            
            data = df[metric].dropna()
            
            if len(data) < 5:
                return {"error": "Insufficient data for outlier detection"}
            
            outliers = {}
            
            # IQR method
            if 'iqr' in methods:
                outliers['iqr'] = self._detect_iqr_outliers(data)
            
            # Z-score method
            if 'zscore' in methods:
                outliers['zscore'] = self._detect_zscore_outliers(data)
            
            # Modified Z-score method (more robust)
            if 'modified_zscore' in methods:
                outliers['modified_zscore'] = self._detect_modified_zscore_outliers(data)
            
            # Consensus outliers (detected by multiple methods)
            consensus_outliers = self._find_consensus_outliers(outliers)
            
            # Outlier statistics
            outlier_stats = {
                "total_data_points": len(data),
                "outliers_by_method": {method: len(outlier_data['indices']) for method, outlier_data in outliers.items()},
                "consensus_outliers": len(consensus_outliers),
                "outlier_percentage": len(consensus_outliers) / len(data) * 100
            }
            
            return {
                "success": True,
                "outliers": outliers,
                "consensus_outliers": consensus_outliers,
                "statistics": outlier_stats,
                "summary": self._generate_outlier_summary(outlier_stats, consensus_outliers, data)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_seasonality_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in financial data.
        """
        try:
            metric = kwargs.get('metric', 'amount')
            period = kwargs.get('period', 30)  # Default to monthly seasonality
            
            if df.empty or 'date' not in df.columns or metric not in df.columns:
                return {"error": "Insufficient data for seasonality analysis"}
            
            # Prepare time series
            ts_data = self._prepare_time_series(df, metric, freq='D')  # Daily frequency
            
            if len(ts_data) < 2 * period:
                return {"error": f"Need at least {2 * period} data points for seasonality analysis"}
            
            # Seasonal decomposition
            try:
                decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                
                seasonal_component = decomposition.seasonal
                trend_component = decomposition.trend
                residual_component = decomposition.resid
                
                # Calculate seasonality strength
                seasonal_strength = 1 - (residual_component.var() / (seasonal_component + residual_component).var())
                
                # Detect seasonal patterns
                seasonal_patterns = self._detect_seasonal_patterns(seasonal_component, period)
                
                # Monthly/weekly patterns if applicable
                monthly_patterns = None
                weekly_patterns = None
                
                if 'date' in df.columns:
                    monthly_patterns = self._analyze_monthly_patterns(df, metric)
                    weekly_patterns = self._analyze_weekly_patterns(df, metric)
                
                return {
                    "success": True,
                    "seasonal_strength": float(seasonal_strength),
                    "has_seasonality": seasonal_strength > 0.1,
                    "seasonal_patterns": seasonal_patterns,
                    "monthly_patterns": monthly_patterns,
                    "weekly_patterns": weekly_patterns,
                    "decomposition": {
                        "trend_exists": not trend_component.dropna().empty,
                        "seasonal_variance": float(seasonal_component.var()),
                        "residual_variance": float(residual_component.var())
                    },
                    "summary": self._generate_seasonality_summary(seasonal_strength, seasonal_patterns, monthly_patterns, weekly_patterns)
                }
                
            except Exception as decomp_error:
                return {"success": False, "error": f"Decomposition failed: {str(decomp_error)}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_budget_variance_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze budget vs actual spending variance.
        """
        try:
            actual_data = kwargs.get('actual_data', df)
            budget_data = kwargs.get('budget_data', {})
            categories = kwargs.get('categories', [])
            
            if not budget_data:
                return {"error": "No budget data provided for variance analysis"}
            
            variance_analysis = {}
            
            # Analyze by category if available
            if 'category' in df.columns and categories:
                for category in categories:
                    category_data = df[df['category'] == category]
                    if category_data.empty:
                        continue
                    
                    actual_spending = category_data['amount'].sum() if 'amount' in category_data.columns else 0
                    budget_amount = budget_data.get(category, 0)
                    
                    if budget_amount > 0:
                        variance = actual_spending - budget_amount
                        variance_percentage = (variance / budget_amount) * 100
                        
                        variance_analysis[category] = {
                            "budget": float(budget_amount),
                            "actual": float(actual_spending),
                            "variance": float(variance),
                            "variance_percentage": float(variance_percentage),
                            "status": self._classify_budget_status(variance_percentage),
                            "utilization_rate": float((actual_spending / budget_amount) * 100)
                        }
            
            # Overall budget analysis
            total_budget = sum(budget_data.values())
            total_actual = df['amount'].sum() if 'amount' in df.columns else 0
            total_variance = total_actual - total_budget
            total_variance_percentage = (total_variance / total_budget * 100) if total_budget > 0 else 0
            
            overall_analysis = {
                "total_budget": float(total_budget),
                "total_actual": float(total_actual),
                "total_variance": float(total_variance),
                "total_variance_percentage": float(total_variance_percentage),
                "overall_status": self._classify_budget_status(total_variance_percentage)
            }
            
            # Performance metrics
            performance_metrics = self._calculate_budget_performance_metrics(variance_analysis)
            
            return {
                "success": True,
                "category_analysis": variance_analysis,
                "overall_analysis": overall_analysis,
                "performance_metrics": performance_metrics,
                "summary": self._generate_budget_variance_summary(variance_analysis, overall_analysis)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _calculate_performance_metrics(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate comprehensive financial performance metrics.
        """
        try:
            metric = kwargs.get('metric', 'amount')
            benchmark_data = kwargs.get('benchmark_data', {})
            
            if df.empty or metric not in df.columns:
                return {"error": "No data available for performance metrics"}
            
            data = df[metric].dropna()
            
            if len(data) < 2:
                return {"error": "Insufficient data for performance calculations"}
            
            # Basic performance metrics
            performance_metrics = {
                "total_value": float(data.sum()),
                "average_value": float(data.mean()),
                "median_value": float(data.median()),
                "volatility": float(data.std()),
                "coefficient_of_variation": float(data.std() / data.mean()) if data.mean() != 0 else 0,
                "min_value": float(data.min()),
                "max_value": float(data.max()),
                "range": float(data.max() - data.min())
            }
            
            # Growth metrics (if time series data)
            if 'date' in df.columns:
                growth_metrics = self._calculate_growth_metrics(df, metric)
                performance_metrics.update(growth_metrics)
            
            # Benchmark comparisons
            benchmark_comparisons = {}
            if benchmark_data:
                for benchmark_name, benchmark_value in benchmark_data.items():
                    if isinstance(benchmark_value, (int, float)):
                        comparison = {
                            "benchmark_value": float(benchmark_value),
                            "actual_value": performance_metrics["average_value"],
                            "difference": performance_metrics["average_value"] - benchmark_value,
                            "percentage_difference": ((performance_metrics["average_value"] - benchmark_value) / benchmark_value * 100) if benchmark_value != 0 else 0,
                            "performance": "above" if performance_metrics["average_value"] > benchmark_value else "below" if performance_metrics["average_value"] < benchmark_value else "equal"
                        }
                        benchmark_comparisons[benchmark_name] = comparison
            
            # Risk metrics
            risk_metrics = {
                "volatility_level": self._classify_volatility(performance_metrics["coefficient_of_variation"]),
                "stability_score": 1 / (1 + performance_metrics["coefficient_of_variation"]),  # Higher = more stable
                "consistency_score": self._calculate_consistency_score(data)
            }
            
            return {
                "success": True,
                "performance_metrics": performance_metrics,
                "benchmark_comparisons": benchmark_comparisons,
                "risk_metrics": risk_metrics,
                "summary": self._generate_performance_summary(performance_metrics, risk_metrics, benchmark_comparisons)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Helper methods for statistical calculations
    
    def _aggregate_by_period(self, df: pd.DataFrame, metric: str, period: str) -> pd.Series:
        """Aggregate data by time period."""
        if 'date' not in df.columns:
            return df[metric]
        
        df = df.set_index('date')
        
        if period == 'daily':
            return df[metric].resample('D').sum()
        elif period == 'weekly':
            return df[metric].resample('W').sum()
        elif period == 'monthly':
            return df[metric].resample('M').sum()
        elif period == 'quarterly':
            return df[metric].resample('Q').sum()
        elif period == 'yearly':
            return df[metric].resample('Y').sum()
        else:
            return df[metric].resample('D').sum()
    
    def _classify_trend_strength(self, correlation: float) -> str:
        """Classify trend strength based on correlation."""
        if correlation >= 0.7:
            return "very_strong"
        elif correlation >= 0.5:
            return "strong"
        elif correlation >= 0.3:
            return "moderate"
        elif correlation >= 0.1:
            return "weak"
        else:
            return "very_weak"
    
    def _classify_volatility(self, cv: float) -> str:
        """Classify volatility level based on coefficient of variation."""
        if cv <= 0.1:
            return "very_low"
        elif cv <= 0.25:
            return "low"
        elif cv <= 0.5:
            return "moderate"
        elif cv <= 1.0:
            return "high"
        else:
            return "very_high"
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength based on absolute value."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _classify_budget_status(self, variance_percentage: float) -> str:
        """Classify budget performance status."""
        if variance_percentage <= -20:
            return "significantly_under_budget"
        elif variance_percentage <= -5:
            return "under_budget"
        elif variance_percentage <= 5:
            return "on_track"
        elif variance_percentage <= 15:
            return "slightly_over_budget"
        elif variance_percentage <= 25:
            return "over_budget"
        else:
            return "significantly_over_budget"
    
    def _detect_change_points(self, values: np.ndarray, dates: pd.DatetimeIndex) -> List[Dict]:
        """Detect significant change points in time series data."""
        try:
            change_points = []
            
            if len(values) < 10:
                return change_points
            
            # Simple change point detection using moving averages
            window_size = max(3, len(values) // 10)
            
            # Calculate moving averages
            moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
            
            # Find points where the trend changes significantly
            for i in range(window_size, len(values) - window_size):
                before_avg = moving_avg.iloc[i-window_size:i].mean()
                after_avg = moving_avg.iloc[i:i+window_size].mean()
                
                if not (np.isnan(before_avg) or np.isnan(after_avg)):
                    change_magnitude = abs(after_avg - before_avg) / before_avg if before_avg != 0 else 0
                    
                    if change_magnitude > 0.2:  # 20% change threshold
                        change_points.append({
                            "date": dates[i].strftime('%Y-%m-%d'),
                            "index": int(i),
                            "change_magnitude": float(change_magnitude),
                            "direction": "increase" if after_avg > before_avg else "decrease",
                            "before_avg": float(before_avg),
                            "after_avg": float(after_avg)
                        })
            
            # Sort by change magnitude and return top 5
            change_points.sort(key=lambda x: x['change_magnitude'], reverse=True)
            return change_points[:5]
            
        except Exception as e:
            self.logger.error(f"Change point detection failed: {e}")
            return []
    
    def _calculate_trend_confidence_intervals(self, x: np.ndarray, y: np.ndarray, model) -> Dict:
        """Calculate confidence intervals for linear trend."""
        try:
            from scipy.stats import t
            
            # Predictions
            y_pred = model.predict(x)
            
            # Calculate residuals and standard error
            residuals = y - y_pred
            mse = np.mean(residuals ** 2)
            se = np.sqrt(mse)
            
            # Degrees of freedom
            df = len(y) - 2
            
            # T-statistic for 95% confidence
            t_val = t.ppf(0.975, df)
            
            # Confidence intervals
            margin_error = t_val * se
            lower_bound = y_pred - margin_error
            upper_bound = y_pred + margin_error
            
            return {
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist(),
                "confidence_level": 0.95,
                "standard_error": float(se)
            }
            
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            return {}
    
    def _calculate_correlation_p_values(self, data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate p-values for correlation matrix."""
        try:
            cols = data.columns
            p_values = pd.DataFrame(np.ones((len(cols), len(cols))), columns=cols, index=cols)
            
            for i, col1 in enumerate(cols):
                for j, col2 in enumerate(cols):
                    if i != j:
                        try:
                            if method == 'pearson':
                                _, p_val = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                            elif method == 'spearman':
                                _, p_val = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
                            else:
                                _, p_val = stats.kendalltau(data[col1].dropna(), data[col2].dropna())
                            
                            p_values.loc[col1, col2] = p_val
                        except:
                            p_values.loc[col1, col2] = 1.0
            
            return p_values
            
        except Exception as e:
            self.logger.error(f"P-value calculation failed: {e}")
            return pd.DataFrame()
    
    def _prepare_time_series(self, df: pd.DataFrame, metric: str, freq: str = 'D') -> pd.Series:
        """Prepare time series data for forecasting."""
        if 'date' not in df.columns:
            return pd.Series()
        
        # Group by date and sum the metric
        ts = df.groupby('date')[metric].sum().sort_index()
        
        # Resample to ensure consistent frequency
        if freq == 'D':
            ts = ts.resample('D').sum().fillna(0)
        elif freq == 'W':
            ts = ts.resample('W').sum().fillna(0)
        elif freq == 'M':
            ts = ts.resample('M').sum().fillna(0)
        
        return ts
    
    def _linear_forecast(self, ts_data: pd.Series, periods: int) -> Dict:
        """Perform linear trend forecasting."""
        try:
            # Prepare data
            x = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data.values
            
            # Fit linear model
            model = LinearRegression()
            model.fit(x, y)
            
            # Generate future dates
            future_x = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_x)
            
            # Generate forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            
            return {
                "method": "linear",
                "forecast_values": forecast_values.tolist(),
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "model_score": float(model.score(x, y)),
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_)
            }
            
        except Exception as e:
            self.logger.error(f"Linear forecasting failed: {e}")
            return {"method": "linear", "error": str(e)}
    
    def _exponential_smoothing_forecast(self, ts_data: pd.Series, periods: int) -> Dict:
        """Perform exponential smoothing forecasting."""
        try:
            # Remove any zero or negative values for exponential smoothing
            ts_clean = ts_data[ts_data > 0]
            
            if len(ts_clean) < 10:
                return {"method": "exponential_smoothing", "error": "Insufficient positive data"}
            
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                ts_clean,
                seasonal_periods=None,
                trend='add',
                seasonal=None
            ).fit(optimized=True)
            
            # Generate forecast
            forecast = model.forecast(periods)
            
            # Generate forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            
            return {
                "method": "exponential_smoothing",
                "forecast_values": forecast.tolist(),
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "aic": float(model.aic),
                "smoothing_parameters": {
                    "alpha": float(model.params['smoothing_level']),
                    "beta": float(model.params.get('smoothing_trend', 0))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Exponential smoothing failed: {e}")
            return {"method": "exponential_smoothing", "error": str(e)}
    
    def _arima_forecast(self, ts_data: pd.Series, periods: int) -> Dict:
        """Perform ARIMA forecasting."""
        try:
            # Remove any missing values
            ts_clean = ts_data.dropna()
            
            if len(ts_clean) < 20:
                return {"method": "arima", "error": "Insufficient data for ARIMA"}
            
            # Fit ARIMA model with automatic parameter selection
            model = ARIMA(ts_clean, order=(1, 1, 1))  # Simple ARIMA(1,1,1)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            
            # Generate forecast dates
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
            
            return {
                "method": "arima",
                "forecast_values": forecast.tolist(),
                "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "order": (1, 1, 1)
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA forecasting failed: {e}")
            return {"method": "arima", "error": str(e)}
    
    def _create_ensemble_forecast(self, forecasts: Dict, periods: int) -> Dict:
        """Create ensemble forecast by averaging multiple methods."""
        try:
            valid_forecasts = []
            methods_used = []
            
            for method, forecast_data in forecasts.items():
                if method != 'ensemble' and 'forecast_values' in forecast_data:
                    valid_forecasts.append(np.array(forecast_data['forecast_values']))
                    methods_used.append(method)
            
            if not valid_forecasts:
                return {"method": "ensemble", "error": "No valid forecasts to ensemble"}
            
            # Average the forecasts
            ensemble_forecast = np.mean(valid_forecasts, axis=0)
            
            # Use dates from the first valid forecast
            forecast_dates = None
            for forecast_data in forecasts.values():
                if 'forecast_dates' in forecast_data:
                    forecast_dates = forecast_data['forecast_dates']
                    break
            
            return {
                "method": "ensemble",
                "forecast_values": ensemble_forecast.tolist(),
                "forecast_dates": forecast_dates,
                "methods_used": methods_used,
                "method_count": len(methods_used)
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble forecasting failed: {e}")
            return {"method": "ensemble", "error": str(e)}
    
    def _calculate_forecast_accuracy(self, historical_data: pd.Series, forecast_data: Dict, periods: int) -> Dict:
        """Calculate forecast accuracy metrics using backtesting."""
        try:
            if len(historical_data) < periods + 10:
                return {"error": "Insufficient data for accuracy calculation"}
            
            # Use last 'periods' data points for testing
            train_data = historical_data[:-periods]
            test_data = historical_data[-periods:]
            
            # Create a simple forecast for the test period (using the same method logic)
            x_train = np.arange(len(train_data)).reshape(-1, 1)
            y_train = train_data.values
            
            model = LinearRegression()
            model.fit(x_train, y_train)
            
            x_test = np.arange(len(train_data), len(historical_data)).reshape(-1, 1)
            predictions = model.predict(x_test)
            
            # Calculate accuracy metrics
            mae = mean_absolute_error(test_data.values, predictions)
            mse = mean_squared_error(test_data.values, predictions)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
            
            return {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "mape": float(mape),
                "accuracy_score": float(max(0, 100 - mape))  # Simple accuracy score
            }
            
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return {"error": str(e)}
    
    def _fit_distributions(self, data: np.ndarray) -> Dict:
        """Fit common statistical distributions to the data."""
        try:
            distributions = {}
            
            # Normal distribution
            try:
                mu, sigma = stats.norm.fit(data)
                ks_stat, p_val = stats.kstest(data, lambda x: stats.norm.cdf(x, mu, sigma))
                distributions['normal'] = {
                    "parameters": {"mu": float(mu), "sigma": float(sigma)},
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "goodness_of_fit": "good" if p_val > 0.05 else "poor"
                }
            except:
                pass
            
            # Lognormal distribution (for positive data)
            if np.all(data > 0):
                try:
                    sigma, loc, scale = stats.lognorm.fit(data)
                    ks_stat, p_val = stats.kstest(data, lambda x: stats.lognorm.cdf(x, sigma, loc, scale))
                    distributions['lognormal'] = {
                        "parameters": {"sigma": float(sigma), "loc": float(loc), "scale": float(scale)},
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_val),
                        "goodness_of_fit": "good" if p_val > 0.05 else "poor"
                    }
                except:
                    pass
            
            # Gamma distribution (for positive data)
            if np.all(data > 0):
                try:
                    a, loc, scale = stats.gamma.fit(data)
                    ks_stat, p_val = stats.kstest(data, lambda x: stats.gamma.cdf(x, a, loc, scale))
                    distributions['gamma'] = {
                        "parameters": {"shape": float(a), "loc": float(loc), "scale": float(scale)},
                        "ks_statistic": float(ks_stat),
                        "p_value": float(p_val),
                        "goodness_of_fit": "good" if p_val > 0.05 else "poor"
                    }
                except:
                    pass
            
            return distributions
            
        except Exception as e:
            self.logger.error(f"Distribution fitting failed: {e}")
            return {}
    
    def _detect_iqr_outliers(self, data: pd.Series) -> Dict:
        """Detect outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            "method": "iqr",
            "outlier_values": outliers.tolist(),
            "outlier_indices": outliers.index.tolist(),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": len(outliers)
        }
    
    def _detect_zscore_outliers(self, data: pd.Series, threshold: float = 3.0) -> Dict:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold]
        
        return {
            "method": "zscore",
            "outlier_values": outliers.tolist(),
            "outlier_indices": outliers.index.tolist(),
            "threshold": threshold,
            "max_zscore": float(z_scores.max()),
            "outlier_count": len(outliers)
        }
    
    def _detect_modified_zscore_outliers(self, data: pd.Series, threshold: float = 3.5) -> Dict:
        """Detect outliers using Modified Z-score method (more robust)."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.std(data)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = data[np.abs(modified_z_scores) > threshold]
        
        return {
            "method": "modified_zscore",
            "outlier_values": outliers.tolist(),
            "outlier_indices": outliers.index.tolist(),
            "threshold": threshold,
            "median": float(median),
            "mad": float(mad),
            "outlier_count": len(outliers)
        }
    
    def _find_consensus_outliers(self, outliers_dict: Dict) -> List:
        """Find outliers detected by multiple methods."""
        if len(outliers_dict) < 2:
            return []
        
        # Get all outlier indices
        all_indices = []
        for method_data in outliers_dict.values():
            all_indices.extend(method_data.get('outlier_indices', []))
        
        # Count occurrences
        from collections import Counter
        index_counts = Counter(all_indices)
        
        # Return indices that appear in at least 2 methods
        consensus_outliers = [idx for idx, count in index_counts.items() if count >= 2]
        return consensus_outliers
    
    def _detect_seasonal_patterns(self, seasonal_component: pd.Series, period: int) -> Dict:
        """Analyze seasonal patterns from decomposition."""
        try:
            # Calculate seasonal indices
            seasonal_clean = seasonal_component.dropna()
            
            if len(seasonal_clean) == 0:
                return {"error": "No seasonal data available"}
            
            # Find peak and trough periods
            seasonal_mean = seasonal_clean.mean()
            peaks = seasonal_clean[seasonal_clean > seasonal_mean + seasonal_clean.std()]
            troughs = seasonal_clean[seasonal_clean < seasonal_mean - seasonal_clean.std()]
            
            return {
                "period": period,
                "seasonal_strength": float(seasonal_clean.std()),
                "peak_periods": peaks.index.tolist()[:5],  # Top 5 peaks
                "trough_periods": troughs.index.tolist()[:5],  # Top 5 troughs
                "seasonal_range": float(seasonal_clean.max() - seasonal_clean.min()),
                "seasonal_amplitude": float(seasonal_clean.std() * 2)  # Approximate amplitude
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame, metric: str) -> Dict:
        """Analyze monthly spending patterns."""
        try:
            df_copy = df.copy()
            df_copy['month'] = pd.to_datetime(df_copy['date']).dt.month
            df_copy['month_name'] = pd.to_datetime(df_copy['date']).dt.strftime('%B')
            
            monthly_avg = df_copy.groupby(['month', 'month_name'])[metric].mean().reset_index()
            monthly_total = df_copy.groupby(['month', 'month_name'])[metric].sum().reset_index()
            
            # Find highest and lowest spending months
            highest_month = monthly_avg.loc[monthly_avg[metric].idxmax()]
            lowest_month = monthly_avg.loc[monthly_avg[metric].idxmin()]
            
            return {
                "monthly_averages": {
                    row['month_name']: float(row[metric]) 
                    for _, row in monthly_avg.iterrows()
                },
                "highest_spending_month": {
                    "month": highest_month['month_name'],
                    "average": float(highest_month[metric])
                },
                "lowest_spending_month": {
                    "month": lowest_month['month_name'],
                    "average": float(lowest_month[metric])
                },
                "monthly_variation": float(monthly_avg[metric].std())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame, metric: str) -> Dict:
        """Analyze weekly spending patterns."""
        try:
            df_copy = df.copy()
            df_copy['day_of_week'] = pd.to_datetime(df_copy['date']).dt.dayofweek
            df_copy['day_name'] = pd.to_datetime(df_copy['date']).dt.strftime('%A')
            
            daily_avg = df_copy.groupby(['day_of_week', 'day_name'])[metric].mean().reset_index()
            
            # Find highest and lowest spending days
            highest_day = daily_avg.loc[daily_avg[metric].idxmax()]
            lowest_day = daily_avg.loc[daily_avg[metric].idxmin()]
            
            return {
                "daily_averages": {
                    row['day_name']: float(row[metric]) 
                    for _, row in daily_avg.iterrows()
                },
                "highest_spending_day": {
                    "day": highest_day['day_name'],
                    "average": float(highest_day[metric])
                },
                "lowest_spending_day": {
                    "day": lowest_day['day_name'],
                    "average": float(lowest_day[metric])
                },
                "weekend_vs_weekday": self._compare_weekend_weekday(df_copy, metric)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compare_weekend_weekday(self, df: pd.DataFrame, metric: str) -> Dict:
        """Compare weekend vs weekday spending."""
        try:
            df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
            
            weekend_avg = df[df['is_weekend']][metric].mean()
            weekday_avg = df[~df['is_weekend']][metric].mean()
            
            return {
                "weekend_average": float(weekend_avg),
                "weekday_average": float(weekday_avg),
                "weekend_higher": weekend_avg > weekday_avg,
                "difference_percentage": float((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg != 0 else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_budget_performance_metrics(self, variance_analysis: Dict) -> Dict:
        """Calculate overall budget performance metrics."""
        if not variance_analysis:
            return {}
        
        # Calculate aggregate metrics
        categories_over_budget = sum(1 for v in variance_analysis.values() if v['variance'] > 0)
        categories_under_budget = sum(1 for v in variance_analysis.values() if v['variance'] < 0)
        total_categories = len(variance_analysis)
        
        avg_variance_pct = np.mean([v['variance_percentage'] for v in variance_analysis.values()])
        avg_utilization = np.mean([v['utilization_rate'] for v in variance_analysis.values()])
        
        # Budget discipline score (0-100)
        discipline_score = max(0, 100 - abs(avg_variance_pct))
        
        return {
            "categories_over_budget": categories_over_budget,
            "categories_under_budget": categories_under_budget,
            "total_categories": total_categories,
            "over_budget_percentage": float(categories_over_budget / total_categories * 100) if total_categories > 0 else 0,
            "average_variance_percentage": float(avg_variance_pct),
            "average_utilization_rate": float(avg_utilization),
            "budget_discipline_score": float(discipline_score),
            "performance_rating": self._rate_budget_performance(discipline_score)
        }
    
    def _rate_budget_performance(self, discipline_score: float) -> str:
        """Rate budget performance based on discipline score."""
        if discipline_score >= 90:
            return "excellent"
        elif discipline_score >= 80:
            return "good"
        elif discipline_score >= 70:
            return "fair"
        elif discipline_score >= 60:
            return "needs_improvement"
        else:
            return "poor"
    
    def _calculate_growth_metrics(self, df: pd.DataFrame, metric: str) -> Dict:
        """Calculate growth metrics for time series data."""
        try:
            ts_data = self._prepare_time_series(df, metric)
            
            if len(ts_data) < 2:
                return {}
            
            # Overall growth
            first_value = ts_data.iloc[0]
            last_value = ts_data.iloc[-1]
            total_growth = (last_value - first_value) / first_value * 100 if first_value != 0 else 0
            
            # Period over period growth rates
            growth_rates = ts_data.pct_change().dropna()
            
            # Monthly growth (if enough data)
            monthly_growth = None
            if len(ts_data) >= 30:
                monthly_data = ts_data.resample('M').sum()
                monthly_growth_rates = monthly_data.pct_change().dropna()
                monthly_growth = float(monthly_growth_rates.mean() * 100) if len(monthly_growth_rates) > 0 else None
            
            return {
                "total_growth_percentage": float(total_growth),
                "average_daily_growth": float(growth_rates.mean() * 100),
                "growth_volatility": float(growth_rates.std() * 100),
                "monthly_growth_rate": monthly_growth,
                "periods_analyzed": len(ts_data),
                "growth_trend": "increasing" if total_growth > 0 else "decreasing" if total_growth < 0 else "stable"
            }
            
        except Exception as e:
            self.logger.error(f"Growth metrics calculation failed: {e}")
            return {}
    
    def _calculate_consistency_score(self, data: pd.Series) -> float:
        """Calculate consistency score based on data stability."""
        if len(data) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        cv = data.std() / data.mean() if data.mean() != 0 else float('inf')
        
        # Convert to consistency score (inverse relationship)
        # Lower CV = higher consistency
        consistency_score = 1 / (1 + cv) if cv != float('inf') else 0.0
        
        return float(consistency_score)
    
    def _recommend_forecast_method(self, accuracy_metrics: Dict) -> str:
        """Recommend the best forecasting method based on accuracy."""
        if not accuracy_metrics:
            return "linear"  # Default
        
        # Find method with lowest MAPE (better accuracy)
        best_method = min(
            accuracy_metrics.items(),
            key=lambda x: x[1].get('mape', float('inf'))
        )[0]
        
        return best_method
    
    # Summary generation methods
    def _generate_trend_summary(self, direction: str, strength: str, is_significant: bool, slope: float, period: str) -> str:
        """Generate human-readable trend summary."""
        significance = "statistically significant" if is_significant else "not statistically significant"
        
        return f"The data shows a {strength} {direction} trend over the {period} period. " \
               f"This trend is {significance}. " \
               f"The rate of change is approximately {abs(slope):.2f} units per period."
    
    def _generate_correlation_summary(self, significant_correlations: List, strong_correlations: List) -> str:
        """Generate correlation analysis summary."""
        if not significant_correlations and not strong_correlations:
            return "No significant correlations found between the analyzed categories."
        
        summary_parts = []
        
        if significant_correlations:
            top_corr = significant_correlations[0]
            summary_parts.append(
                f"Strongest correlation found between {top_corr['category_1']} and {top_corr['category_2']} "
                f"({top_corr['correlation']:.3f}, {top_corr['direction']} relationship)."
            )
        
        if len(significant_correlations) > 1:
            summary_parts.append(f"Found {len(significant_correlations)} significant correlations in total.")
        
        return " ".join(summary_parts)
    
    def _generate_forecast_insights(self, forecasts: Dict, historical_data: pd.Series, periods: int) -> List[str]:
        """Generate insights from forecast results."""
        insights = []

        # Historical trend insight
        recent_avg = historical_data.tail(7).mean()
        overall_avg = historical_data.mean()

        if recent_avg > overall_avg * 1.1:
            insights.append("Recent values are trending higher than the historical average.")
        elif recent_avg < overall_avg * 0.9:
            insights.append("Recent values are trending lower than the historical average.")
        else:
            insights.append("Recent values are consistent with historical trends.")

        # Forecast method comparison
        method_scores = {
            m: v.get("model_score", v.get("accuracy_score", None))
            for m, v in forecasts.items()
            if isinstance(v, dict) and ("model_score" in v or "accuracy_score" in v)
        }
        if method_scores:
            best_method = max(method_scores.items(), key=lambda x: x[1])[0]
            insights.append(f"The most reliable forecast method based on accuracy is '{best_method}'.")

        # Forecast range
        for method, result in forecasts.items():
            if "forecast_values" in result:
                max_val = max(result["forecast_values"])
                min_val = min(result["forecast_values"])
                insights.append(f"The '{method}' method predicts values ranging from {min_val:.2f} to {max_val:.2f}.")

        return insights

    def _generate_performance_summary(self, perf: Dict, risk: Dict, benchmarks: Dict) -> str:
        """Generate summary for performance metrics and risk."""
        summary = []

        total = perf.get("total_value", 0)
        avg = perf.get("average_value", 0)
        vol = perf.get("volatility", 0)
        rating = risk.get("volatility_level", "unknown")

        summary.append(f"The total value is {total:.2f} with an average of {avg:.2f}.")
        summary.append(f"The observed volatility is {vol:.2f}, which is considered '{rating}'.")

        consistency = risk.get("consistency_score")
        if consistency is not None:
            summary.append(f"Consistency score is {consistency:.2f}, indicating {'high' if consistency > 0.7 else 'low' if consistency < 0.4 else 'moderate'} stability.")

        if benchmarks:
            for name, comp in benchmarks.items():
                performance = comp["performance"]
                pct_diff = comp["percentage_difference"]
                summary.append(f"Performance is {performance} the benchmark '{name}' by {pct_diff:.2f}%.")

        return " ".join(summary)

    def _generate_budget_variance_summary(self, categories: Dict, overall: Dict) -> str:
        """Generate summary of budget vs actual spending variance."""
        summary = []

        total_budget = overall.get("total_budget", 0)
        total_actual = overall.get("total_actual", 0)
        variance_pct = overall.get("total_variance_percentage", 0)
        status = overall.get("overall_status", "unknown")

        summary.append(f"Total budget was {total_budget:.2f} and actual spending was {total_actual:.2f}.")
        summary.append(f"This resulted in a variance of {variance_pct:.2f}% categorized as '{status}'.")

        over = sum(1 for v in categories.values() if v.get("variance", 0) > 0)
        under = sum(1 for v in categories.values() if v.get("variance", 0) < 0)

        summary.append(f"{over} categories were over budget, while {under} were under budget.")

        return " ".join(summary)

    def _generate_outlier_summary(self, stats: Dict, consensus_outliers: List, data: pd.Series) -> str:
        """Generate summary for outlier detection."""
        total = stats.get("total_data_points", len(data))
        pct = stats.get("outlier_percentage", 0)
        method_counts = stats.get("outliers_by_method", {})

        summary = [f"A total of {len(consensus_outliers)} consensus outliers were found out of {total} data points ({pct:.2f}%)."]

        for method, count in method_counts.items():
            summary.append(f"{count} outliers detected using the {method.upper()} method.")

        if len(consensus_outliers) > 0:
            summary.append("Multiple methods agree on the presence of significant anomalies.")

        return " ".join(summary)

    def _generate_distribution_summary(self, basic: Dict, shape: Dict, normality: Dict) -> str:
        """Generate summary for distributional characteristics."""
        summary = []

        mean = basic.get("mean", 0)
        std = basic.get("std", 0)
        skew = shape.get("skewness_interpretation", "unknown")
        kurt = shape.get("kurtosis_interpretation", "unknown")

        summary.append(f"The data has a mean of {mean:.2f} and standard deviation of {std:.2f}.")
        summary.append(f"It is {skew} and {kurt}.")

        if "shapiro_wilk" in normality:
            p_val = normality["shapiro_wilk"]["p_value"]
            is_normal = normality["shapiro_wilk"]["is_normal"]
            summary.append(f"Shapiro-Wilk test indicates the distribution is {'normal' if is_normal else 'not normal'} (p = {p_val:.3f}).")

        return " ".join(summary)

    def _interpret_skewness(self, skew: float) -> str:
        """Interpret skewness values."""
        if skew < -1:
            return "highly negatively skewed"
        elif -1 <= skew < -0.5:
            return "moderately negatively skewed"
        elif -0.5 <= skew <= 0.5:
            return "approximately symmetric"
        elif 0.5 < skew <= 1:
            return "moderately positively skewed"
        else:
            return "highly positively skewed"

    def _interpret_kurtosis(self, kurt: float) -> str:
        """Interpret kurtosis values."""
        if kurt < -1:
            return "very platykurtic (flat distribution)"
        elif -1 <= kurt < -0.5:
            return "slightly platykurtic"
        elif -0.5 <= kurt <= 0.5:
            return "mesokurtic (normal-like)"
        elif 0.5 < kurt <= 1:
            return "slightly leptokurtic"
        else:
            return "very leptokurtic (peaked distribution)"