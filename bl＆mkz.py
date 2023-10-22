# -- coding: utf-8 --
import io
import sys

import numpy as np
from scipy.optimize import minimize

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def portfolio_utility(weights, returns, cov_matrix, risk_aversion):
    '''
    Parameters
    ----------
    weights : numpy.ndarray
        一维数组，为投资组合中每只股票的权重
    returns : numpy.ndarray
        一维数组，为投资组合中每只股票的预期收益率
    cov_matrix : numpy.ndarray
        二维数组，为投资组合中每只股票之间的协方差矩阵。
    risk_aversion : float
        风险厌恶系数 (lambda)

    Returns
    -------
    utility : float
        投资组合效用函数的值
    '''
    portfolio_return = np.sum(weights * returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return - 0.5 * risk_aversion * portfolio_variance

def optimize_portfolio(returns, cov_matrix, risk_aversion):
    '''
    Parameters
    ----------
    returns : numpy.ndarray
        一维数组，为投资组合中每只股票的预期收益率
    cov_matrix : numpy.ndarray
        二维数组，为投资组合中每只股票之间的协方差矩阵
    risk_aversion : float
        风险厌恶系数 (lambda)

    Returns
    -------
    optimal_weights : numpy.ndarray
        一维数组，为最优投资组合中每只股票的权重
    '''
    n = len(returns)
    initial_weights = np.ones(n) / n  # 使用等权重进行初始化
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # 权重总和等于1的约束
    bounds = tuple((0, 1) for _ in range(n))  # 单个权重的上下界 (0 <= 权重 <= 1)

    # 效用函数最大化，因为一般使用minimize，故对效用函数取负值
    objective = lambda weights: -portfolio_utility(weights, returns, cov_matrix, risk_aversion)

    # 使用数值优化方法找到最优权重,methon参数选为SLSQP
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = result.x
    return optimal_weights

def black_litterman(expected_market_weights, market_covariance, risk_aversion, tau, P, Q, Omega):
    """
    Parameters
    ----------
    expected_market_weights : numpy.ndarray
        一维数组，市场均衡时市场组合中各支股票的权重(n×1向量)
    market_covariance : numpy.ndarray
        二维数组，市场均衡时股票的协方差矩阵(n×n矩阵)
    risk_aversion : float
        风险厌恶系数
    tau : float
        Black-Litterman 模型的调整参数,通常取非负值
    P : numpy.ndarray
        二维数组，主观观点矩阵(k×n矩阵,k为观点个数、n为股票数量)
    Q : numpy.ndarray
        一维数组，主观观点对应的预期收益率(k×1向量,k为观点个数)
    Omega : numpy.ndarray
        二维数组，主观观点的协方差矩阵(k×k维,反映主观观点的不确定性)

    Returns
    -------
    posterior_expected_returns : numpy.ndarray
        一维数组,基于Black-Litterman模型计算的后验收益率估计(n×1向量)
    posterior_covariance : numpy.ndarray
        二维数组,基于Black-Litterman模型计算的后验协方差矩阵估计(n×n矩阵)
    """
    # 计算市场组合的预期收益率
    market_portfolio_return = risk_aversion * np.dot(market_covariance, expected_market_weights)
    # 计算A矩阵
    A = np.linalg.inv(tau * market_covariance) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)
    # 计算B矩阵
    B =  np.dot(np.linalg.inv(tau * market_covariance),market_portfolio_return) + np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q)
    # 计算后验收益率估计
    posterior_expected_returns = np.dot(np.linalg.inv(A), B)
    # 计算后验协方差矩阵估计(收益率的协方差矩阵 = 收益率期望的协方差矩阵 + 市场组合协方差矩阵)
    posterior_covariance = np.linalg.inv(A) + market_covariance

    return posterior_expected_returns, posterior_covariance

if __name__ == "__main__":
    # 假设有5只股票
    n = 5
    # 市场均衡时的股票权重
    expected_market_weights = np.array([0.2, 0.3, 0.15, 0.2, 0.15])
    # 市场均衡时的股票协方差矩阵
    market_covariance = np.array([[0.04, 0.02, 0.015, 0.01, 0.02],
                                  [0.02, 0.03, 0.015, 0.015, 0.01],
                                  [0.015, 0.015, 0.04, 0.02, 0.015],
                                  [0.01, 0.015, 0.02, 0.05, 0.02],
                                  [0.02, 0.01, 0.015, 0.02, 0.04]])
    # 风险厌恶系数
    risk_aversion = 2.0
    # BL模型的参数
    tau = 0.025
    P = np.array([[1, 0, 0, 0, 0],  # 五个观点，这里只有一个观点，所以只有一行
                  [0, 1, 0, 0, 0], 
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    Q = np.array([0.03, 0.04, 0.02, 0.05, 0.03])
    Omega = np.eye(5) * 0.0001  # 假设观点之间相互独立，故为对角阵
    # 调用BL模型函数计算后验收益率和协方差矩阵
    posterior_expected_returns, posterior_covariance = black_litterman(expected_market_weights, market_covariance, risk_aversion, tau, P, Q, Omega)
    print("后验收益率：")
    print(posterior_expected_returns)
    print("后验协方差矩阵：")
    print(posterior_covariance)
    # 使用最优化函数计算最优股票比例
    optimal_weights = optimize_portfolio(posterior_expected_returns, posterior_covariance, risk_aversion)
    print("最优股票权重：")
    print(optimal_weights)