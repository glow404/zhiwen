"""
鉴权分类模块
弱分类器级联，召回分类器，阈值调整
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AuthenticationClassifier:
    """
    鉴权分类类
    通过弱分类器级联得到强分类器
    """
    
    def __init__(
        self,
        normal_frr_threshold: float = 0.03,
        normal_far_threshold: float = 0.00002,  # 1/5W
        special_frr_threshold: float = 0.05,
        cascade_stages: int = 10,
        recall_classifier_enabled: bool = True
    ):
        """
        初始化鉴权分类器
        
        Args:
            normal_frr_threshold: normal场景FRR阈值
            normal_far_threshold: normal场景FAR阈值
            special_frr_threshold: 特殊场景FRR阈值
            cascade_stages: 级联阶段数
            recall_classifier_enabled: 是否启用召回分类器
        """
        self.normal_frr_threshold = normal_frr_threshold
        self.normal_far_threshold = normal_far_threshold
        self.special_frr_threshold = special_frr_threshold
        self.cascade_stages = cascade_stages
        self.recall_classifier_enabled = recall_classifier_enabled
        
        # 弱分类器列表
        self.weak_classifiers = []
        # 强分类器（AdaBoost）
        self.strong_classifier = None
        # 召回分类器
        self.recall_classifier = None
        
        # 阈值
        self.threshold = 0.5
        self.normal_threshold = 0.5
        self.special_threshold = 0.5
    
    def create_weak_classifier(self) -> DecisionTreeClassifier:
        """
        创建弱分类器
        
        Returns:
            DecisionTreeClassifier: 弱分类器
        """
        return DecisionTreeClassifier(max_depth=1)
    
    def train_cascade(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        """
        训练级联分类器
        
        Args:
            features: 特征矩阵 (N, M)
            labels: 标签 (N,) 0或1
        """
        # 使用AdaBoost训练强分类器
        self.strong_classifier = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=self.cascade_stages,
            learning_rate=1.0,
            algorithm='SAMME'
        )
        
        self.strong_classifier.fit(features, labels)
        
        # 提取弱分类器
        self.weak_classifiers = self.strong_classifier.estimators_
    
    def train_recall_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        """
        训练召回分类器
        
        Args:
            features: 特征矩阵
            labels: 标签
        """
        if not self.recall_classifier_enabled:
            return
        
        # 召回分类器使用更宽松的阈值，提高召回率
        self.recall_classifier = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=5,
            learning_rate=0.5,
            algorithm='SAMME'
        )
        
        self.recall_classifier.fit(features, labels)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            features: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率 (N, 2)
        """
        if self.strong_classifier is None:
            # 如果没有训练，返回默认值
            return np.array([[0.5, 0.5]] * len(features))
        
        # 使用强分类器预测
        proba = self.strong_classifier.predict_proba(features)
        
        # 如果有召回分类器，结合预测
        if self.recall_classifier is not None:
            recall_proba = self.recall_classifier.predict_proba(features)
            # 取较大值（提高召回率）
            proba = np.maximum(proba, recall_proba)
        
        return proba
    
    def predict(self, features: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        预测类别
        
        Args:
            features: 特征矩阵
            threshold: 阈值，如果None则使用self.threshold
            
        Returns:
            np.ndarray: 预测类别 (0或1)
        """
        if threshold is None:
            threshold = self.threshold
        
        proba = self.predict_proba(features)
        predictions = (proba[:, 1] >= threshold).astype(int)
        
        return predictions
    
    def set_threshold_for_security_level(
        self,
        security_level: str = "normal",
        frr_target: Optional[float] = None,
        far_target: Optional[float] = None
    ):
        """
        根据安全等级设置阈值
        
        Args:
            security_level: 安全等级（"normal", "high", "low"）
            frr_target: 目标FRR
            far_target: 目标FAR
        """
        if security_level == "normal":
            if frr_target is None:
                frr_target = self.normal_frr_threshold
            if far_target is None:
                far_target = self.normal_far_threshold
            self.threshold = self.normal_threshold
        elif security_level == "special":
            # 钢化膜/高低温/湿手场景
            if frr_target is None:
                frr_target = self.special_frr_threshold
            if far_target is None:
                far_target = self.normal_far_threshold * 2  # 放宽FAR
            self.threshold = self.special_threshold
        else:
            self.threshold = 0.5
    
    def authenticate(
        self,
        similarity_score: float,
        condition: str = "normal"
    ) -> Tuple[bool, float]:
        """
        鉴权判断
        
        Args:
            similarity_score: 相似度分数（0-1）
            condition: 采集条件（"normal", "special"）
            
        Returns:
            Tuple[bool, float]: (是否通过, 置信度)
        """
        # 根据条件选择阈值
        if condition == "special":
            threshold = self.special_threshold
        else:
            threshold = self.normal_threshold
        
        # 判断是否通过
        passed = similarity_score >= threshold
        
        # 计算置信度（距离阈值的距离）
        confidence = abs(similarity_score - threshold)
        
        return passed, confidence
    
    def adjust_threshold(
        self,
        similarity_scores: List[float],
        labels: List[bool],
        target_frr: Optional[float] = None,
        target_far: Optional[float] = None
    ) -> float:
        """
        调整阈值以达到目标FRR/FAR
        
        Args:
            similarity_scores: 相似度分数列表
            labels: 标签列表（True=同一人，False=不同人）
            target_frr: 目标FRR
            target_far: 目标FAR
            
        Returns:
            float: 调整后的阈值
        """
        scores = np.array(similarity_scores)
        labels = np.array(labels)
        
        # 同一人的分数
        positive_scores = scores[labels]
        # 不同人的分数
        negative_scores = scores[~labels]
        
        if len(positive_scores) == 0 or len(negative_scores) == 0:
            return self.threshold
        
        # 二分搜索找到合适的阈值
        min_score = min(scores.min(), 0.0)
        max_score = max(scores.max(), 1.0)
        
        best_threshold = self.threshold
        best_error = float('inf')
        
        # 尝试不同的阈值
        for threshold in np.linspace(min_score, max_score, 100):
            # 计算FRR和FAR
            frr = np.sum(positive_scores < threshold) / len(positive_scores) if len(positive_scores) > 0 else 0.0
            far = np.sum(negative_scores >= threshold) / len(negative_scores) if len(negative_scores) > 0 else 0.0
            
            # 计算误差
            error = 0.0
            if target_frr is not None:
                error += abs(frr - target_frr)
            if target_far is not None:
                error += abs(far - target_far)
            
            if error < best_error:
                best_error = error
                best_threshold = threshold
        
        self.threshold = best_threshold
        return best_threshold

