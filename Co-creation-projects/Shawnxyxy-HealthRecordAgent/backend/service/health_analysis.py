"""
健康分析工作流服务
负责串联多个 Agent，完成一次完整的健康报告分析
"""

import asyncio
from typing import Dict, Any
from uuid import uuid4

from agents.planner import PlannerAgent
from agents.health_indicator import HealthIndicatorAgent
from agents.risk_assess import RiskAssessmentAgent
from agents.advice import AdviceAgent
from agents.report import ReportAgent
from agents.base import create_task, update_agent_state, complete_task

class HealthAnalysisService:
    def __init__(self, task_id: str = None):
        self.task_id = task_id or str(uuid4())
        # 任务初始化
        create_task(self.task_id)

        self.planner = PlannerAgent(task_id=self.task_id)
        self.indicator_agent = HealthIndicatorAgent(task_id=self.task_id)
        self.risk_agent = RiskAssessmentAgent(task_id=self.task_id)
        self.advice_agent = AdviceAgent(task_id=self.task_id)
        self.report_agent = ReportAgent(task_id=self.task_id)

    async def run(self, report_text: str) -> Dict[str, Any]:
        """
        执行完整的健康分析流程
        """

        # 1.任务规划
        update_agent_state(self.task_id, "PlannerAgent", "running")
        plan_result = await self.planner.run({"goal": f"分析以下体检报告并制定执行计划：\n{report_text}"})
        update_agent_state(self.task_id, "PlannerAgent", "completed")

        # 2.健康指标分析
        update_agent_state(self.task_id, "HealthIndicatorAgent", "running")
        indicator_result = await self.indicator_agent.run({
            "report_text": report_text,
            "plan": plan_result
        })
        update_agent_state(self.task_id, "HealthIndicatorAgent", "completed", partial_report={"indicator_results": indicator_result})

        # 3. 风险评估
        update_agent_state(self.task_id, "RiskAssessmentAgent", "running")
        risk_result = await self.risk_agent.run({
            "indicator_results": indicator_result
        })
        update_agent_state(self.task_id, "RiskAssessmentAgent", "completed", partial_report={"risk_assessment": risk_result})

        # 4. 健康建议生成
        update_agent_state(self.task_id, "AdviceAgent", "running")
        advice_result = await self.advice_agent.run({
            "risk_assessment": risk_result
        })
        update_agent_state(self.task_id, "AdviceAgent", "completed", partial_report={"advice": advice_result})


        # 5. 报告汇总
        update_agent_state(self.task_id, "ReportAgent", "running")
        final_report = await self.report_agent.run({
            "indicators": indicator_result,
            "risk_assessment": risk_result,
            "advice": advice_result
        })
        update_agent_state(self.task_id, "ReportAgent", "completed")
        complete_task(self.task_id, final_report)

        return self.task_id

# ---------- 临时本地验证入口 ----------

async def _demo():
    demo_text = """
        男性，28岁，BMI 27.3，血压 145/95 mmHg，
        总胆固醇 6.2 mmol/L，空腹血糖 6.1 mmol/L。
        """

    workflow = HealthAnalysisService()
    result = await workflow.run(demo_text)
    print(result)

if __name__ == "__main__":
    asyncio.run(_demo())
