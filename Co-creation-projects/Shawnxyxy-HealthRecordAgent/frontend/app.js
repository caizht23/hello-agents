const API_BASE = "http://127.0.0.1:8000";
// 显示多Agent执行状态
function showAgentProgress(agentContainer, agents, statusFunc) {
    agentContainer.innerHTML = "";
    agents.forEach(agent => {
        const li = document.createElement("li");
        const status = typeof statusFunc === "function" ? statusFunc(agent.key) : statusFunc;
        li.textContent = `${agent.label}: ${status}`;
        agentContainer.appendChild(li);
    });
}

// 公共函数：提交任务并轮询状态
async function submitAndPollTask(url, body, agents, resultCard, reportDiv, analysisDiv, progressList, loadingText, doneText, errorText) {
    reportDiv.innerHTML = "";
    analysisDiv.innerText = loadingText;
    progressList.classList.remove("hidden");
    showAgentProgress(progressList, agents, () => "⏳ 执行中...");
    resultCard.classList.add("hidden");

    try {
        const response = await fetch(url, body);
        if (!response.ok) throw new Error(`服务器返回错误状态：${response.status}`);

        const data = await response.json();
        const taskId = data.task_id;

        let taskStatus = await fetch(`${API_BASE}/api/health/task_status/${taskId}`).then(r => r.json());
        while (taskStatus.state !== "completed") {
            showAgentProgress(progressList, agents, agentKey => taskStatus.agents?.[agentKey] ?? "⏳ 执行中...");
            await new Promise(res => setTimeout(res, 1000));
            taskStatus = await fetch(`${API_BASE}/api/health/task_status/${taskId}`).then(r => r.json());
        }
        // 任务完成后刷新一次 agent 状态，保证 ReportAgent 也显示 completed
        showAgentProgress(progressList, agents, agentKey => taskStatus.agents?.[agentKey] ?? "⏳ 执行中...");
        // 显示最终报告
        const summary = taskStatus.report?.report?.summary || "<p>❌ 未返回报告内容</p>";
        reportDiv.innerHTML = typeof summary === "string" ? summary : JSON.stringify(summary, null, 2);
        analysisDiv.innerText = doneText;
        resultCard.classList.remove("hidden");

    } catch (error) {
        const errorMessage = error?.message || JSON.stringify(error);
        console.error("任务提交或轮询出错:", errorMessage);
        reportDiv.innerHTML = `<p>❌ ${errorText}: ${errorMessage}</p>`;
        analysisDiv.innerText = `❌ ${errorText}`;
        progressList.innerHTML = "";
    }
}

// 文本报告分析
async function analyze() {
    const reportText = document.getElementById("reportText").value;
    if (!reportText) {
        alert("请输入体检报告内容");
        return;
    }

    const resultCard = document.getElementById("resultCard");
    const reportDiv = document.getElementById("report");
    const analysisDiv = document.getElementById("analysis");
    const progressList = document.getElementById("progressList");

    const agents = [
        { key: "PlannerAgent", label: "PlannerAgent 规划分析" },
        { key: "HealthIndicatorAgent", label: "HealthIndicatorAgent 指标分析" },
        { key: "RiskAssessmentAgent", label: "RiskAssessmentAgent 风险评估" },
        { key: "AdviceAgent", label: "AdviceAgent 建议生成" },
        { key: "ReportAgent", label: "ReportAgent 报告生成" }
    ];

    await submitAndPollTask(
        `${API_BASE}/api/health/analysis`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ report_text: reportText })
        },
        agents,
        resultCard,
        reportDiv,
        analysisDiv,
        progressList,
        "⏳ 正在分析文本报告，请稍候...",
        "✅ 文本分析完成",
        "报告生成失败"
    );
}

// PDF报告分析
async function uploadPDF() {
    const fileInput = document.getElementById("pdfFile");
    const file = fileInput.files[0];
    if (!file) {
        alert("请选择PDF文件");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const resultCard = document.getElementById("resultCard");
    const reportDiv = document.getElementById("report");
    const analysisDiv = document.getElementById("analysis");
    const progressList = document.getElementById("progressList");

    const agents = [
        { key: "PlannerAgent", label: "PlannerAgent 规划分析" },
        { key: "HealthIndicatorAgent", label: "HealthIndicatorAgent 指标分析" },
        { key: "RiskAssessmentAgent", label: "RiskAssessmentAgent 风险评估" },
        { key: "AdviceAgent", label: "AdviceAgent 建议生成" },
        { key: "ReportAgent", label: "ReportAgent 报告生成" }
    ];

    await submitAndPollTask(
        `${API_BASE}/api/health/analysis/pdf`,
        { method: "POST", body: formData },
        agents,
        resultCard,
        reportDiv,
        analysisDiv,
        progressList,
        "⏳ 正在分析 PDF 报告，请稍候...",
        "✅ PDF分析完成",
        "上传失败"
    );
}

// 绑定按钮事件
document.getElementById("analyzeBtn")?.addEventListener("click", analyze);
document.getElementById("uploadBtn")?.addEventListener("click", uploadPDF);