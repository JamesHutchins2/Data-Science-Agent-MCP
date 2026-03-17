

UI = """
<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>ADCES MVP</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; max-width: 920px; }
            h1 { margin-bottom: 8px; }
            p { color: #444; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
            textarea { width: 100%; min-height: 84px; }
            input[type=file] { margin-top: 8px; }
            button { padding: 10px 16px; border-radius: 6px; border: 1px solid #333; cursor: pointer; }
            ul { margin-top: 8px; }
            a { word-break: break-all; }
            #logs { background: #111; color: #e8e8e8; padding: 12px; border-radius: 8px; min-height: 140px; white-space: pre-wrap; }
            .muted { color: #666; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>ADCES MVP</h1>
        <p>Upload CSV files, run cleaning, and get EDA reports, cleaned CSVs, and notebook outputs.</p>

        <div class="card">
            <h3>1) Inputs</h3>
            <label>CSV files (one or more)</label><br/>
            <input id="files" type="file" accept=".csv" multiple />
            <br/><br/>
            <label>User request</label>
            <textarea id="prompt">Clean these files for downstream analytics while preserving useful business columns.</textarea>
            <br/>
            <label>System prompt</label>
            <textarea id="systemPrompt">You are an expert data cleaning agent.</textarea>
            <br/>
            <label>Optional ML objective</label>
            <textarea id="mlPrompt" placeholder="Example: Build and test classification models to predict Exited."></textarea>
            <br/>
            <button id="runBtn">Run Pipeline</button>
            <div class="muted" id="runMeta"></div>
        </div>

        <div class="card">
            <h3>2) Live Progress</h3>
            <div id="logs">No run started.</div>
        </div>

        <div class="card">
            <h3>3) EDA Reports (live)</h3>
            <ul id="reports"></ul>
        </div>

        <div class="card">
            <h3>4) Outputs</h3>
            <ul id="outputs"></ul>
        </div>

        <script>
            const logs = document.getElementById('logs');
            const reports = document.getElementById('reports');
            const outputs = document.getElementById('outputs');
            const runMeta = document.getElementById('runMeta');
            const runBtn = document.getElementById('runBtn');

            let currentRunId = null;
            let evtSource = null;

            function appendLog(line) {
                logs.textContent += (logs.textContent === 'No run started.' ? '' : '\\n') + line;
            }

            function addUniqueLink(listNode, href, label) {
                const key = href;
                if ([...listNode.querySelectorAll('a')].some(a => a.getAttribute('href') === key)) return;
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = href;
                a.textContent = label;
                a.target = '_blank';
                li.appendChild(a);
                listNode.appendChild(li);
            }

            async function pollOutputs(runId) {
                const statusResp = await fetch(`/pipeline/status/${runId}`);
                const status = await statusResp.json();
                const outResp = await fetch(`/pipeline/outputs/${runId}`);
                const outData = await outResp.json();
                const reportsResp = await fetch(`/pipeline/reports/${runId}`);
                const reportsData = await reportsResp.json();

                outputs.innerHTML = '';
                for (const name of outData.outputs || []) {
                    addUniqueLink(outputs, `/files/${runId}/output/${encodeURIComponent(name)}`, name);
                }
                for (const name of outData.plans || []) {
                    addUniqueLink(outputs, `/files/${runId}/output/${encodeURIComponent(name)}`, `plan: ${name}`);
                }
                for (const name of reportsData.reports || []) {
                    addUniqueLink(reports, `/reports/${runId}/${encodeURIComponent(name)}`, name);
                }

                if (status.status === 'completed' || status.status === 'completed_with_validation_errors' || status.status === 'failed') {
                    appendLog(`Run ended with status: ${status.status}`);
                    if (evtSource) evtSource.close();
                    return;
                }
                setTimeout(() => pollOutputs(runId), 3000);
            }

            runBtn.onclick = async () => {
                const filesInput = document.getElementById('files');
                const prompt = document.getElementById('prompt').value.trim();
                const mlPrompt = document.getElementById('mlPrompt').value.trim();
                const systemPrompt = document.getElementById('systemPrompt').value.trim();

                if (!filesInput.files.length) {
                    alert('Please upload at least one CSV file.');
                    return;
                }

                logs.textContent = '';
                reports.innerHTML = '';
                outputs.innerHTML = '';

                const form = new FormData();
                form.append('prompt', prompt);
                form.append('ml_prompt', mlPrompt);
                form.append('system_prompt', systemPrompt);
                for (const f of filesInput.files) form.append('files', f);

                runBtn.disabled = true;
                runBtn.textContent = 'Starting...';

                try {
                    const res = await fetch('/pipeline/start', { method: 'POST', body: form });
                    if (!res.ok) {
                        const text = await res.text();
                        throw new Error(text || 'Failed to start pipeline');
                    }

                    const data = await res.json();
                    currentRunId = data.run_id;
                    runMeta.textContent = `Run ID: ${currentRunId}`;
                    appendLog(`Run started: ${currentRunId}`);

                    evtSource = new EventSource(`/pipeline/events/${currentRunId}`);
                    evtSource.onmessage = (event) => {
                        try {
                            const payload = JSON.parse(event.data);
                            appendLog(`[${payload.stage}] ${payload.message}`);
                            if (payload.payload && payload.payload.report_url) {
                                const url = payload.payload.report_url;
                                addUniqueLink(reports, url, url.split('/').pop());
                            }
                        } catch {
                            appendLog(event.data);
                        }
                    };
                    evtSource.onerror = () => {
                        appendLog('Event stream closed.');
                    };

                    pollOutputs(currentRunId);
                } catch (err) {
                    appendLog(`Error: ${err.message}`);
                } finally {
                    runBtn.disabled = false;
                    runBtn.textContent = 'Run Pipeline';
                }
            };
        </script>
    </body>
</html>
                """