/**
 * Main Application Class with On-Screen Debugging
 */
class App {
    constructor() {
        this.currentSection = 'dashboard';
        this.ws = null;
        this.cameras = [];
        this.models = [];
        this.pipelineStatusInterval = null;
        document.addEventListener('DOMContentLoaded', () => this.init());
    }

    // --- ON-SCREEN DEBUG LOGGER ---
    log(message, type = 'info') {
        const content = document.getElementById('debug-log-content');
        if (!content) return;
        const entry = document.createElement('div');
        entry.className = `log-${type}`;
        entry.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong>: ${message}`;
        content.prepend(entry);
    }

    async init() {
        this.log("App initializing in DIAGNOSTIC MODE...");
        this.setupNavigation();
        this.setupModals();
        
        const toggleBtn = document.getElementById('sidebar-toggle');
        const sidebar = document.getElementById('main-sidebar');
        if (toggleBtn && sidebar) {
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
            });
        }
        
        await Promise.all([this.loadCameras(), this.loadModels()]);
        const hash = window.location.hash.replace('#', '') || 'dashboard';
        this.navigate(hash);
        this.log("App initialization complete.");
    }

    // --- Core UI and API ---
    setupNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.navigate(item.dataset.target);
            });
        });
    }

    navigate(section) {
        this.log(`Navigating to section: <strong>${section}</strong>`);
        // Cleanup previous section
        if (this.pipelineStatusInterval) {
            clearInterval(this.pipelineStatusInterval);
            this.pipelineStatusInterval = null;
        }
        if (this.currentSection === 'live' && this.ws) {
            this.ws.close();
        }

        // Activate the nav item
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelector(`.nav-item[data-target="${section}"]`)?.classList.add('active');

        // Show the section
        document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
        const targetEl = document.getElementById(section);
        if (targetEl) {
            targetEl.classList.add('active');
        } else {
            this.log(`Section element #${section} not found in DOM!`, 'error');
        }

        this.currentSection = section;
        window.location.hash = section;

        const initFn = `init${section.charAt(0).toUpperCase() + section.slice(1)}`;
        if (typeof this[initFn] === 'function') {
            this.log(`Running initializer: ${initFn}`);
            this[initFn]();
        } else {
            this.log(`No initializer found for section: ${section}`, 'warn');
        }
    }

    async api(method, endpoint, data = null) {
        try {
            const options = { method, headers: {} };
            if (data instanceof FormData) {
                options.body = data;
            } else if (data) {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(data);
            }
            const res = await fetch(`/api/${endpoint}`, options);
            if (!res.ok) {
                const err = await res.json().catch(() => ({ detail: `HTTP Error ${res.status}` }));
                throw new Error(err.detail);
            }
            return res.status !== 204 ? await res.json() : null;
        } catch (error) {
            this.showToast(error.message, 'error');
            throw error;
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    setupModals() {
        const overlay = document.getElementById('modal-overlay');
        const closeBtn = document.getElementById('modal-close-btn');
        if (closeBtn) closeBtn.addEventListener('click', () => this.closeModal());
        if (overlay) overlay.addEventListener('click', (e) => { if (e.target === overlay) this.closeModal(); });
    }

    showModal(htmlContent) {
        document.getElementById('modal-body').innerHTML = htmlContent;
        document.getElementById('modal-overlay').classList.remove('hidden');
    }

    closeModal() {
        document.getElementById('modal-overlay').classList.add('hidden');
    }

    async loadCameras() {
        try {
            this.cameras = await this.api('GET', 'cameras') || [];
            this.log(`Loaded ${this.cameras.length} cameras.`);
        } catch (e) {
            this.cameras = [];
            this.log(`Failed to load cameras: ${e.message}`, 'error');
        }
    }

    async loadModels() {
        try {
            this.models = await this.api('GET', 'models') || [];
            this.log(`Loaded ${this.models.length} models.`);
        } catch (e) {
            this.models = [];
            this.log(`Failed to load models: ${e.message}`, 'error');
        }
    }

    // =========================================================
    // Section Initializers
    // =========================================================
    initDashboard() {
        const statsContainer = document.getElementById('dashboard-stats');
        if (!statsContainer) return this.log('Dashboard stats container #dashboard-stats missing.', 'error');
        statsContainer.innerHTML = '<p>Loading stats...</p>';
        this.api('GET', 'inference/stats')
            .then(stats => {
                statsContainer.innerHTML = `
                    <div class="stats-grid">
                        <div class="stat-card"><span class="stat-label">Status</span><span class="stat-value">${stats.is_running ? '🟢 Running' : '🔴 Stopped'}</span></div>
                        <div class="stat-card"><span class="stat-label">Camera</span><span class="stat-value">${stats.camera_id || 'N/A'}</span></div>
                        <div class="stat-card"><span class="stat-label">Total Inspected</span><span class="stat-value">${stats.total_inspected || 0}</span></div>
                        <div class="stat-card"><span class="stat-label">OK</span><span class="stat-value" style="color:#4caf50">${stats.ok_count || 0}</span></div>
                        <div class="stat-card"><span class="stat-label">NG</span><span class="stat-value" style="color:#f44336">${stats.ng_count || 0}</span></div>
                        <div class="stat-card"><span class="stat-label">FPS</span><span class="stat-value">${stats.fps || 0}</span></div>
                    </div>
                `;
                
                // Draw Chart
                this.api('GET', 'events/stats').then(eventStats => {
                    const ctx = document.getElementById('statsChart');
                    if (!ctx) return;
                    if (this.statsChartInstance) {
                        this.statsChartInstance.destroy();
                    }
                    this.statsChartInstance = new Chart(ctx, {
                        type: 'pie',
                        data: {
                            labels: ['OK', 'NG'],
                            datasets: [{
                                data: [eventStats.ok_count || 0, eventStats.ng_count || 0],
                                backgroundColor: ['#4caf50', '#f44336'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: { position: 'bottom', labels: { color: '#ffffff' } },
                                title: { display: true, text: 'Overall Inspection Results', color: '#ffffff' }
                            }
                        }
                    });
                }).catch(err => this.log('Chart load error: ' + err.message, 'error'));
            })
            .catch(err => {
                statsContainer.innerHTML = `<p style="color:orange">Could not load stats: ${err.message}</p>`;
            });
    }

    initCameras() {
        this.loadCamerasTable();
        const addBtn = document.getElementById('btn-add-camera');
        if (addBtn) addBtn.onclick = () => this.showAddCameraModal();
    }

    initLive() {
        // Populate camera/model selects
        const camSelect = document.getElementById('live-camera-select');
        const modelSelect = document.getElementById('live-model-select');
        if (camSelect) {
            camSelect.innerHTML = '<option value="">-- Select Camera --</option>' +
                this.cameras.map(c => `<option value="${c.id}">${c.name} (${c.rtsp_url})</option>`).join('');
        }
        if (modelSelect) {
            modelSelect.innerHTML = '<option value="">-- Select Model --</option>' +
                this.models.filter(m => m.type === 'embedder').map(m => `<option value="${m.id}">${m.name}</option>`).join('');
        }
        // Bind start/stop buttons (matching HTML IDs)
        const startBtn = document.getElementById('btn-start-pipeline');
        const stopBtn = document.getElementById('btn-stop-pipeline');
        if (startBtn) startBtn.onclick = () => this.startPipeline();
        if (stopBtn) stopBtn.onclick = () => this.stopPipeline();

        // Threshold sliders
        const confSlider = document.getElementById('live-conf-thresh');
        const confVal = document.getElementById('live-conf-val');
        if (confSlider && confVal) {
            confSlider.oninput = () => { confVal.textContent = confSlider.value; };
        }
        const simSlider = document.getElementById('live-sim-thresh');
        const simVal = document.getElementById('live-sim-val');
        if (simSlider && simVal) {
            simSlider.oninput = () => { simVal.textContent = simSlider.value; };
        }

        this.updatePipelineStatus();
    }

    initFileinfer() {
        const modelSelect = document.getElementById('file-model-select');
        if (modelSelect) {
            modelSelect.innerHTML = '<option value="">-- Select Model --</option>' +
                this.models.filter(m => m.type === 'embedder').map(m => `<option value="${m.id}">${m.name}</option>`).join('');
        }

        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('btn-upload');
        if (uploadBtn) {
            uploadBtn.onclick = async () => {
                if (!fileInput || !fileInput.files.length) return this.showToast('No files selected.', 'warn');
                const modelId = modelSelect?.value;
                if (!modelId) return this.showToast('Please select a model.', 'warn');
                
                const form = new FormData();
                form.append('model_id', modelId);
                for (const f of fileInput.files) form.append('files', f);
                
                try {
                    const firstFile = fileInput.files[0];
                    const isVideo = firstFile.type.startsWith('video/');
                    const endpoint = isVideo ? 'inference/video' : 'inference/image';
                    this.showToast(`Processing ${isVideo ? 'video' : 'image'}...`, 'info');
                    
                    const results = await this.api('POST', endpoint, form);
                    if (isVideo) {
                        this.showToast('Video processed successfully.', 'success');
                        const container = document.getElementById('file-results');
                        if (container) {
                            container.innerHTML = `
                                <h3>Video Results</h3>
                                <p>OK: ${results.summary.ok_count}, NG: ${results.summary.ng_count}</p>
                                <a href="${results.export_url}" target="_blank" class="btn btn-primary mt-4">Download Result Video</a>
                            `;
                        }
                    } else {
                        this.displayFileResults(results.results);
                    }
                } catch (e) {
                    this.log(`File inference failed: ${e.message}`, 'error');
                }
            };
        }
    }

    initTraining() {
        const trainBtn = document.getElementById('btn-train');
        if (trainBtn) {
            trainBtn.onclick = async () => {
                const fileInput = document.getElementById('train-file-input');
                const nameInput = document.getElementById('train-model-name');
                if (!fileInput || !fileInput.files.length) return this.showToast('Please select OK reference images.', 'warn');
                
                const form = new FormData();
                if (nameInput?.value) form.append('model_name', nameInput.value.trim());
                for (const f of fileInput.files) form.append('files', f);
                
                try {
                    this.showToast('Uploading images...', 'info');
                    const res = await this.api('POST', 'training/upload', form);
                    this.showToast(res.message, 'success');
                    
                    // Auto start training
                    await this.api('POST', `training/start?session_id=${res.session_id}`);
                    this.showToast('Training started in background!', 'success');
                    
                    // Refresh models after a short delay
                    setTimeout(async () => {
                        await this.loadModels();
                        this.showToast('Models refreshed.', 'info');
                    }, 5000);
                } catch (e) {
                    this.log(`Training failed: ${e.message}`, 'error');
                }
            };
        }
    }

    initModels() {
        const tableBody = document.getElementById('models-body');
        if (!tableBody) return this.log('Models table body #models-body not found.', 'error');
        tableBody.innerHTML = '';
        if (this.models.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="5" style="text-align:center; opacity:0.6;">No models uploaded yet.</td></tr>';
            return;
        }
        this.models.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${m.id}</td>
                <td>${m.name}</td>
                <td>${m.type}</td>
                <td>${m.is_active ? '✅ Active' : '—'}</td>
                <td><button class="btn btn-sm btn-primary" onclick="app.activateModel(${m.id})">Activate</button></td>
            `;
            tableBody.appendChild(tr);
        });
    }

    initEvents() {
        const container = document.querySelector('#events .glass-panel');
        if (!container) return;
        container.innerHTML = '<p>Loading events...</p>';
        this.api('GET', 'events')
            .then(events => {
                if (!events || events.length === 0) {
                    container.innerHTML = '<p style="opacity:0.6">No events recorded yet.</p>';
                    return;
                }
                let html = '<div class="table-responsive"><table class="data-table"><thead><tr><th><input type="checkbox" id="selectAllEvents"></th><th>ID</th><th>Camera</th><th>Result</th><th>Similarity</th><th>Time</th><th>Image</th><th>Action</th></tr></thead><tbody>';
                events.forEach(ev => {
                    const color = ev.result === 'OK' ? '#4caf50' : '#f44336';
                    const imgHtml = ev.image_path ? `<img src="${ev.image_path}" style="height:40px; border-radius:4px;">` : 'No Image';
                    html += `<tr>
                        <td><input type="checkbox" class="event-cb" value="${ev.id}"></td>
                        <td>${ev.id}</td>
                        <td>Cam ${ev.camera_id}</td>
                        <td style="color:${color};font-weight:bold">${ev.result}</td>
                        <td>${ev.similarity_score ? ev.similarity_score.toFixed(2) : '-'}</td>
                        <td>${new Date(ev.occurred_at).toLocaleString()}</td>
                        <td>${imgHtml}</td>
                        <td>
                            <button class="btn btn-sm btn-primary" onclick="app.editEvent(${ev.id}, '${ev.result}')">Edit</button>
                            <button class="btn btn-sm btn-danger" onclick="app.deleteEvent(${ev.id})">Delete</button>
                        </td>
                    </tr>`;
                });
                html += '</tbody></table></div>';
                container.innerHTML = html;
                
                const selectAll = document.getElementById('selectAllEvents');
                if (selectAll) {
                    selectAll.onchange = (e) => {
                        document.querySelectorAll('.event-cb').forEach(cb => cb.checked = e.target.checked);
                    };
                }
                const btnDeleteSelected = document.getElementById('btn-delete-selected');
                if (btnDeleteSelected) {
                    btnDeleteSelected.onclick = () => {
                        const checked = Array.from(document.querySelectorAll('.event-cb:checked')).map(cb => parseInt(cb.value));
                        if (checked.length === 0) return this.showToast('Select events to delete', 'warn');
                        if (!confirm(`Delete ${checked.length} selected events?`)) return;
                        this.api('DELETE', 'events/batch', { ids: checked })
                            .then(() => { this.showToast('Events deleted', 'success'); this.initEvents(); })
                            .catch(err => this.showToast(err.message, 'error'));
                    };
                }
                const btnDeleteAll = document.getElementById('btn-delete-all');
                if (btnDeleteAll) {
                    btnDeleteAll.onclick = () => {
                        if (!confirm('Are you sure you want to delete ALL events? This cannot be undone.')) return;
                        this.api('DELETE', 'events/all')
                            .then(() => { this.showToast('All events deleted', 'success'); this.initEvents(); })
                            .catch(err => this.showToast(err.message, 'error'));
                    };
                }
            })
            .catch(err => {
                container.innerHTML = `<p style="color:orange">Failed to load events: ${err.message}</p>`;
            });
    }

    initSettings() {
        this.log('Settings section not yet implemented.');
    }

    deleteEvent(id) {
        if (!confirm('Are you sure you want to delete this event?')) return;
        this.api('DELETE', `events/${id}`)
            .then(() => {
                this.showToast('Event deleted.', 'success');
                this.initEvents();
            })
            .catch(err => this.showToast(`Delete failed: ${err.message}`, 'error'));
    }

    editEvent(id, currentResult) {
        const newResult = prompt('Enter new result (OK or NG):', currentResult);
        if (!newResult) return;
        if (newResult !== 'OK' && newResult !== 'NG') {
            return this.showToast('Result must be OK or NG', 'error');
        }
        this.api('PUT', `events/${id}`, { result: newResult })
            .then(() => {
                this.showToast('Event updated.', 'success');
                this.initEvents();
            })
            .catch(err => this.showToast(`Update failed: ${err.message}`, 'error'));
    }

    // =========================================================
    // UI Helpers
    // =========================================================
    loadCamerasTable() {
        const tbody = document.getElementById('cameras-body');
        if (!tbody) return;
        tbody.innerHTML = '';
        if (this.cameras.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; opacity:0.6;">No cameras added yet. Click "+ Add Camera" above.</td></tr>';
            return;
        }
        this.cameras.forEach(cam => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${cam.id}</td>
                <td>${cam.name}</td>
                <td>${cam.rtsp_url}</td>
                <td>${cam.is_active ? '✅ Active' : '❌ Inactive'}</td>
                <td>
                    <button class="btn btn-sm btn-primary" onclick="app.editCamera(${cam.id})">Edit</button>
                    <button class="btn btn-sm btn-danger" onclick="app.deleteCamera(${cam.id})">Delete</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    updatePipelineStatus() {
        // Use inference/stats instead of non-existent pipeline/status
        const poll = async () => {
            try {
                const stats = await this.api('GET', 'inference/stats');
                const dot = document.getElementById('global-pipeline-status');
                const txt = document.getElementById('global-pipeline-text');
                const liveStatus = document.getElementById('live-status');
                const liveFps = document.getElementById('live-fps');

                if (stats.is_running) {
                    if (dot) dot.className = 'dot running';
                    if (txt) txt.textContent = `Running (Cam ${stats.camera_id})`;
                    if (liveStatus) { liveStatus.textContent = 'Running'; liveStatus.style.color = '#4caf50'; }
                    if (liveFps) liveFps.textContent = stats.fps || 0;
                } else {
                    if (dot) dot.className = 'dot stopped';
                    if (txt) txt.textContent = 'System Idle';
                    if (liveStatus) { liveStatus.textContent = 'Stopped'; liveStatus.style.color = '#f44336'; }
                    if (liveFps) liveFps.textContent = '0';
                }
            } catch (e) {
                // Silently ignore polling errors to avoid spamming debug log
            }
        };
        poll();
        this.pipelineStatusInterval = setInterval(poll, 5000);
    }

    displayFileResults(results) {
        const container = document.getElementById('file-results');
        if (!container) return;
        container.innerHTML = '';
        if (!results || results.length === 0) {
            container.innerHTML = '<p>No results.</p>';
            return;
        }
        results.forEach(res => {
            const div = document.createElement('div');
            div.className = 'file-result-item';
            const sim = typeof res.similarity === 'number' ? res.similarity.toFixed(2) : 'N/A';
            div.innerHTML = `
                <p><strong>File:</strong> ${res.filename}</p>
                <p><strong>Result:</strong> ${res.result} (Similarity: ${sim})</p>
                ${res.image_data ? `<img src="data:image/jpeg;base64,${res.image_data}" style="max-width:200px; border-radius:8px; margin-top:8px;"/>` : ''}
            `;
            container.appendChild(div);
        });
    }

    // =========================================================
    // Camera CRUD
    // =========================================================
    editCamera(id) {
        this.showToast('Edit camera not implemented yet.', 'info');
    }

    deleteCamera(id) {
        if (!confirm('Delete this camera?')) return;
        this.api('DELETE', `cameras/${id}`)
            .then(() => {
                this.cameras = this.cameras.filter(c => c.id !== id);
                this.loadCamerasTable();
                this.showToast('Camera deleted.', 'success');
            })
            .catch(err => this.log(`Delete camera failed: ${err.message}`, 'error'));
    }

    showAddCameraModal() {
        const html = `
            <h3>Add New Camera</h3>
            <div class="form-group"><label>Name</label><input id="new-camera-name" class="form-input" placeholder="e.g. Camera 1"/></div>
            <div class="form-group"><label>RTSP URL</label><input id="new-camera-rtsp" class="form-input" placeholder="rtsp://..."/></div>
            <button class="btn btn-primary" id="save-camera-btn">Save</button>
        `;
        this.showModal(html);
        document.getElementById('save-camera-btn').onclick = async () => {
            const name = document.getElementById('new-camera-name').value.trim();
            const rtsp = document.getElementById('new-camera-rtsp').value.trim();
            if (!name || !rtsp) return this.showToast('Name and RTSP URL required.', 'error');
            try {
                const newCam = await this.api('POST', 'cameras', { name, rtsp_url: rtsp, is_active: true });
                this.cameras.push(newCam);
                this.loadCamerasTable();
                this.closeModal();
                this.showToast('Camera added.', 'success');
            } catch (e) {
                this.log(`Add camera failed: ${e.message}`, 'error');
            }
        };
    }

    // =========================================================
    // Model Actions
    // =========================================================
    activateModel(modelId) {
        this.api('POST', `models/${modelId}/activate`)
            .then(() => {
                this.showToast('Model activated.', 'success');
                this.loadModels().then(() => this.initModels());
            })
            .catch(err => this.log(`Activate model failed: ${err.message}`, 'error'));
    }

    // =========================================================
    // Live Pipeline Controls
    // =========================================================
    async startPipeline() {
        const camId = document.getElementById('live-camera-select')?.value;
        const modelId = document.getElementById('live-model-select')?.value;
        const confThresh = document.getElementById('live-conf-thresh')?.value || '0.5';
        const simThresh = document.getElementById('live-sim-thresh')?.value || '0.85';

        if (!camId) return this.showToast('Please select a camera.', 'error');
        if (!modelId) return this.showToast('Please select a model.', 'error');

        this.log(`Starting pipeline: cam=${camId}, model=${modelId}, conf=${confThresh}, sim=${simThresh}`);

        try {
            await this.api('POST', `inference/start/${camId}`, {
                model_id: parseInt(modelId),
                confidence_threshold: parseFloat(confThresh),
                similarity_threshold: parseFloat(simThresh),
            });
            this.showToast('Pipeline started!', 'success');
            // Toggle buttons
            const startBtn = document.getElementById('btn-start-pipeline');
            const stopBtn = document.getElementById('btn-stop-pipeline');
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'inline-block';
            // Connect WebSocket for live frames
            this.connectLiveStream(camId);
            // Instantly update dashboard
            this.initDashboard();
        } catch (e) {
            this.log(`Failed to start pipeline: ${e.message}`, 'error');
        }
    }

    async stopPipeline() {
        try {
            await this.api('POST', 'inference/stop');
            this.showToast('Pipeline stopped.', 'info');
            const startBtn = document.getElementById('btn-start-pipeline');
            const stopBtn = document.getElementById('btn-stop-pipeline');
            if (startBtn) startBtn.style.display = 'inline-block';
            if (stopBtn) stopBtn.style.display = 'none';
            if (this.ws) { this.ws.close(); this.ws = null; }
            // Show no-signal message
            const noSignal = document.getElementById('no-signal-msg');
            if (noSignal) noSignal.style.display = 'flex';
            // Instantly update dashboard
            this.initDashboard();
        } catch (e) {
            this.log(`Failed to stop pipeline: ${e.message}`, 'error');
        }
    }

    connectLiveStream(cameraId) {
        if (this.ws) this.ws.close();
        const wsUrl = `ws://${window.location.host}/ws/live/${cameraId}`;
        this.log(`Connecting to WebSocket: ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);

        const canvas = document.getElementById('live-canvas');
        const ctx = canvas?.getContext('2d');
        const noSignal = document.getElementById('no-signal-msg');

        if (!ctx) return this.log('Canvas 2D context not found!', 'error');

        this.ws.onopen = () => {
            this.log('WebSocket connection established.', 'success');
            if (noSignal) noSignal.style.display = 'none';
        };
        this.ws.onclose = () => this.log('WebSocket connection closed.', 'warn');
        this.ws.onerror = (err) => this.log(`WebSocket error`, 'error');

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'frame' && message.data) {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.naturalWidth || 1280;
                        canvas.height = img.naturalHeight || 720;
                        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    };
                    img.src = 'data:image/jpeg;base64,' + message.data;
                } else if (message.type === 'event' && message.data) {
                    if (message.data.image) {
                        const logContainer = document.getElementById('classification-log-container');
                        const emptyMsg = document.getElementById('classification-empty-msg');
                        if (emptyMsg) emptyMsg.style.display = 'none';
                        
                        if (logContainer) {
                            // Create a small card for the classification
                            const card = document.createElement('div');
                            card.style.display = 'flex';
                            card.style.flexDirection = 'column';
                            card.style.alignItems = 'center';
                            card.style.background = 'rgba(255, 255, 255, 0.05)';
                            card.style.padding = '8px';
                            card.style.borderRadius = '8px';
                            card.style.minWidth = '80px';
                            
                            const imgEl = document.createElement('img');
                            imgEl.src = 'data:image/jpeg;base64,' + message.data.image;
                            imgEl.style.width = '80px';
                            imgEl.style.height = '80px';
                            imgEl.style.objectFit = 'contain';
                            imgEl.style.border = message.data.result === 'OK' ? '2px solid #10b981' : '2px solid #ef4444';
                            imgEl.style.borderRadius = '8px';
                            
                            const label = document.createElement('span');
                            label.style.marginTop = '4px';
                            label.style.fontSize = '12px';
                            label.style.fontWeight = 'bold';
                            label.style.color = message.data.result === 'OK' ? '#10b981' : '#ef4444';
                            label.textContent = `ID:${message.data.track_id} ${message.data.result}`;
                            
                            card.title = `Sim: ${message.data.similarity?.toFixed(2)} - ${new Date().toLocaleTimeString()}`;
                            card.appendChild(imgEl);
                            card.appendChild(label);
                            
                            logContainer.prepend(card);
                            
                            // Keep max 20 images
                            while (logContainer.children.length > 20) {
                                logContainer.removeChild(logContainer.lastChild);
                            }
                        }
                    }
                }
            } catch (e) {
                // Ignore parse errors on binary frames
            }
        };
    }
}

window.app = new App();
