/**
 * Modular Dashboard Application
 *
 * This is the main orchestrator that initializes and coordinates all dashboard modules.
 * The previous monolithic approach has been refactored into separate, focused modules.
 */

class DashboardManager {
    constructor() {
        // Initialize core modules
        this.state = new DashboardState();
        this.dom = new DOMUtils();
        this.api = new APIClient(this.state);
        this.websocket = new WebSocketManager(this.state);
        this.charts = new ChartManager(this.state, this.api);
        this.navigation = new NavigationManager(this.state, this.api, this.dom);
        this.sessions = new SessionManager(this.state, this.api, this.dom);

        // Configuration
        this.config = {
            updateInterval: 1000,
            websocketReconnectDelay: 3000,
            multiRolloutAnalysisDelay: 2000
        };

        // Intervals
        this.updateIntervalId = null;

        // Bind methods
        this.updateDashboard = this.updateDashboard.bind(this);
        this.refreshAllSections = this.refreshAllSections.bind(this);

        // Make globally accessible for backward compatibility
        window.dashboardManager = this;
        window.navigationManager = this.navigation;
        window.sessionManager = this.sessions;

        // Global functions for HTML onclick handlers
        window.navigateExample = this.navigation.navigateExample;
        window.setFilter = this.navigation.setFilter;
        window.switchSession = this.sessions.switchSession;
    }

    /**
     * Initialize the dashboard
     */
    async initialize() {
        try {

            // Setup session selector
            this.sessions.setupSessionSelector();

            // Setup chart toggle buttons
            this.setupChartControls();

            // Initialize WebSocket connection
            this.websocket.connect();

            // Load initial data
            await this.loadInitialData();

            // Setup periodic updates
            this.setupPeriodicUpdates();

            // Mark as initialized
            this.state.set('isInitialized', true);


            } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
        }
    }

    /**
     * Load initial dashboard data
     */
    async loadInitialData() {
        try {
            // Load sessions and initial context
            await Promise.all([
                this.sessions.loadAvailableSessions(),
                this.updateEvaluationContext(),
                this.updateProgressTracking(),
                this.updateKeyResults(),
                this.navigation.updateExampleNavigator(),
                this.charts.updateProgressChart()
            ]);

            // Update chart toggle visibility after data is loaded
            await this.updateChartToggleVisibility();

            // Load multi-rollout analysis after a delay to ensure data is available
            setTimeout(() => {
                this.updateMultiRolloutAnalysis();
            }, this.config.multiRolloutAnalysisDelay);

        } catch (error) {
            console.error('❌ Error loading initial data:', error);
        }
    }

    /**
     * Setup chart toggle controls and adaptive display
     */
    setupChartControls() {
        const buttons = document.querySelectorAll('.chart-control-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const chartType = btn.dataset.chartType;

                // Update active state
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Switch chart type
                this.charts.switchChartType(chartType);
            });
        });
    }

    /**
     * Detect reward structure and show/hide chart toggle
     */
    async updateChartToggleVisibility() {
        try {
            const sessionId = this.state.get('currentSessionId');
            const { data: rollouts } = await this.api.getEvaluationHistory({
                limit: 1000,
                sessionId
            });

            if (!rollouts || rollouts.length === 0) {
                // Hide toggle if no data
                document.getElementById('chart-controls').style.display = 'none';
                return;
            }

            // Check if rewards are binary (only 0.0 and 1.0)
            const uniqueRewards = new Set(rollouts.map(r => r.reward || 0));
            const isBinary = uniqueRewards.size <= 2 &&
                           [...uniqueRewards].every(r => r === 0.0 || r === 1.0);

            // Show toggle only for non-binary rewards
            const chartControls = document.getElementById('chart-controls');
            if (isBinary) {
                chartControls.style.display = 'none';
            } else {
                chartControls.style.display = 'flex';
            }
        } catch (error) {
            console.error('❌ Error updating chart toggle visibility:', error);
        }
    }

    /**
     * Setup periodic updates
     */
    setupPeriodicUpdates() {
        if (this.updateIntervalId) {
            clearInterval(this.updateIntervalId);
        }

        this.updateIntervalId = setInterval(() => {
            this.updateProgressTracking();
            this.updateKeyResults();
            this.charts.updateProgressChart();

            // Only update example navigator if user isn't actively using it
            if (!this.state.get('userIsNavigating')) {
                this.navigation.updateExampleNavigator();
            }
        }, this.config.updateInterval);
    }

    /**
     * Main dashboard update method (called by WebSocket)
     */
    updateDashboard(metrics) {
        this.refreshAllSections();
    }

    /**
     * Refresh all dashboard sections
     */
    async refreshAllSections() {
        try {
            await Promise.all([
                this.updateEvaluationContext(),
                this.updateProgressTracking(),
                this.updateKeyResults(),
                this.navigation.updateExampleNavigator(),
                this.charts.updateProgressChart(),
                this.updateMultiRolloutAnalysis(),
                this.updateChartToggleVisibility()
            ]);
        } catch (error) {
            console.error('❌ Error refreshing dashboard sections:', error);
        }
    }

    /**
     * Update evaluation context section
     */
    async updateEvaluationContext() {
        try {

            const [envInfo, sessions, progress] = await Promise.all([
                this.api.getEnvironmentInfo(),
                this.api.getSessions(),
                this.api.getProgress()
            ]);

            // Get current session for environment and model info
            const currentSessionId = this.state.get('currentSessionId');
            let currentSession = null;
            let modelName = 'Unknown Model';
            let providerName = 'API';

            if (currentSessionId === 'current' && Array.isArray(sessions) && sessions.length > 0) {
                currentSession = sessions[0];
                modelName = currentSession.model_name || currentSession.model || 'Unknown Model';
            } else if (currentSessionId !== 'current') {
                currentSession = sessions.find(s => s.session_id === currentSessionId);
                if (currentSession) {
                    modelName = currentSession.model_name || currentSession.model || 'Unknown Model';
                }
            }

            // Update environment type from session (prefer env_id if available)
            // Helper function to format environment names (math-python → Math Python)
            const formatEnvironmentName = (envType) => {
                return envType
                    .replace(/-/g, '_')  // Convert hyphens to underscores
                    .split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
            };

            // Prefer env_id (e.g., "math-python") over environment_type (e.g., "ToolEnv")
            const sessionEnvId = currentSession?.env_id;
            const sessionEnvType = currentSession?.environment_type;
            const envType = sessionEnvId || sessionEnvType || envInfo.env_id || envInfo.env_type || 'Unknown';
            const friendlyName = envType !== 'Unknown' ? formatEnvironmentName(envType) : 'Unknown Environment';

            // Update DOM elements
            this.dom.updateText('evaluation-title', `${friendlyName} Evaluation`);
            this.dom.updateText('environment-name', envType);  // Keep raw format (math-python)

            this.dom.updateText('model-name', modelName);
            // Note: provider-name element doesn't exist in current template
            // this.dom.updateText('provider-name', providerName);
            this.dom.updateText('progress-display', `${progress.completed || 0}/${progress.total || 0}`);

            // Store environment info in state
            this.state.set('environmentInfo', envInfo);

            // Update session selector button with current context
            if (this.sessions && this.sessions.updateCurrentSessionButton) {
                this.sessions.updateCurrentSessionButton();
            }

            } catch (error) {
            console.error('❌ Error updating evaluation context:', error);
            // Set fallback values
            this.dom.updateText('evaluation-title', 'Evaluation Dashboard');
            this.dom.updateText('environment-name', 'Loading...');
            this.dom.updateText('model-name', 'Loading...');
            // Note: provider-name element doesn't exist in current template
            // this.dom.updateText('provider-name', 'Loading...');
            this.dom.updateText('progress-display', '0/0');
        }
    }

    /**
     * Update progress tracking section
     */
    async updateProgressTracking() {
        try {
            const progress = await this.api.getProgress();

                if (progress.error) return;

            // Update state
            this.state.update('progress', progress);

                // Update context progress
            this.dom.updateText('progress-display', `${progress.completed || 0}/${progress.total || 0}`);

            // Show/hide progress section based on status
                const isInProgress = progress.completed < progress.total && progress.total > 0;

            if (isInProgress) {
                // Note: progress-section element doesn't exist in current template
                // this.dom.show('progress-section');
                this.dom.updateStyle('progress-bar', 'width', `${progress.percentage || 0}%`);
                this.dom.updateText('progress-text', `Processing example ${progress.completed}/${progress.total}`);

                    if (progress.eta_seconds && progress.eta_seconds > 0) {
                        const minutes = Math.floor(progress.eta_seconds / 60);
                        const seconds = Math.floor(progress.eta_seconds % 60);
                    this.dom.updateText('progress-eta', `ETA: ${minutes}m ${seconds}s`);
                    } else {
                    this.dom.updateText('progress-eta', 'Calculating ETA...');
                }

                this.dom.updateText('progress-throughput', `${(progress.throughput || 0).toFixed(1)} examples/sec`);

                // Update status badge
                this.dom.updateText('status-badge', 'In Progress');
                this.dom.updateAttribute('status-badge', 'class', 'status-badge in-progress');

                } else if (progress.completed > 0) {
                    // Evaluation completed
                // Note: progress-section element doesn't exist in current template
                // this.dom.hide('progress-section');
                this.dom.updateText('status-badge', 'Completed');
                this.dom.updateAttribute('status-badge', 'class', 'status-badge completed');
                }

            } catch (error) {
            console.error('❌ Error updating progress:', error);
        }
    }

    /**
     * Update key results section
     */
    async updateKeyResults() {
        try {

            const sessionId = this.state.get('currentSessionId');
            const [evalResult, progress] = await Promise.all([
                this.api.getEvaluationHistory({ limit: 1000, sessionId }),
                this.api.getProgress()
            ]);

                const evalData = evalResult.data || evalResult;

                if (Array.isArray(evalData) && evalData.length > 0) {
                // Primary: Success Rate
                    const rewards = evalData.map(d => d.reward || 0);
                    const successCount = rewards.filter(r => r > 0.5).length;
                    const successRate = (successCount / rewards.length) * 100;

                this.dom.updateText('primary-score', successRate.toFixed(0) + '%');
                this.dom.updateAttribute('primary-score', 'class',
                    `result-value ${successRate > 50 ? 'stat-positive' : 'stat-neutral'}`);

                    // Show progress context during evaluation
                    if (progress.completed < progress.total) {
                    this.dom.updateText('primary-detail', `${successCount} of ${evalData.length} processed so far`);
                    } else {
                    this.dom.updateText('primary-detail', `${successCount} of ${rewards.length} examples correct`);
                    }

                // Speed (average response time)
                    const avgTime = evalData.reduce((sum, d) => sum + (d.rollout_time || 0), 0) / evalData.length;
                this.dom.updateText('speed-score', avgTime.toFixed(1) + 's');
                this.dom.updateText('speed-detail', 'average response time');

                // Consistency (% of examples with low variance + range)
                    // Group rollouts by example_number to calculate per-example variance
                    const exampleGroups = {};
                    evalData.forEach(d => {
                        const exNum = d.example_number || 0;
                        if (!exampleGroups[exNum]) {
                            exampleGroups[exNum] = [];
                        }
                        exampleGroups[exNum].push(d.reward || 0);
                    });

                    const exampleCount = Object.keys(exampleGroups).length;
                    const rolloutsPerExample = evalData.length / exampleCount;

                    if (rolloutsPerExample > 1) {
                        // Calculate consistency: % of examples with low variance
                        const consistentExamples = Object.values(exampleGroups).filter(rollouts => {
                            if (rollouts.length <= 1) return true;

                            const mean = rollouts.reduce((a, b) => a + b, 0) / rollouts.length;
                            const variance = rollouts.reduce((sum, r) =>
                                sum + Math.pow(r - mean, 2), 0) / rollouts.length;
                            const stdDev = Math.sqrt(variance);

                            // Consider consistent if σ < 0.15 (low variance threshold)
                            return stdDev < 0.15;
                        }).length;

                        const consistencyRate = (consistentExamples / exampleCount) * 100;
                        const minReward = Math.min(...rewards);
                        const maxReward = Math.max(...rewards);

                        this.dom.updateText('consistency-score', `${consistencyRate.toFixed(0)}%`);
                        this.dom.updateText('consistency-detail', `Range: ${minReward.toFixed(1)} - ${maxReward.toFixed(1)}`);
                        this.dom.updateText('rollouts-meta', `${Math.round(rolloutsPerExample)}x`);
                    } else {
                        // Single rollout - no consistency data
                        this.dom.updateText('consistency-score', '—');
                        this.dom.updateText('consistency-detail', 'single rollout');
                        this.dom.updateText('rollouts-meta', '1x');
                    }

                } else {
                    // Show progress even when no completed examples yet
                    if (!progress.error && progress.total > 0) {
                    this.dom.updateText('primary-score', '0%');
                    this.dom.updateText('primary-detail', `Starting evaluation of ${progress.total} examples...`);
                    this.dom.updateText('speed-score', '--');
                    this.dom.updateText('speed-detail', 'calculating...');
                    this.dom.updateText('consistency-score', '--');
                    this.dom.updateText('consistency-detail', 'calculating...');
                    this.dom.updateText('rollouts-meta', '--');
                    }
                }

            } catch (error) {
                console.error('❌ Error updating key results:', error);
            this.dom.updateText('primary-score', 'Error');
            this.dom.updateText('primary-detail', 'Failed to load data');
        }
    }

    /**
     * Update multi-rollout analysis section
     */
    async updateMultiRolloutAnalysis() {
        try {

            const sessionId = this.state.get('currentSessionId');
            const { data: rollouts } = await this.api.getEvaluationHistory({
                limit: 1000,
                sessionId
            });

            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.dom.hide('multi-rollout-section');
                    return;
                }

            // Calculate multi-rollout analysis client-side
            const analysis = this.calculateMultiRolloutAnalysis(rollouts);

            if (analysis.error || analysis.multi_rollout_prompts === 0) {
                this.dom.hide('multi-rollout-section');
                return;
            }

            // Show multi-rollout section
            this.dom.show('multi-rollout-section');

            let contentHtml = `
                <div class="config-grid">
                    <div class="config-item">
                        <div class="config-label">Prompts with Multiple Rollouts</div>
                        <div class="config-value">${analysis.multi_rollout_prompts} of ${analysis.total_prompts}</div>
                                        </div>
                    <div class="config-item">
                        <div class="config-label">Best-of-N Improvement</div>
                        <div class="config-value ${analysis.best_of_n_improvement > 0.1 ? 'stat-positive' : 'stat-neutral'}">
                            +${(analysis.best_of_n_improvement * 100).toFixed(1)}%
                        </div>
                        </div>
                    <div class="config-item clickable-metric" onclick="window.navigationManager.filterByVariance('high')" style="cursor: pointer;">
                        <div class="config-label">High Variance Prompts</div>
                        <div class="config-value ${analysis.consistency_stats.high_variance > 0 ? 'stat-neutral' : 'stat-positive'}">
                            ${analysis.consistency_stats.high_variance} prompts →
                    </div>
                    </div>
                    <div class="config-item clickable-metric" onclick="window.navigationManager.filterByVariance('low')" style="cursor: pointer;">
                        <div class="config-label">Consistent Prompts</div>
                        <div class="config-value stat-positive">
                            ${analysis.consistency_stats.low_variance} prompts →
                    </div>
                    </div>
                    </div>
                `;

            // Show detailed prompt analysis if we have unstable prompts
            if (analysis.consistency_stats.high_variance > 0) {
                const unstablePrompts = analysis.prompt_analysis.filter(p => p.std_dev > 0.3);
                contentHtml += `
                    <div style="margin-top: 24px;">
                        <div class="config-label" style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div>Unstable Prompts</div>
                                <div style="font-size: 11px; color: var(--color-text-tertiary); margin-top: 2px;">High Variance</div>
                            </div>
                            <span class="clickable-link" onclick="window.navigationManager.filterByVariance('high')" style="font-size: 13px; color: var(--color-primary); cursor: pointer;">View all →</span>
                        </div>
                `;

                unstablePrompts.slice(0, 5).forEach(prompt => {
                    contentHtml += `
                        <div class="config-item" style="border-left: 3px solid var(--color-info); background: var(--color-bg-tertiary); border-top: 1px solid var(--color-border-default); border-right: 1px solid var(--color-border-default); border-bottom: 1px solid var(--color-border-default); border-radius: 6px; padding: 12px;">
                            <div class="config-label">Answer: "${prompt.answer}"</div>
                            <div style="font-size: 13px; color: var(--color-info); margin-top: 4px;">
                                Rewards: [${prompt.rewards.map(r => r.toFixed(1)).join(', ')}]
                                (σ = ${prompt.std_dev.toFixed(3)})
                                </div>
                            </div>
                        `;
                    });

                contentHtml += `</div>`;
            }

            this.dom.updateHTML('multi-rollout-content', contentHtml);

            } catch (error) {
            console.error('❌ Error updating multi-rollout analysis:', error);
            }
        }

    /**
     * Calculate multi-rollout analysis from rollouts data
     */
    calculateMultiRolloutAnalysis(rollouts) {
            // Group rollouts by prompt_hash
            const promptGroups = {};

            rollouts.forEach(rollout => {
                let promptHash = rollout.prompt_hash;
                if (!promptHash || promptHash === "" || promptHash === "0") {
                    // Use prompt content for grouping (unique per example)
                    promptHash = rollout.prompt
                        ? `prompt_${JSON.stringify(rollout.prompt)}`
                        : `fallback_${rollout.id}`; // Last resort: unique ID (no grouping)
                }

                if (!promptGroups[promptHash]) {
                    promptGroups[promptHash] = [];
                }
                promptGroups[promptHash].push(rollout);
            });

            const totalPrompts = Object.keys(promptGroups).length;
            const multiRolloutGroups = Object.entries(promptGroups).filter(([hash, group]) => group.length > 1);
            const multiRolloutPrompts = multiRolloutGroups.length;

            if (multiRolloutPrompts === 0) {
                return { error: "No multi-rollout prompts found" };
            }

            // Analyze consistency and variance
            const consistencyStats = {
                high_variance: 0,
                medium_variance: 0,
                low_variance: 0
            };

            const promptAnalysis = [];
            const improvements = [];

            multiRolloutGroups.forEach(([promptHash, group]) => {
                const rewards = group.map(r => r.reward || 0);
                const avgReward = rewards.reduce((sum, r) => sum + r, 0) / rewards.length;
                const variance = rewards.reduce((sum, r) => sum + Math.pow(r - avgReward, 2), 0) / rewards.length;
                const stdDev = Math.sqrt(variance);

                // Categorize variance
                if (stdDev > 0.3) {
                    consistencyStats.high_variance++;
                } else if (stdDev > 0.1) {
                    consistencyStats.medium_variance++;
                } else {
                    consistencyStats.low_variance++;
                }

                // Best-of-N improvement
                const firstReward = rewards[0];
                const bestReward = Math.max(...rewards);
                improvements.push(bestReward - firstReward);

                // Store analysis
                promptAnalysis.push({
                    prompt_hash: promptHash,
                    rollout_count: group.length,
                    rewards: rewards,
                    avg_reward: avgReward,
                    std_dev: stdDev,
                    min_reward: Math.min(...rewards),
                    max_reward: Math.max(...rewards),
                    answer: group[0].answer || ""
                });
            });

            const avgImprovement = improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length;

            return {
                total_prompts: totalPrompts,
                multi_rollout_prompts: multiRolloutPrompts,
                consistency_stats: consistencyStats,
                prompt_analysis: promptAnalysis,
                best_of_n_improvement: avgImprovement
            };
        }

    /**
     * Clean up dashboard resources
     */
    cleanup() {
        if (this.updateIntervalId) {
            clearInterval(this.updateIntervalId);
        }

        this.websocket.disconnect();
        this.charts.cleanup();
        this.sessions.cleanup();
        this.dom.clearCache();
            }
        }

        // Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', async function() {

    // Create dashboard manager instance
    const dashboard = new DashboardManager();

    // Initialize the dashboard
    await dashboard.initialize();

});
