/**
 * Session management for switching between evaluation sessions
 */
class SessionManager {
    constructor(state, apiClient, domUtils) {
        this.state = state;
        this.api = apiClient;
        this.dom = domUtils;

        // Bind methods to preserve context
        this.switchSession = this.switchSession.bind(this);
        this.loadAvailableSessions = this.loadAvailableSessions.bind(this);

        // Subscribe to session changes
        this.state.subscribe('currentSessionId', (sessionId) => {
            this.onSessionChanged(sessionId);
        });
    }

    /**
     * Group sessions by environment
     */
    groupSessionsByEnvironment(sessions) {
        const grouped = {};

        sessions.forEach(session => {
            const env = session.environment_type || 'unknown';
            if (!grouped[env]) {
                grouped[env] = [];
            }
            grouped[env].push(session);
        });

        // Sort sessions within each group by timestamp (most recent first)
        Object.keys(grouped).forEach(env => {
            grouped[env].sort((a, b) => {
                const timeA = new Date(a.started_at || a.timestamp * 1000);
                const timeB = new Date(b.started_at || b.timestamp * 1000);
                return timeB - timeA; // Descending (newest first)
            });
        });

        return grouped;
    }

    /**
     * Load available sessions and populate selector
     */
    async loadAvailableSessions() {
        try {
            const sessions = await this.api.getSessions();
            this.state.set('availableSessions', sessions);

            // Render custom session selector
            this.renderSessionSelector(sessions);

        } catch (error) {
            console.error('❌ Error loading sessions:', error);
        }
    }

    /**
     * Render custom session selector with environment grouping
     */
    renderSessionSelector(sessions) {
        const wrapper = document.querySelector('.session-selector-wrapper');
        if (!wrapper) return;

        // Group sessions by environment
        const grouped = this.groupSessionsByEnvironment(sessions);
        const envs = Object.keys(grouped).sort();

        // Get current session info for button and dropdown
        const currentButton = this.getCurrentSessionInfo();
        const currentDetails = this.getCurrentSessionDetails();

        // Build HTML
        let html = `
            <div class="session-dropdown">
                <button class="session-dropdown-toggle" onclick="window.sessionManager.toggleDropdown(event)">
                    <span class="session-current-label">${String(currentButton)}</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="dropdown-icon"><polyline points="6 9 12 15 18 9"></polyline></svg>
                </button>

                <div class="session-dropdown-menu" id="session-dropdown-menu" style="display: none;">
                    <div class="session-section">
                        <div class="session-section-header">CURRENT</div>
                        <div class="session-item current">
                            <span class="session-info">${currentDetails}</span>
                        </div>
                    </div>

                    ${envs.length > 0 ? `
                        <div class="session-section">
                            <div class="session-section-header">RECENT</div>
                            ${this.renderEnvironmentGroups(grouped, 5)}
                        </div>
                    ` : ''}

                    ${sessions.length > 15 ? `
                        <div class="session-section">
                            <button class="session-action-btn" onclick="alert('Search feature coming soon!')">
                                Search ${sessions.length - 15} more sessions
                            </button>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        // Replace old selector
        const oldSelector = document.getElementById('session-selector');
        if (oldSelector) {
            oldSelector.style.display = 'none';
        }

        wrapper.innerHTML = html;
    }

    /**
     * Render environment groups for dropdown
     */
    renderEnvironmentGroups(grouped, maxPerEnv = 3) {
        const envs = Object.keys(grouped).sort();
        let html = '';

        // If only "unknown", don't show group header (cleaner)
        const onlyUnknown = envs.length === 1 && envs[0] === 'unknown';

        envs.forEach(env => {
            const sessions = grouped[env].slice(0, maxPerEnv);
            const envName = this.formatEnvironmentName(env);
            const isUnknown = env === 'unknown';

            // Skip showing "Unknown" header if it's the only environment
            if (onlyUnknown || isUnknown) {
                // Just show sessions without group header
                html += sessions.map(session => this.renderSessionItem(session)).join('');
            } else {
                // Show group with header for known environments
                html += `
                    <div class="env-group">
                        <div class="env-group-header">${envName}</div>
                        ${sessions.map(session => this.renderSessionItem(session)).join('')}
                    </div>
                `;
            }
        });

        return html;
    }

    /**
     * Render a single session item with colored status dot
     */
    renderSessionItem(session) {
        const successRate = session.final_reward_mean != null
            ? Math.round(session.final_reward_mean * 100)
            : null;

        // Determine status color class
        let statusClass = 'status-unknown';
        if (successRate !== null) {
            if (successRate >= 80) statusClass = 'status-success';
            else if (successRate >= 50) statusClass = 'status-warning';
            else statusClass = 'status-danger';
        }

        const model = session.model_name || session.model || 'Unknown';
        const numEx = session.num_examples || 0;
        const timestamp = session.started_at || (session.timestamp ? new Date(session.timestamp * 1000) : null);
        const timeAgo = timestamp ? this.getRelativeTime(timestamp) : '';

        // Build parts array for clean formatting
        const parts = [model];
        if (successRate !== null) parts.push(`${successRate}%`);
        parts.push(`${numEx} ex`);
        if (timeAgo) parts.push(timeAgo);

        return `
            <div class="session-item" onclick="window.sessionManager.selectSession('${session.session_id}')">
                <span class="session-status-dot ${statusClass}"></span>
                <span class="session-details">${parts.join('  ·  ')}</span>
            </div>
        `;
    }

    /**
     * Get current session info for display (button shows "Current Session")
     */
    getCurrentSessionInfo() {
        // Always show "Current Session" for the button (simple and clear)
        return 'Current Session';
    }

    /**
     * Get current session details for dropdown
     */
    getCurrentSessionDetails() {
        // Try to get from state first (most reliable)
        const currentSessionId = this.state.get('currentSessionId');

        if (currentSessionId && currentSessionId !== 'current') {
            const availableSessions = this.state.get('availableSessions') || [];
            const session = availableSessions.find(s => s.session_id === currentSessionId);
            if (session && session.model_name) {
                // Use env_id (like "math-python") if available, otherwise fallback to formatted environment_type (like "ToolEnv" → "Tool Env")
                const envName = session.env_id || this.formatEnvironmentName(session.environment_type);
                return `${envName}  ·  ${session.model_name}`;
            }
        }

        // Fallback: Try to get environment info from availableSessions for 'current' session
        if (currentSessionId === 'current') {
            const availableSessions = this.state.get('availableSessions') || [];
            // Find the most recent session
            if (availableSessions.length > 0) {
                const recentSession = availableSessions[0]; // Already sorted by timestamp
                if (recentSession) {
                    const envName = recentSession.env_id || this.formatEnvironmentName(recentSession.environment_type);
                    const modelElement = document.getElementById('model-name');
                    const model = modelElement ? modelElement.textContent.trim() : null;

                    if (model && model !== '...') {
                        return `${envName}  ·  ${model}`;
                    }
                    return envName;
                }
            }
        }

        // Fallback: Try to get from page context
        const modelElement = document.getElementById('model-name');

        if (modelElement) {
            const model = modelElement.textContent.trim();

            // Only use DOM if it's not loading state
            if (model && model !== '...') {
                // Try to get environment name from recent sessions
                const availableSessions = this.state.get('availableSessions') || [];
                if (availableSessions.length > 0) {
                    const envName = availableSessions[0].env_id || this.formatEnvironmentName(availableSessions[0].environment_type);
                    return `${envName}  ·  ${model}`;
                }

                return model;
            }
        }

        // Final fallback
        return 'No active session';
    }

    /**
     * Toggle dropdown visibility
     */
    toggleDropdown(event) {
        event.stopPropagation();
        const menu = document.getElementById('session-dropdown-menu');
        if (!menu) return;

        const isVisible = menu.style.display !== 'none';
        menu.style.display = isVisible ? 'none' : 'block';

        // Close on outside click
        if (!isVisible) {
            setTimeout(() => {
                document.addEventListener('click', this.closeDropdownOnOutsideClick.bind(this), { once: true });
            }, 0);
        }
    }

    /**
     * Close dropdown when clicking outside
     */
    closeDropdownOnOutsideClick(event) {
        const menu = document.getElementById('session-dropdown-menu');
        const dropdown = document.querySelector('.session-dropdown');

        if (menu && dropdown && !dropdown.contains(event.target)) {
            menu.style.display = 'none';
        }
    }

    /**
     * Select a session from dropdown
     */
    selectSession(sessionId) {
        // Close dropdown
        const menu = document.getElementById('session-dropdown-menu');
        if (menu) {
            menu.style.display = 'none';
        }

        // Update state
        this.state.set('currentSessionId', sessionId);

        // Update old selector for backward compatibility
        const oldSelector = document.getElementById('session-selector');
        if (oldSelector) {
            oldSelector.value = sessionId;
        }
    }

    /**
     * Get relative time string
     */
    getRelativeTime(timestamp) {
        const now = new Date();
        const then = new Date(timestamp);
        const diffMs = now - then;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays}d ago`;
        return then.toLocaleDateString();
    }

    /**
     * Format session label for display (clean, professional)
     */
    formatSessionLabel(session) {
        const env = this.formatEnvironmentName(session.environment_type);
        const model = session.model_name || session.model || 'Unknown';
        const numEx = session.num_examples || 0;

        // Get timestamp
        const timestamp = session.started_at || (session.timestamp ? new Date(session.timestamp * 1000) : null);
        const timeAgo = timestamp ? this.getRelativeTime(timestamp) : 'Unknown';

        // Get success rate if available
        const successRate = session.final_reward_mean != null
            ? `${Math.round(session.final_reward_mean * 100)}%`
            : '';

        // Format: Math Python  ·  gpt-4o-mini  ·  73%  ·  10 ex  ·  4h ago
        const parts = [env, model];
        if (successRate) parts.push(successRate);
        parts.push(`${numEx} ex`, timeAgo);

        return parts.join('  ·  ');
    }

    /**
     * Format environment name for display
     */
    formatEnvironmentName(envType) {
        // Convert snake_case to Title Case
        return envType
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Update the current session button text
     */
    updateCurrentSessionButton() {
        const button = document.querySelector('.session-current-label');
        if (!button) return;

        const currentInfo = this.getCurrentSessionInfo();
        if (button.textContent !== currentInfo) {
            button.textContent = currentInfo;
        }
    }

    /**
     * Switch to a different session
     */
    switchSession() {
        const selector = document.getElementById('session-selector');
        if (!selector) return;

        const selectedSessionId = selector.value;
        // Update state
        this.state.set('currentSessionId', selectedSessionId);
    }

    /**
     * Handle session change event
     */
    onSessionChanged(sessionId) {
        // Update selector if needed
        const selector = document.getElementById('session-selector');
        if (selector && selector.value !== sessionId) {
            selector.value = sessionId;
        }

        // Trigger dashboard refresh
        if (window.dashboardManager && window.dashboardManager.refreshAllSections) {
            window.dashboardManager.refreshAllSections();
        }
    }

    /**
     * Get current session data object (for export/API use)
     */
    getCurrentSessionData() {
        const currentSessionId = this.state.get('currentSessionId');
        const availableSessions = this.state.get('availableSessions');

        if (currentSessionId === 'current') {
            return {
                id: 'current',
                label: 'Current Session',
                isCurrent: true
            };
        }

        const session = availableSessions.find(s => s.session_id === currentSessionId);
        if (session) {
            return {
                id: session.session_id,
                label: this.formatSessionLabel(session),
                isCurrent: false,
                session
            };
        }

        return {
            id: currentSessionId,
            label: 'Unknown Session',
            isCurrent: false
        };
    }

    /**
     * Get session statistics
     */
    async getSessionStatistics(sessionId = null) {
        try {
            const targetSessionId = sessionId || this.state.get('currentSessionId');

            const { data: rollouts } = await this.api.getEvaluationHistory({
                sessionId: targetSessionId,
                limit: 10000
            });

            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                return {
                    totalRollouts: 0,
                    uniqueExamples: 0,
                    averageReward: 0,
                    successRate: 0,
                    averageResponseTime: 0
                };
            }

            const rewards = rollouts.map(r => r.reward || 0);
            const responseTimes = rollouts.map(r => (r.rollout_time || 0) * 1000); // Convert to ms
            const uniqueExamples = new Set(rollouts.map(r => r.example_number)).size;

            const stats = {
                totalRollouts: rollouts.length,
                uniqueExamples,
                averageReward: rewards.reduce((sum, r) => sum + r, 0) / rewards.length,
                successRate: rewards.filter(r => r > 0.5).length / rewards.length,
                averageResponseTime: responseTimes.reduce((sum, t) => sum + t, 0) / responseTimes.length,
                rewardDistribution: {
                    min: Math.min(...rewards),
                    max: Math.max(...rewards),
                    std: this.calculateStandardDeviation(rewards)
                }
            };

            return stats;

        } catch (error) {
            console.error('❌ Error getting session statistics:', error);
            return { error: error.message };
        }
    }

    /**
     * Calculate standard deviation
     */
    calculateStandardDeviation(values) {
        if (values.length < 2) return 0;

        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }

    /**
     * Compare two sessions
     */
    async compareSessions(sessionId1, sessionId2) {
        try {
            const [stats1, stats2] = await Promise.all([
                this.getSessionStatistics(sessionId1),
                this.getSessionStatistics(sessionId2)
            ]);

            if (stats1.error || stats2.error) {
                throw new Error('Failed to get session statistics');
            }

            return {
                session1: { id: sessionId1, stats: stats1 },
                session2: { id: sessionId2, stats: stats2 },
                comparison: {
                    rewardDifference: stats2.averageReward - stats1.averageReward,
                    successRateDifference: stats2.successRate - stats1.successRate,
                    responseTimeDifference: stats2.averageResponseTime - stats1.averageResponseTime,
                    rolloutCountDifference: stats2.totalRollouts - stats1.totalRollouts
                }
            };

        } catch (error) {
            console.error('❌ Error comparing sessions:', error);
            return { error: error.message };
        }
    }

    /**
     * Export session data
     */
    async exportSession(sessionId = null) {
        try {
            const targetSessionId = sessionId || this.state.get('currentSessionId');
            const sessionInfo = this.getCurrentSessionData();

            const [rollouts, envInfo, stats] = await Promise.all([
                this.api.getEvaluationHistory({ sessionId: targetSessionId, limit: 10000 }),
                this.api.getEnvironmentInfo(),
                this.getSessionStatistics(targetSessionId)
            ]);

            const exportData = {
                sessionInfo,
                statistics: stats,
                environmentInfo: envInfo,
                rollouts: rollouts.data || rollouts,
                exportedAt: new Date().toISOString(),
                exportVersion: '1.0'
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `session_${targetSessionId}_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('❌ Error exporting session:', error);
            throw error;
        }
    }

    /**
     * Delete session (if supported by backend)
     */
    async deleteSession(sessionId) {
        try {
            // This would need backend support
            console.warn('⚠️ Session deletion not implemented in backend');
            return { error: 'Session deletion not supported' };
        } catch (error) {
            console.error('❌ Error deleting session:', error);
            return { error: error.message };
        }
    }

    /**
     * Get session metadata for display
     */
    getSessionMetadata(sessionId = null) {
        const targetSessionId = sessionId || this.state.get('currentSessionId');
        const availableSessions = this.state.get('availableSessions');

        if (targetSessionId === 'current') {
            return {
                id: 'current',
                label: 'Current Session',
                isCurrent: true,
                canExport: true,
                canDelete: false
            };
        }

        const session = availableSessions.find(s => s.session_id === targetSessionId);
        if (session) {
            return {
                id: session.session_id,
                label: this.formatSessionLabel(session),
                modelName: session.model_name || session.model,
                numExamples: session.num_examples,
                startedAt: session.started_at || session.timestamp,
                isCurrent: false,
                canExport: true,
                canDelete: false, // Would need backend support
                session
            };
        }

        return null;
    }

    /**
     * Refresh session list
     */
    async refreshSessions() {
        await this.loadAvailableSessions();
    }

    /**
     * Set up session selector event listener
     */
    setupSessionSelector() {
        const selector = document.getElementById('session-selector');
        if (selector) {
            selector.addEventListener('change', this.switchSession);
        }
    }

    /**
     * Clean up event listeners
     */
    cleanup() {
        const selector = document.getElementById('session-selector');
        if (selector) {
            selector.removeEventListener('change', this.switchSession);
        }
    }
}

// Export for use in other modules
window.SessionManager = SessionManager;
