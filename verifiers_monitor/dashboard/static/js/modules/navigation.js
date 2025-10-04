/**
 * Example navigation and filtering functionality
 */
class NavigationManager {
    constructor(state, apiClient, domUtils) {
        this.state = state;
        this.api = apiClient;
        this.dom = domUtils;

        // Bind methods to preserve context
        this.navigateExample = this.navigateExample.bind(this);
        this.setFilter = this.setFilter.bind(this);
        this.renderCurrentExample = this.renderCurrentExample.bind(this);

        // Subscribe to state changes
        this.state.subscribe('currentExampleIndex', () => this.renderCurrentExample());
        this.state.subscribe('filteredData', () => this.renderCurrentExample());
        this.state.subscribe('currentFilter', () => this.updateFilterButtons());

        // Set up event delegation for collapsible sections (only once, survives re-renders)
        this.setupCollapsibleDelegation();

        // Bind pagination methods
        this.showMoreFailed = this.showMoreFailed.bind(this);
        this.showMorePassed = this.showMorePassed.bind(this);
        this.toggleShowAll = this.toggleShowAll.bind(this);

        // Track if we've calculated initial page size
        this.hasCalculatedInitialPageSize = false;
    }

    /**
     * Calculate optimal items per page based on viewport size
     */
    calculateInitialItemsPerPage() {
        const container = document.querySelector('.example-list-wrapper');
        if (!container) {
            return 20; // fallback if container doesn't exist yet
        }

        const containerHeight = container.clientHeight || 600;
        const itemHeight = 41; // Height of each list item
        const headerHeight = 36; // Height of section headers
        const padding = 20; // Extra padding/spacing

        // Calculate available height (accounting for 2 headers: failed + passed)
        const availableHeight = containerHeight - (headerHeight * 2) - padding;
        const itemsFit = Math.floor(availableHeight / itemHeight);

        // Clamp between 10 and 50
        const optimalCount = Math.max(10, Math.min(itemsFit, 50));

        return optimalCount;
    }

    /**
     * Initialize pagination with calculated page size (called once)
     */
    initializePaginationSize() {
        if (this.hasCalculatedInitialPageSize) {
            return; // Already calculated
        }

        const optimalCount = this.calculateInitialItemsPerPage();
        this.state.set('pagination.failedItemsPerPage', optimalCount);
        this.state.set('pagination.passedItemsPerPage', optimalCount);
        this.hasCalculatedInitialPageSize = true;

    }

    setupCollapsibleDelegation() {
        // Use document-level event delegation (always works, survives all re-renders)
        document.addEventListener('click', (e) => {
            // Check if click is on observability toggle
            const observabilityToggle = e.target.closest('.observability-toggle');
            if (observabilityToggle) {
                e.preventDefault();
                e.stopPropagation();
                const section = observabilityToggle.closest('[data-collapsible]');
                if (section) {
                    section.classList.toggle('collapsed');
                }
                return;
            }

            // Legacy: Check if click is on old reward-group-collapsible header
            const header = e.target.closest('.reward-group-collapsible');
            if (header) {
                e.preventDefault();
                e.stopPropagation();
                const group = header.closest('.reward-group');
                if (group) {
                    const wasCollapsed = group.classList.contains('collapsed');
                    group.classList.toggle('collapsed');
                }
            }
        }, { capture: true }); // Use capture phase to ensure we catch it first
    }

    /**
     * Load and update example data
     */
    async updateExampleNavigator() {
        try {

            const sessionId = this.state.get('currentSessionId');
            const { data: rollouts } = await this.api.getEvaluationHistory({
                limit: 1000,
                sessionId
            });

            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.showNoExamples();
                return;
            }

            // Group rollouts by example_number and calculate aggregate statistics
            const groupedByExample = {};
            rollouts.forEach(rollout => {
                const exampleNum = rollout.example_number;
                if (!groupedByExample[exampleNum]) {
                    groupedByExample[exampleNum] = {
                        example_number: exampleNum,
                        rollouts: [],
                        aggregate: {
                            count: 0,
                            sum: 0,
                            min: Infinity,
                            max: -Infinity,
                            rewards: []
                        }
                    };
                }

                // Store rollout
                groupedByExample[exampleNum].rollouts.push(rollout);

                // Update aggregate stats
                const reward = rollout.reward || 0;
                groupedByExample[exampleNum].aggregate.count++;
                groupedByExample[exampleNum].aggregate.sum += reward;
                groupedByExample[exampleNum].aggregate.min = Math.min(groupedByExample[exampleNum].aggregate.min, reward);
                groupedByExample[exampleNum].aggregate.max = Math.max(groupedByExample[exampleNum].aggregate.max, reward);
                groupedByExample[exampleNum].aggregate.rewards.push(reward);
            });

            // Calculate final aggregate stats for each example
            Object.values(groupedByExample).forEach(example => {
                const agg = example.aggregate;
                agg.avg = agg.sum / agg.count;
                agg.pass_rate = agg.rewards.filter(r => r > 0.5).length / agg.count;

                // Calculate variance
                const mean = agg.avg;
                const squaredDiffs = agg.rewards.map(r => Math.pow(r - mean, 2));
                agg.variance = squaredDiffs.reduce((a, b) => a + b, 0) / agg.count;
                agg.std_dev = Math.sqrt(agg.variance);

                // Copy first rollout's metadata (question, prompt) to example level
                if (example.rollouts.length > 0) {
                    const first = example.rollouts[0];
                    example.prompt = first.prompt;
                    example.task = first.task;
                    example.answer = first.answer;
                }

                // Sort rollouts by reward (worst first for debugging)
                example.rollouts.sort((a, b) => (a.reward || 0) - (b.reward || 0));
            });

            // Convert to array and sort by worst performance (failure-first)
            const exampleData = Object.values(groupedByExample)
                .sort((a, b) => a.aggregate.min - b.aggregate.min);

            // Update state
            this.state.set('exampleData', exampleData);
            this.state.updateFilteredData();

            // Update UI
            this.updateFilterControls();
            this.renderCurrentExample();


        } catch (error) {
            console.error('❌ Error updating example navigator:', error);
            this.showError('Failed to load examples');
        }
    }

    /**
     * Navigate to next/previous example
     */
    navigateExample(direction) {

        const success = this.state.navigateExample(direction);
        if (success) {
            const newIndex = this.state.get('currentExampleIndex');
        } else {
        }
    }

    /**
     * Set filter and update display
     */
    setFilter(filterType) {
        this.state.setFilter(filterType);
        // Reset both sections to optimal calculated size when filtering
        const optimalCount = this.calculateInitialItemsPerPage();
        this.state.set('pagination.failedItemsPerPage', optimalCount);
        this.state.set('pagination.passedItemsPerPage', optimalCount);
        this.updateFilterButtons();
    }

    /**
     * Show more failed examples
     */
    showMoreFailed() {
        const current = this.state.get('pagination.failedItemsPerPage');
        this.state.set('pagination.failedItemsPerPage', current + 10);
        this.renderCurrentExample();
    }

    /**
     * Show more passed examples
     */
    showMorePassed() {
        const current = this.state.get('pagination.passedItemsPerPage');
        this.state.set('pagination.passedItemsPerPage', current + 10);
        this.renderCurrentExample();
    }

    /**
     * Toggle between paginated and "Show All" mode
     */
    toggleShowAll() {
        const enabled = this.state.get('pagination.enabled');
        this.state.set('pagination.enabled', !enabled);
        if (enabled) {
            // If re-enabling pagination, reset to optimal calculated size
            const optimalCount = this.calculateInitialItemsPerPage();
            this.state.set('pagination.failedItemsPerPage', optimalCount);
            this.state.set('pagination.passedItemsPerPage', optimalCount);
        }
        this.renderCurrentExample();
    }


    /**
     * Filter examples by variance level and navigate to Examples tab
     */
    filterByVariance(varianceLevel) {

        // Navigate to Examples tab
        window.location.hash = '#examples';

        // Get current example data
        const exampleData = this.state.get('exampleData');
        if (!exampleData || exampleData.length === 0) {
            return;
        }

        // Filter examples based on variance level
        const filtered = exampleData.filter(example => {
            // Skip single rollouts
            if (!example.aggregate || example.aggregate.count < 2) {
                return false;
            }

            const stdDev = example.aggregate.std_dev;

            switch (varianceLevel) {
                case 'high':
                    return stdDev > 0.3;
                case 'medium':
                    return stdDev >= 0.1 && stdDev <= 0.3;
                case 'low':
                    return stdDev < 0.1;
                default:
                    return false;
            }
        });


        // Update filtered data in state
        this.state.set('filteredData', filtered);
        this.state.set('currentExampleIndex', 0);

        // Update filter buttons to show custom filter
        this.state.set('currentFilter', `variance-${varianceLevel}`);
        this.updateFilterButtons();
    }

    /**
     * Update filter control buttons
     */
    updateFilterControls() {
        const exampleData = this.state.get('exampleData');
        if (exampleData.length === 0) return;

        // Count examples by type using aggregate stats
        const failedExamples = exampleData.filter(e => {
            const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
            return minReward <= 0.5;
        });
        const passedExamples = exampleData.filter(e => {
            const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
            return minReward > 0.5;
        });

        // Update counts
        this.dom.updateText('count-all', exampleData.length);
        this.dom.updateText('count-failed', failedExamples.length);
        this.dom.updateText('count-passed', passedExamples.length);

        // Show filter controls
        this.dom.show('filter-controls');

        // Analyze and display failure types
        this.updateFailureSummary(failedExamples);
    }

    /**
     * Update failure type analysis
     */
    updateFailureSummary(failedExamples) {
        if (failedExamples.length === 0) {
            this.dom.updateHTML('failure-summary', '🎉 No failures detected!');
            return;
        }

        const failureTypes = {
            parsing: 0,
            format: 0,
            logic: 0,
            tool: 0
        };

        failedExamples.forEach(example => {
            if (example.parsing_analysis) {
                const parsing = example.parsing_analysis;
                if (!parsing.parsing_successful) {
                    failureTypes.parsing++;
                } else if (!parsing.format_compliance) {
                    failureTypes.format++;
                } else {
                    failureTypes.logic++;
                }
            } else {
                failureTypes.logic++;
            }

            if (example.response_analysis?.tool_calls_made > example.response_analysis?.tool_calls_successful) {
                failureTypes.tool++;
            }
        });

        let summaryHtml = '';
        if (failureTypes.parsing > 0) summaryHtml += `<span class="failure-type">Parsing: ${failureTypes.parsing}</span>`;
        if (failureTypes.format > 0) summaryHtml += `<span class="failure-type">Format: ${failureTypes.format}</span>`;
        if (failureTypes.logic > 0) summaryHtml += `<span class="failure-type">Logic: ${failureTypes.logic}</span>`;
        if (failureTypes.tool > 0) summaryHtml += `<span class="failure-type">Tool: ${failureTypes.tool}</span>`;

        this.dom.updateHTML('failure-summary', summaryHtml || 'No specific failure patterns detected');
    }

    /**
     * Update filter button states
     */
    updateFilterButtons() {
        const currentFilter = this.state.get('currentFilter');

        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
            const btnFilter = btn.getAttribute('data-filter');
            if (btnFilter === currentFilter) {
                btn.classList.add('active');
            }
        });
    }

    /**
     * Render the current example (NEW: List + Detail view)
     */
    renderCurrentExample() {
        const filteredData = this.state.get('filteredData');
        const currentIndex = this.state.get('currentExampleIndex');

        if (filteredData.length === 0) {
            this.showNoMatchingExamples();
            return;
        }

        // Render list view
        this.renderExampleList(filteredData, currentIndex);

        // Render detail view
        const example = filteredData[currentIndex];
        if (example) {
            this.renderExampleDetail(example, currentIndex, filteredData.length);
        }
    }


    /**
     * Render compact example list with pagination (UPDATED)
     */
    renderExampleList(examples, selectedIndex) {
        // Calculate optimal page size on first render
        this.initializePaginationSize();

        // Group examples by pass/fail status (using aggregate stats if available)
        const failed = examples.filter(e => {
            const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
            return minReward <= 0.5;
        });
        const passed = examples.filter(e => {
            const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
            return minReward > 0.5;
        });

        // Render with pagination
        this.renderDirectly(failed, passed, examples, selectedIndex);
    }

    /**
     * Render list directly with per-section pagination
     */
    renderDirectly(failed, passed, examples, selectedIndex) {
        const paginationEnabled = this.state.get('pagination.enabled');
        const failedItemsPerPage = this.state.get('pagination.failedItemsPerPage');
        const passedItemsPerPage = this.state.get('pagination.passedItemsPerPage');

        // Paginate failed examples (show first 10, then 20, etc.)
        const failedPaginated = paginationEnabled
            ? failed.slice(0, failedItemsPerPage)
            : failed;

        // Paginate passed examples (show first 10, then 20, etc.)
        const passedPaginated = paginationEnabled
            ? passed.slice(0, passedItemsPerPage)
            : passed;

        let listHtml = '<div class="example-list">';

        // Failures section (show paginated)
        if (failedPaginated.length > 0) {
            listHtml += `
                <div class="example-list-group">
                    <div class="list-group-header">
                        <span class="group-icon">⚠️</span>
                        <span class="group-title">FAILURES</span>
                        <span class="group-count">${failed.length}</span>
                    </div>
            `;

            failedPaginated.forEach((example) => {
                const globalIndex = examples.findIndex(e => e.example_number === example.example_number);
                const isSelected = globalIndex === selectedIndex;
                listHtml += this.renderListItem(example, globalIndex, isSelected, true);
            });

            // Add "Show More" button for failed examples if needed
            if (paginationEnabled && failedPaginated.length < failed.length) {
                const remaining = failed.length - failedPaginated.length;
                listHtml += `
                    <div class="list-show-more" onclick="window.navigationManager.showMoreFailed()">
                        Show ${remaining} more failed example${remaining !== 1 ? 's' : ''}...
                    </div>
                `;
            }

            listHtml += '</div>';
        }

        // Passed section (show paginated)
        if (passedPaginated.length > 0) {
            listHtml += `
                <div class="example-list-group">
                    <div class="list-group-header">
                        <span class="group-icon">✅</span>
                        <span class="group-title">PASSED</span>
                        <span class="group-count">${passed.length}</span>
                    </div>
            `;

            passedPaginated.forEach((example) => {
                const globalIndex = examples.findIndex(e => e.example_number === example.example_number);
                const isSelected = globalIndex === selectedIndex;
                listHtml += this.renderListItem(example, globalIndex, isSelected, false);
            });

            // Add "Show More" button for passed examples if needed
            if (paginationEnabled && passedPaginated.length < passed.length) {
                const remaining = passed.length - passedPaginated.length;
                listHtml += `
                    <div class="list-show-more" onclick="window.navigationManager.showMorePassed()">
                        Show ${remaining} more passed example${remaining !== 1 ? 's' : ''}...
                    </div>
                `;
            }

            listHtml += '</div>';
        }

        listHtml += '</div>';

        this.dom.updateHTML('example-list-container', listHtml);
    }

    /**
     * Render a single list item (minimal design)
     */
    renderListItem(example, index, isSelected, isFailed) {
        const questionPreview = this.getQuestionPreview(example);
        const fullQuestion = this.getFullQuestionText(example);
        const statusIcon = isFailed ? '⚫' : '✅';

        // Calculate average response time from rollouts
        let responseTime = '';
        if (example.rollouts && example.rollouts.length > 0) {
            const avgTime = example.rollouts.reduce((sum, r) => sum + (r.rollout_time || 0), 0) / example.rollouts.length;
            responseTime = avgTime > 0 ? `${avgTime.toFixed(1)}s` : '';
        }

        return `
            <div class="example-list-item ${isSelected ? 'selected' : ''} ${isFailed ? 'failed' : 'passed'}"
                 onclick="window.navigationManager.selectExample(${index})"
                 title="${this.dom.escapeHtml(fullQuestion)}">
                <span class="item-status">${statusIcon}</span>
                <span class="item-id">#${example.example_number}</span>
                <span class="item-preview">${questionPreview}</span>
                ${responseTime ? `<span class="item-time">${responseTime}</span>` : ''}
            </div>
        `;
    }

    /**
     * Get full question text (for tooltip)
     */
    getFullQuestionText(example) {
        try {
            let prompt = example.prompt;

            // Parse JSON string if needed
            if (typeof prompt === 'string') {
                try {
                    prompt = JSON.parse(prompt);
                } catch {
                    return prompt || '[No question]';
                }
            }

            // Chat-style messages
            if (Array.isArray(prompt)) {
                const userMsg = prompt.find(m => m.role === 'user');
                if (userMsg) {
                    if (Array.isArray(userMsg.content)) {
                        const textPart = userMsg.content.find(c => c.type === 'text');
                        if (textPart?.text) return textPart.text;
                    }
                    if (typeof userMsg.content === 'string') {
                        return userMsg.content;
                    }
                }
                const firstMsg = prompt[0];
                if (firstMsg?.content && typeof firstMsg.content === 'string') {
                    return firstMsg.content;
                }
            }

            // Multi-turn or dataset question
            if (example.info?.questions?.[0]) return example.info.questions[0];
            if (example.question) return example.question;

            return '[No question]';
        } catch (error) {
            return '[Preview error]';
        }
    }

    /**
     * Extract question/task preview from prompt (universal across all env types)
     *
     * Design principles:
     * 1. Extract user intent, not system instructions
     * 2. Handle: chat, completion, multimodal, multi-turn
     * 3. Graceful degradation with informative fallbacks
     * 4. No assumptions about structure
     *
     * Priority order:
     * 1. User message from chat prompt (most common)
     * 2. Completion-style string prompt
     * 3. Multi-turn info.questions
     * 4. Dataset question field
     */
    getQuestionPreview(example) {
        try {
            let prompt = example.prompt;

            // Parse JSON string if needed
            if (typeof prompt === 'string') {
                try {
                    prompt = JSON.parse(prompt);
                } catch {
                    // Not JSON - treat as completion-style prompt (raw string)
                    return this.truncateAtWord(prompt);
                }
            }

            // Chat-style messages (array of message objects)
            if (Array.isArray(prompt)) {
                // Find first user message (skip system prompts)
                const userMsg = prompt.find(m => m.role === 'user');
                if (userMsg) {
                    // Multimodal content (array of content items)
                    if (Array.isArray(userMsg.content)) {
                        // Extract text part from multimodal content
                        const textPart = userMsg.content.find(c => c.type === 'text');
                        if (textPart?.text) {
                            return this.truncateAtWord(textPart.text);
                        }
                        // No text found - check for images
                        const hasImage = userMsg.content.some(c => c.type === 'image_url');
                        return hasImage ? '[Multimodal: Image + Text]' : '[Multimodal]';
                    }

                    // Standard text content
                    if (typeof userMsg.content === 'string') {
                        return this.truncateAtWord(userMsg.content);
                    }
                }

                // Fallback: no user message found, try first message
                const firstMsg = prompt[0];
                if (firstMsg?.content && typeof firstMsg.content === 'string') {
                    return this.truncateAtWord(firstMsg.content);
                }
            }

            // Multi-turn: check info.questions
            if (example.info?.questions?.[0]) {
                return this.truncateAtWord(example.info.questions[0]);
            }

            // Dataset question field (direct access)
            if (example.question) {
                return this.truncateAtWord(example.question);
            }

            return '[No preview]';
        } catch (error) {
            console.warn('⚠️ Error extracting question preview:', error, example);
            return '[Preview error]';
        }
    }

    /**
     * Truncate text at word boundary or first sentence (semantic truncation)
     * Updated to prefer first sentence, then 200 chars
     */
    truncateAtWord(text, maxLength = 200) {
        if (!text || typeof text !== 'string') return '';

        // Remove XML/HTML tags for cleaner preview
        text = text.replace(/<[^>]*>/g, ' ').trim();

        // Already short enough
        if (text.length <= maxLength) return this.dom.escapeHtml(text);

        // Try to break at first sentence (. ! ?)
        const sentenceEndRegex = /[.!?]\s+/g;
        let match;
        while ((match = sentenceEndRegex.exec(text)) !== null) {
            const endPos = match.index + match[0].length;
            // If first sentence is reasonable length (20% to 100% of maxLength)
            if (endPos >= maxLength * 0.2 && endPos <= maxLength) {
                return this.dom.escapeHtml(text.substring(0, endPos).trim());
            }
            // First sentence is too long, break below
            if (endPos > maxLength) break;
        }

        // Fall back to word boundary within maxLength
        const truncated = text.substring(0, maxLength);
        const lastSpace = truncated.lastIndexOf(' ');
        const minAcceptable = maxLength * 0.6;

        const finalText = (lastSpace > minAcceptable)
            ? truncated.substring(0, lastSpace)
            : truncated;

        return this.dom.escapeHtml(finalText) + '...';
    }

    /**
     * Build rollout context summary (shows all rollout statuses)
     */
    buildRolloutContext(example, selectedIndex) {
        if (!example.rollouts || example.rollouts.length <= 1) {
            return ''; // No context needed for single rollout
        }

        const rollouts = example.rollouts;
        const passedCount = rollouts.filter(r => r.reward > 0.5).length;
        const failedCount = rollouts.length - passedCount;

        // Build status icons for all rollouts
        const statusIcons = rollouts.map((r, idx) => {
            const icon = r.reward > 0.5 ? '✅' : '❌';
            const isSelected = idx === selectedIndex;
            return isSelected ? `<strong>${icon}</strong>` : icon;
        }).join(' ');

        // Generate insight message
        let insight = '';
        if (passedCount === rollouts.length) {
            insight = 'All passed';
        } else if (failedCount === rollouts.length) {
            insight = 'All failed';
        } else {
            insight = `${passedCount}/${rollouts.length} passed`;
        }

        return `
            <div class="status-rollout-context">
                <span class="context-label">Other rollouts:</span>
                <span class="context-icons">${statusIcons}</span>
                <span class="context-separator">•</span>
                <span class="context-insight">${insight}</span>
            </div>
        `;
    }

    /**
     * Select a specific example from list (NEW)
     */
    selectExample(index) {
        this.state.set('currentExampleIndex', index);
    }

    /**
     * Select a specific rollout within an example
     */
    selectRollout(exampleNumber, rolloutIndex) {
        const rolloutIndices = this.state.get('selectedRolloutIndices') || {};
        rolloutIndices[exampleNumber] = rolloutIndex;
        this.state.set('selectedRolloutIndices', rolloutIndices);

        // Re-render the detail view
        this.renderCurrentExample();
    }

    /**
     * Render detailed example view with multi-rollout support
     */
    async renderExampleDetail(example, currentIndex, totalExamples) {
        // Check if this is a multi-rollout example
        const isMultiRollout = example.aggregate && example.aggregate.count > 1;

        // Get the current rollout (default to worst rollout for debugging)
        let currentRollout = example;
        let selectedRolloutIndex = 0;

        if (isMultiRollout) {
            // Store selected rollout index in state (default to 0 = worst)
            if (!this.state.get('selectedRolloutIndices')) {
                this.state.set('selectedRolloutIndices', {});
            }
            const rolloutIndices = this.state.get('selectedRolloutIndices');
            selectedRolloutIndex = rolloutIndices[example.example_number] || 0;
            currentRollout = example.rollouts[selectedRolloutIndex];
        }

        const reward = currentRollout.reward || 0;
        const rewardClass = reward > 0.5 ? 'stat-positive' : 'stat-neutral';


        // Format prompt and completion for CURRENT rollout
        const promptHtml = this.formatMessages(currentRollout.prompt || example.prompt, 'No prompt available');
        const completionHtml = this.formatMessages(currentRollout.completion, 'No completion available');

        // Fetch reward breakdown for CURRENT rollout (async)
        const rewardBreakdownHtml = await this.renderRewardBreakdown(currentRollout);

        // Build rollout context summary
        const rolloutContext = this.buildRolloutContext(example, selectedRolloutIndex);

        // Determine status
        const statusIcon = reward > 0.5 ? '✅' : '❌';
        const statusText = reward > 0.5 ? 'Passed' : 'Failed';
        const statusClass = reward > 0.5 ? 'status-success' : 'status-failed';

        // Build rollout navigation buttons for status bar (if multi-rollout)
        let rolloutNavButtons = '';
        if (isMultiRollout) {
            const canGoPrev = selectedRolloutIndex > 0;
            const canGoNext = selectedRolloutIndex < example.rollouts.length - 1;
            rolloutNavButtons = `
                <button class="rollout-switch-btn"
                        onclick="window.navigationManager.selectRollout(${example.example_number}, ${selectedRolloutIndex - 1})"
                        ${!canGoPrev ? 'disabled' : ''}
                        title="Previous rollout">
                    ◀
                </button>
                <button class="rollout-switch-btn"
                        onclick="window.navigationManager.selectRollout(${example.example_number}, ${selectedRolloutIndex + 1})"
                        ${!canGoNext ? 'disabled' : ''}
                        title="Next rollout">
                    ▶
                </button>
            `;
        }

        // Build detail HTML with professional layout
        const detailHtml = `
            <div class="example-detail">
                <!-- Status Bar -->
                <div class="detail-status-bar">
                    <div class="status-info">
                        <span class="status-counter">Example ${currentIndex + 1} of ${totalExamples}</span>
                        <span class="status-separator">•</span>
                        <span class="status-id">Rollout ${selectedRolloutIndex + 1} of ${example.rollouts?.length || 1}</span>
                        ${rolloutNavButtons}
                    </div>
                    <div class="status-nav">
                        <button class="status-nav-btn" onclick="window.navigationManager.navigateExample(-1)" ${currentIndex === 0 ? 'disabled' : ''}>
                            ← Previous
                        </button>
                        <button class="status-nav-btn" onclick="window.navigationManager.navigateExample(1)" ${currentIndex >= totalExamples - 1 ? 'disabled' : ''}>
                            Next →
                        </button>
                    </div>
                </div>

                <!-- Status Banner -->
                <div class="detail-status-banner ${statusClass}">
                    <div class="status-main">
                        <span class="status-icon">${statusIcon}</span>
                        <span class="status-label">${statusText}</span>
                        <span class="status-separator">•</span>
                        <span class="status-score">${reward.toFixed(2)} / 1.0</span>
                    </div>
                    ${rolloutContext}
                    </div>

                <!-- Content Grid (Prompt | Completion) -->
                <div class="detail-content-grid">
                    <div class="content-panel content-panel-prompt">
                        <div class="panel-header">
                            <span class="panel-title">PROMPT</span>
                </div>
                        <div class="panel-body">${promptHtml}</div>
                    </div>
                    <div class="content-panel content-panel-completion">
                        <div class="panel-header">
                            <span class="panel-title">COMPLETION</span>
                        </div>
                        <div class="panel-body">${completionHtml}</div>
                    </div>
                </div>

                <!-- Reward Breakdown -->
                <div class="detail-metrics">
                    ${rewardBreakdownHtml}
                </div>

                <!-- Metadata Footer -->
                <div class="detail-footer">
                    <span>${this.formatResponseTime(currentRollout.rollout_time)}</span>
                    <span class="footer-separator">•</span>
                    <span>${this.formatLength(this.calculateCompletionLength(currentRollout.completion))}</span>
                    <span class="footer-separator">•</span>
                    <span>${this.countToolCalls(currentRollout.completion)} tool calls</span>
                </div>
            </div>
        `;

        this.dom.updateHTML('example-detail-container', detailHtml);
        // Event listeners handled by delegation in constructor - no need to re-attach
    }

    /**
     * Render reward breakdown for an example (Option A: Clean, Professional, No Emojis)
     */
    async renderRewardBreakdown(example) {
        try {

            // Fetch reward system info
            const rewardInfo = await this.api.get('/api/reward-system/info');

            if (!rewardInfo || rewardInfo.error || !rewardInfo.reward_functions || rewardInfo.reward_functions.length === 0) {
                console.warn('⚠️ No reward functions available, skipping breakdown');
                return '';
            }

            const functions = rewardInfo.reward_functions;
            const weights = rewardInfo.reward_weights || [];
            const maxReward = rewardInfo.max_possible_reward || 1.0;
            const totalReward = example.reward || 0;
            const parsedMetrics = example.metrics || {};

            // Calculate percentage
            const percentage = maxReward > 0 ? (totalReward / maxReward * 100).toFixed(0) : 0;

            // Determine overall status (based on 0.5 threshold)
            let statusClass, statusSymbol, statusText;
            if (totalReward > maxReward * 0.5) {
                if (percentage >= 90) {
                    statusClass = 'reward-excellent';
                    statusSymbol = '✓';
                    statusText = 'Excellent';
                } else if (percentage >= 70) {
                    statusClass = 'reward-good';
                    statusSymbol = '✓';
                    statusText = 'Good';
                } else {
                    statusClass = 'reward-pass';
                    statusSymbol = '✓';
                    statusText = 'Pass';
                }
            } else if (percentage >= 30) {
                statusClass = 'reward-partial';
                statusSymbol = '~';
                statusText = 'Partial';
            } else {
                statusClass = 'reward-fail';
                statusSymbol = '✗';
                statusText = 'Fail';
            }

            // Check if we have metric data
            const hasMetrics = Object.keys(parsedMetrics).length > 0;

            if (!hasMetrics) {
                return `
                    <div class="reward-breakdown ${statusClass}">
                        <div class="reward-header">
                            <span class="reward-symbol">${statusSymbol}</span>
                            <span class="reward-value">${totalReward.toFixed(2)} / ${maxReward.toFixed(1)}</span>
                            <span class="reward-percent">(${percentage}%)</span>
                        </div>
                        <div class="reward-body">
                            <p class="reward-no-data">Detailed metrics not available</p>
                        </div>
                    </div>
                `;
            }

            // Group functions by weight category
            const primaryFuncs = [];    // weight >= 0.5
            const secondaryFuncs = [];  // 0 < weight < 0.5
            const observabilityFuncs = []; // weight === 0.0

            functions.forEach((funcName, idx) => {
                const weight = weights[idx] !== undefined ? weights[idx] : 1.0;
                const score = parsedMetrics[funcName];
                const hasScore = score !== undefined && score !== null;
                const contribution = hasScore ? (weight * score) : 0;
                const maxContribution = weight;

                const funcData = { funcName, weight, score: hasScore ? score : 0, contribution, hasScore, maxContribution };

                if (weight >= 0.5) {
                    primaryFuncs.push(funcData);
                } else if (weight > 0) {
                    secondaryFuncs.push(funcData);
                } else {
                    observabilityFuncs.push(funcData);
                }
            });

            // Detect metric type for observability functions
            const detectMetricType = (funcName, score) => {
                // Check function name patterns
                if (funcName.includes('_count') || funcName.includes('_calls')) {
                    return 'count';
                }
                if (funcName.includes('_reward') || funcName.includes('_rate')) {
                    return 'rate';
                }

                // Check value patterns
                if (Number.isInteger(score) && score >= 0) {
                    return 'count';
                }
                if (score >= 0 && score <= 1) {
                    return 'rate';
                }

                return 'value';
            };

            // Render a weighted function (shows contribution)
            const renderWeightedFunc = (funcData) => {
                const { funcName, weight, score, contribution, hasScore, maxContribution } = funcData;

                if (!hasScore) {
                    return `
                        <div class="reward-func reward-func-missing">
                            <span class="func-symbol">–</span>
                            <span class="func-name">${funcName}</span>
                            <span class="func-score">N/A</span>
                        </div>
                    `;
                }

                // Determine status based on contribution (percentage of max possible contribution)
                const contributionPercent = maxContribution > 0 ? (contribution / maxContribution) : 0;
                let funcClass, funcSymbol;

                if (contributionPercent >= 0.7) {
                    funcClass = 'reward-func-pass';
                    funcSymbol = '✓';
                } else if (contributionPercent >= 0.3) {
                    funcClass = 'reward-func-partial';
                    funcSymbol = '~';
                } else {
                    funcClass = 'reward-func-fail';
                    funcSymbol = '✗';
                }

                return `
                    <div class="reward-func ${funcClass}">
                        <span class="func-symbol">${funcSymbol}</span>
                        <span class="func-name">${funcName}</span>
                        <span class="func-contribution">${contribution.toFixed(2)}</span>
                        <span class="func-formula">(${weight.toFixed(2)} × ${score.toFixed(2)})</span>
                    </div>
                `;
            };

            // Render an observability metric (shows raw value with type label)
            const renderObservabilityFunc = (funcData) => {
                const { funcName, score, hasScore } = funcData;

                if (!hasScore) {
                    return '';
                }

                const metricType = detectMetricType(funcName, score);

                return `
                    <div class="reward-metric">
                        <span class="metric-name">${funcName}</span>
                        <span class="metric-value">${score.toFixed(2)}</span>
                        <span class="metric-type">${metricType}</span>
                    </div>
                `;
            };

            // Build HTML structure with visual hierarchy (no explicit labels)
            let sectionsHTML = '';

            // Primary functions - full prominence, no label
            if (primaryFuncs.length > 0) {
                sectionsHTML += `
                    <div class="reward-section reward-section-primary">
                        ${primaryFuncs.map(f => renderWeightedFunc(f)).join('')}
                    </div>
                `;
            }

            // Secondary functions - visual de-emphasis via CSS
            if (secondaryFuncs.length > 0) {
                sectionsHTML += `
                    <div class="reward-section reward-section-secondary">
                        ${secondaryFuncs.map(f => renderWeightedFunc(f)).join('')}
                    </div>
                `;
            }

            // Observability metrics - minimal toggle, no explicit label
            if (observabilityFuncs.length > 0) {
                const autoCollapse = observabilityFuncs.length > 3;
                sectionsHTML += `
                    <div class="reward-section reward-section-observability ${autoCollapse ? 'collapsed' : ''}" data-collapsible>
                        <div class="observability-toggle">
                            <span class="toggle-icon">▸</span>
                            <span class="toggle-count">${observabilityFuncs.length} observability metrics</span>
                        </div>
                        <div class="section-content">
                            ${observabilityFuncs.map(f => renderObservabilityFunc(f)).join('')}
                        </div>
                    </div>
                `;
            }


            // Assemble final HTML
            return `
                <div class="reward-breakdown ${statusClass}">
                    <div class="reward-header">
                        <span class="reward-label">REWARD:</span>
                        <span class="reward-value">${totalReward.toFixed(2)} / ${maxReward.toFixed(1)}</span>
                        <span class="reward-percent">(${percentage}%)</span>
                        <span class="reward-status"><span class="status-symbol">${statusSymbol}</span>${statusText}</span>
                    </div>
                    <div class="reward-body">
                        ${sectionsHTML}
                    </div>
                </div>
            `;
        } catch (error) {
            console.error('❌ Error rendering reward breakdown:', error);
            return '';
        }
    }

    /**
     * Format messages for display (universal across all environment types)
     *
     * Design principles:
     * 1. Single Responsibility - each helper formats one thing
     * 2. Structural clarity - clear visual hierarchy
     * 3. Type-aware - handles chat, completion, tool, multimodal
     * 4. Defensive parsing - multiple fallback strategies
     */
    formatMessages(messageData, fallback) {
        if (!messageData) return fallback;

        // Parse JSON string with multiple strategies
        const parsedData = this.parseMessageData(messageData);
        if (parsedData === null) return fallback;

        // Route to appropriate formatter
        if (typeof parsedData === 'string') {
            // Completion-style: raw text
            return this.formatCompletionMessage(parsedData);
        }

        if (Array.isArray(parsedData)) {
            // Chat-style: array of messages
            return this.formatChatMessages(parsedData);
        }

        return fallback;
    }

    /**
     * Parse message data with defensive strategies
     */
    parseMessageData(data) {
        if (typeof data !== 'string') {
            return data; // Already parsed
        }

        // Strategy 1: Direct JSON parse
        try {
            return JSON.parse(data);
            } catch (e) {
            // Strategy 2: Unescape and parse
                try {
                const unescaped = data.replace(/\\"/g, '"').replace(/\\\\/g, '\\');
                return JSON.parse(unescaped);
                } catch (e2) {
                // Strategy 3: Treat as raw completion string
                return data;
            }
        }
    }

    /**
     * Format completion-style message (simple string)
     */
    formatCompletionMessage(text) {
        return `<div class="msg msg-completion">${this.dom.escapeHtml(text)}</div>`;
    }

    /**
     * Format chat-style messages (array of message objects)
     */
    formatChatMessages(messages) {
        if (!Array.isArray(messages) || messages.length === 0) {
            return '<div class="msg-empty">No messages</div>';
        }

        return messages.map((msg, idx) => {
            // Skip empty messages
            if (!msg || typeof msg !== 'object') return '';

            return this.formatSingleMessage(msg, idx);
        }).filter(html => html).join('');
    }

    /**
     * Format a single message with role, content, tool calls, etc.
     */
    formatSingleMessage(msg, index) {
        const role = (msg.role || 'unknown').toUpperCase();
        const roleClass = `msg-role-${msg.role || 'unknown'}`;

        let html = `<div class="msg ${roleClass}" data-index="${index}">`;
        html += `<div class="msg-role-label">${role}</div>`;

        // Handle content
        if (msg.content) {
            html += this.formatMessageContent(msg.content);
        } else if (!msg.tool_calls && msg.role !== 'assistant') {
            // No content and no tool calls - show placeholder
            // Skip for assistant (they often have only tool calls, no content text)
            html += '<div class="msg-content msg-empty">[Empty message]</div>';
        }

        // Handle tool calls (function calling)
        if (msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
            html += '<div class="msg-tool-calls">';
            msg.tool_calls.forEach((tc, tcIdx) => {
                html += this.formatToolCall(tc, tcIdx);
            });
            html += '</div>';
        }

        // Hide tool_call_id - it's debug info, not needed for users
        // (tool results are already visually associated with their calls via indentation)

        html += '</div>';
        return html;
    }

    /**
     * Format message content (handles text, multimodal, etc.)
     */
    formatMessageContent(content) {
        // Standard text content
        if (typeof content === 'string') {
            return `<div class="msg-content">${this.dom.escapeHtml(content)}</div>`;
        }

        // Multimodal content (array of content items)
        if (Array.isArray(content)) {
            return '<div class="msg-content msg-multimodal">' +
                content.map(item => this.formatContentItem(item)).join('') +
                '</div>';
        }

        // Unknown content type
        return `<div class="msg-content msg-unknown">${this.dom.escapeHtml(JSON.stringify(content))}</div>`;
    }

    /**
     * Format a single content item (text, image, etc.)
     */
    formatContentItem(item) {
        if (!item || typeof item !== 'object') return '';

        switch (item.type) {
            case 'text':
                return `<div class="content-text">${this.dom.escapeHtml(item.text || '')}</div>`;

            case 'image_url':
                const url = item.image_url?.url || '';
                const isBase64 = url.startsWith('data:');
                if (isBase64) {
                    return `<div class="content-image"><img src="${url}" alt="Embedded image" style="max-width: 200px; max-height: 200px; border-radius: 4px;" /></div>`;
                }
                return `<div class="content-image">[Image: ${this.dom.escapeHtml(url.substring(0, 50))}...]</div>`;

            default:
                return `<div class="content-unknown">[${item.type || 'unknown'}]</div>`;
        }
    }

    /**
     * Format a tool call (function calling)
     */
    formatToolCall(toolCall, index) {
        if (!toolCall?.function) return '';

        const name = toolCall.function.name || 'unknown';
        const args = toolCall.function.arguments || '{}';

        // Pretty-print JSON args
        let argsFormatted;
        try {
            const argsObj = typeof args === 'string' ? JSON.parse(args) : args;
            argsFormatted = JSON.stringify(argsObj, null, 2);
        } catch {
            argsFormatted = String(args);
        }

        return `
            <div class="msg-tool-call" data-index="${index}">
                <div class="tool-call-header">
                    <span class="tool-call-icon">⚡</span>
                    <span class="tool-call-name">${this.dom.escapeHtml(name)}</span>
                </div>
                <pre class="tool-call-args"><code>${this.dom.escapeHtml(argsFormatted)}</code></pre>
                                </div>
                            `;
                        }

    /**
     * Format response time for display
     */
    formatResponseTime(rolloutTime) {
        if (!rolloutTime || rolloutTime === 0) return '0s';

        if (rolloutTime < 1) {
            // Less than 1 second - show in milliseconds
            return `${(rolloutTime * 1000).toFixed(0)}ms`;
        } else {
            // 1 second or more - show in seconds
            return `${rolloutTime.toFixed(1)}s`;
        }
    }

    /**
     * Format length for display (e.g., 3200 -> "3.2k chars")
     */
    formatLength(length) {
        if (length === 0) return '0 chars';
        if (length < 1000) return `${length} chars`;
        return `${(length / 1000).toFixed(1)}k chars`;
    }

    /**
     * Calculate total character count from completion messages
     */
    calculateCompletionLength(completion) {
        if (!completion) return 0;

        try {
            // Parse if string
            const messages = typeof completion === 'string' ? JSON.parse(completion) : completion;

            if (!Array.isArray(messages)) return 0;

            // Sum up all content lengths
            let totalLength = 0;
            for (const msg of messages) {
                if (msg.content) {
                    if (typeof msg.content === 'string') {
                        totalLength += msg.content.length;
                    } else if (Array.isArray(msg.content)) {
                        // Multimodal content
                        for (const item of msg.content) {
                            if (item.text) {
                                totalLength += item.text.length;
                            }
                        }
                    }
                }

                // Include tool call arguments length
                if (msg.tool_calls) {
                    for (const tc of msg.tool_calls) {
                        if (tc.function?.arguments) {
                            totalLength += tc.function.arguments.length;
                        }
                    }
                }
            }

            return totalLength;
        } catch (e) {
            return 0;
        }
    }

    /**
     * Count tool calls in completion messages
     */
    countToolCalls(completion) {
        if (!completion) return 0;

        try {
            // Parse if string
            const messages = typeof completion === 'string' ? JSON.parse(completion) : completion;

            if (!Array.isArray(messages)) return 0;

            // Count all tool_calls across all messages
            let count = 0;
            for (const msg of messages) {
                if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
                    count += msg.tool_calls.length;
                }
            }

            return count;
        } catch (e) {
            return 0;
        }
    }

    /**
     * Get tool call count from example (legacy method for backwards compatibility)
     */
    getToolCallCount(example) {
        return example.response_analysis?.tool_calls_made || 0;
    }

    /**
     * Get parsing status indicator
     */
    getParsingStatus(example) {
        if (!example.parsing_analysis) return '';

        try {
            const parsing = typeof example.parsing_analysis === 'string'
                ? JSON.parse(example.parsing_analysis)
                : example.parsing_analysis;
            const status = parsing.parsing_successful ? '✓' : '✗';
            return ` | Parsing: ${status}`;
        } catch (e) {
            return ' | Parsing: ?';
        }
    }

    /**
     * Show "no examples" message
     */
    showNoExamples() {
        this.dom.updateHTML('example-list-container', '<div class="no-data">No examples available</div>');
        this.dom.updateHTML('example-detail-container', '<div class="no-data">No examples to display</div>');
        this.dom.hide('filter-controls');
    }

    /**
     * Show "no matching examples" message
     */
    showNoMatchingExamples() {
        this.dom.updateHTML('example-list-container', '<div class="no-data">No examples match the current filter</div>');
        this.dom.updateHTML('example-detail-container', '<div class="no-data">Try changing your filter</div>');
    }

    /**
     * Show error message
     */
    showError(message) {
        this.dom.updateHTML('example-list-container', `<div class="no-data" style="color: #ef4444;">${message}</div>`);
        this.dom.updateHTML('example-detail-container', `<div class="no-data" style="color: #ef4444;">Error loading example</div>`);
    }

    /**
     * Get navigation statistics
     */
    getNavigationStats() {
        const exampleData = this.state.get('exampleData');
        const filteredData = this.state.get('filteredData');
        const currentIndex = this.state.get('currentExampleIndex');
        const currentFilter = this.state.get('currentFilter');

        return {
            totalExamples: exampleData.length,
            filteredExamples: filteredData.length,
            currentIndex,
            currentFilter,
            hasNext: currentIndex < filteredData.length - 1,
            hasPrevious: currentIndex > 0,
            currentExample: filteredData[currentIndex] || null
        };
    }

    /**
     * Jump to specific example by number
     */
    jumpToExample(exampleNumber) {
        const filteredData = this.state.get('filteredData');
        const index = filteredData.findIndex(e => e.example_number === exampleNumber);

        if (index !== -1) {
            this.state.set('currentExampleIndex', index);
            return true;
        }
        return false;
    }

    /**
     * Export current example data
     */
    exportCurrentExample() {
        const example = this.state.getCurrentExample();
        if (example) {
            const blob = new Blob([JSON.stringify(example, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `example_${example.example_number}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    }
}

// Export for use in other modules
window.NavigationManager = NavigationManager;
