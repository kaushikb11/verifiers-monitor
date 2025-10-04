/**
 * Chart management using Plotly.js
 */
class ChartManager {
    constructor(state, apiClient) {
        this.state = state;
        this.api = apiClient;
        this.charts = new Map();
        this.currentChartType = 'line'; // Track current chart type
        this.defaultConfig = {
            displayModeBar: false,
            responsive: true
        };
        this.defaultLayout = {
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            font: {
                family: 'Inter, sans-serif',
                size: 12,
                color: '#525252'
            },
            margin: { t: 20, r: 20, b: 50, l: 50 }
        };
    }

    /**
     * Create or update progress chart with fancy example grouping
     */
    async updateProgressChart() {
        try {
            const sessionId = this.state.get('currentSessionId');
            const { data: rollouts } = await this.api.getEvaluationHistory({
                limit: 1000,
                sessionId
            });

            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.showNoDataMessage('progress-chart', 'No evaluation data yet');
                return;
            }

            // Sort by timestamp to ensure chronological order
            rollouts.sort((a, b) => a.timestamp - b.timestamp);

            // Create sequential indices
            const indices = rollouts.map((_, i) => i + 1);
            const rewards = rollouts.map(d => d.reward || 0);

            // Detect example boundaries for visual grouping
            const exampleBoundaries = [];
            const exampleLabels = [];
            let currentExample = null;
            let exampleStartIdx = 0;

            rollouts.forEach((rollout, idx) => {
                if (currentExample !== rollout.example_number) {
                    if (currentExample !== null) {
                        // Mark boundary between examples
                        exampleBoundaries.push(idx);
                        // Store label position (middle of example group)
                        const midPoint = (exampleStartIdx + idx) / 2;
                        exampleLabels.push({
                            x: midPoint,
                            example: currentExample,
                            start: exampleStartIdx,
                            end: idx - 1
                        });
                    }
                    currentExample = rollout.example_number;
                    exampleStartIdx = idx;
                }
            });

            // Add final example label
            if (currentExample !== null) {
                const midPoint = (exampleStartIdx + rollouts.length) / 2;
                exampleLabels.push({
                    x: midPoint,
                    example: currentExample,
                    start: exampleStartIdx,
                    end: rollouts.length - 1
                });
            }

            // Build shapes for alternating backgrounds and dividers
            const shapes = [];
            const annotations = [];

            // Alternating background shading
            exampleLabels.forEach((label, idx) => {
                if (idx % 2 === 0) {
                    shapes.push({
                        type: 'rect',
                        xref: 'x',
                        yref: 'paper',
                        x0: label.start + 0.5,
                        x1: label.end + 1.5,
                        y0: 0,
                        y1: 1,
                        fillcolor: '#f9fafb',
                        line: { width: 0 },
                        layer: 'below'
                    });
                }

                // Example number annotations at top - REMOVED per user request
                // annotations.push({
                //     x: label.x + 0.5,
                //     y: 1.02,
                //     xref: 'x',
                //     yref: 'paper',
                //     text: `Ex ${label.example}`,
                //     showarrow: false,
                //     font: {
                //         size: 11,
                //         color: '#737373'
                //     }
                // });
            });

            // Vertical divider lines between examples
            exampleBoundaries.forEach(boundary => {
                shapes.push({
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: boundary + 0.5,
                    x1: boundary + 0.5,
                    y0: 0,
                    y1: 1,
                    line: {
                        color: '#d1d5db',
                        width: 1,
                        dash: 'dot'
                    },
                    layer: 'below'
                });
            });

            // Success threshold line at y=0.5
            shapes.push({
                type: 'line',
                xref: 'paper',
                yref: 'y',
                x0: 0,
                x1: 1,
                y0: 0.5,
                y1: 0.5,
                line: {
                    color: '#fbbf24',
                    width: 1,
                    dash: 'dash'
                },
                layer: 'below'
            });

            const trace = {
                x: indices,
                y: rewards,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Reward',
                line: {
                    color: '#16a34a',
                    width: 2,
                    shape: 'linear'
                },
                marker: {
                    size: 8,
                    color: rewards.map(r => r > 0.5 ? '#16a34a' : '#a3a3a3'),
                    line: { color: 'white', width: 2 }
                },
                customdata: rollouts.map(d => [d.example_number, d.rollout_number]),
                hovertemplate: 'Rollout #%{x}<br>Reward: %{y:.3f}<br>Example: %{customdata[0]}<br>Rollout: %{customdata[1]}<extra></extra>'
            };

            const layout = {
                ...this.defaultLayout,
                xaxis: {
                    title: 'Sequential Rollout',
                    gridcolor: '#f5f5f5',
                    gridwidth: 1,
                    showgrid: true,
                    zeroline: false
                },
                yaxis: {
                    title: 'Reward',
                    gridcolor: '#f5f5f5',
                    gridwidth: 1,
                    showgrid: true,
                    zeroline: false,
                    range: [-0.05, 1.05]
                },
                shapes: shapes,
                annotations: annotations
            };

            await this.createChart('progress-chart', [trace], layout);

        } catch (error) {
            console.error('❌ Error updating progress chart:', error);
            this.showErrorMessage('progress-chart', 'Failed to load chart data');
        }
    }

    /**
     * Switch between chart types (line/distribution)
     */
    async switchChartType(chartType) {
        try {
            this.currentChartType = chartType;

            const sessionId = this.state.get('currentSessionId');
            const { data: rollouts } = await this.api.getEvaluationHistory({
                limit: 1000,
                sessionId
            });

            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.showNoDataMessage('progress-chart', 'No evaluation data yet');
                return;
            }

            if (chartType === 'distribution') {
                await this.createRewardDistribution('progress-chart', rollouts);
            } else {
                await this.updateProgressChart();
            }

        } catch (error) {
            console.error('❌ Error switching chart type:', error);
        }
    }

    /**
     * Create reward distribution histogram
     */
    async createRewardDistribution(containerId, rollouts) {
        try {
            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.showNoDataMessage(containerId, 'No data available');
                return;
            }

            const rewards = rollouts.map(d => d.reward || 0);

            const trace = {
                x: rewards,
                type: 'histogram',
                nbinsx: 20,
                name: 'Reward Distribution',
                marker: {
                    color: '#3b82f6',
                    opacity: 0.7,
                    line: {
                        color: '#1d4ed8',
                        width: 1
                    }
                },
                hovertemplate: 'Reward: %{x}<br>Count: %{y}<extra></extra>'
            };

            const layout = {
                ...this.defaultLayout,
                title: 'Reward Distribution',
                xaxis: { title: 'Reward' },
                yaxis: { title: 'Frequency' }
            };

            await this.createChart(containerId, [trace], layout);

        } catch (error) {
            console.error('❌ Error creating reward distribution:', error);
            this.showErrorMessage(containerId, 'Failed to create distribution chart');
        }
    }

    /**
     * Create response time scatter plot
     */
    async createResponseTimeChart(containerId, rollouts) {
        try {
            if (!Array.isArray(rollouts) || rollouts.length === 0) {
                this.showNoDataMessage(containerId, 'No data available');
                return;
            }

            const responseTimes = rollouts.map(d => (d.rollout_time || 0) * 1000); // Convert to ms
            const rewards = rollouts.map(d => d.reward || 0);
            const examples = rollouts.map(d => d.example_number || 0);

            const trace = {
                x: responseTimes,
                y: rewards,
                type: 'scatter',
                mode: 'markers',
                name: 'Response Time vs Reward',
                marker: {
                    size: 8,
                    color: rewards,
                    colorscale: 'RdYlGn',
                    showscale: true,
                    colorbar: {
                        title: 'Reward'
                    }
                },
                text: examples.map(e => `Example ${e}`),
                hovertemplate: 'Response Time: %{x}ms<br>Reward: %{y}<br>%{text}<extra></extra>'
            };

            const layout = {
                ...this.defaultLayout,
                title: 'Response Time vs Reward',
                xaxis: { title: 'Response Time (ms)' },
                yaxis: { title: 'Reward' }
            };

            await this.createChart(containerId, [trace], layout);

        } catch (error) {
            console.error('❌ Error creating response time chart:', error);
            this.showErrorMessage(containerId, 'Failed to create response time chart');
        }
    }

    /**
     * Create multi-rollout variance chart
     */
    async createVarianceChart(containerId, analysis) {
        try {
            if (!analysis || !analysis.prompt_analysis || analysis.prompt_analysis.length === 0) {
                this.showNoDataMessage(containerId, 'No multi-rollout data available');
                return;
            }

            const prompts = analysis.prompt_analysis;
            const stdDevs = prompts.map(p => p.std_dev);
            const avgRewards = prompts.map(p => p.avg_reward);
            const rolloutCounts = prompts.map(p => p.rollout_count);

            const trace = {
                x: avgRewards,
                y: stdDevs,
                type: 'scatter',
                mode: 'markers',
                name: 'Prompt Variance',
                marker: {
                    size: rolloutCounts.map(c => Math.min(c * 5, 30)), // Scale size by rollout count
                    color: stdDevs,
                    colorscale: 'Reds',
                    showscale: true,
                    colorbar: {
                        title: 'Std Dev'
                    }
                },
                text: prompts.map(p => `${p.rollout_count} rollouts`),
                hovertemplate: 'Avg Reward: %{x}<br>Std Dev: %{y}<br>%{text}<extra></extra>'
            };

            const layout = {
                ...this.defaultLayout,
                title: 'Prompt Consistency Analysis',
                xaxis: { title: 'Average Reward' },
                yaxis: { title: 'Standard Deviation' }
            };

            await this.createChart(containerId, [trace], layout);

        } catch (error) {
            console.error('❌ Error creating variance chart:', error);
            this.showErrorMessage(containerId, 'Failed to create variance chart');
        }
    }

    /**
     * Create reward function breakdown chart
     */
    async createRewardBreakdownChart(containerId, breakdown) {
        try {
            if (!breakdown || !breakdown.reward_functions) {
                this.showNoDataMessage(containerId, 'No reward function data available');
                return;
            }

            const functions = Object.keys(breakdown.reward_functions);
            const avgScores = functions.map(f => breakdown.reward_functions[f].avg_score || 0);
            const colors = avgScores.map(score => score > 0.5 ? '#16a34a' : '#ef4444');

            const trace = {
                x: functions,
                y: avgScores,
                type: 'bar',
                name: 'Reward Functions',
                marker: {
                    color: colors,
                    opacity: 0.8,
                    line: {
                        color: 'white',
                        width: 1
                    }
                },
                hovertemplate: 'Function: %{x}<br>Avg Score: %{y}<extra></extra>'
            };

            const layout = {
                ...this.defaultLayout,
                title: 'Reward Function Performance',
                xaxis: { title: 'Reward Function' },
                yaxis: { title: 'Average Score' }
            };

            await this.createChart(containerId, [trace], layout);

        } catch (error) {
            console.error('❌ Error creating reward breakdown chart:', error);
            this.showErrorMessage(containerId, 'Failed to create breakdown chart');
        }
    }

    /**
     * Generic chart creation method
     */
    async createChart(containerId, traces, layout, config = {}) {
        try {
            const element = document.getElementById(containerId);
            if (!element) {
                console.warn(`⚠️ Chart container '${containerId}' not found`);
                return;
            }

            const finalLayout = { ...this.defaultLayout, ...layout };
            const finalConfig = { ...this.defaultConfig, ...config };

            await Plotly.newPlot(containerId, traces, finalLayout, finalConfig);

            // Store chart reference
            this.charts.set(containerId, {
                traces,
                layout: finalLayout,
                config: finalConfig,
                lastUpdated: Date.now()
            });

        } catch (error) {
            console.error(`❌ Error creating chart '${containerId}':`, error);
            this.showErrorMessage(containerId, 'Failed to create chart');
        }
    }

    /**
     * Update existing chart with new data
     */
    async updateChart(containerId, traces, layout = {}) {
        try {
            const element = document.getElementById(containerId);
            if (!element) {
                console.warn(`⚠️ Chart container '${containerId}' not found`);
                return;
            }

            const existingChart = this.charts.get(containerId);
            if (existingChart) {
                const newLayout = { ...existingChart.layout, ...layout };
                await Plotly.react(containerId, traces, newLayout, existingChart.config);

                // Update stored reference
                this.charts.set(containerId, {
                    ...existingChart,
                    traces,
                    layout: newLayout,
                    lastUpdated: Date.now()
                });
            } else {
                // Create new chart if it doesn't exist
                await this.createChart(containerId, traces, layout);
            }

        } catch (error) {
            console.error(`❌ Error updating chart '${containerId}':`, error);
        }
    }

    /**
     * Remove chart and clean up
     */
    removeChart(containerId) {
        try {
            const element = document.getElementById(containerId);
            if (element) {
                Plotly.purge(containerId);
            }
            this.charts.delete(containerId);
        } catch (error) {
            console.error(`❌ Error removing chart '${containerId}':`, error);
        }
    }

    /**
     * Resize all charts (useful for responsive layouts)
     */
    resizeAllCharts() {
        this.charts.forEach((chart, containerId) => {
            try {
                Plotly.Plots.resize(containerId);
            } catch (error) {
                console.error(`❌ Error resizing chart '${containerId}':`, error);
            }
        });
    }

    /**
     * Show "no data" message in chart container
     */
    showNoDataMessage(containerId, message = 'No data available') {
        const element = document.getElementById(containerId);
        if (element) {
            element.innerHTML = `<div class="no-data">${message}</div>`;
        }
    }

    /**
     * Show error message in chart container
     */
    showErrorMessage(containerId, message = 'Error loading chart') {
        const element = document.getElementById(containerId);
        if (element) {
            element.innerHTML = `<div class="no-data" style="color: #ef4444;">${message}</div>`;
        }
    }

    /**
     * Get chart data for export
     */
    getChartData(containerId) {
        return this.charts.get(containerId) || null;
    }

    /**
     * Clean up all charts
     */
    cleanup() {
        this.charts.forEach((chart, containerId) => {
            this.removeChart(containerId);
        });
        this.charts.clear();
    }
}

// Export for use in other modules
window.ChartManager = ChartManager;
