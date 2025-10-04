/**
 * API client with consistent error handling and response formatting
 */
class APIClient {
    constructor(state) {
        this.state = state;
        this.baseURL = '';
        this.defaultTimeout = 10000;

        // Request interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];
    }

    /**
     * Make HTTP request with consistent error handling
     */
    async request(url, options = {}) {
        const config = {
            timeout: this.defaultTimeout,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Apply request interceptors
        for (const interceptor of this.requestInterceptors) {
            await interceptor(config);
        }

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout);

            const response = await fetch(this.baseURL + url, {
                ...config,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            // Apply response interceptors
            for (const interceptor of this.responseInterceptors) {
                await interceptor(response);
            }

            if (!response.ok) {
                throw new APIError(`HTTP ${response.status}: ${response.statusText}`, response.status, response);
            }

            const data = await response.json();
            return data;

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new APIError('Request timeout', 408);
            }

            if (error instanceof APIError) {
                throw error;
            }

            throw new APIError(`Network error: ${error.message}`, 0, null, error);
        }
    }

    /**
     * GET request
     */
    async get(url, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const fullUrl = queryString ? `${url}?${queryString}` : url;
        return this.request(fullUrl, { method: 'GET' });
    }

    /**
     * POST request
     */
    async post(url, data = {}) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT request
     */
    async put(url, data = {}) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     */
    async delete(url) {
        return this.request(url, { method: 'DELETE' });
    }

    // Dashboard-specific API methods

    /**
     * Get current metrics
     */
    async getMetrics() {
        try {
            return await this.get('/api/metrics');
        } catch (error) {
            console.error('❌ Error fetching metrics:', error);
            throw error;
        }
    }

    /**
     * Get evaluation history with session support
     */
    async getEvaluationHistory(options = {}) {
        const {
            limit = 1000,
            offset = 0,
            sessionId = this.state.get('currentSessionId')
        } = options;

        const params = { limit, offset };
        if (sessionId && sessionId !== 'current') {
            params.session_id = sessionId;
        }

        try {
            const result = await this.get('/api/evaluation_history', params);

            // Normalize response format (handle both old and new API formats)
            if (result.data) {
                return {
                    data: result.data,
                    pagination: result.pagination || {}
                };
            } else if (Array.isArray(result)) {
                return {
                    data: result,
                    pagination: { total: result.length, limit, offset }
                };
            } else {
                return { data: [], pagination: {} };
            }
        } catch (error) {
            console.error('❌ Error fetching evaluation history:', error);
            return { data: [], pagination: {}, error: error.message };
        }
    }

    /**
     * Get training history
     */
    async getTrainingHistory(limit = 1000) {
        try {
            return await this.get('/api/training_history', { limit });
        } catch (error) {
            console.error('❌ Error fetching training history:', error);
            return [];
        }
    }

    /**
     * Get environment information
     */
    async getEnvironmentInfo() {
        try {
            return await this.get('/api/environment_info');
        } catch (error) {
            console.error('❌ Error fetching environment info:', error);
            return {};
        }
    }

    /**
     * Get evaluation progress
     */
    async getProgress() {
        try {
            return await this.get('/api/progress');
        } catch (error) {
            console.error('❌ Error fetching progress:', error);
            return { error: error.message };
        }
    }

    /**
     * Get available sessions
     */
    async getSessions() {
        try {
            const sessions = await this.get('/api/sessions');
            return Array.isArray(sessions) ? sessions : [];
        } catch (error) {
            console.error('❌ Error fetching sessions:', error);
            return [];
        }
    }

    /**
     * Get reward breakdown analysis
     */
    async getRewardBreakdown(sessionId = this.state.get('currentSessionId')) {
        try {
            const params = {};
            if (sessionId && sessionId !== 'current') {
                params.session_id = sessionId;
            }
            return await this.get('/api/reward_breakdown', params);
        } catch (error) {
            console.error('❌ Error fetching reward breakdown:', error);
            return { error: error.message };
        }
    }

    /**
     * Get multi-rollout analysis
     */
    async getMultiRolloutAnalysis(sessionId = this.state.get('currentSessionId')) {
        try {
            const params = {};
            if (sessionId && sessionId !== 'current') {
                params.session_id = sessionId;
            }
            return await this.get('/api/multi_rollout_analysis', params);
        } catch (error) {
            console.error('❌ Error fetching multi-rollout analysis:', error);
            return { error: error.message };
        }
    }

    /**
     * Get dashboard status
     */
    async getStatus() {
        try {
            return await this.get('/api/status');
        } catch (error) {
            console.error('❌ Error fetching status:', error);
            return { error: error.message };
        }
    }

    /**
     * Batch multiple API calls
     */
    async batch(requests) {
        try {
            const promises = requests.map(({ method, url, params, data }) => {
                switch (method.toLowerCase()) {
                    case 'get':
                        return this.get(url, params);
                    case 'post':
                        return this.post(url, data);
                    case 'put':
                        return this.put(url, data);
                    case 'delete':
                        return this.delete(url);
                    default:
                        throw new Error(`Unsupported method: ${method}`);
                }
            });

            const results = await Promise.allSettled(promises);

            return results.map((result, index) => ({
                success: result.status === 'fulfilled',
                data: result.status === 'fulfilled' ? result.value : null,
                error: result.status === 'rejected' ? result.reason : null,
                request: requests[index]
            }));

        } catch (error) {
            console.error('❌ Error in batch request:', error);
            throw error;
        }
    }

    /**
     * Add request interceptor
     */
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }

    /**
     * Add response interceptor
     */
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    /**
     * Set default timeout
     */
    setTimeout(timeout) {
        this.defaultTimeout = timeout;
    }
}

/**
 * Custom API Error class
 */
class APIError extends Error {
    constructor(message, status = 0, response = null, originalError = null) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.response = response;
        this.originalError = originalError;
    }

    /**
     * Check if error is due to network issues
     */
    isNetworkError() {
        return this.status === 0 || this.status === 408;
    }

    /**
     * Check if error is a server error (5xx)
     */
    isServerError() {
        return this.status >= 500 && this.status < 600;
    }

    /**
     * Check if error is a client error (4xx)
     */
    isClientError() {
        return this.status >= 400 && this.status < 500;
    }
}

// Export classes
window.APIClient = APIClient;
window.APIError = APIError;
