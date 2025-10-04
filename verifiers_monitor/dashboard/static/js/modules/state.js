/**
 * Centralized state management for the dashboard
 */
class DashboardState {
    constructor() {
        this.state = {
            // WebSocket connection
            ws: null,
            isConnected: false,

            // Session management
            currentSessionId: 'current',
            availableSessions: [],
            sessionData: {},

            // Example navigation
            currentExampleIndex: 0,
            exampleData: [],
            filteredData: [],
            currentFilter: 'all',
            userIsNavigating: false,

            // Pagination state (per-section)
            pagination: {
                enabled: true,  // false = "Show All" mode
                failedItemsPerPage: 10,
                passedItemsPerPage: 10
            },

            // Progress tracking
            progress: {
                completed: 0,
                total: 0,
                percentage: 0,
                eta_seconds: 0,
                throughput: 0
            },

            // Environment info
            environmentInfo: {},

            // UI state
            isInitialized: false,
            lastUpdateTime: null
        };

        this.listeners = new Map();
    }

    /**
     * Get current state or specific property
     */
    get(path) {
        if (!path) return { ...this.state };

        return path.split('.').reduce((obj, key) => obj?.[key], this.state);
    }

    /**
     * Set state property and notify listeners
     */
    set(path, value) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        const target = keys.reduce((obj, key) => {
            if (!obj[key]) obj[key] = {};
            return obj[key];
        }, this.state);

        const oldValue = target[lastKey];
        target[lastKey] = value;

        // Notify listeners
        this.notifyListeners(path, value, oldValue);

        return this;
    }

    /**
     * Update nested object properties
     */
    update(path, updates) {
        const current = this.get(path) || {};
        const newValue = { ...current, ...updates };
        this.set(path, newValue);
        return this;
    }

    /**
     * Subscribe to state changes
     */
    subscribe(path, callback) {
        if (!this.listeners.has(path)) {
            this.listeners.set(path, new Set());
        }
        this.listeners.get(path).add(callback);

        // Return unsubscribe function
        return () => {
            const pathListeners = this.listeners.get(path);
            if (pathListeners) {
                pathListeners.delete(callback);
                if (pathListeners.size === 0) {
                    this.listeners.delete(path);
                }
            }
        };
    }

    /**
     * Notify listeners of state changes
     */
    notifyListeners(path, newValue, oldValue) {
        // Notify exact path listeners
        const pathListeners = this.listeners.get(path);
        if (pathListeners) {
            pathListeners.forEach(callback => {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    console.error('Error in state listener:', error);
                }
            });
        }

        // Notify parent path listeners (e.g., 'progress' when 'progress.completed' changes)
        const pathParts = path.split('.');
        for (let i = pathParts.length - 1; i > 0; i--) {
            const parentPath = pathParts.slice(0, i).join('.');
            const parentListeners = this.listeners.get(parentPath);
            if (parentListeners) {
                const parentValue = this.get(parentPath);
                parentListeners.forEach(callback => {
                    try {
                        callback(parentValue, undefined, parentPath);
                    } catch (error) {
                        console.error('Error in parent state listener:', error);
                    }
                });
            }
        }
    }

    /**
     * Reset state to initial values
     */
    reset() {
        const initialState = {
            ws: null,
            isConnected: false,
            currentSessionId: 'current',
            availableSessions: [],
            sessionData: {},
            currentExampleIndex: 0,
            exampleData: [],
            filteredData: [],
            currentFilter: 'all',
            userIsNavigating: false,
            pagination: {
                enabled: true,
                failedItemsPerPage: 10,
                passedItemsPerPage: 10
            },
            progress: {
                completed: 0,
                total: 0,
                percentage: 0,
                eta_seconds: 0,
                throughput: 0
            },
            environmentInfo: {},
            isInitialized: false,
            lastUpdateTime: null
        };

        Object.keys(initialState).forEach(key => {
            this.set(key, initialState[key]);
        });
    }

    /**
     * Get filtered examples based on current filter
     */
    getFilteredExamples() {
        const examples = this.get('exampleData');
        const filter = this.get('currentFilter');

        switch (filter) {
            case 'failed':
                // Use aggregate.min to determine if any rollout failed
                return examples.filter(e => {
                    const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
                    return minReward <= 0.5;
                });
            case 'passed':
                // Use aggregate.min to determine if all rollouts passed
                return examples.filter(e => {
                    const minReward = e.aggregate ? e.aggregate.min : (e.reward || 0);
                    return minReward > 0.5;
                });
            default:
                return [...examples];
        }
    }

    /**
     * Update filtered data and adjust current index if needed
     */
    updateFilteredData() {
        const filteredData = this.getFilteredExamples();
        this.set('filteredData', filteredData);

        // Adjust current index if it's out of bounds
        const currentIndex = this.get('currentExampleIndex');
        if (currentIndex >= filteredData.length) {
            this.set('currentExampleIndex', Math.max(0, filteredData.length - 1));
        }
    }

    /**
     * Get current example being viewed
     */
    getCurrentExample() {
        const filteredData = this.get('filteredData');
        const currentIndex = this.get('currentExampleIndex');
        return filteredData[currentIndex] || null;
    }

    /**
     * Navigate to next/previous example
     */
    navigateExample(direction) {
        const filteredData = this.get('filteredData');
        const currentIndex = this.get('currentExampleIndex');
        const newIndex = currentIndex + direction;

        if (newIndex >= 0 && newIndex < filteredData.length) {
            this.set('currentExampleIndex', newIndex);
            return true;
        }
        return false;
    }

    /**
     * Set filter and update filtered data
     */
    setFilter(filterType) {
        this.set('currentFilter', filterType);
        this.set('currentExampleIndex', 0); // Reset to first example
        this.updateFilteredData();
    }

    /**
     * Debug helper to log current state
     */
    debug() {
        // Debug logging removed for production
    }
}

// Export singleton instance
window.DashboardState = DashboardState;
