/**
 * WebSocket connection management for real-time dashboard updates
 */
class WebSocketManager {
    constructor(state) {
        this.state = state;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000;
        this.isConnecting = false;

        // Bind methods to preserve context
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.onOpen = this.onOpen.bind(this);
        this.onMessage = this.onMessage.bind(this);
        this.onClose = this.onClose.bind(this);
        this.onError = this.onError.bind(this);
    }

    /**
     * Establish WebSocket connection
     */
    connect() {
        if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
            return;
        }

        this.isConnecting = true;

        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = this.onOpen;
            this.ws.onmessage = this.onMessage;
            this.ws.onclose = this.onClose;
            this.ws.onerror = this.onError;

            // Store in state
            this.state.set('ws', this.ws);

        } catch (error) {
            console.error('❌ WebSocket connection failed:', error);
            this.isConnecting = false;
            this.scheduleReconnect();
        }
    }

    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.state.set('ws', null);
            this.state.set('isConnected', false);
        }
    }

    /**
     * Handle WebSocket open event
     */
    onOpen(event) {
        this.isConnecting = false;
        this.reconnectAttempts = 0;

        // Update state
        this.state.set('isConnected', true);

        // Update UI
        this.updateConnectionStatus(true);
    }

    /**
     * Handle incoming WebSocket messages
     */
    onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            // Trigger dashboard update with received metrics
            if (window.dashboardManager && window.dashboardManager.updateDashboard) {
                window.dashboardManager.updateDashboard(data);
            }

        } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
        }
    }

    /**
     * Handle WebSocket close event
     */
    onClose(event) {
        this.isConnecting = false;

        // Update state
        this.state.set('isConnected', false);
        this.state.set('ws', null);
        this.ws = null;

        // Update UI
        this.updateConnectionStatus(false);

        // Schedule reconnection if it wasn't a clean close
        if (event.code !== 1000) { // 1000 = normal closure
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket error event
     */
    onError(error) {
        console.error('❌ WebSocket error:', error);
        this.isConnecting = false;
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('❌ Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1); // Exponential backoff

        setTimeout(() => {
            if (!this.state.get('isConnected')) {
                this.connect();
            }
        }, delay);
    }

    /**
     * Update connection status in UI
     */
    updateConnectionStatus(isConnected) {
        const statusElement = document.getElementById('connection-status');
        const statusDot = document.querySelector('.status-dot');

        if (statusElement) {
            statusElement.textContent = isConnected ? 'Connected' : 'Disconnected';
        }

        if (statusDot) {
            statusDot.style.background = isConnected ? '#16a34a' : '#ef4444';
        }
    }

    /**
     * Send message through WebSocket
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                const data = typeof message === 'string' ? message : JSON.stringify(message);
                this.ws.send(data);
                return true;
            } catch (error) {
                console.error('❌ Error sending WebSocket message:', error);
                return false;
            }
        } else {
            console.warn('⚠️ WebSocket not connected, cannot send message');
            return false;
        }
    }

    /**
     * Get current connection state
     */
    getConnectionState() {
        if (!this.ws) return 'CLOSED';

        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'CONNECTING';
            case WebSocket.OPEN: return 'OPEN';
            case WebSocket.CLOSING: return 'CLOSING';
            case WebSocket.CLOSED: return 'CLOSED';
            default: return 'UNKNOWN';
        }
    }

    /**
     * Check if WebSocket is connected
     */
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Reset reconnection attempts (useful for manual reconnection)
     */
    resetReconnectAttempts() {
        this.reconnectAttempts = 0;
    }
}

// Export for use in other modules
window.WebSocketManager = WebSocketManager;
