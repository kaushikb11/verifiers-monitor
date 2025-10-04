/**
 * DOM manipulation utilities and helpers
 */
class DOMUtils {
    constructor() {
        // Cache frequently used elements
        this.elementCache = new Map();
        this.cacheTimeout = 5000; // 5 seconds
    }

    /**
     * Get element by ID with caching
     */
    getElementById(id, useCache = true) {
        if (useCache && this.elementCache.has(id)) {
            const cached = this.elementCache.get(id);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                return cached.element;
            }
        }

        const element = document.getElementById(id);
        if (useCache && element) {
            this.elementCache.set(id, {
                element,
                timestamp: Date.now()
            });
        }

        return element;
    }

    /**
     * Update text content of element
     */
    updateText(elementId, text) {
        const element = this.getElementById(elementId);
        if (element) {
            element.textContent = text;
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for text update`);
        return false;
    }

    /**
     * Update HTML content of element
     */
    updateHTML(elementId, html) {
        const element = this.getElementById(elementId);
        if (element) {
            element.innerHTML = html;
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for HTML update`);
        return false;
    }

    /**
     * Update element attribute
     */
    updateAttribute(elementId, attribute, value) {
        const element = this.getElementById(elementId);
        if (element) {
            element.setAttribute(attribute, value);
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for attribute update`);
        return false;
    }

    /**
     * Update element style
     */
    updateStyle(elementId, property, value) {
        const element = this.getElementById(elementId);
        if (element) {
            element.style[property] = value;
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for style update`);
        return false;
    }

    /**
     * Add CSS class to element
     */
    addClass(elementId, className) {
        const element = this.getElementById(elementId);
        if (element) {
            element.classList.add(className);
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for class addition`);
        return false;
    }

    /**
     * Remove CSS class from element
     */
    removeClass(elementId, className) {
        const element = this.getElementById(elementId);
        if (element) {
            element.classList.remove(className);
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for class removal`);
        return false;
    }

    /**
     * Toggle CSS class on element
     */
    toggleClass(elementId, className) {
        const element = this.getElementById(elementId);
        if (element) {
            element.classList.toggle(className);
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for class toggle`);
        return false;
    }

    /**
     * Show element
     */
    show(elementId, display = 'block') {
        return this.updateStyle(elementId, 'display', display);
    }

    /**
     * Hide element
     */
    hide(elementId) {
        return this.updateStyle(elementId, 'display', 'none');
    }

    /**
     * Check if element exists
     */
    exists(elementId) {
        return this.getElementById(elementId) !== null;
    }

    /**
     * Get element value (for inputs)
     */
    getValue(elementId) {
        const element = this.getElementById(elementId);
        if (element) {
            return element.value || element.textContent || '';
        }
        return null;
    }

    /**
     * Set element value (for inputs)
     */
    setValue(elementId, value) {
        const element = this.getElementById(elementId);
        if (element) {
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA' || element.tagName === 'SELECT') {
                element.value = value;
            } else {
                element.textContent = value;
            }
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for value update`);
        return false;
    }

    /**
     * Create element with attributes and content
     */
    createElement(tagName, attributes = {}, content = '') {
        const element = document.createElement(tagName);

        // Set attributes
        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'className') {
                element.className = value;
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(element.style, value);
            } else {
                element.setAttribute(key, value);
            }
        });

        // Set content
        if (content) {
            if (typeof content === 'string') {
                element.innerHTML = content;
            } else if (content instanceof Node) {
                element.appendChild(content);
            }
        }

        return element;
    }

    /**
     * Append element to parent
     */
    appendChild(parentId, child) {
        const parent = this.getElementById(parentId);
        if (parent) {
            parent.appendChild(child);
            return true;
        }
        console.warn(`⚠️ Parent element '${parentId}' not found for appendChild`);
        return false;
    }

    /**
     * Remove element
     */
    removeElement(elementId) {
        const element = this.getElementById(elementId);
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
            this.elementCache.delete(elementId);
            return true;
        }
        return false;
    }

    /**
     * Clear element content
     */
    clearContent(elementId) {
        const element = this.getElementById(elementId);
        if (element) {
            element.innerHTML = '';
            return true;
        }
        return false;
    }

    /**
     * Add event listener to element
     */
    addEventListener(elementId, event, handler, options = {}) {
        const element = this.getElementById(elementId);
        if (element) {
            element.addEventListener(event, handler, options);
            return true;
        }
        console.warn(`⚠️ Element '${elementId}' not found for event listener`);
        return false;
    }

    /**
     * Remove event listener from element
     */
    removeEventListener(elementId, event, handler) {
        const element = this.getElementById(elementId);
        if (element) {
            element.removeEventListener(event, handler);
            return true;
        }
        return false;
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        if (typeof text !== 'string') return text;

        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Format number with commas
     */
    formatNumber(num, decimals = 0) {
        if (typeof num !== 'number') return num;
        return num.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    /**
     * Format percentage
     */
    formatPercentage(value, decimals = 1) {
        if (typeof value !== 'number') return value;
        return `${(value * 100).toFixed(decimals)}%`;
    }

    /**
     * Format time duration
     */
    formatDuration(milliseconds) {
        if (typeof milliseconds !== 'number') return milliseconds;

        if (milliseconds < 1000) {
            return `${Math.round(milliseconds)}ms`;
        } else if (milliseconds < 60000) {
            return `${(milliseconds / 1000).toFixed(1)}s`;
        } else if (milliseconds < 3600000) {
            const minutes = Math.floor(milliseconds / 60000);
            const seconds = Math.floor((milliseconds % 60000) / 1000);
            return `${minutes}m ${seconds}s`;
        } else {
            const hours = Math.floor(milliseconds / 3600000);
            const minutes = Math.floor((milliseconds % 3600000) / 60000);
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (typeof bytes !== 'number') return bytes;

        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let size = bytes;
        let unitIndex = 0;

        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }

        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }

    /**
     * Animate element (simple fade in/out)
     */
    fadeIn(elementId, duration = 300) {
        const element = this.getElementById(elementId);
        if (!element) return Promise.reject(new Error('Element not found'));

        return new Promise(resolve => {
            element.style.opacity = '0';
            element.style.display = 'block';

            const start = performance.now();
            const animate = (currentTime) => {
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);

                element.style.opacity = progress.toString();

                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };

            requestAnimationFrame(animate);
        });
    }

    /**
     * Animate element fade out
     */
    fadeOut(elementId, duration = 300) {
        const element = this.getElementById(elementId);
        if (!element) return Promise.reject(new Error('Element not found'));

        return new Promise(resolve => {
            const start = performance.now();
            const startOpacity = parseFloat(element.style.opacity) || 1;

            const animate = (currentTime) => {
                const elapsed = currentTime - start;
                const progress = Math.min(elapsed / duration, 1);

                element.style.opacity = (startOpacity * (1 - progress)).toString();

                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    element.style.display = 'none';
                    resolve();
                }
            };

            requestAnimationFrame(animate);
        });
    }

    /**
     * Scroll element into view smoothly
     */
    scrollIntoView(elementId, options = { behavior: 'smooth', block: 'center' }) {
        const element = this.getElementById(elementId);
        if (element) {
            element.scrollIntoView(options);
            return true;
        }
        return false;
    }

    /**
     * Get element dimensions
     */
    getDimensions(elementId) {
        const element = this.getElementById(elementId);
        if (element) {
            const rect = element.getBoundingClientRect();
            return {
                width: rect.width,
                height: rect.height,
                top: rect.top,
                left: rect.left,
                right: rect.right,
                bottom: rect.bottom
            };
        }
        return null;
    }

    /**
     * Clear element cache
     */
    clearCache() {
        this.elementCache.clear();
    }

    /**
     * Batch DOM updates to improve performance
     */
    batchUpdate(updates) {
        // Use document fragment for multiple DOM operations
        const fragment = document.createDocumentFragment();
        const results = [];

        updates.forEach(update => {
            try {
                const result = this[update.method](...update.args);
                results.push({ success: true, result });
            } catch (error) {
                results.push({ success: false, error: error.message });
            }
        });

        return results;
    }

    /**
     * Debounce function for performance
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Throttle function for performance
     */
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Export for use in other modules
window.DOMUtils = DOMUtils;
